#!/usr/bin/env python3
"""Vendor the hardened ORCHID modules into vendor/orchid/, and check the copy.

    python scripts/sync_orchid.py [--orchid PATH] [--dry-run]   # (re)write vendor/orchid at ORCHID_COMMIT
    python scripts/sync_orchid.py --check [--orchid PATH]       # verify vendor/orchid (offline unless --orchid)

Files are copied byte for byte from git objects in a local ORCHID clone (path from --orchid,
else $ORCHID_DIR, else ../orchid next to this repo).  The clone's working tree and branch are
never touched: only `git rev-parse`, `git cat-file` and `git ls-tree` are run against it.

Each vendored module brings __init__.py, README.md (its API docs) and <name>_template.py;
_stress_test.py and requirements.txt are not copied.  Optional `algorithms.*` imports inside the
templates are followed automatically (they are all guarded with local fallbacks); a dependency
listed in HELD is left out, which is only allowed when every import of it is guarded.

To enable a held module once its ORCHID suite has converged: move its line from HELD to
URBAN_READY (or CORE_EXTRA), add its import to engine/orchid.py, re-run this script, run
--check, and sanity-check a known answer before showing any number from it.

Exit codes: 0 ok, 1 check failed, 2 source or config error.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath

# ------------------------------------------------------------------- config (edit here) ---
ORCHID_REPO = "https://github.com/Orthant-Research-Group/orchid"
ORCHID_COMMIT = "81022c75f9bfc039f713efdd8fa056fd64c429c8"  # origin/main, merge of PR #10 (policy_compiler rounds 6-9, public_voice rounds 6-7)
FILES_PER_MODULE = ("__init__.py", "README.md", "{name}_template.py")

URBAN = "algorithms/domains/urban/"

# Urban modules whose ORCHID suites are complete (URBAN_HANDOVER.md s1; URBAN_PICKUP.md baseline).
URBAN_READY = {
    URBAN + "street_network": "plan s2: RoadGraph, HIGHWAY_DEFAULTS, clean_network, EdgeIndex, map_match, snap_point, to_geojson",
    URBAN + "traffic_assignment": "plan s2: user_equilibrium, detect_braess (suite pins Braess 83/498 -> 92/552)",
    URBAN + "travel_demand": "plan s2: gravity_model, gls_matrix_estimation",
    URBAN + "pedestrian_flow": "plan s2: crossing_delay, fruin_los",
    URBAN + "emissions_environment": "plan s0: display-only emissions layer",
    URBAN + "traffic_flow": "hardened; not called by the plan",
    URBAN + "accessibility_equity": "hardened; not called by the plan",
    URBAN + "signal_control": "hardened; not called by the plan",
    # Enabled by decision on 2026-11-07 with hunt rounds 6-8 merged (ORCHID #4, #6, #7). Not converged:
    # classify_wording (the wording_ambiguous gate) still finds bugs, so set clause strength by hand.
    URBAN + "policy_compiler": "plan s2 rules: compile_policy, verify_span, Gazetteer, AbstentionQueue, policy_check",
    # Vendored BEFORE convergence at the team's request (2026-11-07): hunt rounds 6 and 7 found and
    # fixed 5 and 4 template bugs (history 12, 8, 5, 5, 5, 5, 4), so a later round may find more.
    # Sanity-pin every number shown from it.  Round 6 is ORCHID PR #5 (merged); round 7 is PR #8.
    URBAN + "public_voice": (
        "plan s2/s4 (NOT converged, see comment): ingest_concerns, classify_theme, find_near_duplicates, "
        "concern_field, local_getis_ord, hotspot_clusters, voice_objective, voice_improvement"),
}

# Core modules the plan calls directly.
CORE_EXTRA = {
    "algorithms/core/ml/data_splitting": "plan s3: GroupKFold, TimeSeriesSplit",
    "algorithms/core/ml/ensembles": "plan s3: GradientBoostingRegressor fallback (no quantile loss)",
    "algorithms/core/optimisation/multi_objective": "plan s4: hypervolume, knee_point, find_pareto_front",
    "algorithms/core/optimisation/stochastic_optimisation": "plan s4: scenario_reduction, conditional_value_at_risk",
}

# Never copied.  Importing one through engine.orchid raises ImportError with this reason.
HELD = {
    URBAN + "network_design": "pending: not converged, bug-hunt round 3 still found 4 template bugs (URBAN_PICKUP.md)",
    URBAN + "resilience": "pending: not converged, bug-hunt round 7 still found 4 template bugs (URBAN_PICKUP.md)",
    URBAN + "green_infrastructure": (
        "held: green_min_cut_tree's adapter assumes parent < child, but core gomory_hu_tree (fixed in "
        "07a4271) can return parent > child, so .parent/.cut_value are wrong whenever core "
        "network_optimisation is importable; the plan does not use green_infrastructure"),
    "algorithms/core/probabilistic": (
        "held: documented, unfixed erf defect (URBAN_HANDOVER.md s5); emissions_environment and "
        "travel_demand fall back to their own stdlib code"),
}

# Empty in ORCHID at ORCHID_COMMIT (each dir holds only a 0-byte __init__.py).
NOT_BUILT = (
    "urban_optimisation", "mosaic_pipeline", "transit", "active_travel", "safety",
    "land_use", "parking_curbside", "realtime_operations", "microsimulation", "geospatial_urban",
)
NOT_BUILT_REASON = ("not built in ORCHID yet; plan v2 s0 replaces urban_optimisation and "
                    "mosaic_pipeline with team glue")
# -------------------------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
VENDOR = REPO_ROOT / "vendor" / "orchid"
FACADE = REPO_ROOT / "engine" / "orchid.py"
GENERATED = ("MANIFEST.json", "README.md")
ALLOWED_GIT = ("rev-parse", "cat-file", "ls-tree")
_TRY_NODES = tuple(t for t in (getattr(ast, "Try", None), getattr(ast, "TryStar", None)) if t)
_IMPORT_ERRORS = ("ImportError", "ModuleNotFoundError", "Exception", "BaseException")


class SyncError(Exception):
    """A source or configuration problem (exit code 2)."""


# ------------------------------------------------------------------------------ helpers ---
def short(path: str) -> str:
    return PurePosixPath(path).name


def template_path(mod: str) -> str:
    return "%s/%s_template.py" % (mod, short(mod))


def import_name(mod: str) -> str:
    return "%s.%s_template" % (mod.replace("/", "."), short(mod))


def blob_id(data: bytes) -> str:
    """Git's blob id for raw bytes, so files can be checked without git."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def git(orchid: Path, *args: str) -> bytes:
    """Run an allowlisted, read-only git command in the ORCHID clone; return raw stdout."""
    if not args or args[0] not in ALLOWED_GIT:
        raise SyncError("refusing to run git %r (allowed: %s)" % (args[:1], ", ".join(ALLOWED_GIT)))
    try:
        proc = subprocess.run(["git", "-C", str(orchid), *args], capture_output=True)
    except FileNotFoundError:
        raise SyncError("git is not on PATH")
    if proc.returncode != 0:
        raise SyncError("git %s failed: %s" % (" ".join(args), proc.stderr.decode("utf-8", "replace").strip()))
    return proc.stdout


def locate_orchid(arg: Path | None) -> Path:
    raw = arg or (Path(os.environ["ORCHID_DIR"]) if os.environ.get("ORCHID_DIR") else REPO_ROOT.parent / "orchid")
    orchid = raw.expanduser().resolve()
    if not (orchid / ".git").exists():
        raise SyncError("no git clone at %s (pass --orchid PATH or set ORCHID_DIR)" % orchid)
    if orchid == REPO_ROOT or REPO_ROOT in orchid.parents or orchid in REPO_ROOT.parents:
        raise SyncError("ORCHID clone %s and this repo %s must not contain each other" % (orchid, REPO_ROOT))
    return orchid


def resolve_commit(orchid: Path) -> str:
    try:
        sha = git(orchid, "rev-parse", "--verify", "--quiet", ORCHID_COMMIT + "^{commit}").decode().strip()
    except SyncError:
        raise SyncError("commit %s is not in %s -- run `git -C %s fetch origin` first (this script never fetches)"
                        % (ORCHID_COMMIT[:7], orchid, orchid))
    if sha != ORCHID_COMMIT:
        raise SyncError("ORCHID_COMMIT must be a full 40-character SHA (got %s -> %s)" % (ORCHID_COMMIT, sha))
    return sha


def list_tree(orchid: Path, commit: str) -> dict:
    """{path: (blob_id, size)} for every blob under algorithms/ at the commit."""
    out = git(orchid, "ls-tree", "-r", "-l", "-z", commit, "--", "algorithms")
    tree = {}
    for rec in out.split(b"\0"):
        if not rec:
            continue
        meta, _, path = rec.partition(b"\t")
        _mode, kind, oid, size = meta.split()
        if kind == b"blob":
            tree[path.decode("utf-8")] = (oid.decode("ascii"), int(size))
    if not tree:
        raise SyncError("no algorithms/ tree at %s" % commit)
    return tree


# ------------------------------------------------------------------ dependency closure ---
def _catches_import_error(handler: ast.ExceptHandler) -> bool:
    t = handler.type
    if t is None:  # bare `except:`
        return True
    for e in (t.elts if isinstance(t, ast.Tuple) else [t]):
        name = e.id if isinstance(e, ast.Name) else getattr(e, "attr", None)
        if name in _IMPORT_ERRORS:
            return True
    return False


def _is_guarded(node: ast.AST, parents: dict) -> bool:
    """True if the import sits in the body of a try whose handlers catch ImportError."""
    child, cur = node, parents.get(node)
    while cur is not None:
        if isinstance(cur, _TRY_NODES) and any(child is n for n in cur.body):
            if any(_catches_import_error(h) for h in cur.handlers):
                return True
        child, cur = cur, parents.get(cur)
    return False


def _module_dir(dotted: str, tree: dict) -> str | None:
    parts = dotted.split(".")
    if parts[-1].endswith("_template"):
        parts = parts[:-1]
    mod = "/".join(parts)
    return mod if template_path(mod) in tree else None


def algorithms_imports(src: bytes, where: str, tree: dict) -> list:
    """[(module_dir, line, guarded)] for every `algorithms.*` import in a template."""
    root = ast.parse(src, filename=where)
    parents = {}
    for node in ast.walk(root):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    found = []
    for node in ast.walk(root):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "__import__":
            raise SyncError("%s:%d uses __import__ -- cannot resolve its dependencies safely" % (where, node.lineno))
        if isinstance(node, ast.ImportFrom):
            if node.level:
                raise SyncError("%s:%d relative import -- cannot vendor safely" % (where, node.lineno))
            module = node.module or ""
            if module == "importlib" or module.startswith("importlib."):
                raise SyncError("%s:%d imports importlib -- cannot resolve its dependencies safely" % (where, node.lineno))
            if module != "algorithms" and not module.startswith("algorithms."):
                continue
            dirs = {_module_dir(module, tree)} - {None}
            if not dirs:  # form: from algorithms.x.y import y_template
                dirs = {_module_dir("%s.%s" % (module, a.name), tree) for a in node.names} - {None}
            if not dirs:
                raise SyncError("%s:%d cannot resolve `from %s import ...` to a module dir" % (where, node.lineno, module))
        elif isinstance(node, ast.Import):
            names = [a.name for a in node.names]
            if any(n == "importlib" or n.startswith("importlib.") for n in names):
                raise SyncError("%s:%d imports importlib -- cannot resolve its dependencies safely" % (where, node.lineno))
            names = [n for n in names if n == "algorithms" or n.startswith("algorithms.")]
            if not names:
                continue
            dirs = {_module_dir(n, tree) for n in names}
            if None in dirs:
                raise SyncError("%s:%d cannot resolve `import %s` to a module dir" % (where, node.lineno, ", ".join(names)))
        else:
            continue
        guarded = _is_guarded(node, parents)
        found.extend((d, node.lineno, guarded) for d in sorted(dirs))
    return found


class Source:
    """Read-only view of the ORCHID clone at ORCHID_COMMIT, with a blob cache."""

    def __init__(self, orchid: Path):
        self.orchid = orchid
        self.commit = resolve_commit(orchid)
        self.tree = list_tree(orchid, self.commit)
        self._blobs = {}

    def read(self, path: str) -> bytes:
        if path not in self._blobs:
            if path not in self.tree:
                raise SyncError("%s does not exist at %s" % (path, self.commit[:7]))
            oid, size = self.tree[path]
            data = git(self.orchid, "cat-file", "blob", oid)  # raw bytes: no eol/textconv filters
            if len(data) != size or blob_id(data) != oid:
                raise SyncError("blob check failed for %s" % path)
            self._blobs[path] = data
        return self._blobs[path]


def resolve_modules(src: Source) -> tuple:
    """Breadth-first closure from the seeds.  Returns (modules, held_skips)."""
    seeds = {m: ("urban", r) for m, r in URBAN_READY.items()}
    seeds.update({m: ("core", r) for m, r in CORE_EXTRA.items()})
    clash = sorted(set(seeds) & set(HELD))
    if clash:
        raise SyncError("listed as both ready and held: %s" % ", ".join(clash))
    for m in seeds:
        if template_path(m) not in src.tree:
            raise SyncError("%s has no %s at %s" % (m, short(template_path(m)), src.commit[:7]))

    modules = {m: {"role": role, "reason": reason, "requires": set(), "required_by": set()}
               for m, (role, reason) in seeds.items()}
    held_skips = {}
    queue = sorted(seeds)
    while queue:
        mod = queue.pop(0)
        for dep, line, guarded in algorithms_imports(src.read(template_path(mod)), template_path(mod), src.tree):
            if dep == mod:
                continue
            if dep in HELD:
                if not guarded:
                    raise SyncError("%s:%d imports held module %s WITHOUT an ImportError guard -- it cannot be "
                                    "left out; vendor it or hold %s too" % (template_path(mod), line, dep, mod))
                held_skips.setdefault(dep, set()).add(mod)
                continue
            if dep not in modules:
                modules[dep] = {"role": "dependency", "reason": "optional core import of vendored modules",
                                "requires": set(), "required_by": set()}
                queue.append(dep)
            modules[mod]["requires"].add(dep)
            modules[dep]["required_by"].add(mod)

    by_name = {}
    for m in sorted(modules):
        if short(m) in by_name:
            raise SyncError("two vendored modules share the short name %r: %s and %s" % (short(m), by_name[short(m)], m))
        by_name[short(m)] = m
    return modules, held_skips


def file_set(modules: dict, tree: dict) -> list:
    files = set()
    for mod in modules:
        for pattern in FILES_PER_MODULE:
            path = "%s/%s" % (mod, pattern.format(name=short(mod)))
            if path not in tree:
                raise SyncError("%s is missing upstream" % path)
            files.add(path)
        parts = mod.split("/")
        for i in range(1, len(parts)):  # package markers, only where they exist upstream
            init = "/".join(parts[:i]) + "/__init__.py"
            if init in tree:
                files.add(init)
    return sorted(files)


# ----------------------------------------------------------------- manifest and README ---
def build_manifest(src: Source, modules: dict, held_skips: dict, files: list) -> dict:
    return {
        "generated_by": "scripts/sync_orchid.py -- do not edit; change the lists in that script and re-run it",
        "source": {"repo": ORCHID_REPO, "commit": src.commit},
        "files_per_module": list(FILES_PER_MODULE),
        "modules": [{
            "name": short(m), "path": m, "import": import_name(m), "role": info["role"], "reason": info["reason"],
            "requires": sorted(short(x) for x in info["requires"]),
            "required_by": sorted(short(x) for x in info["required_by"]),
        } for m, info in sorted(modules.items())],
        "held": [{
            "name": short(m), "path": m, "reason": reason,
            "optional_import_skipped_by": sorted(short(x) for x in held_skips.get(m, ())),
        } for m, reason in sorted(HELD.items())],
        "not_built": [{"name": n, "path": URBAN + n, "reason": NOT_BUILT_REASON} for n in NOT_BUILT],
        "files": {p: src.tree[p][0] for p in files},
    }


def build_readme(manifest: dict) -> str:
    lines = [
        "# vendor/orchid -- generated, do not edit",
        "",
        "Verbatim copies of ORCHID (%s) at commit `%s`, written by `scripts/sync_orchid.py`."
        % (manifest["source"]["repo"], manifest["source"]["commit"]),
        "Import them through `engine/orchid.py`, e.g. `from engine.orchid import street_network`.",
        "",
        "To change what is vendored, edit the lists at the top of `scripts/sync_orchid.py`, re-run it,",
        "then run `python scripts/sync_orchid.py --check`. `MANIFEST.json` records every module, why it",
        "is here, what is held back and why, and the git blob id of every file.",
        "",
        "| Module | Role | Why |",
        "|---|---|---|",
    ]
    lines += ["| `%s` | %s | %s |" % (m["name"], m["role"], m["reason"]) for m in manifest["modules"]]
    lines += ["", "Held back (not copied):", ""]
    lines += ["- `%s` -- %s" % (h["name"], h["reason"]) for h in manifest["held"]]
    return "\n".join(lines) + "\n"


def write_vendor(src: Source, files: list, manifest: dict) -> None:
    if not (REPO_ROOT / ".git").exists():
        raise SyncError("%s is not a git checkout -- refusing to write" % REPO_ROOT)
    target = VENDOR.resolve()
    if target.parent != (REPO_ROOT / "vendor").resolve() or target.name != "orchid":
        raise SyncError("refusing to write outside %s" % (REPO_ROOT / "vendor" / "orchid"))
    if target.exists():
        shutil.rmtree(target)
    for path in files:
        dest = target.joinpath(*PurePosixPath(path).parts)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(src.read(path))
    (target / "MANIFEST.json").write_bytes((json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    (target / "README.md").write_bytes(build_readme(manifest).encode("utf-8"))


# ------------------------------------------------------------------------------- check ---
def check_facade(manifest: dict) -> list:
    """engine/orchid.py must import exactly the exposed (non-dependency) modules, by short name."""
    if not FACADE.is_file():
        return ["engine/orchid.py is missing"]
    imported = {}
    for node in ast.parse(FACADE.read_bytes(), filename=str(FACADE)).body:
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("algorithms."):
            for a in node.names:
                imported["%s.%s" % (node.module, a.name)] = a.asname or a.name
    exposed = {m["import"]: m["name"] for m in manifest["modules"] if m["role"] != "dependency"}
    problems = ["engine/orchid.py does not import %s" % x for x in sorted(set(exposed) - set(imported))]
    problems += ["engine/orchid.py imports %s, which is not an exposed vendored module" % x
                 for x in sorted(set(imported) - set(exposed))]
    problems += ["engine/orchid.py imports %s as %r; expected %r" % (x, imported[x], exposed[x])
                 for x in sorted(set(imported) & set(exposed)) if imported[x] != exposed[x]]
    return problems


def check(orchid: Path | None) -> list:
    mpath = VENDOR / "MANIFEST.json"
    if not mpath.is_file():
        return ["vendor/orchid/MANIFEST.json is missing -- run: python scripts/sync_orchid.py"]
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    files = manifest["files"]
    problems = []
    for path, oid in sorted(files.items()):
        f = VENDOR.joinpath(*PurePosixPath(path).parts)
        if not f.is_file():
            problems.append("missing: %s" % path)
        elif blob_id(f.read_bytes()) != oid:
            problems.append("modified: %s" % path)
    for f in sorted(VENDOR.rglob("*")):
        if f.is_dir() or "__pycache__" in f.parts or f.suffix in (".pyc", ".pyo"):
            continue
        rel = f.relative_to(VENDOR).as_posix()
        if rel not in files and rel not in GENERATED:
            problems.append("unexpected file: %s" % rel)
    problems += ["held module present: %s" % m for m in sorted(HELD) if VENDOR.joinpath(*m.split("/")).exists()]
    if manifest["source"]["commit"] != ORCHID_COMMIT:
        problems.append("MANIFEST commit %s != ORCHID_COMMIT %s -- re-run the sync"
                        % (manifest["source"]["commit"][:7], ORCHID_COMMIT[:7]))
    seeds = set(URBAN_READY) | set(CORE_EXTRA)
    vendored = {m["path"] for m in manifest["modules"] if m["role"] != "dependency"}
    if vendored != seeds:
        problems.append("config lists differ from MANIFEST (%s) -- re-run the sync" % ", ".join(sorted(seeds ^ vendored)))
    if {h["path"] for h in manifest["held"]} != set(HELD):
        problems.append("HELD differs from MANIFEST -- re-run the sync")
    problems += check_facade(manifest)
    if orchid is not None:
        src_tree = list_tree(orchid, resolve_commit(orchid))
        problems += ["differs from ORCHID %s: %s" % (ORCHID_COMMIT[:7], p)
                     for p, oid in sorted(files.items()) if src_tree.get(p, (None,))[0] != oid]
    return problems


# -------------------------------------------------------------------------------- main ---
def report(modules: dict, held_skips: dict, files: list, src: Source) -> None:
    total = sum(src.tree[p][1] for p in files)
    for role in ("urban", "core", "dependency"):
        names = sorted(short(m) for m, info in modules.items() if info["role"] == role)
        print("%-10s (%d): %s" % (role, len(names), ", ".join(names)))
    for m, importers in sorted(held_skips.items()):
        print("held, skipped optional import of %s by: %s" % (short(m), ", ".join(sorted(short(x) for x in importers))))
    print("%d modules, %d files, %d bytes from ORCHID %s" % (len(modules), len(files), total, src.commit[:7]))


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description="Vendor hardened ORCHID modules into vendor/orchid/.",
                                 epilog=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orchid", type=Path, help="local ORCHID clone (default: $ORCHID_DIR, else ../orchid)")
    ap.add_argument("--dry-run", action="store_true", help="resolve and print what would be vendored; write nothing")
    ap.add_argument("--check", action="store_true", help="verify vendor/orchid against MANIFEST.json "
                                                         "(and against ORCHID when --orchid is given)")
    args = ap.parse_args(argv)
    try:
        if args.check:
            problems = check(locate_orchid(args.orchid) if args.orchid else None)
        else:
            src = Source(locate_orchid(args.orchid))
            modules, held_skips = resolve_modules(src)
            files = file_set(modules, src.tree)
            for path in files:
                src.read(path)  # fetch and hash-check everything before touching the disk
            report(modules, held_skips, files, src)
            if args.dry_run:
                print("dry run: nothing written")
                return 0
            write_vendor(src, files, build_manifest(src, modules, held_skips, files))
            print("wrote %s" % VENDOR.relative_to(REPO_ROOT).as_posix())
            problems = check(src.orchid)
    except SyncError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2
    for p in problems:
        print("CHECK FAILED: %s" % p)
    if problems:
        return 1
    print("check OK: vendor/orchid matches MANIFEST.json%s" % (" and ORCHID" if (args.orchid or not args.check) else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
