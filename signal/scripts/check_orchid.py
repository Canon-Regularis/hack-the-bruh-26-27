"""Confirm the orchid entry points named in v2 section 2 and print their signatures.

Usage:
    python3 signal/scripts/check_orchid.py [DIR_CONTAINING_THE_ORCHID_PACKAGE]

Exits 0 only if every entry point is found. This only reads orchid, it never
changes it. Not part of check.sh, because orchid is not always installed.
"""

import importlib
import inspect
import sys

# v2 gives orchid/algorithms/domains/urban for the urban modules, but writes
# core/ml/... with no prefix, so the core modules are tried at both locations.
URBAN = ["orchid.algorithms.domains.urban.{}"]
CORE = ["orchid.algorithms.core.ml.{}", "orchid.core.ml.{}"]

# (module name, candidate import paths, names v2 relies on)
ENTRY_POINTS = [
    (
        "street_network",
        URBAN,
        ["RoadGraph", "clean_network", "EdgeIndex", "map_match", "snap_point"],
    ),
    ("data_splitting", CORE, ["GroupKFold", "TimeSeriesSplit"]),
    ("ensembles", CORE, ["GradientBoostingRegressor"]),
    ("traffic_assignment", URBAN, ["user_equilibrium"]),
]


def import_first(module_name, templates):
    """Return the first candidate module that imports, or None."""
    for template in templates:
        try:
            return importlib.import_module(template.format(module_name))
        except ImportError:
            continue
    return None


def describe(obj):
    """Return a printable signature for a function or class."""
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return "(signature not available)"


def main():
    if len(sys.argv) > 1:
        sys.path.insert(0, sys.argv[1])

    missing = 0
    for module_name, templates, names in ENTRY_POINTS:
        module = import_first(module_name, templates)
        if module is None:
            print(f"MISSING MODULE {module_name}: tried {templates}")
            missing += len(names)
            continue
        print(f"module {module.__name__}")
        for name in names:
            if hasattr(module, name):
                print(f"  {name}{describe(getattr(module, name))}")
            else:
                print(f"  MISSING {name}")
                missing += 1

    print(f"missing entry points: {missing}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
