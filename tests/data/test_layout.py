"""Check that data and secrets stay out of git."""

import subprocess
from pathlib import Path

import pytest

DATA_ROOT = Path(__file__).resolve().parents[2] / "data"


def is_ignored(path):
    # `git check-ignore` exits 0 only when the path is ignored.
    result = subprocess.run(
        ["git", "check-ignore", "-q", path], cwd=DATA_ROOT, check=False
    )
    return result.returncode == 0


@pytest.mark.parametrize(
    "path",
    ["raw/pings.jsonl", "raw/osm.json", "raw/sources/ETRO-25-23-order.txt", "rules/clauses.json",
     "derived/speeds.parquet", "graph/graph.pkl", ".env"],
)
def test_sensitive_paths_are_gitignored(path):
    assert is_ignored(path)


def test_sample_data_is_not_gitignored():
    assert not is_ignored("sample/pings.jsonl")


@pytest.mark.parametrize("name", ["meta.json", "network.geojson", "streets.json", "rules.json", "designs.json"])
def test_the_sites_data_is_not_gitignored(name):
    assert not is_ignored(f"../frontend/public/data/{name}")


@pytest.mark.parametrize(
    "path",
    [
        "demand/od_matrix.json", "demand/baseline_user_equilibrium.csv", "demand/scenarios.json",
        "concerns/concern_field.csv", "rules/restriction_flags.csv", "rules/policy_check.json",
        "decide/design_search_checkpoint.jsonl", "decide/best_trade_offs.json", "decide/designs/knee.csv",
        "decide/candidates.json",
    ],
)
def test_pipeline_output_is_gitignored(path):
    assert is_ignored(path)
