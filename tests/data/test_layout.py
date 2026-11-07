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
    ["raw/pings.jsonl", "derived/speeds.parquet", ".env"],
)
def test_sensitive_paths_are_gitignored(path):
    assert is_ignored(path)


def test_sample_data_is_not_gitignored():
    assert not is_ignored("sample/pings.jsonl")
