"""Guards on the repo layout that would otherwise fail in confusing ways."""

import subprocess
from pathlib import Path

import pytest

DATA_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_DIRS = [
    "collector",
    "matching",
    "speeds",
    "features",
    "model",
    "export",
    "tests",
    "raw",
    "derived",
    "sample",
    "reports",
    "scripts",
]


@pytest.mark.parametrize("folder", EXPECTED_DIRS)
def test_expected_folder_exists(folder):
    assert (DATA_ROOT / folder).is_dir()


@pytest.mark.parametrize(
    "path",
    ["raw/pings.jsonl", "derived/speeds.parquet", ".env"],
)
def test_sensitive_paths_are_gitignored(path):
    # `git check-ignore` exits 0 only when the path is ignored.
    result = subprocess.run(
        ["git", "check-ignore", "-q", path], cwd=DATA_ROOT, check=False
    )
    assert result.returncode == 0


def test_sample_data_is_not_gitignored():
    result = subprocess.run(
        ["git", "check-ignore", "-q", "sample/pings.jsonl"],
        cwd=DATA_ROOT,
        check=False,
    )
    assert result.returncode == 1
