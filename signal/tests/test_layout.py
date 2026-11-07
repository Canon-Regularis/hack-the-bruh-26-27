"""Guards on the repo layout that would otherwise fail in confusing ways."""

import subprocess
from pathlib import Path

import pytest

SIGNAL_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_DIRS = [
    "collector",
    "matching",
    "speeds",
    "features",
    "model",
    "export",
    "tests",
    "data/raw",
    "data/derived",
    "data/sample",
    "reports",
    "scripts",
]


def test_signal_folder_is_not_a_package():
    # signal/__init__.py would shadow Python's built-in `signal` module for
    # anything run from the repo root, which breaks subprocess and asyncio.
    assert not (SIGNAL_ROOT / "__init__.py").exists()


@pytest.mark.parametrize("folder", EXPECTED_DIRS)
def test_expected_folder_exists(folder):
    assert (SIGNAL_ROOT / folder).is_dir()


@pytest.mark.parametrize(
    "path",
    ["data/raw/pings.jsonl", "data/derived/speeds.parquet", ".env"],
)
def test_sensitive_paths_are_gitignored(path):
    # `git check-ignore` exits 0 only when the path is ignored.
    result = subprocess.run(
        ["git", "check-ignore", "-q", path], cwd=SIGNAL_ROOT, check=False
    )
    assert result.returncode == 0


def test_sample_data_is_not_gitignored():
    result = subprocess.run(
        ["git", "check-ignore", "-q", "data/sample/pings.jsonl"],
        cwd=SIGNAL_ROOT,
        check=False,
    )
    assert result.returncode == 1
