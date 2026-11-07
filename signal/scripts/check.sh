#!/usr/bin/env bash
# One command to check all Signal work. Exits non-zero on any failure.
# Usage: signal/scripts/check.sh   (works from any directory)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON="${PYTHON:-python3}"

echo "== (a) lint =="
"$PYTHON" -m ruff check .

echo "== (b) tests =="
"$PYTHON" -m pytest -q

echo "== (c) smoke =="
# scripts/smoke.sh arrives once two pipeline stages exist. It runs the fixed
# sample in data/sample/ from raw pings to the latest finished stage.
if [[ -e scripts/smoke.sh ]]; then
  bash scripts/smoke.sh
else
  echo "SKIPPED: no scripts/smoke.sh yet (fewer than two pipeline stages exist)"
fi

echo "check.sh: PASSED"
