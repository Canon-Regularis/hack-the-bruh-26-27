"""Compute sampled edge betweenness on the frozen road graph once, and save it.

Run from the repo root, with the same import paths as pytest.ini:
    PYTHONPATH=.:data python3 scripts/compute_betweenness.py

Writes data/derived/betweenness.json, which git ignores.
"""

import json
import time
from pathlib import Path

from features.betweenness import SAMPLE_SOURCES, sampled_betweenness
from src.graph.build_graph import load_graph

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "derived" / "betweenness.json"


def main():
    graph = load_graph()
    started = time.monotonic()
    values = sampled_betweenness(graph)
    seconds = time.monotonic() - started
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(values), encoding="utf-8")
    print(f"{len(values)} edges, {SAMPLE_SOURCES} sampled sources, {seconds:.0f} s -> {OUT_PATH}")


if __name__ == "__main__":
    main()
