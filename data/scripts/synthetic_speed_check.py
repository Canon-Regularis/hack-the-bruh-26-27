"""Print the speeds recovered from the fake bus. All numbers here are synthetic.

Usage: python3 data/scripts/synthetic_speed_check.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from speeds.dwell import is_dwell  # noqa: E402
from speeds.edge_speeds import edge_speed_table  # noqa: E402
from speeds.synthetic import STOP_EDGE, TRUE_SPEED_KMH, synthetic_bus  # noqa: E402

CASES = [
    ("dwell lines up with pings, no noise", 0, 0.0),
    ("dwell falls between pings, 2 m GPS noise", 15, 2.0),
]


def dwell_detection_rate(noise_m, runs=200):
    """Return the share of fully stationary pairs recognised as dwell."""
    caught = total = 0
    for seed in range(runs):
        pairs, stops, _ = synthetic_bus(0, noise_m, seed)
        stationary = [p for p in pairs if sum(m for _, m in p.edges) == 0]
        caught += sum(is_dwell(p, stops) for p in stationary)
        total += len(stationary)
    return caught / total


def main():
    print(f"SYNTHETIC. True speed {TRUE_SPEED_KMH} km/h, stop on {STOP_EDGE}.")
    for label, first_ping_s, noise_m in CASES:
        pairs, stops, names = synthetic_bus(first_ping_s, noise_m)
        table = edge_speed_table(pairs, stops, names)
        print(f"\n{label}")
        print(f"  ping pairs: {len(pairs)}")
        for row in table.itertuples():
            print(
                f"  {row.edge_id}: {row.mean_bus_observed_speed_kmh:.2f} km/h "
                f"({row.total_metres:.0f} m in {row.total_seconds:.0f} s, "
                f"dwell {row.dwell_seconds:.0f} s)"
            )

    print("\nStationary pairs recognised as dwell, 200 runs per noise level")
    for noise_m in (2.0, 5.0, 10.0):
        print(f"  {noise_m:.0f} m GPS noise: {dwell_detection_rate(noise_m):.1%}")


if __name__ == "__main__":
    main()
