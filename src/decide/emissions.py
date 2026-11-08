"""Display-only hot-exhaust emissions for the three picked designs (ALG-S3-9, #40).

Per edge, orchid's COPERT fleet factors are applied to each design's baseline-conditions
UE flows and times, from #29's designs/<role>.csv, and to #24's baseline:
speed_kmh = 3.6 * length_m / ue_time_s and distance_km = length_m / 1000. With vehicles =
the UE flow in veh/h, each value is grams per hour of the modelled hour. Speeds outside a
curve's range are clamped by orchid. The fleet mix is an assumption, so every number is
labelled modelled, not measured. Not part of the optimiser.

Run from the repo root after #29: ``python -m src.decide.emissions``.
"""

import json
from pathlib import Path

import pandas as pd

from engine.orchid import emissions_environment as ee
from src.decide.baseline_user_equilibrium import UE_PATH
from src.decide.best_trade_offs import DESIGNS_DIR, TRADE_OFFS_PATH

ROOT = Path(__file__).resolve().parents[2]
EMISSIONS_PATH = ROOT / "data" / "decide" / "emissions.json"
EDGE_EMISSIONS_PATH = ROOT / "data" / "decide" / "edge_emissions.csv"

POLLUTANTS = ("NOx", "CO2")
BASELINE = "baseline"

# Assumed urban fleet by (category, Euro class); the shares are illustrative, not measured.
FLEET = ee.FleetComposition({
    ("passenger_car_petrol", "Euro 6"): 0.38,
    ("passenger_car_petrol", "Euro 5"): 0.07,
    ("passenger_car_diesel", "Euro 6"): 0.22,
    ("passenger_car_diesel", "Euro 5"): 0.11,
    ("lgv_diesel", "Euro 6"): 0.12,
    ("bus_diesel", "Euro 6"): 0.04,
    ("motorcycle", "Euro 4"): 0.06,
}, name="assumed urban fleet")


def edge_emissions(ue: pd.DataFrame, fleet: ee.FleetComposition = FLEET,
                   pollutants: tuple = POLLUTANTS) -> pd.DataFrame:
    """Return edge_id, speed_kmh, flow_veh_h and grams per hour of each pollutant per UE row.

    An edge with no flow emits nothing; an edge with flow but no positive travel time raises.
    """
    if ue[["flow_veh_h", "length_m"]].isna().any().any():
        raise ValueError("a UE row is missing its flow or length")
    if ((ue["flow_veh_h"] > 0) & ~(ue["ue_time_s"] > 0)).any():
        raise ValueError("an edge with flow has no positive UE travel time")
    out = pd.DataFrame({"edge_id": ue["edge_id"].astype(str),
                        "speed_kmh": 3.6 * ue["length_m"] / ue["ue_time_s"].where(ue["ue_time_s"] > 0),
                        "flow_veh_h": ue["flow_veh_h"]})
    for pollutant in pollutants:
        out[f"{pollutant}_g_h"] = [
            ee.fleet_hot_emissions(fleet, pollutant, speed, length / 1000, vehicles=flow) if flow > 0 else 0.0
            for speed, length, flow in zip(out["speed_kmh"], ue["length_m"], ue["flow_veh_h"])]
    return out


def totals(edges: pd.DataFrame, pollutants: tuple = POLLUTANTS) -> dict[str, float]:
    """Network total of each pollutant in grams per hour."""
    return {p: float(edges[f"{p}_g_h"].sum()) for p in pollutants}


def compare(designs: dict[str, pd.DataFrame], pollutants: tuple = POLLUTANTS) -> dict:
    """Totals per design plus each design's change from the baseline, absolute and in percent."""
    sums = {name: totals(edges, pollutants) for name, edges in designs.items()}
    base = sums[BASELINE]
    return {name: {"total_g_h": t,
                   "change_g_h": {p: t[p] - base[p] for p in pollutants},
                   "change_pct": {p: (100 * (t[p] - base[p]) / base[p]) if base[p] else None for p in pollutants}}
            for name, t in sums.items()}


def read_ue(path: Path) -> pd.DataFrame:
    """Read a #24-format UE CSV, keeping edge ids as strings."""
    return pd.read_csv(path, dtype={"edge_id": str})


def main() -> None:
    """Write per-edge and total emissions for the baseline and each pick, and print the totals."""
    picks = json.loads(TRADE_OFFS_PATH.read_text(encoding="utf-8"))["picks"]
    designs = {BASELINE: edge_emissions(read_ue(UE_PATH))}
    for role, pick in picks.items():
        designs[role] = edge_emissions(read_ue(DESIGNS_DIR.parent / pick["flows_file"]))
    summary = compare(designs)

    EMISSIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([edges.assign(design=name) for name, edges in designs.items()]).to_csv(
        EDGE_EMISSIONS_PATH, index=False)
    EMISSIONS_PATH.write_text(json.dumps({
        "label": "modelled hot-exhaust emissions from UE flows under an assumed fleet; display only, "
                 "not measured and not part of the optimiser",
        "fleet": {f"{c} {e}": share for (c, e), share in FLEET.shares.items()},
        "units": "grams per hour of the modelled hour",
        "designs": summary,
    }, indent=1), encoding="utf-8")
    for name, s in summary.items():
        changes = ", ".join(f"{p} {s['change_pct'][p]:+.1f}%" for p in POLLUTANTS
                            if s["change_pct"][p] is not None)
        print(f"{name}: " + ", ".join(f"{p} {s['total_g_h'][p]:,.0f} g/h" for p in POLLUTANTS)
              + (f" ({changes} vs baseline)" if name != BASELINE else ""))


if __name__ == "__main__":
    main()
