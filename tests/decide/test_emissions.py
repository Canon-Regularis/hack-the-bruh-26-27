import json
import math

import pandas as pd
import pytest

from engine.orchid import emissions_environment as ee
from src.decide import emissions as em


def ue(rows):
    """A #24-format UE table: (edge_id, length_m, ue_time_s, flow_veh_h).

    Free-flow time is half the UE time, so using the wrong column changes every speed.
    """
    return pd.DataFrame([{"edge_id": e, "length_m": length, "free_flow_time_s": t / 2, "ue_time_s": t,
                          "flow_veh_h": flow, "vc": 0.5} for e, length, t, flow in rows])


def test_edge_value_is_orchids_fleet_emissions_at_the_ue_speed():
    out = em.edge_emissions(ue([("7", 1000.0, 100.0, 500.0)]))
    assert out.loc[0, "speed_kmh"] == pytest.approx(36.0)
    for p in em.POLLUTANTS:
        expected = ee.fleet_hot_emissions(em.FLEET, p, 36.0, 1.0, vehicles=500.0)
        assert out.loc[0, f"{p}_g_h"] == pytest.approx(expected, rel=1e-12)
        assert expected > 0


def test_emissions_are_linear_in_flow_and_length():
    one = em.edge_emissions(ue([("1", 500.0, 50.0, 100.0)]))
    double_flow = em.edge_emissions(ue([("1", 500.0, 50.0, 200.0)]))
    double_length = em.edge_emissions(ue([("1", 1000.0, 100.0, 100.0)]))
    for p in em.POLLUTANTS:
        assert double_flow.loc[0, f"{p}_g_h"] == pytest.approx(2 * one.loc[0, f"{p}_g_h"])
        assert double_length.loc[0, f"{p}_g_h"] == pytest.approx(2 * one.loc[0, f"{p}_g_h"])


def test_an_edge_with_no_flow_emits_nothing_even_with_no_time():
    out = em.edge_emissions(ue([("1", 300.0, 0.0, 0.0), ("2", 300.0, 30.0, 0.0)]))
    assert out["NOx_g_h"].tolist() == [0.0, 0.0] and out["CO2_g_h"].tolist() == [0.0, 0.0]


def test_flow_without_a_positive_time_raises():
    with pytest.raises(ValueError, match="positive UE travel time"):
        em.edge_emissions(ue([("1", 300.0, 0.0, 10.0)]))


@pytest.mark.parametrize("column", ["flow_veh_h", "length_m"])
def test_a_missing_flow_or_length_raises_instead_of_counting_as_zero(column):
    table = ue([("1", 300.0, 30.0, 10.0)])
    table.loc[0, column] = float("nan")
    with pytest.raises(ValueError, match="missing its flow or length"):
        em.edge_emissions(table)


def test_integer_edge_ids_come_out_as_strings():
    assert em.edge_emissions(ue([(1, 100.0, 10.0, 5.0)]))["edge_id"].tolist() == ["1"]


def test_a_crawling_edge_is_clamped_to_a_finite_value():
    out = em.edge_emissions(ue([("1", 100.0, 1e6, 50.0)]))
    assert out.loc[0, "speed_kmh"] < 1
    assert all(math.isfinite(out.loc[0, f"{p}_g_h"]) and out.loc[0, f"{p}_g_h"] > 0 for p in em.POLLUTANTS)


def test_read_ue_keeps_edge_ids_as_strings(tmp_path):
    path = tmp_path / "ue.csv"
    ue([("007", 100.0, 10.0, 5.0)]).to_csv(path, index=False)
    assert em.read_ue(path)["edge_id"].tolist() == ["007"]


def test_compare_reports_changes_against_the_baseline():
    base = em.edge_emissions(ue([("1", 1000.0, 100.0, 400.0), ("2", 1000.0, 100.0, 400.0)]))
    closed = em.edge_emissions(ue([("1", 1000.0, 100.0, 400.0), ("2", 1000.0, 100.0, 0.0)]))
    out = em.compare({em.BASELINE: base, "car_best": closed})
    assert out[em.BASELINE]["change_g_h"] == {"NOx": 0.0, "CO2": 0.0}
    for p in em.POLLUTANTS:
        assert out["car_best"]["change_pct"][p] == pytest.approx(-50.0)
        assert out["car_best"]["total_g_h"][p] == pytest.approx(out[em.BASELINE]["total_g_h"][p] / 2)


def test_compare_gives_no_percentage_against_a_zero_baseline():
    empty = em.edge_emissions(ue([("1", 100.0, 10.0, 0.0)]))
    some = em.edge_emissions(ue([("1", 100.0, 10.0, 5.0)]))
    out = em.compare({em.BASELINE: empty, "knee": some})
    assert out["knee"]["change_pct"] == {"NOx": None, "CO2": None}
    assert out["knee"]["change_g_h"]["NOx"] > 0


def test_main_writes_labelled_totals_and_per_edge_rows(tmp_path, monkeypatch, capsys):
    designs = tmp_path / "decide" / "designs"
    designs.mkdir(parents=True)
    ue([("1", 1000.0, 100.0, 400.0), ("2", 800.0, 60.0, 300.0)]).to_csv(tmp_path / "base.csv", index=False)
    ue([("1", 1000.0, 90.0, 500.0), ("2", 800.0, 60.0, 0.0)]).to_csv(designs / "car_best.csv", index=False)
    picks = {"picks": {"car_best": {"mask": 1, "flows_file": "designs/car_best.csv"}}}
    (tmp_path / "decide" / "best_trade_offs.json").write_text(json.dumps(picks), encoding="utf-8")
    monkeypatch.setattr(em, "UE_PATH", tmp_path / "base.csv")
    monkeypatch.setattr(em, "TRADE_OFFS_PATH", tmp_path / "decide" / "best_trade_offs.json")
    monkeypatch.setattr(em, "DESIGNS_DIR", designs)
    monkeypatch.setattr(em, "EMISSIONS_PATH", tmp_path / "out" / "emissions.json")
    monkeypatch.setattr(em, "EDGE_EMISSIONS_PATH", tmp_path / "out" / "edge_emissions.csv")
    em.main()

    out = json.loads((tmp_path / "out" / "emissions.json").read_text(encoding="utf-8"))
    assert "display only" in out["label"] and "not measured" in out["label"]
    assert set(out["designs"]) == {em.BASELINE, "car_best"}
    assert math.fsum(out["fleet"].values()) == pytest.approx(1.0)
    edges = pd.read_csv(tmp_path / "out" / "edge_emissions.csv", dtype={"edge_id": str})
    assert sorted(edges["design"].unique()) == [em.BASELINE, "car_best"]
    assert len(edges) == 4
    base_nox = edges.loc[edges["design"] == em.BASELINE, "NOx_g_h"].sum()
    assert out["designs"][em.BASELINE]["total_g_h"]["NOx"] == pytest.approx(base_nox)
    printed = capsys.readouterr().out
    assert "baseline: NOx" in printed and "vs baseline" in printed
