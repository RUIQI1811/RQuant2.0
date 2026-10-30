from __future__ import annotations

import tempfile

import numpy as np
import pandas as pd
import pytest

from rquant.errors import DataContractError
from rquant.factors.catalog import get_catalog
from rquant.factors.engine import FactorBuildConfig, KunQuantFactorEngine
from rquant.io import atomic_write_json, sha256_file, stable_hash


@pytest.fixture(scope="module")
def tiny_engine():
    with tempfile.TemporaryDirectory(prefix="c469-", ignore_cleanup_errors=True) as directory:
        yield KunQuantFactorEngine(directory, FactorBuildConfig(factor_set="china469_pit", universe="all", workers=1))


def test_registered_source_timing_factors() -> None:
    catalog = get_catalog()
    specs = catalog.select("china469_pit")
    assert [(s.canonical_name, s.source_name) for s in specs] == [
        ("cnax_001", "Cop"), ("cnax_002", "Cla"), ("cnax_003", "Claq"), ("cnax_004", "Cbe"),
        ("cnax_005", "Opa"), ("cnax_006", "Ope"), ("cnax_007", "Ola"), ("cnax_008", "Ole"),
        ("cnax_009", "Olaq"), ("cnax_010", "Oleq"), ("cnax_011", "Pm"),
        ("cnax_012", "Tbi"), ("cnax_013", "Bl"), ("cnax_014", "Poa"), ("cnax_015", "Pta"),
        ("cnax_016", "Cto"), ("cnax_017", "Gla"), ("cnax_018", "Ta")
    ]


def test_reference_loader_pins_and_checks_calendar_and_fundamentals(tmp_path) -> None:
    from rquant.factors.libraries.china469_pit import load_reference

    reference = tmp_path / "reference"
    reference.mkdir()
    fundamentals = pd.DataFrame({
        "instrument": ["S1"], "end_date": ["2023-12-31"], "ann_date": ["2024-04-01"],
        "total_assets": [10.], "operate_profit": [3.], "n_income": [2.], "n_cashflow_act": [1.],
        "total_hldr_eqy_inc_min_int": [5.],
        "income_tax": [1.], "total_revenue": [8.], "revenue": [7.], "oper_cost": [4.],
        "n_cashflow_inv_act": [-1.], "n_cash_flows_fnc_act": [1.],
    })
    fundamentals.to_parquet(reference / "fundamentals.parquet", index=False)
    days = pd.date_range("2024-05-01", "2024-08-10")
    calendar = pd.DataFrame({"exchange": "SSE", "cal_date": days.strftime("%Y%m%d"),
                             "is_open": (days.dayofweek < 5).astype(int)})
    calendar.to_parquet(reference / "trade_cal.parquet", index=False)
    atomic_write_json(tmp_path / "manifest.json", {"status": "complete", "raw_fingerprint": "raw"})
    manifest = {"status": "complete", "raw_fingerprint": "raw", "files": {
        "fundamentals.parquet": sha256_file(reference / "fundamentals.parquet"),
        "trade_cal.parquet": sha256_file(reference / "trade_cal.parquet"),
    }}
    manifest["fingerprint"] = stable_hash(manifest)
    atomic_write_json(reference / "manifest.json", manifest)
    loaded, hashes = load_reference(tmp_path, pd.bdate_range("2024-05-01", "2024-07-31"), ["S1"])
    assert len(loaded["fundamentals"]) == 1
    assert hashes["china469_trade_calendar"] == sha256_file(reference / "trade_cal.parquet")
    calendar[calendar.cal_date <= "20240630"].to_parquet(reference / "trade_cal.parquet", index=False)
    with pytest.raises(DataContractError, match="differs"):
        load_reference(tmp_path, pd.bdate_range("2024-05-01", "2024-07-31"), ["S1"])
    manifest["files"]["trade_cal.parquet"] = sha256_file(reference / "trade_cal.parquet")
    manifest.pop("fingerprint")
    manifest["fingerprint"] = stable_hash(manifest)
    atomic_write_json(reference / "manifest.json", manifest)
    with pytest.raises(DataContractError, match="next-month open"):
        load_reference(tmp_path, pd.bdate_range("2024-05-01", "2024-06-28"), ["S1"])


def test_legacy_calendar_is_checked_against_hashed_raw_tushare_partition(tmp_path) -> None:
    from rquant.factors.libraries.china469_pit import load_reference

    canonical, raw = tmp_path / "canonical", tmp_path / "raw"
    reference, partition = canonical / "reference", raw / "trade_cal" / "range=fixture"
    reference.mkdir(parents=True)
    partition.mkdir(parents=True)
    records = pd.DataFrame({"instrument": ["S1"], "end_date": ["2023-12-31"], "ann_date": ["2024-04-01"],
                            "total_assets": [1.], "operate_profit": [1.], "n_income": [1.], "n_cashflow_act": [1.],
                            "total_hldr_eqy_inc_min_int": [1.], "income_tax": [1.], "total_revenue": [1.],
                            "revenue": [1.], "oper_cost": [1.], "n_cashflow_inv_act": [1.],
                            "n_cash_flows_fnc_act": [1.]})
    records.to_parquet(reference / "fundamentals.parquet", index=False)
    days = pd.date_range("2024-04-01", "2024-06-30")
    calendar = pd.DataFrame({"exchange": "SSE", "cal_date": days.strftime("%Y%m%d"),
                             "is_open": (days.dayofweek < 5).astype(int)})
    calendar.loc[calendar.cal_date.eq("20240501"), "is_open"] = 0
    calendar.to_parquet(partition / "data.parquet", index=False)
    calendar.to_parquet(reference / "trade_cal.parquet", index=False)
    atomic_write_json(partition / "manifest.json", {"status": "complete", "endpoint": "trade_cal",
                                                    "sha256": sha256_file(partition / "data.parquet")})
    atomic_write_json(raw / "sync_manifest.json", {"status": "complete", "fingerprint": "raw"})
    atomic_write_json(canonical / "manifest.json", {"status": "complete", "raw_fingerprint": "raw"})
    manifest = {"status": "complete", "raw_fingerprint": "raw",
                "files": {"fundamentals.parquet": sha256_file(reference / "fundamentals.parquet")}}
    manifest["fingerprint"] = stable_hash(manifest)
    atomic_write_json(reference / "manifest.json", manifest)
    output = pd.DatetimeIndex(["2024-05-02"])
    load_reference(canonical, output, ["S1"])
    calendar.loc[calendar.cal_date.eq("20240501"), "is_open"] = 1
    calendar.to_parquet(reference / "trade_cal.parquet", index=False)
    with pytest.raises(DataContractError, match="raw trade calendar"):
        load_reference(canonical, output, ["S1"])


def test_statement_graph_matches_hand_calculation_and_frozen_formation(tiny_engine) -> None:
    frame = pd.DataFrame([
        ("S1", "2021-12-31", "2022-04-01", 100., 20., 10., 20.),
        ("S1", "2022-12-31", "2023-04-01", 200., 40., 20., 30.),
        ("S1", "2023-09-30", "2023-10-20", 220., 30., 9., 15.),
        ("S1", "2023-12-31", "2024-04-10", 240., 50., 15., 30.),
        ("S1", "2024-03-31", "2024-05-10", 250., 12., 4., 6.),
        ("S2", "2022-12-31", "2023-04-01", 150., 30., 15., 15.),
        ("S2", "2023-12-31", "2024-07-01", 180., 35., 15., 20.),
    ], columns=["instrument", "end_date", "ann_date", "total_assets", "operate_profit", "n_income", "n_cashflow_act"])
    frame["total_hldr_eqy_inc_min_int"] = [50., 80., 95., 120., 125., 0., 0.]
    frame["income_tax"] = [3., 5., 2., 7., 1., 4., 5.]
    frame["total_revenue"] = [100., 400., 450., 500., 550., 300., 360.]
    frame["revenue"] = [90., 390., 430., 450., 500., 280., 330.]
    frame["oper_cost"] = [60., 250., 270., 300., 340., 180., 220.]
    frame["n_cashflow_inv_act"] = [-2., -4., -6., -10., -2., -3., -4.]
    frame["n_cash_flows_fnc_act"] = [1., 3., 2., 5., 1., 2., 4.]
    frame["end_date"] = pd.to_datetime(frame.end_date)
    frame["ann_date"] = pd.to_datetime(frame.ann_date)
    calendar = pd.bdate_range("2020-01-01", "2025-12-31")
    dates = calendar[(calendar >= "2023-01-01") & (calendar <= "2025-01-31")]
    instruments = pd.Index(["S1", "S2"])
    engine = tiny_engine
    modules = engine.compile_statements()
    results = engine.run_statements({"fundamentals": frame, "formation_calendar": calendar},
                                    dates, instruments, modules)
    cop = pd.DataFrame(results.daily("y_cnax_cop"), index=dates, columns=instruments)
    cla = pd.DataFrame(results.daily("y_cnax_cla"), index=dates, columns=instruments)
    claq = pd.DataFrame(results.daily("q_cnax_claq"), index=dates, columns=instruments)
    cbe = pd.DataFrame(results.daily("y_cnax_cbe"), index=dates, columns=instruments)
    extra = {name: pd.DataFrame(results.daily(name), index=dates, columns=instruments) for name in
             ("y_cnax_opa", "y_cnax_ope", "y_cnax_ola", "y_cnax_ole", "q_cnax_olaq", "q_cnax_oleq",
              "y_cnax_pm", "y_cnax_tbi", "y_cnax_bl", "y_cnax_poa", "y_cnax_pta", "y_cnax_cto", "y_cnax_gla",
              "y_cnax_ta")}
    assert np.isclose(cop.loc["2023-06-30", "S1"], 50 / 200)
    assert np.isclose(cbe.loc["2023-06-30", "S1"], 50 / 80)
    assert pd.isna(cbe.loc["2023-06-30", "S2"])
    assert np.isclose(cbe.loc["2024-06-28", "S1"], 65 / 120)
    assert np.isclose(extra["y_cnax_opa"].loc["2023-06-30", "S1"], 40 / 200)
    assert np.isclose(extra["y_cnax_ope"].loc["2023-06-30", "S1"], 40 / 80)
    assert np.isclose(extra["y_cnax_ola"].loc["2023-06-30", "S1"], 40 / 100)
    assert np.isclose(extra["y_cnax_ole"].loc["2023-06-30", "S1"], 40 / 50)
    assert np.isclose(extra["q_cnax_olaq"].loc["2024-05-01", "S1"], 20 / 220)
    assert np.isclose(extra["q_cnax_oleq"].loc["2024-05-01", "S1"], 20 / 95)
    assert np.isclose(extra["y_cnax_pm"].loc["2023-06-30", "S1"], 40 / 400)
    assert np.isclose(extra["y_cnax_tbi"].loc["2023-06-30", "S1"], 25 / 20)
    assert np.isclose(extra["y_cnax_bl"].loc["2023-06-30", "S1"], 200 / 80)
    assert np.isclose(extra["y_cnax_cto"].loc["2023-06-30", "S1"], 400 / 100)
    assert np.isclose(extra["y_cnax_poa"].loc["2023-06-30", "S1"], -10 / 20)
    assert np.isclose(extra["y_cnax_pta"].loc["2023-06-30", "S1"], (20 - 30 + 4 - 3) / 20)
    assert np.isclose(extra["y_cnax_gla"].loc["2023-06-30", "S1"], 140 / 100)
    assert np.isclose(extra["y_cnax_ta"].loc["2023-06-30", "S1"], (20 - 30 + 4 - 3) / 100)
    for field, value in (("total_assets", 0.), ("ann_date", pd.Timestamp("2023-07-01"))):
        altered = frame.copy()
        altered.loc[(altered.instrument == "S1") & (altered.end_date == pd.Timestamp("2021-12-31")), field] = value
        changed = engine.run_statements({"fundamentals": altered, "formation_calendar": calendar},
                                        dates, instruments, modules)
        assert np.isnan(changed.daily("y_cnax_gla")[dates.get_loc("2023-06-30"), 0])
        assert np.isnan(changed.daily("y_cnax_ta")[dates.get_loc("2023-06-30"), 0])
    assert pd.isna(extra["y_cnax_gla"].loc["2023-06-30", "S2"])
    assert pd.isna(extra["y_cnax_ope"].loc["2023-06-30", "S2"])
    assert pd.isna(extra["y_cnax_ola"].loc["2023-06-30", "S2"])
    assert pd.isna(cbe.loc["2024-06-28", "S2"])
    assert np.isclose(cla.loc["2023-06-30", "S1"], 50 / 100)
    assert np.isclose(cop.loc["2023-06-30", "S2"], 30 / 150)
    assert pd.isna(cla.loc["2023-06-30", "S2"])
    assert pd.isna(cop.loc["2024-06-28", "S2"])
    assert pd.isna(cop.loc["2024-07-02", "S2"])
    assert np.isclose(claq.loc["2024-05-01", "S1"], (20 - 6 + 15) / 220)
    assert np.isclose(claq.loc["2024-08-01", "S1"], (12 - 4 + 6) / 240)
    for cutoff in ("2023-06-29", "2023-06-30", "2024-05-01", "2024-06-28", "2024-08-01"):
        prefix = dates[dates <= cutoff]
        short_frame = frame[frame.ann_date <= pd.Timestamp(cutoff)]
        shortened = engine.run_statements(
            {"fundamentals": short_frame, "formation_calendar": calendar}, prefix, instruments, modules
        )
        for name in ("y_cnax_cop", "y_cnax_cla", "q_cnax_claq", "y_cnax_cbe", *extra):
            expected = results.daily(name)[:len(prefix)]
            actual = shortened.daily(name)
            np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
            np.testing.assert_allclose(actual, expected, equal_nan=True)


def test_nonfinite_statement_denominators_never_produce_zero(tiny_engine) -> None:
    frame = pd.DataFrame([
        ("S1", "2021-12-31", "2022-04-01", np.inf, 20., 10., 5., np.inf),
        ("S1", "2022-12-31", "2023-04-01", -np.inf, 25., 12., 6., -np.inf),
        ("S1", "2023-09-30", "2023-10-20", np.inf, 30., 15., 9., np.inf),
        ("S1", "2023-12-31", "2024-04-10", np.inf, 50., 20., 15., np.inf),
    ], columns=["instrument", "end_date", "ann_date", "total_assets", "operate_profit", "n_income",
                "n_cashflow_act", "total_hldr_eqy_inc_min_int"])
    frame["end_date"] = pd.to_datetime(frame.end_date)
    frame["ann_date"] = pd.to_datetime(frame.ann_date)
    frame["income_tax"] = np.inf
    frame["total_revenue"] = np.inf
    frame["revenue"] = np.inf
    frame["oper_cost"] = 1.
    frame["n_cashflow_inv_act"] = 1.
    frame["n_cash_flows_fnc_act"] = 1.
    calendar = pd.bdate_range("2020-01-01", "2025-12-31")
    dates = calendar[(calendar >= "2023-01-01") & (calendar <= "2024-06-28")]
    results = tiny_engine.run_statements({"fundamentals": frame, "formation_calendar": calendar}, dates,
                                         pd.Index(["S1"]))
    for name in ("y_cnax_cop", "y_cnax_cla", "y_cnax_cbe", "y_cnax_opa", "y_cnax_ope", "y_cnax_ola", "y_cnax_ole",
                 "y_cnax_pm", "y_cnax_tbi", "y_cnax_bl", "y_cnax_cto", "y_cnax_gla", "y_cnax_ta"):
        assert np.isnan(results.daily(name)[dates.get_loc("2023-06-30"), 0]), name
    for name in ("q_cnax_claq", "q_cnax_olaq", "q_cnax_oleq"):
        assert np.isnan(results.daily(name)[dates.get_loc("2024-05-01"), 0]), name


def test_tbi_requires_positive_pretax_and_net_income(tiny_engine) -> None:
    frame = pd.DataFrame({"instrument": ["S1", "S2", "S3", "S4"],
                          "end_date": pd.to_datetime(["2022-12-31"] * 4),
                          "ann_date": pd.to_datetime(["2023-04-01"] * 4),
                          "total_assets": [100.] * 4, "total_hldr_eqy_inc_min_int": [50.] * 4,
                          "operate_profit": [10.] * 4, "n_cashflow_act": [3.] * 4,
                          "n_income": [5., -5., 5., 0.], "income_tax": [1., 10., -10., 3.],
                          "total_revenue": [30.] * 4, "revenue": [25.] * 4, "oper_cost": [10.] * 4,
                          "n_cashflow_inv_act": [1., 1., np.nan, 1.], "n_cash_flows_fnc_act": [2.] * 4})
    calendar = pd.bdate_range("2022-01-01", "2024-12-31")
    dates = calendar[(calendar >= "2023-06-01") & (calendar <= "2023-07-31")]
    result = tiny_engine.run_statements({"fundamentals": frame, "formation_calendar": calendar}, dates,
                                        pd.Index(["S1", "S2", "S3", "S4"]))
    value = result.daily("y_cnax_tbi")[dates.get_loc("2023-06-30")]
    assert np.isclose(value[0], 1.2)
    assert np.isnan(value[1:]).all()
    poa = result.daily("y_cnax_poa")[dates.get_loc("2023-06-30")]
    np.testing.assert_allclose(poa[:3], [0.4, -1.6, 0.4])
    assert np.isnan(poa[3])
    pta = result.daily("y_cnax_pta")[dates.get_loc("2023-06-30")]
    np.testing.assert_allclose(pta[:2], [-0.2, -2.2])
    assert np.isnan(pta[2:]).all()


def test_real_engine_path_builds_three_factors_from_small_canonical_fixture(tmp_path, tiny_engine) -> None:
    canonical, reference = tmp_path / "canonical", tmp_path / "canonical" / "reference"
    reference.mkdir(parents=True)
    dates = pd.bdate_range("2024-05-01", "2024-07-31")
    path = canonical / "bars" / "year=2024"
    path.mkdir(parents=True)
    bars = pd.DataFrame({"datetime": dates, "instrument": "SH600000", "close": 10.})
    bars.loc[bars.datetime.eq(pd.Timestamp("2024-07-01")), "close"] = np.inf
    bars.to_parquet(path / "bars.parquet", index=False)
    frame = pd.DataFrame([
        ("SH600000", "2022-12-31", "2023-04-01", 200., 40., 20., 30.),
        ("SH600000", "2023-09-30", "2023-10-20", 220., 30., 9., 15.),
        ("SH600000", "2023-12-31", "2024-04-10", 240., 50., 15., 30.),
    ], columns=["instrument", "end_date", "ann_date", "total_assets", "operate_profit", "n_income", "n_cashflow_act"])
    frame["total_hldr_eqy_inc_min_int"] = [80., 95., 120.]
    frame["income_tax"] = [5., 2., 7.]
    frame["total_revenue"] = [400., 450., 500.]
    frame["revenue"] = [390., 430., 450.]
    frame["oper_cost"] = [250., 270., 300.]
    frame["n_cashflow_inv_act"] = [-4., -6., -10.]
    frame["n_cash_flows_fnc_act"] = [3., 2., 5.]
    frame.to_parquet(reference / "fundamentals.parquet", index=False)
    days = pd.date_range("2020-01-01", "2025-12-31")
    pd.DataFrame({"exchange": "SSE", "cal_date": days.strftime("%Y%m%d"),
                  "is_open": (days.dayofweek < 5).astype(int)}).to_parquet(reference / "trade_cal.parquet", index=False)
    atomic_write_json(canonical / "manifest.json", {"status": "complete", "raw_fingerprint": "raw"})
    manifest = {"status": "complete", "raw_fingerprint": "raw",
                "files": {"fundamentals.parquet": sha256_file(reference / "fundamentals.parquet"),
                          "trade_cal.parquet": sha256_file(reference / "trade_cal.parquet")}}
    manifest["fingerprint"] = stable_hash(manifest)
    atomic_write_json(reference / "manifest.json", manifest)
    result = tiny_engine.build_from_canonical(canonical, tmp_path / "factors")
    assert result["columns"] == 18
    values = pd.read_parquet(tmp_path / "factors" / "china469_pit" / "year=2024" / "factors.parquet")
    june = values.set_index("datetime").loc["2024-06-28"]
    assert np.isclose(june["cnax_001"], 65 / 240)
    assert np.isclose(june["cnax_002"], 65 / 200)
    assert np.isclose(june["cnax_003"], 29 / 220)
    assert np.isclose(june["cnax_004"], 65 / 120)
    assert np.isclose(june["cnax_005"], 50 / 240)
    assert np.isclose(june["cnax_006"], 50 / 120)
    assert np.isclose(june["cnax_007"], 50 / 200)
    assert np.isclose(june["cnax_008"], 50 / 80)
    assert np.isclose(june["cnax_009"], 20 / 220)
    assert np.isclose(june["cnax_010"], 20 / 95)
    assert np.isclose(june["cnax_011"], 50 / 500)
    assert np.isclose(june["cnax_012"], 22 / 15)
    assert np.isclose(june["cnax_013"], 240 / 120)
    assert np.isclose(june["cnax_014"], -15 / 15)
    assert np.isclose(june["cnax_015"], (15 - 30 + 10 - 5) / 15)
    assert np.isclose(june["cnax_016"], 500 / 200)
    assert np.isclose(june["cnax_017"], 150 / 200)
    assert np.isclose(june["cnax_018"], (15 - 30 + 10 - 5) / 200)
    invalid_quote = values.set_index("datetime").loc["2024-07-01"]
    assert invalid_quote[[f"cnax_{n:03d}" for n in range(1, 19)]].isna().all()
