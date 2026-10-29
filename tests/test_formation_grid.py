from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rquant.errors import DataContractError
from rquant.factors.formation_grid import FormationGrid


def trading_days() -> pd.DatetimeIndex:
    return pd.bdate_range("2020-01-01", "2026-12-31").difference(pd.to_datetime(["2024-05-01", "2024-05-02"]))


def records() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("S1", "2020-03-31", "2026-01-01", 2001.0),
            ("S1", "2021-12-31", "2023-07-03", 2021.0),
            ("S1", "2022-12-31", "2023-06-30", 2022.0),
            ("S1", "2023-12-31", "2024-07-01", 2023.0),
            ("S1", "2024-12-31", "2025-04-30", 2024.0),
            ("S2", "2020-03-31", "2026-01-01", 2001.0),
            ("S2", "2021-12-31", "2022-04-15", 2021.0),
            ("S2", "2022-12-31", "2023-04-15", 2022.0),
            ("S3", "2023-06-30", "2023-08-01", 2302.0),
            ("S3", "2023-09-30", "2023-11-01", 2303.0),
            ("S3", "2023-12-31", "2024-05-03", 2304.0),
            ("S3", "2024-03-31", "2024-05-15", 2401.0),
            ("S4", "2023-06-30", "2025-01-01", 2302.0),
            ("S4", "2023-09-30", "2023-11-01", 2303.0),
            ("S4", "2023-12-31", "2024-04-10", 2304.0),
            ("S4", "2024-03-31", "2024-08-01", 2401.0),
            ("S5", "2023-12-31", None, 2304.0),
            ("S6", None, "2024-01-02", 9999.0),
        ],
        columns=["instrument", "end_date", "ann_date", "marker"],
    )


def panels(frame: pd.DataFrame, dates: pd.DatetimeIndex, mode: str) -> pd.DataFrame:
    instruments = pd.Index(["S1", "S2", "S3", "S4", "S5", "S6"])
    grid = FormationGrid(frame, dates, instruments, full_calendar=trading_days(), mode=mode)
    values = grid.value("marker").T
    return pd.DataFrame(grid.gather(values, annual=mode.startswith("annual")), index=dates, columns=instruments)


def test_equivalent_anchors_share_read_only_storage() -> None:
    days = trading_days()
    grid = FormationGrid(records(), days, pd.Index(["S1", "S2"]), full_calendar=days, mode="annual_current")
    assert grid.anchor is grid.annual_anchor
    assert not grid.anchor.flags.writeable


def test_annual_freeze_late_announcements_and_missing_year_reset() -> None:
    days = trading_days()
    current = panels(records(), days, "annual_current")
    previous = panels(records(), days, "annual_previous")
    assert current.loc["2023-06-29", "S1"] != current.loc["2023-06-29", "S1"]
    assert current.loc["2023-06-30", "S1"] == 2022
    assert pd.isna(previous.loc["2023-06-30", "S1"])
    assert previous.loc["2023-06-30", "S2"] == 2022
    assert current.loc["2024-06-27", "S1"] == 2022
    assert pd.isna(current.loc["2024-06-28", "S1"])
    assert pd.isna(current.loc["2024-07-01", "S1"])
    assert current.loc["2025-06-30", "S1"] == 2024
    assert previous.loc["2025-06-30", "S1"] == 2024
    assert pd.isna(current.loc["2026-06-30", "S1"])
    assert pd.isna(previous.loc["2026-06-30", "S1"])


def test_monthly_cutoff_previous_quarter_fallback_and_unrelated_late_report() -> None:
    data = panels(records(), trading_days(), "monthly_previous")
    assert data.loc["2024-02-01", "S3"] == 2303
    assert data.loc["2024-05-03", "S3"] == 2303
    assert data.loc["2024-06-03", "S3"] == 2304
    assert data.loc["2024-07-01", "S3"] == 2304
    assert data.loc["2024-08-01", "S3"] == 2401
    assert pd.isna(data.loc["2024-02-01", "S4"])
    assert data.loc["2024-05-03", "S4"] == 2304
    assert data.loc["2024-08-01", "S4"] == 2304
    assert data.loc["2024-09-02", "S4"] == 2401


def test_missing_previous_quarter_requires_fallback_without_erasing_older_data() -> None:
    frame = records()
    frame = frame[~((frame.instrument == "S3") & (frame.end_date == "2023-12-31"))]
    days = trading_days()
    data = panels(frame, days, "monthly_previous")
    assert data.loc["2024-08-01", "S3"] == 2303
    assert data.loc["2024-09-02", "S3"] == 2303


def test_latest_eligible_report_with_missing_field_must_not_fallback() -> None:
    frame = records()
    mask = (frame.instrument == "S3") & (frame.end_date == "2023-12-31")
    frame.loc[mask, "marker"] = np.nan
    data = panels(frame, trading_days(), "monthly_previous")
    assert data.loc["2024-05-03", "S3"] == 2303
    assert pd.isna(data.loc["2024-06-03", "S3"])
    assert pd.isna(data.loc["2024-07-01", "S3"])


def test_subrange_inherits_the_latest_full_calendar_formation() -> None:
    days = trading_days()
    full = panels(records(), days, "monthly_previous")
    selected = days[(days >= "2024-06-14") & (days <= "2024-07-15")]
    part = panels(records(), selected, "monthly_previous")
    np.testing.assert_array_equal(part.isna(), full.loc[selected].isna())
    np.testing.assert_allclose(part, full.loc[selected], equal_nan=True)


def test_annual_modes_retain_latest_single_report_when_previous_missing() -> None:
    frame = records()
    frame = frame[~((frame.instrument == "S2") & (frame.end_date == "2021-12-31"))]
    days = trading_days()
    assert panels(frame, days, "annual_current").loc["2023-06-30", "S2"] == 2022
    assert pd.isna(panels(frame, days, "annual_previous").loc["2023-06-30", "S2"])


@pytest.mark.parametrize("mode", ["annual_current", "annual_previous", "monthly_previous"])
@pytest.mark.parametrize("cutoff", ["2024-05-03", "2024-06-27", "2024-06-28", "2024-08-01"])
def test_report_and_calendar_prefix_match_original_values_and_nan_masks(mode: str, cutoff: str) -> None:
    days = trading_days()
    frame = records()
    full = panels(frame, days, mode).loc[:cutoff]
    cutoff_frame = frame[pd.to_datetime(frame.ann_date) <= pd.Timestamp(cutoff)]
    shortened = panels(cutoff_frame, days[days <= cutoff], mode)
    np.testing.assert_array_equal(full.isna(), shortened.isna())
    np.testing.assert_allclose(full, shortened, equal_nan=True)


def test_calendar_prefix_cannot_turn_penultimate_june_day_into_formation() -> None:
    days = trading_days()
    data = panels(records(), days[days <= "2024-06-27"], "annual_current")
    assert data.loc["2024-06-27", "S1"] == 2022


def test_empty_data_and_missing_announcement_produce_no_signal() -> None:
    days = trading_days()
    data = panels(records().iloc[0:0], days[:30], "monthly_previous")
    assert data.isna().all().all()
    result = panels(records(), days, "monthly_previous")
    assert result["S5"].isna().all() and result["S6"].isna().all()


def test_invalid_calendar_modes_and_duplicate_reports_fail_explicitly() -> None:
    days = trading_days()
    instruments = pd.Index(["S1"])
    with pytest.raises(DataContractError):
        FormationGrid(records(), days, instruments, full_calendar=days, mode="invalid")
    with pytest.raises(DataContractError):
        FormationGrid(records(), days, instruments, full_calendar=days[::-1], mode="annual_current")
    with pytest.raises(DataContractError):
        FormationGrid(pd.concat([records(), records().iloc[:1]]), days, instruments,
                      full_calendar=days, mode="annual_current")


def test_engine_uses_library_statement_grid_hook(monkeypatch) -> None:
    from types import SimpleNamespace

    from KunQuant.runner import KunRunner

    from rquant.factors.engine import FactorBuildConfig, KunQuantFactorEngine

    frame, days, instruments = records(), trading_days(), pd.Index(["S1", "S2"])
    grid = FormationGrid(frame, days, instruments, full_calendar=days, mode="annual_current")
    calls = []

    class TestLibrary:
        family = "sample"
        statement_outputs = ("y_value",)
        statement_fields = ("marker",)

        def make_statement_grid(self, inputs, output_dates, output_instruments):
            calls.append(inputs["full_calendar"])
            assert output_dates.equals(days) and output_instruments.equals(instruments)
            return grid

    engine = object.__new__(KunQuantFactorEngine)
    engine.config = FactorBuildConfig(workers=1)
    engine.libraries = (TestLibrary(),)
    monkeypatch.setattr(KunRunner, "createMultiThreadExecutor", lambda workers: None)
    monkeypatch.setattr(
        KunRunner, "runGraph", lambda executor, module, arrays, start, end: {"y_value": arrays["marker"]}
    )
    result = engine.run_statements({"fundamentals": frame, "full_calendar": days}, days, instruments,
                                   {"sample": SimpleNamespace(blocking_len=1)})
    assert len(calls) == 1
    expected = grid.gather(grid.value("marker").T, annual=True)
    np.testing.assert_allclose(result.daily("y_value"), expected, equal_nan=True)
