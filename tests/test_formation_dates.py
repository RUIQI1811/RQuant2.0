from __future__ import annotations

import pandas as pd
import pytest

from rquant.errors import DataContractError
from rquant.factors.formation_dates import formation_dates


def calendar() -> pd.DatetimeIndex:
    days = pd.bdate_range("2023-01-02", "2025-12-31")
    return days.difference(pd.to_datetime(["2024-05-01", "2024-05-02", "2024-06-28"]))


def records() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "instrument": ["S1", "S1", "S2", "S3", "S4", "S5", "S6", "S7"],
            "end_date": pd.to_datetime([
                "2023-12-31", "2023-12-31", "2023-12-31", "2023-12-31", "2023-09-30",
                "2023-12-31", None, "2023-12-31",
            ]),
            "ann_date": pd.to_datetime([
                "2024-04-20", "2024-05-03", "2024-06-27", "2024-06-28", "2023-11-01",
                None, "2024-02-01", "2026-01-01",
            ]),
            "amount": [1, 2, 3, 4, 5, 6, 7, 8],
        },
        index=[9, 7, 3, 2, 8, 6, 5, 4],
    )


def test_monthly_uses_four_calendar_months_and_announcement_before_formation() -> None:
    frame = records()
    result = formation_dates(frame, calendar())
    assert result.loc[9, "monthly_available"] == pd.Timestamp("2024-05-03")
    assert result.loc[7, "monthly_available"] == pd.Timestamp("2024-06-03")
    assert result.loc[8, "monthly_available"] == pd.Timestamp("2024-02-01")
    assert result.index.equals(frame.index)
    pd.testing.assert_frame_equal(result[frame.columns], frame)
    assert pd.isna(result.loc[6, "monthly_available"])
    assert pd.isna(result.loc[5, "monthly_available"])
    assert pd.isna(result.loc[4, "monthly_available"])


def test_annual_uses_last_june_trading_day_and_does_not_backfill_late_report() -> None:
    result = formation_dates(records(), calendar())
    assert result.loc[9, "annual_available"] == pd.Timestamp("2024-06-27")
    assert result.loc[3, "annual_available"] == pd.Timestamp("2024-06-27")
    assert result.loc[2, "annual_available"] == pd.Timestamp("2025-06-30")
    for key in (8, 6, 5, 4):
        assert pd.isna(result.loc[key, "annual_available"])


def test_incomplete_first_month_and_final_june_are_not_fabricated() -> None:
    frame = pd.DataFrame({"instrument": ["S1"], "end_date": ["2023-12-31"], "ann_date": ["2024-04-01"]})
    truncated_start = calendar()[calendar() >= "2024-05-10"]
    result = formation_dates(frame, truncated_start)
    assert result.loc[0, "monthly_available"] == pd.Timestamp("2024-06-03")
    truncated_end = calendar()[calendar() <= "2024-06-14"]
    assert pd.isna(formation_dates(frame, truncated_end).loc[0, "annual_available"])


def test_truncating_report_rows_cannot_change_preexisting_formation_dates() -> None:
    frame = records()
    full = formation_dates(frame, calendar())
    earlier = formation_dates(frame.iloc[[0, 1, 4]], calendar())
    pd.testing.assert_frame_equal(earlier, full.loc[earlier.index])


def test_available_dates_respect_truncated_calendar_prefix() -> None:
    frame = records()
    full = formation_dates(frame, calendar())
    prefix = calendar()[calendar() <= "2024-08-01"]
    early = formation_dates(frame, prefix)
    for name in ("monthly_available", "annual_available"):
        expected = full[name].where(full[name] <= prefix[-1])
        pd.testing.assert_series_equal(early[name], expected, check_names=False)


def test_older_eligible_quarter_is_preserved_until_late_new_quarter_can_form() -> None:
    frame = pd.DataFrame({
        "instrument": ["S1", "S1"],
        "end_date": pd.to_datetime(["2023-09-30", "2023-12-31"]),
        "ann_date": pd.to_datetime(["2023-11-01", "2024-05-03"]),
    })
    result = formation_dates(frame, calendar())
    eligible = result[result.monthly_available <= "2024-05-03"]
    assert eligible.end_date.max() == pd.Timestamp("2023-09-30")
    assert result.loc[1, "monthly_available"] == pd.Timestamp("2024-06-03")


def test_leap_year_and_calendar_start_on_first_day() -> None:
    frame = pd.DataFrame({
        "instrument": ["S1", "S2"],
        "end_date": pd.to_datetime(["2024-01-31", "2023-12-31"]),
        "ann_date": pd.to_datetime(["2024-02-01", "2024-03-01"]),
    })
    days = pd.bdate_range("2024-05-01", "2024-07-31")
    result = formation_dates(frame, days)
    assert result.loc[0, "monthly_available"] == pd.Timestamp("2024-06-03")
    assert result.loc[1, "monthly_available"] == pd.Timestamp("2024-05-01")


def test_empty_calendar_has_no_fabricated_formation_days() -> None:
    result = formation_dates(records(), pd.DatetimeIndex([]))
    assert result.annual_available.isna().all()
    assert result.monthly_available.isna().all()


def test_integer_yyyymmdd_dates_are_not_interpreted_as_epoch_nanoseconds() -> None:
    frame = pd.DataFrame({"instrument": ["S1"], "end_date": [20231231], "ann_date": [20240420]})
    result = formation_dates(frame, calendar())
    assert result.loc[0, "monthly_available"] == pd.Timestamp("2024-05-03")
    assert result.loc[0, "annual_available"] == pd.Timestamp("2024-06-27")
    numeric_calendar = calendar().strftime("%Y%m%d").astype(int)
    converted = formation_dates(frame, numeric_calendar)
    assert converted.loc[0, "monthly_available"] == pd.Timestamp("2024-05-03")


def test_invalid_calendar_or_missing_columns_raise_explicit_contract_error() -> None:
    with pytest.raises(DataContractError):
        formation_dates(records(), pd.DatetimeIndex(["2024-01-03", "2024-01-02"]))
    with pytest.raises(DataContractError):
        formation_dates(records(), pd.DatetimeIndex(["2024-01-02", "2024-01-02"]))
    with pytest.raises(DataContractError):
        formation_dates(records().drop(columns="ann_date"), calendar())
    with pytest.raises(DataContractError):
        formation_dates(records().assign(monthly_available="existing"), calendar())
