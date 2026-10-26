"""Research universes: which canonical bars the factor stores, models and backtests may use on each day.

* ``csi300``: point-in-time CSI300 members.
* ``main``: Shanghai and Shenzhen main-board A shares that any ordinary account can trade. ChiNext (30xxxx), STAR
  Market (68xxxx) and the Beijing exchange need extra trading permissions, and B shares are not A shares, so all of
  them are left out. On each day a main-board stock is also left out while it carries an ST, *ST or delisting name,
  and during its first ``MAIN_NEW_LISTING_EXCLUDED_DAYS`` trading days.
* ``sh_main``: the Shanghai half of ``main`` (60xxxx only), with the same ST and new-listing screens.
* ``all``: every canonical bar.
"""

from __future__ import annotations

from typing import Any

from rquant.errors import DataContractError

SUPPORTED_UNIVERSES = ("csi300", "main", "sh_main", "all")

# Board-screened universes and their Qlib code prefixes: Shanghai 60xxxx and Shenzhen 00xxxx, including the former SME
# board (002/003, merged into the main board in 2021).
BOARD_PREFIXES = {"main": ("SH60", "SZ00"), "sh_main": ("SH60",)}
MAIN_NEW_LISTING_EXCLUDED_DAYS = 60

# Canonical bar columns each universe's membership is derived from, besides ``instrument``.
SCREEN_COLUMNS = ("is_st", "listed_days")
UNIVERSE_COLUMNS = {"csi300": ("in_csi300",), "all": (), **dict.fromkeys(BOARD_PREFIXES, SCREEN_COLUMNS)}


def validate_universe(universe: str) -> str:
    if universe not in SUPPORTED_UNIVERSES:
        raise DataContractError(f"Unsupported universe {universe!r}; expected one of {SUPPORTED_UNIVERSES}")
    return universe


def on_board(instruments: Any, universe: str) -> Any:
    """Return a boolean Series marking Qlib instrument codes on the boards of a board-screened ``universe``."""
    return instruments.astype(str).str.upper().str.startswith(BOARD_PREFIXES[universe])


def universe_mask(frame: Any, universe: str) -> Any:
    """Return a boolean Series marking the rows of canonical ``frame`` that are inside ``universe`` on their date."""
    validate_universe(universe)
    if universe == "all":
        return frame["instrument"].notna()
    if universe == "csi300":
        return frame["in_csi300"].fillna(False).astype(bool)
    board = on_board(frame["instrument"], universe)
    unknown_st = board & frame["is_st"].isna()
    if unknown_st.any():
        raise DataContractError(
            f"The {universe} universe needs point-in-time ST history, which the canonical bars lack; "
            "run `rquant data sync` and `rquant data build-qlib` again to fetch Tushare namechange"
        )
    seasoned = frame["listed_days"].gt(MAIN_NEW_LISTING_EXCLUDED_DAYS)
    return board & ~frame["is_st"].fillna(True).astype(bool) & seasoned


def universe_intervals(frame: Any, universe: str) -> list[tuple[str, Any, Any]]:
    """Collapse per-bar membership into Qlib instrument spans of consecutive member bars.

    Days without a bar (suspensions) do not split a span; a bar outside the universe does.
    """
    import numpy as np
    import pandas as pd

    if frame.empty:
        return []
    frame = frame.sort_values(["instrument", "datetime"])
    member = universe_mask(frame, universe).to_numpy(dtype=bool)
    instruments = frame["instrument"].astype(str).to_numpy()
    dates = pd.DatetimeIndex(frame["datetime"])
    first_bar = np.r_[True, instruments[1:] != instruments[:-1]]
    last_bar = np.r_[first_bar[1:], True]
    previous_member = np.r_[False, member[:-1]] & ~first_bar
    next_member = np.r_[member[1:], False] & ~last_bar
    starts = np.flatnonzero(member & ~previous_member)
    ends = np.flatnonzero(member & ~next_member)
    return [(instruments[first], dates[first], dates[last]) for first, last in zip(starts, ends, strict=True)]
