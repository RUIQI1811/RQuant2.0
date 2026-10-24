"""Daily Liu-Stambaugh-Yuan CH-3 and CH-4 factor returns ("Size and Value in China", JFE 2019).

The factors are rebuilt from the full A-share cross-section in Tushare and follow the paper's construction, re-dated
to daily returns:

* Portfolios are formed on the first trading day of each month from data through the previous trading day (the last
  trading day of the prior month) and held for the month.
* The universe is Shanghai and Shenzhen main-board and ChiNext stocks (codes 60, 00 and 30) listed for at least six
  months, with at least 120 trading days over the past 250 trading days and at least 15 in the most recent month
  (three quarters of the month's trading days when the market itself was open fewer than 20 days).
  The smallest 30% by total market value, ranked before the trading filters, is removed as shell-value contaminated.
* Size is total market value (all shares, tradable or not). Earnings-to-price is the reciprocal of Tushare's trailing
  ``pe_ttm``; Tushare leaves ``pe_ttm`` empty for loss-making firms, and those stocks are placed in the growth group
  as the paper does for negative EP. The paper uses annualised net profit excluding non-recurring items, which
  Tushare does not publish for the whole market at the daily level, so trailing twelve-month net profit stands in.
* Abnormal turnover is average daily turnover (share volume over total shares) over the last 20 trading days divided
  by its 250-day average.
* Size is split at the median and EP and abnormal turnover at the 30th and 70th percentiles of the remaining
  universe. Daily portfolio returns are weighted by the previous day's total market value.

``mkt`` is the value-weighted universe return, ``smb`` and ``vmg`` are the CH-3 factors from the size x EP sort,
``pmo`` (pessimistic minus optimistic) is long low and short high abnormal turnover within size groups, and ``smb4``
is the CH-4 size factor: the average of the EP-neutral and the turnover-neutral SMB. Returns are not reduced by the
deposit rate, which over daily horizons only shifts regression intercepts.
"""

from __future__ import annotations

from collections import deque
from typing import Any

CH4_FACTORS = ("mkt", "smb", "vmg", "pmo", "smb4")
CH4_BOARDS = ("60", "00", "30")
CH4_SHELL_FRACTION = 0.3
CH4_SEASONING_DAYS = 183
CH4_YEAR_DAYS = 250
CH4_YEAR_MIN_DAYS = 120
CH4_MONTH_MIN_DAYS = 15
CH4_MONTH_MIN_SHARE = 0.75
CH4_TURNOVER_DAYS = 20
CH4_MIN_STOCKS = 100


class CH4FactorBuilder:
    """Stream trading days in order and return each day's CH-3/CH-4 factor returns.

    ``update`` must be called for every trading day. Its inputs are the day's close-to-close returns, the day's
    ``total_mv`` and ``pe_ttm`` and the day's turnover, all indexed by Tushare code; a stock without a ``daily`` row
    did not trade. A day's returns are weighted with the previous day's market values and grouped by the assignment
    formed at the start of the month, so a row never uses data from after its own close.
    """

    def __init__(self, list_dates: dict[str, Any]) -> None:
        import pandas as pd

        self.list_dates = pd.Series(list_dates, dtype="datetime64[ns]")
        self.history: deque[tuple[Any, Any, Any]] = deque(maxlen=CH4_YEAR_DAYS)
        self.previous: Any = None
        self.previous_date: Any = None
        self.groups: Any = None
        self.month: tuple[int, int] | None = None

    def update(self, trade_date: Any, returns: Any, state: Any, turnover: Any) -> dict[str, Any]:
        import numpy as np

        row: dict[str, Any] = {"datetime": trade_date, **{name: np.nan for name in CH4_FACTORS}, "ch4_stocks": 0}
        if self.previous is not None:
            month = (trade_date.year, trade_date.month)
            if month != self.month:
                self.groups = self._form()
                self.month = month
            row.update(self._returns(returns))
        self.history.append((trade_date, returns.index, turnover))
        self.previous = state
        self.previous_date = trade_date
        return row

    def _form(self) -> Any:
        """Assign size, EP and abnormal-turnover groups from data through the previous trading day."""
        import numpy as np
        import pandas as pd

        state = self.previous
        codes = state.index.astype(str)
        board = codes.str[:2].isin(CH4_BOARDS) & ~codes.str.endswith(".BJ")
        size = state["total_mv"].where(board & state["total_mv"].gt(0).to_numpy()).dropna()
        if size.empty:
            return None
        size = size[size > size.quantile(CH4_SHELL_FRACTION)]
        listed = self.list_dates.reindex(size.index)
        seasoned = ((self.previous_date - listed).dt.days >= CH4_SEASONING_DAYS).fillna(False).to_numpy()

        last_month = (self.previous_date.year, self.previous_date.month)
        year_days = pd.Series(0, index=size.index)
        month_days = pd.Series(0, index=size.index)
        market_days = 0
        for day, traded, _ in self.history:
            present = size.index.isin(traded).astype(int)
            year_days += present
            if (day.year, day.month) == last_month:
                month_days += present
                market_days += 1
        # A Spring Festival February can have fewer than 15 trading days, so the month rule scales with short months.
        month_minimum = min(CH4_MONTH_MIN_DAYS, int(np.ceil(CH4_MONTH_MIN_SHARE * market_days)))
        traded = (year_days >= CH4_YEAR_MIN_DAYS) & (month_days >= month_minimum)
        eligible = seasoned & traded.to_numpy()
        size = size[eligible]
        if len(size) < CH4_MIN_STOCKS:
            return None

        turnovers = pd.concat([turnover.reindex(size.index) for _, _, turnover in self.history], axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            abnormal = turnovers.iloc[:, -CH4_TURNOVER_DAYS:].mean(axis=1) / turnovers.mean(axis=1)
        abnormal = abnormal.where(np.isfinite(abnormal))
        pe = state["pe_ttm"].reindex(size.index) if "pe_ttm" in state.columns else pd.Series(np.nan, index=size.index)
        # Loss-making firms have no pe_ttm; they sort below every positive EP, into the growth group.
        ep = (1.0 / pe.where(pe > 0)).fillna(-1.0)

        groups = pd.DataFrame(index=size.index)
        groups["size"] = np.where(size <= size.median(), "S", "B")
        low, high = ep.quantile([0.3, 0.7])
        groups["value"] = np.where(ep <= low, "G", np.where(ep > high, "V", "M"))
        sentiment = pd.Series("", index=size.index, dtype=object)
        if abnormal.notna().sum() >= CH4_MIN_STOCKS:
            low, high = abnormal.quantile([0.3, 0.7])
            sentiment = pd.Series(
                np.where(abnormal <= low, "P", np.where(abnormal > high, "O", "M")), index=size.index, dtype=object
            ).where(abnormal.notna(), "")
        groups["sentiment"] = sentiment
        return groups

    def _returns(self, returns: Any) -> dict[str, Any]:
        import numpy as np

        groups = self.groups
        if groups is None:
            return {}
        weights = self.previous["total_mv"].reindex(groups.index)
        today = returns.reindex(groups.index)
        valid = (today.notna() & weights.gt(0)).to_numpy()
        if valid.sum() < CH4_MIN_STOCKS:
            return {"ch4_stocks": int(valid.sum())}
        groups, weights, today = groups[valid], weights[valid].to_numpy(), today[valid].to_numpy()

        def portfolio(members: Any) -> float:
            members = np.asarray(members)
            if not members.any():
                return np.nan
            return float((today[members] * weights[members]).sum() / weights[members].sum())

        size = groups["size"].to_numpy()
        value = groups["value"].to_numpy()
        sentiment = groups["sentiment"].to_numpy()
        ep = {(s, v): portfolio((size == s) & (value == v)) for s in "SB" for v in "VMG"}
        turn = {(s, t): portfolio((size == s) & (sentiment == t)) for s in "SB" for t in "POM"}
        smb = np.mean([ep["S", v] for v in "VMG"]) - np.mean([ep["B", v] for v in "VMG"])
        vmg = (ep["S", "V"] + ep["B", "V"]) / 2.0 - (ep["S", "G"] + ep["B", "G"]) / 2.0
        pmo = (turn["S", "P"] + turn["B", "P"]) / 2.0 - (turn["S", "O"] + turn["B", "O"]) / 2.0
        smb_turnover = np.mean([turn["S", t] for t in "POM"]) - np.mean([turn["B", t] for t in "POM"])
        return {
            "mkt": portfolio(np.ones(len(today), dtype=bool)),
            "smb": float(smb),
            "vmg": float(vmg),
            "pmo": float(pmo),
            "smb4": float((smb + smb_turnover) / 2.0),
            "ch4_stocks": int(len(today)),
        }
