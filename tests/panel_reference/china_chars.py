"""Pandas reference implementation of the china_chars factor set.

RQuant computes china_chars with KunQuant (``rquant.factors.libraries.china_chars``). This calculator, the original
implementation, is kept for the parity tests that check the KunQuant graphs against it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from rquant.errors import DataContractError
from rquant.factors.libraries.china_chars import CHINA_CHARS_NAMES, HALF_YEAR, MARKET_FACTORS, MONTH, QUARTER, YEAR
from rquant.factors.panel_operators import replace_inf, rolling_regression, safe_div
from rquant.factors.statements import QuarterGrid, growth, lag, nan_mean, ratio, rolling_quarter_std

Panel = pd.DataFrame


def calculate(inputs: Mapping[str, Any]) -> dict[str, Panel]:
    """Every characteristic by source name, as dates x instruments panels."""
    calculator = ChinaChars(inputs)
    return {name: calculator.calculate(name) for name in CHINA_CHARS_NAMES}


class ChinaChars:
    """Calculator with shared intermediates; ``calculate`` returns one characteristic panel."""

    def __init__(self, inputs: Mapping[str, Any]) -> None:
        self.inputs = inputs
        close = inputs["close"]
        self.dates = pd.DatetimeIndex(close.index)
        self.instruments = pd.Index(close.columns)
        self.trading = close.notna()
        self.price = close.ffill()
        self.returns = close / close.shift(1) - 1.0
        self._cache: dict[str, Any] = {}
        self._grid: QuarterGrid | None = None

    # ------------------------------------------------------------------ infrastructure
    def calculate(self, name: str) -> Panel:
        method: Callable[[], Panel] = getattr(self, f"_char_{name}")
        result = method()
        if isinstance(result, np.ndarray):
            result = Panel(result, index=self.dates, columns=self.instruments)
        result = replace_inf(result.astype(float))
        return result.where(self.trading)

    def _panel(self, key: str) -> Panel:
        value = self.inputs.get(key)
        if value is None:
            raise DataContractError(f"china_chars input is missing: {key}")
        return value

    def _series(self, key: str) -> np.ndarray:
        return np.asarray(pd.Series(self._panel(key)).reindex(self.dates), dtype=float)

    @property
    def grid(self) -> QuarterGrid:
        if self._grid is None:
            self._grid = QuarterGrid(self._panel("fundamentals"), self.dates, self.instruments)
        return self._grid

    def _turnover(self) -> Panel:
        return self._panel("basic_turnover_rate")

    def _float_value(self) -> Panel:
        return self._panel("basic_circ_mv")

    def _cap(self) -> Panel:
        return self._panel("cap")

    def _rolling(self, panel: Panel, window: int, minimum: int) -> Any:
        return panel.rolling(window, min_periods=minimum)

    def _regression(self, key: str, regressors: tuple[str, ...], window: int, minimum: int, **kwargs: Any) -> dict:
        if key not in self._cache:
            x = np.column_stack([self._series(name) for name in regressors])
            self._cache[key] = rolling_regression(
                self.returns.to_numpy(dtype=float), x, window, minimum, **kwargs
            )
        return self._cache[key]

    def _capm(self) -> dict:
        return self._regression("capm", ("mkt",), HALF_YEAR, 40)

    def _ff3(self) -> dict:
        return self._regression("ff3", MARKET_FACTORS, HALF_YEAR, 40)

    def _momentum(self, start: int, end: int) -> Panel:
        """Return from ``start`` trading days ago to ``end`` trading days ago (start > end)."""
        return self.price.shift(end) / self.price.shift(start) - 1.0

    def _market_panel(self, values: np.ndarray) -> Panel:
        return Panel(np.repeat(values[:, None], len(self.instruments), axis=1), index=self.dates,
                     columns=self.instruments)

    # ------------------------------------------------------------------ liquidity
    def _char_size(self) -> Panel:
        return np.log(self._float_value().where(self._float_value() > 0))

    def _char_size3(self) -> Panel:
        return self._char_size() ** 3

    def _monthly_turnover(self, window: int, minimum: int) -> Panel:
        mean = self._rolling(self._turnover(), window, minimum).mean()
        return np.log((mean * MONTH).where(mean > 0))

    def _char_turnm(self) -> Panel:
        return self._monthly_turnover(MONTH, 15)

    def _char_turnq(self) -> Panel:
        return self._monthly_turnover(QUARTER, 40)

    def _char_turna(self) -> Panel:
        return self._monthly_turnover(YEAR, 120)

    def _char_vturn(self) -> Panel:
        return self._rolling(self._turnover(), HALF_YEAR, 40).std()

    def _char_cvturn(self) -> Panel:
        rolling = self._rolling(self._turnover(), HALF_YEAR, 40)
        return safe_div(rolling.std(), rolling.mean())

    def _char_abturn(self) -> Panel:
        turnover = self._turnover()
        return safe_div(self._rolling(turnover, MONTH, 7).mean(), self._rolling(turnover, 250, 85).mean())

    def _monthly_value(self, window: int, minimum: int) -> Panel:
        mean = self._rolling(self._panel("amount"), window, minimum).mean()
        return np.log((mean * MONTH).where(mean > 0))

    def _char_dtvm(self) -> Panel:
        return self._monthly_value(MONTH, 15)

    def _char_dtvq(self) -> Panel:
        return self._monthly_value(QUARTER, 40)

    def _char_dtva(self) -> Panel:
        return self._monthly_value(YEAR, 120)

    def _char_vdtv(self) -> Panel:
        return self._rolling(self._panel("amount"), HALF_YEAR, 40).std()

    def _char_cvd(self) -> Panel:
        rolling = self._rolling(self._panel("amount"), HALF_YEAR, 40)
        return safe_div(rolling.std(), rolling.mean())

    def _char_ami(self) -> Panel:
        amount = self._panel("amount")
        illiquidity = safe_div(self.returns.abs(), amount.where(amount > 0))
        return self._rolling(illiquidity, MONTH, 15).mean()

    # ------------------------------------------------------------------ risk
    def _char_idvc(self) -> np.ndarray:
        return self._capm()["resid_std"]

    def _char_idvff(self) -> np.ndarray:
        return self._ff3()["resid_std"]

    def _char_tv(self) -> Panel:
        return self._rolling(self.returns, HALF_YEAR, 40).std()

    def _char_idsff(self) -> np.ndarray:
        return self._ff3()["resid_skew"]

    def _char_idsc(self) -> np.ndarray:
        return self._capm()["resid_skew"]

    def _char_ts(self) -> Panel:
        return self._rolling(self.returns, HALF_YEAR, 40).skew()

    def _char_cs(self) -> np.ndarray:
        return self._capm()["coskew"]

    def _char_dbeta(self) -> np.ndarray:
        window, minimum = 240, 50
        market = self._series("mkt")
        returns = self.returns.to_numpy(dtype=float)
        result = np.full(returns.shape, np.nan)
        if len(market) < window:
            return result
        market_windows = np.lib.stride_tricks.sliding_window_view(market, window)
        with np.errstate(invalid="ignore"):
            down = market_windows < nan_mean(market_windows, 1)[0][:, None]
        market_values = np.where(np.isfinite(market_windows), market_windows, 0.0)
        for column in range(returns.shape[1]):
            stock = returns[:, column]
            observed = np.isfinite(stock)
            mask = np.lib.stride_tricks.sliding_window_view(observed, window) & down
            values = np.lib.stride_tricks.sliding_window_view(np.where(observed, stock, 0.0), window) * mask
            m = market_values * mask
            count = mask.sum(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                mean_r = values.sum(axis=1) / count
                mean_m = m.sum(axis=1) / count
                covariance = (values * m).sum(axis=1) / count - mean_r * mean_m
                variance = (m * m).sum(axis=1) / count - mean_m**2
                beta = covariance / variance
            result[window - 1 :, column] = np.where((count >= minimum) & (variance > 0), beta, np.nan)
        return result

    def _char_betafp(self) -> Panel:
        log_returns = np.log1p(self.returns)
        log_market = np.log1p(self._series("mkt"))
        market = self._market_panel(log_market)
        sigma_stock = self._rolling(log_returns, HALF_YEAR, 40).std()
        sigma_market = self._rolling(market.where(log_returns.notna()), HALF_YEAR, 40).std()
        stock_3d = log_returns.rolling(3, min_periods=3).sum()
        market_3d = self._market_panel(pd.Series(log_market).rolling(3, min_periods=3).sum().to_numpy())
        correlation = stock_3d.rolling(HALF_YEAR, min_periods=40).corr(market_3d)
        return correlation.clip(-1.0, 1.0) * safe_div(sigma_stock, sigma_market)

    def _char_betadm(self) -> np.ndarray:
        if "dimson" not in self._cache:
            market = self._series("mkt")
            lagged = np.concatenate([[np.nan], market[:-1]])
            lead = np.concatenate([market[1:], [np.nan]])
            fit = rolling_regression(
                self.returns.to_numpy(dtype=float), np.column_stack([lagged, market, lead]), HALF_YEAR, 40
            )
            # A window ending on day d uses the market return of d + 1, so its estimate is published on d + 1.
            shifted = np.full(fit["slopes"].shape, np.nan)
            shifted[1:] = fit["slopes"][:-1]
            self._cache["dimson"] = shifted
        return self._cache["dimson"]

    def _char_beta(self) -> np.ndarray:
        return self._capm()["beta"]

    # ------------------------------------------------------------------ past returns
    def _char_m1(self) -> Panel:
        return self._momentum(MONTH, 0)

    def _char_m11(self) -> Panel:
        return self._momentum(YEAR, MONTH)

    def _char_m60(self) -> Panel:
        return self._momentum(5 * YEAR, YEAR)

    def _char_m6(self) -> Panel:
        return self._momentum(6 * MONTH, MONTH)

    def _char_m3(self) -> Panel:
        return self._momentum(3 * MONTH, MONTH)

    def _char_indmom(self) -> Panel:
        sector = self._panel("sector")
        returns = self._momentum(6 * MONTH, 0)
        weights = self._float_value().where(returns.notna() & self._float_value().gt(0))
        result = Panel(np.nan, index=self.dates, columns=self.instruments)
        codes = pd.unique(sector.to_numpy().ravel())
        for code in codes[np.isfinite(codes)]:
            members = sector.eq(code)
            member_weights = weights.where(members)
            industry = (returns * member_weights).sum(axis=1) / member_weights.sum(axis=1)
            result = result.mask(members, np.broadcast_to(industry.to_numpy()[:, None], result.shape))
        return result

    def _char_m24(self) -> Panel:
        return self._momentum(3 * YEAR, YEAR)

    def _char_mchg(self) -> Panel:
        return self._momentum(6 * MONTH, MONTH) - self._momentum(YEAR, 6 * MONTH)

    def _residual_momentum(self) -> dict:
        slices = {"im12": (720 - YEAR, 720 - MONTH), "im6": (720 - 6 * MONTH, 720 - MONTH)}
        return self._regression("ff3_momentum", MARKET_FACTORS, 720, 360, residual_slices=slices)

    def _char_im12(self) -> np.ndarray:
        return self._residual_momentum()["im12"]

    def _char_im6(self) -> np.ndarray:
        return self._residual_momentum()["im6"]

    def _char_w52(self) -> Panel:
        return safe_div(self.price, self.price.rolling(240, min_periods=120).max())

    def _char_mdr(self) -> np.ndarray:
        window, minimum, top = MONTH, 15, 5
        returns = self.returns.to_numpy(dtype=float)
        result = np.full(returns.shape, np.nan)
        if len(returns) < window:
            return result
        for first in range(0, returns.shape[1], 64):
            block = np.lib.stride_tricks.sliding_window_view(returns[:, first : first + 64], window, axis=0)
            count = np.isfinite(block).sum(axis=2)
            largest = -np.sort(np.where(np.isfinite(block), -block, np.inf), axis=2)[:, :, :top]
            mean, _ = nan_mean(largest, 2)
            result[window - 1 :, first : first + 64] = np.where(count >= minimum, mean, np.nan)
        return result

    def _char_pr(self) -> Panel:
        return self._panel("basic_close")

    def _char_abr(self) -> Panel:
        frame = self._panel("fundamentals")
        abnormal = self.returns.sub(self._series("mkt"), axis=0)
        window_sum = abnormal.rolling(4, min_periods=3).sum().to_numpy()
        announcement = np.searchsorted(self.dates.to_numpy(), pd.to_datetime(frame["ann_date"]).to_numpy())
        columns = self.instruments.get_indexer(frame["instrument"])
        published = announcement + 1
        keep = (columns >= 0) & (published < len(self.dates))
        events = np.full(window_sum.shape, np.nan)
        events[published[keep], columns[keep]] = window_sum[published[keep], columns[keep]]
        return Panel(events, index=self.dates, columns=self.instruments).ffill()

    def _char_season(self) -> Panel:
        monthly = self._momentum(MONTH, 0)
        stacked = np.stack([monthly.shift(YEAR * k - MONTH).to_numpy() for k in range(1, 6)], axis=2)
        mean, count = nan_mean(stacked, 2)
        return np.where(count >= 3, mean, np.nan)

    # ------------------------------------------------------------------ fundamentals helpers
    def _q(self, values: np.ndarray) -> Panel:
        return self.grid.panel(values)

    def _y(self, values: np.ndarray) -> Panel:
        return self.grid.panel(values, annual=True)

    def _book_equity(self) -> np.ndarray:
        g = self.grid
        return g.value("total_hldr_eqy_exc_min_int") - g.value("oth_eqt_tools_p_shr")

    def _debt(self) -> np.ndarray:
        g = self.grid
        return g.value("st_borr") + g.value("non_cur_liab_due_1y") + g.value("lt_borr") + g.value("bond_payable")

    def _net_operating_assets(self) -> np.ndarray:
        g = self.grid
        return (
            g.value("total_hldr_eqy_inc_min_int")
            + self._debt()
            - g.value("money_cap")
            - g.value("trad_asset")
        )

    def _operating_profit(self, flow: Callable[[str], np.ndarray]) -> np.ndarray:
        return flow("revenue") - flow("oper_cost") - flow("sell_exp") - flow("admin_exp") - flow("fin_exp")

    def _roe(self) -> np.ndarray:
        return ratio(self.grid.single("n_income_attr_p"), lag(self._book_equity(), 1))

    def _roa(self) -> np.ndarray:
        return ratio(self.grid.single("n_income"), lag(self.grid.value("total_assets"), 1))

    def _lagged_assets_change(self, values: np.ndarray) -> np.ndarray:
        assets = self.grid.value("total_assets")
        return ratio(values - lag(values, 4), lag(assets, 4))

    # ------------------------------------------------------------------ profitability
    def _char_roe(self) -> Panel:
        return self._q(self._roe())

    def _char_droe(self) -> Panel:
        roe = self._roe()
        return self._q(roe - lag(roe, 4))

    def _char_roa(self) -> Panel:
        return self._q(self._roa())

    def _char_droa(self) -> Panel:
        roa = self._roa()
        return self._q(roa - lag(roa, 4))

    def _char_rna(self) -> Panel:
        return self._q(ratio(self.grid.single("operate_profit"), lag(self._net_operating_assets(), 1)))

    def _char_pm(self) -> Panel:
        g = self.grid
        return self._q(ratio(g.single("operate_profit"), g.single("revenue")))

    def _char_ato(self) -> Panel:
        return self._q(ratio(self.grid.single("revenue"), lag(self._net_operating_assets(), 1)))

    def _char_ct(self) -> Panel:
        g = self.grid
        return self._q(ratio(g.single("revenue"), lag(g.value("total_assets"), 1)))

    def _char_gpa(self) -> Panel:
        g = self.grid
        return self._y(ratio(g.annual("revenue") - g.annual("oper_cost"), g.value("total_assets")))

    def _char_gpla(self) -> Panel:
        g = self.grid
        return self._q(ratio(g.single("revenue") - g.single("oper_cost"), lag(g.value("total_assets"), 1)))

    def _char_ope(self) -> Panel:
        return self._y(ratio(self._operating_profit(self.grid.annual), self._book_equity()))

    def _char_ople(self) -> Panel:
        return self._q(ratio(self._operating_profit(self.grid.single), lag(self._book_equity(), 1)))

    def _char_opa(self) -> Panel:
        return self._y(ratio(self._operating_profit(self.grid.annual), self.grid.value("total_assets")))

    def _char_opla(self) -> Panel:
        g = self.grid
        return self._q(ratio(self._operating_profit(g.single), lag(g.value("total_assets"), 1)))

    def _char_tbi(self) -> Panel:
        g = self.grid
        return self._q(ratio(g.single("total_profit"), g.single("n_income")))

    def _char_bl(self) -> Panel:
        return self._q(ratio(self.grid.value("total_assets"), self._book_equity()))

    def _char_sg(self) -> Panel:
        revenue = self.grid.annual("revenue")
        return self._y(growth(revenue, lag(revenue, 4)))

    def _char_sgq(self) -> Panel:
        revenue = self.grid.single("revenue")
        return self._q(growth(revenue, lag(revenue, 4)))

    def _char_fscore(self) -> Panel:
        g = self.grid
        assets = g.value("total_assets")
        income = g.single("n_income")
        cash_flow = g.single("n_cashflow_act")
        revenue = g.single("revenue")
        roa = ratio(income, lag(assets, 1))
        leverage = ratio(g.value("lt_borr") + g.value("bond_payable"), assets)
        current = ratio(g.value("total_cur_assets"), g.value("total_cur_liab"))
        margin = ratio(revenue - g.single("oper_cost"), revenue)
        turnover = ratio(revenue, lag(assets, 1))
        shares = g.value("total_share")
        components = (
            (roa, roa > 0),
            (cash_flow, cash_flow > 0),
            (roa - lag(roa, 4), roa - lag(roa, 4) > 0),
            (cash_flow - income, cash_flow > income),
            (leverage - lag(leverage, 4), leverage < lag(leverage, 4)),
            (current - lag(current, 4), current > lag(current, 4)),
            (shares - lag(shares, 4), shares <= lag(shares, 4)),
            (margin - lag(margin, 4), margin > lag(margin, 4)),
            (turnover - lag(turnover, 4), turnover > lag(turnover, 4)),
        )
        known = np.all([np.isfinite(value) for value, _ in components], axis=0)
        score = np.sum([signal.astype(float) for _, signal in components], axis=0)
        return self._q(np.where(known, score, np.nan))

    def _char_oscore(self) -> Panel:
        g = self.grid
        assets = g.value("total_assets")
        liabilities = g.value("total_liab")
        current_assets = g.value("total_cur_assets")
        current_liabilities = g.value("total_cur_liab")
        income = g.ttm("n_income")
        previous = lag(income, 4)
        with np.errstate(invalid="ignore", divide="ignore"):
            log_assets = np.log(np.where(assets > 0, assets, np.nan))
        score = (
            -1.32
            - 0.407 * log_assets
            + 6.03 * ratio(liabilities, assets)
            - 1.43 * ratio(current_assets - current_liabilities, assets)
            + 0.076 * ratio(current_liabilities, current_assets)
            - 1.72 * np.where(np.isfinite(liabilities - assets), (liabilities > assets).astype(float), np.nan)
            - 2.37 * ratio(income, assets)
            - 1.83 * ratio(g.ttm("n_cashflow_act"), liabilities)
            + 0.285 * np.where(np.isfinite(income + previous), ((income < 0) & (previous < 0)).astype(float), np.nan)
            - 0.521 * ratio(income - previous, np.abs(income) + np.abs(previous))
        )
        return self._q(score)

    # ------------------------------------------------------------------ value
    def _to_price(self, numerator: Panel) -> Panel:
        cap = self._cap()
        return safe_div(numerator, cap.where(cap > 0))

    def _char_bm(self) -> Panel:
        return self._to_price(self._q(self._book_equity()))

    def _char_dm(self) -> Panel:
        return self._to_price(self._q(self.grid.value("total_liab")))

    def _char_am(self) -> Panel:
        return self._to_price(self._q(self.grid.value("total_assets")))

    def _char_ep(self) -> Panel:
        return self._to_price(self._q(self.grid.ttm("n_income_attr_p")))

    def _depreciation_ttm(self) -> np.ndarray:
        g = self.grid
        return g.ttm("depr_fa_coga_dpba") + g.ttm("amort_intang_assets") + g.ttm("lt_amort_deferred_exp")

    def _char_cfp(self) -> Panel:
        return self._to_price(self._q(self.grid.ttm("n_income") + self._depreciation_ttm()))

    def _char_sr(self) -> Panel:
        revenue = self.grid.annual("revenue")
        total = np.zeros(revenue.shape)
        for year in range(1, 6):
            total = total + (6 - year) * growth(lag(revenue, 4 * (year - 1)), lag(revenue, 4 * year))
        return self._y(total / 15.0)

    def _char_em(self) -> Panel:
        g = self.grid
        net_debt = self._q(self._debt() + g.value("oth_eqt_tools_p_shr") + g.value("minority_int")
                           - g.value("money_cap") - g.value("trad_asset"))
        ebitda = self._q(g.ttm("total_profit") + g.ttm("fin_exp") + self._depreciation_ttm())
        return safe_div(self._cap() + net_debt, ebitda.where(ebitda > 0))

    def _char_sp(self) -> Panel:
        return self._to_price(self._q(self.grid.ttm("revenue")))

    def _char_ocfp(self) -> Panel:
        return self._to_price(self._q(self.grid.ttm("n_cashflow_act")))

    def _char_de(self) -> Panel:
        return self._y(ratio(self.grid.value("total_liab"), self._book_equity()))

    def _net_debt(self) -> np.ndarray:
        g = self.grid
        return self._debt() + g.value("oth_eqt_tools_p_shr") - g.value("money_cap") - g.value("trad_asset")

    def _char_ebp(self) -> Panel:
        net_debt = self._q(self._net_debt())
        book = self._q(self._book_equity())
        return safe_div(net_debt + book, net_debt + self._cap())

    def _char_ndp(self) -> Panel:
        return self._to_price(self._q(self._net_debt()))

    # ------------------------------------------------------------------ investment
    def _char_ag(self) -> Panel:
        assets = self.grid.value("total_assets")
        return self._q(growth(assets, lag(assets, 4)))

    def _char_dpia(self) -> Panel:
        g = self.grid
        return self._q(self._lagged_assets_change(g.value("fix_assets") + g.value("inventories")))

    def _char_noa(self) -> Panel:
        return self._q(ratio(self._net_operating_assets(), lag(self.grid.value("total_assets"), 4)))

    def _char_dnoa(self) -> Panel:
        return self._q(self._lagged_assets_change(self._net_operating_assets()))

    def _char_ig(self) -> Panel:
        capex = self.grid.annual("c_pay_acq_const_fiolta")
        return self._y(growth(capex, lag(capex, 4)))

    def _char_cei(self) -> Panel:
        cap = self._cap().ffill()
        with np.errstate(invalid="ignore", divide="ignore"):
            value_growth = np.log(safe_div(cap, cap.shift(YEAR)))
            price_growth = np.log(safe_div(self.price, self.price.shift(YEAR)))
        return value_growth - price_growth

    def _char_cdi(self) -> Panel:
        liabilities = self.grid.value("total_liab")
        base = lag(liabilities, 20)
        with np.errstate(invalid="ignore", divide="ignore"):
            result = np.log(ratio(liabilities, np.where(base > 0, base, np.nan)))
        return self._y(result)

    def _char_ivg(self) -> Panel:
        inventory = self.grid.value("inventories")
        return self._y(growth(inventory, lag(inventory, 4)))

    def _char_ivchg(self) -> Panel:
        g = self.grid
        inventory = g.value("inventories")
        assets = g.value("total_assets")
        return self._y(ratio(inventory - lag(inventory, 4), (assets + lag(assets, 4)) / 2.0))

    def _char_oacc(self) -> Panel:
        g = self.grid
        return self._y(ratio(g.annual("n_income") - g.annual("n_cashflow_act"), lag(g.value("total_assets"), 4)))

    def _working_capital(self) -> tuple[np.ndarray, np.ndarray]:
        g = self.grid
        operating_assets = g.value("total_cur_assets") - g.value("money_cap") - g.value("trad_asset")
        operating_liabilities = g.value("total_cur_liab") - g.value("st_borr") - g.value("non_cur_liab_due_1y")
        return operating_assets, operating_liabilities

    def _non_current(self) -> tuple[np.ndarray, np.ndarray]:
        g = self.grid
        assets = g.value("total_assets") - g.value("total_cur_assets") - g.value("lt_eqt_invest")
        liabilities = g.value("total_liab") - g.value("total_cur_liab") - g.value("lt_borr") - g.value("bond_payable")
        return assets, liabilities

    def _financial(self) -> np.ndarray:
        g = self.grid
        return g.value("trad_asset") + g.value("lt_eqt_invest") - self._debt() - g.value("oth_eqt_tools_p_shr")

    def _char_tacc(self) -> Panel:
        current_assets, current_liabilities = self._working_capital()
        non_current_assets, non_current_liabilities = self._non_current()
        working = current_assets - current_liabilities
        total = working + (non_current_assets - non_current_liabilities) + self._financial()
        return self._y(self._lagged_assets_change(total))

    def _char_dwc(self) -> Panel:
        current_assets, current_liabilities = self._working_capital()
        return self._y(self._lagged_assets_change(current_assets - current_liabilities))

    def _char_dcoa(self) -> Panel:
        return self._y(self._lagged_assets_change(self._working_capital()[0]))

    def _char_dcol(self) -> Panel:
        return self._y(self._lagged_assets_change(self._working_capital()[1]))

    def _char_dnco(self) -> Panel:
        assets, liabilities = self._non_current()
        return self._y(self._lagged_assets_change(assets - liabilities))

    def _char_dnca(self) -> Panel:
        return self._y(self._lagged_assets_change(self._non_current()[0]))

    def _char_dncl(self) -> Panel:
        return self._y(self._lagged_assets_change(self._non_current()[1]))

    def _char_dfin(self) -> Panel:
        return self._y(self._lagged_assets_change(self._financial()))

    def _char_dbe(self) -> Panel:
        return self._y(self._lagged_assets_change(self._book_equity()))

    # ------------------------------------------------------------------ other
    def _char_rdm(self) -> Panel:
        return self._to_price(self._q(self.grid.single("rd_exp")))

    def _char_rds(self) -> Panel:
        g = self.grid
        return self._q(ratio(g.single("rd_exp"), g.single("revenue")))

    def _char_ol(self) -> Panel:
        g = self.grid
        costs = g.annual("oper_cost") + g.annual("sell_exp") + g.annual("admin_exp")
        return self._y(ratio(costs, g.value("total_assets")))

    def _char_age(self) -> Panel:
        listed = pd.Series(self._panel("list_date")).reindex(self.instruments)
        days = self.dates.to_numpy()[:, None] - pd.to_datetime(listed).to_numpy()[None, :]
        months = days / np.timedelta64(1, "D") / 30.4375
        return np.where(months >= 0, months, np.nan)

    def _annual_change_gap(self, first: np.ndarray, second: np.ndarray) -> Panel:
        return self._y(growth(first, lag(first, 4)) - growth(second, lag(second, 4)))

    def _char_dsi(self) -> Panel:
        g = self.grid
        return self._annual_change_gap(g.annual("revenue"), g.value("inventories"))

    def _char_dsa(self) -> Panel:
        g = self.grid
        return self._annual_change_gap(g.annual("revenue"), g.value("accounts_receiv"))

    def _char_dgs(self) -> Panel:
        g = self.grid
        return self._annual_change_gap(g.annual("revenue") - g.annual("oper_cost"), g.annual("revenue"))

    def _char_dss(self) -> Panel:
        g = self.grid
        return self._annual_change_gap(g.annual("revenue"), g.annual("sell_exp") + g.annual("admin_exp"))

    def _char_etr(self) -> Panel:
        g = self.grid
        rate = ratio(g.annual("income_tax"), g.annual("total_profit"))
        average, _ = nan_mean(np.stack([lag(rate, 4 * k) for k in (1, 2, 3)], axis=2), 2)
        income = g.annual("n_income_attr_p")
        earnings_change = self._y(income - lag(income, 4))
        return self._y(rate - average) * self._to_price(earnings_change)

    def _char_tan(self) -> Panel:
        g = self.grid
        tangible = (
            g.value("money_cap")
            + 0.715 * g.value("accounts_receiv")
            + 0.547 * g.value("inventories")
            + 0.535 * g.value("fix_assets")
        )
        return self._q(ratio(tangible, g.value("total_assets")))

    def _char_vcf(self) -> Panel:
        g = self.grid
        return self._q(rolling_quarter_std(ratio(g.single("n_cashflow_act"), g.single("revenue")), 16, 8))

    def _char_cta(self) -> Panel:
        g = self.grid
        return self._q(ratio(g.value("money_cap"), g.value("total_assets")))

    def _char_esm(self) -> Panel:
        g = self.grid
        lagged_assets = lag(g.value("total_assets"), 1)
        earnings = rolling_quarter_std(ratio(g.single("n_income"), lagged_assets), 16, 8)
        cash = rolling_quarter_std(ratio(g.single("n_cashflow_act"), lagged_assets), 16, 8)
        return self._q(ratio(earnings, cash))

    def _asset_liquidity(self) -> np.ndarray:
        g = self.grid
        cash = g.value("money_cap")
        current = g.value("total_cur_assets")
        assets = g.value("total_assets")
        return cash + 0.75 * (current - cash) + 0.5 * (assets - current - g.value("goodwill") - g.value("intan_assets"))

    def _char_ala(self) -> Panel:
        return self._q(ratio(self._asset_liquidity(), lag(self.grid.value("total_assets"), 1)))

    def _char_alm(self) -> Panel:
        g = self.grid
        market_assets = self._q(g.value("total_assets") - self._book_equity()) + self._cap()
        return safe_div(self._q(self._asset_liquidity()), market_assets.where(market_assets > 0))

    def _surprise(self, values: np.ndarray) -> np.ndarray:
        change = values - lag(values, 4)
        return ratio(change, rolling_quarter_std(change, 8, 6))

    def _char_sue(self) -> Panel:
        return self._q(self._surprise(self.grid.single("n_income_attr_p")))

    def _char_rs(self) -> Panel:
        return self._q(self._surprise(self.grid.single("revenue")))

    def _char_tes(self) -> Panel:
        g = self.grid
        tax = g.single("income_tax")
        return self._q(ratio(tax - lag(tax, 4), lag(g.value("total_assets"), 4)))
