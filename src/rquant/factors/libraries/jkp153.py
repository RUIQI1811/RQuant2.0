"""The Jensen-Kelly-Pedersen (JKP) global factor characteristics, evaluated as daily point-in-time A-share panels.

The catalog follows the 153 characteristics of "Is There a Replication Crisis in Finance?" (Jensen, Kelly and Pedersen,
Journal of Finance 2023) as listed in the authors' ``factor_details`` table, with the constructions of their public
code base. This module is an independent re-implementation: no upstream code or data is used, inputs come from
Tushare, and every characteristic is re-dated to the trading day on which it is known.

Daily adaptation
----------------
* Rows are trading dates and columns are instruments. A month is 21 trading days and a year 252; JKP's monthly
  windows (``_21d``, ``_126d``, ``_252d``, ``_1260d``, ``ret_x_y``, ``seas_*``) become the same number of trading days
  or of 21-day blocks ending on the row.
* Returns are close-to-close returns of RQuant's backward-adjusted prices; the market, SMB and HML returns come from
  ``reference/market_factors.parquet``. Returns are not reduced by a risk-free rate.
* Accounting characteristics use the first-reported statements of ``reference/fundamentals.parquet`` on the quarter
  grid of ``china_chars``: a quarter is usable once it and every earlier quarter are announced. Flows are
  trailing-twelve-month sums of single quarters (JKP's ``_x`` annual items, refreshed quarterly), ``q`` items are the
  latest single quarter, balance-sheet items the latest quarter end, and JKP's 12-month lags are four quarters.
  Compustat items map to Tushare lines as documented in ``ITEMS``; preferred stock is ``oth_eqt_tools_p_shr``, and
  share buybacks, which Chinese cash-flow statements do not report separately, are zero.
* Dividends are the trailing-twelve-month dividend yield of daily_basic times the total market value.
* The cross-sectional composites (mispricing and quality-minus-junk) rank only the stocks that are in the CSI 300 on
  that day, so later index members do not shape earlier cross-sections.
* Every output is missing on days the stock has no quote.

Left out because Tushare data at this account level cannot build them: employee counts (``emp_gr1``,
``sale_emp_gr1``), the q-factor model moments (``ivol_hxz4_21d``, ``iskew_hxz4_21d``), seasonality beyond ten years
(``seas_11_15*``, ``seas_16_20*``, before the data starts), and ``eqpo_me``, which equals ``div12m_me`` without
buybacks.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rquant.errors import DataContractError
from rquant.factors.libraries.base import FactorSpec, PanelFactorLibrary
from rquant.factors.panel_operators import replace_inf, rolling_regression, safe_div
from rquant.factors.statements import QuarterGrid, lag, nan_mean, ratio, rolling_quarter_std
from rquant.io import sha256_file, stable_hash

Panel = pd.DataFrame
MONTH = 21
YEAR = 252
JKP_DAILY_BASIC = {"turnover_rate": 0.01, "close": 1.0, "total_share": 10_000.0, "dv_ttm": 0.01}
MARKET_FACTORS = ("mkt", "smb", "hml")
DISCOUNT = 0.12

# Compustat item: Tushare statement lines summed into it.
ITEMS: dict[str, tuple[str, ...]] = {
    "at": ("total_assets",),
    "act": ("total_cur_assets",),
    "lct": ("total_cur_liab",),
    "lt": ("total_liab",),
    "che": ("money_cap", "trad_asset"),
    "ivst": ("trad_asset",),
    "ivao": ("lt_eqt_invest", "fa_avail_for_sale", "htm_invest"),
    "rect": ("accounts_receiv", "notes_receiv"),
    "invt": ("inventories",),
    "ppegt": ("fix_assets", "cip"),
    "ppent": ("fix_assets",),
    "intan": ("intan_assets", "goodwill"),
    "nca": ("total_nca",),
    "ncl": ("total_ncl",),
    "dlc": ("st_borr", "non_cur_liab_due_1y"),
    "dltt": ("lt_borr", "bond_payable"),
    "ap": ("acct_payable", "notes_payable"),
    "txp": ("taxes_payable",),
    "txditc": ("defer_tax_liab",),
    "pstk": ("oth_eqt_tools_p_shr",),
    "seq": ("total_hldr_eqy_exc_min_int",),
    "mib": ("minority_int",),
    "re": ("surplus_rese", "undistr_porfit"),
}
FLOWS: dict[str, tuple[str, ...]] = {
    "sale": ("revenue",),
    "cogs": ("oper_cost",),
    "xsga": ("sell_exp", "admin_exp", "rd_exp"),
    "xrd": ("rd_exp",),
    "dp": ("depr_fa_coga_dpba", "amort_intang_assets", "lt_amort_deferred_exp"),
    "xint": ("fin_exp_int_exp",),
    "txt": ("income_tax",),
    "pi": ("total_profit",),
    "ni": ("n_income_attr_p",),
    "oancf": ("n_cashflow_act",),
    "capx": ("c_pay_acq_const_fiolta",),
    "sstk": ("c_recp_cap_contrib",),
    "dltis": ("c_recp_borrow", "proc_issue_bonds"),
    "dltr": ("c_prepay_amt_borr",),
    "fincf": ("n_cash_flows_fnc_act",),
}
# Statement lines that mean "none reported" when blank, keyed by the line that shows the statement exists.
ZERO_FILL = {
    "total_assets": ("money_cap", "trad_asset", "lt_eqt_invest", "fa_avail_for_sale", "htm_invest",
                     "accounts_receiv", "notes_receiv", "inventories", "fix_assets", "cip", "intan_assets",
                     "goodwill", "st_borr", "non_cur_liab_due_1y", "lt_borr", "bond_payable", "acct_payable",
                     "notes_payable", "taxes_payable", "defer_tax_liab", "oth_eqt_tools_p_shr", "minority_int",
                     "surplus_rese", "undistr_porfit"),
    "revenue": ("sell_exp", "admin_exp", "rd_exp", "fin_exp_int_exp", "oper_cost"),
    "n_cashflow_act": ("depr_fa_coga_dpba", "amort_intang_assets", "lt_amort_deferred_exp", "c_pay_acq_const_fiolta",
                       "c_recp_cap_contrib", "c_recp_borrow", "proc_issue_bonds", "c_prepay_amt_borr",
                       "n_cash_flows_fnc_act"),
}
STATEMENT_LINES = tuple(sorted({line for lines in (*ITEMS.values(), *FLOWS.values()) for line in lines}))

# (source name, JKP theme, description, maximum lookback in trading days). Order is the stable public column order.
JKP_CHARS: tuple[tuple[str, str, str, int], ...] = (
    # Momentum
    ("niq_su", "momentum", "standardized earnings surprise over the prior eight quarters", 1),
    ("ret_6_1", "momentum", "return from day -126 to -21", 126),
    ("ret_12_1", "momentum", "return from day -252 to -21", YEAR),
    ("saleq_su", "momentum", "standardized revenue surprise over the prior eight quarters", 1),
    ("tax_gr1a", "momentum", "one-year change in tax expense over assets", 1),
    ("ni_inc8q", "momentum", "consecutive quarters (of the last eight) with year-on-year earnings increases", 1),
    ("prc_highprc_252d", "momentum", "price over its 252-day high", YEAR),
    ("resff3_6_1", "momentum", "FF3 residual momentum, months t-6..t-1 of a 756-day fit", 756),
    ("resff3_12_1", "momentum", "FF3 residual momentum, months t-12..t-1 of a 756-day fit", 756),
    # Value
    ("be_me", "value", "book equity over market equity", 1),
    ("debt_me", "value", "debt over market equity", 1),
    ("at_me", "value", "assets over market equity", 1),
    ("ret_60_12", "value", "return from day -1260 to -252", 5 * YEAR),
    ("ni_me", "value", "trailing earnings over market equity", 1),
    ("fcf_me", "value", "trailing free cash flow over market equity", 1),
    ("div12m_me", "value", "trailing twelve-month dividends over market equity", 1),
    ("eqnpo_me", "value", "dividends minus net equity issuance, over market equity", 1),
    ("sale_gr3", "value", "three-year sales growth", 1),
    ("sale_gr1", "value", "one-year sales growth", 1),
    ("ebitda_mev", "value", "EBITDA over enterprise value", 1),
    ("sale_me", "value", "sales over market equity", 1),
    ("ocf_me", "value", "operating cash flow over market equity", 1),
    ("ival_me", "value", "residual-income intrinsic value over market equity", 1),
    ("bev_mev", "value", "book enterprise value over enterprise value", 1),
    ("netdebt_me", "value", "net debt over market equity", 1),
    ("eq_dur", "value", "equity duration from ten-year AR(1) projections of ROE and growth", 1),
    # Investment
    ("capex_abn", "investment", "capex-to-sales over its average one to three years ago, minus 1", 1),
    ("at_gr1", "investment", "one-year asset growth", 1),
    ("ppeinv_gr1a", "investment", "one-year change in PP&E plus inventory over lagged assets", 1),
    ("noa_at", "investment", "net operating assets over lagged assets", 1),
    ("noa_gr1a", "investment", "one-year change in net operating assets over lagged assets", 1),
    ("lnoa_gr1a", "investment", "one-year change in long-term net operating assets over current plus lagged assets", 1),
    ("capx_gr1", "investment", "one-year capex growth", 1),
    ("capx_gr2", "investment", "two-year capex growth", 1),
    ("capx_gr3", "investment", "three-year capex growth", 1),
    ("chcsho_12m", "investment", "252-day change in total shares", YEAR),
    ("eqnpo_12m", "investment", "252-day log return minus log change in market equity", YEAR),
    ("debt_gr3", "investment", "three-year debt growth", 1),
    ("inv_gr1", "investment", "one-year inventory growth", 1),
    ("inv_gr1a", "investment", "one-year change in inventory over assets", 1),
    ("oaccruals_at", "investment", "operating accruals over assets", 1),
    ("taccruals_at", "investment", "total accruals over assets", 1),
    ("cowc_gr1a", "investment", "one-year change in current operating working capital over assets", 1),
    ("coa_gr1a", "investment", "one-year change in current operating assets over assets", 1),
    ("col_gr1a", "investment", "one-year change in current operating liabilities over assets", 1),
    ("nncoa_gr1a", "investment", "one-year change in net non-current operating assets over assets", 1),
    ("ncoa_gr1a", "investment", "one-year change in non-current operating assets over assets", 1),
    ("ncol_gr1a", "investment", "one-year change in non-current operating liabilities over assets", 1),
    ("nfna_gr1a", "investment", "one-year change in net financial assets over assets", 1),
    ("sti_gr1a", "investment", "one-year change in short-term investments over assets", 1),
    ("lti_gr1a", "investment", "one-year change in long-term investments over assets", 1),
    ("fnl_gr1a", "investment", "one-year change in financial liabilities over assets", 1),
    ("be_gr1a", "investment", "one-year change in book equity over assets", 1),
    ("oaccruals_ni", "investment", "operating accruals over absolute net income", 1),
    ("taccruals_ni", "investment", "total accruals over absolute net income", 1),
    ("netis_at", "investment", "net equity plus debt issuance over assets", 1),
    ("eqnetis_at", "investment", "net equity issuance over assets", 1),
    ("dbnetis_at", "investment", "net debt issuance over assets", 1),
    # Profitability
    ("niq_be", "profitability", "quarterly earnings over book equity a quarter earlier", 1),
    ("niq_be_chg1", "profitability", "one-year change in niq_be", 1),
    ("niq_at", "profitability", "quarterly earnings over assets a quarter earlier", 1),
    ("niq_at_chg1", "profitability", "one-year change in niq_at", 1),
    ("ebit_bev", "profitability", "EBIT over book enterprise value", 1),
    ("ebit_sale", "profitability", "EBIT over sales", 1),
    ("sale_bev", "profitability", "sales over book enterprise value", 1),
    ("at_turnover", "profitability", "sales over average assets", 1),
    ("gp_at", "profitability", "gross profit over assets", 1),
    ("gp_atl1", "profitability", "gross profit over lagged assets", 1),
    ("ope_be", "profitability", "operating profit after interest over book equity", 1),
    ("ope_bel1", "profitability", "operating profit after interest over lagged book equity", 1),
    ("op_at", "profitability", "operating profit before R&D over assets", 1),
    ("op_atl1", "profitability", "operating profit before R&D over lagged assets", 1),
    ("cop_at", "profitability", "cash-based operating profit over assets", 1),
    ("cop_atl1", "profitability", "cash-based operating profit over lagged assets", 1),
    ("f_score", "profitability", "Piotroski F-score", 1),
    ("o_score", "profitability", "Ohlson O-score", 1),
    ("z_score", "profitability", "Altman Z-score", 1),
    ("pi_nix", "profitability", "pretax income over net income", 1),
    ("at_be", "profitability", "assets over book equity", 1),
    ("saleq_gr1", "profitability", "quarterly sales growth over four quarters", 1),
    # Intangibles
    ("rd_me", "intangibles", "R&D expense over market equity", 1),
    ("rd_sale", "intangibles", "R&D expense over sales", 1),
    ("opex_at", "intangibles", "operating expenses over assets", 1),
    ("rd5_at", "intangibles", "five-year depreciated R&D capital over assets", 1),
    ("age", "intangibles", "months since listing", 1),
    ("dsale_dinv", "intangibles", "% change in sales minus % change in inventory (vs two-year average)", 1),
    ("dsale_drec", "intangibles", "% change in sales minus % change in receivables", 1),
    ("dgp_dsale", "intangibles", "% change in gross profit minus % change in sales", 1),
    ("dsale_dsga", "intangibles", "% change in sales minus % change in SG&A", 1),
    ("tangibility", "intangibles", "weighted liquid and fixed assets over assets", 1),
    ("kz_index", "intangibles", "Kaplan-Zingales index", 1),
    ("ocfq_saleq_std", "intangibles", "std of quarterly operating cash flow over sales, 16 quarters", 1),
    ("cash_at", "intangibles", "cash and short-term investments over assets", 1),
    ("ni_ar1", "intangibles", "AR(1) slope of earnings over assets, five annual pairs", 1),
    ("ni_ivol", "intangibles", "residual std of the earnings AR(1)", 1),
    ("earnings_variability", "intangibles", "std of ROA over std of cash ROA, five years", 1),
    ("aliq_at", "intangibles", "asset liquidity over lagged assets", 1),
    ("aliq_mat", "intangibles", "asset liquidity over market assets a year earlier", YEAR),
    ("seas_1_1an", "intangibles", "return in the same month a year ago", YEAR),
    ("seas_1_1na", "intangibles", "average return in the other months of the last year", YEAR),
    ("seas_2_5an", "intangibles", "average same-month return two to five years ago", 5 * YEAR),
    ("seas_2_5na", "intangibles", "average other-month return two to five years ago", 5 * YEAR),
    ("seas_6_10an", "intangibles", "average same-month return six to ten years ago", 10 * YEAR),
    ("seas_6_10na", "intangibles", "average other-month return six to ten years ago", 10 * YEAR),
    # Trading frictions
    ("market_equity", "frictions", "total market value", 1),
    ("ivol_ff3_21d", "frictions", "FF3 residual volatility, 21 days", MONTH),
    ("ivol_capm_252d", "frictions", "CAPM residual volatility, 252 days", YEAR),
    ("ivol_capm_21d", "frictions", "CAPM residual volatility, 21 days", MONTH),
    ("rvol_21d", "frictions", "return volatility, 21 days", MONTH),
    ("beta_60m", "frictions", "market beta from sixty 21-day returns", 60 * MONTH),
    ("betabab_1260d", "frictions", "Frazzini-Pedersen beta: 1260-day correlation times 252-day volatility ratio",
     5 * YEAR + 2),
    ("beta_dimson_21d", "frictions", "Dimson beta with lead and lag market returns, 21 days", MONTH + 1),
    ("turnover_126d", "frictions", "average daily turnover, 126 days", 126),
    ("turnover_var_126d", "frictions", "coefficient of variation of daily turnover, 126 days", 126),
    ("dolvol_126d", "frictions", "average traded value, 126 days", 126),
    ("dolvol_var_126d", "frictions", "coefficient of variation of traded value, 126 days", 126),
    ("prc", "frictions", "unadjusted share price", 1),
    ("ami_126d", "frictions", "Amihud illiquidity |r| / value * 1e6, 126 days", 126),
    ("zero_trades_21d", "frictions", "days without trading, turnover tiebreaker, 21 days", MONTH),
    ("zero_trades_126d", "frictions", "days without trading, turnover tiebreaker, 126 days scaled to 21", 126),
    ("zero_trades_252d", "frictions", "days without trading, turnover tiebreaker, 252 days scaled to 21", YEAR),
    ("rmax1_21d", "frictions", "largest daily return, 21 days", MONTH),
    ("rskew_21d", "frictions", "return skewness, 21 days", MONTH),
    ("iskew_capm_21d", "frictions", "CAPM residual skewness, 21 days", MONTH),
    ("iskew_ff3_21d", "frictions", "FF3 residual skewness, 21 days", MONTH),
    ("coskew_21d", "frictions", "co-skewness with the market, 21 days", MONTH),
    ("ret_1_0", "frictions", "return over the last 21 days", MONTH),
    ("betadown_252d", "frictions", "market beta on days the market falls, 252 days", YEAR),
    ("bidaskhl_21d", "frictions", "Corwin-Schultz high-low bid-ask spread, 21 days", MONTH + 1),
    # New
    ("ret_3_1", "new", "return from day -63 to -21", 3 * MONTH),
    ("ret_9_1", "new", "return from day -189 to -21", 9 * MONTH),
    ("ret_12_7", "new", "return from day -252 to -126", YEAR),
    ("corr_1260d", "new", "correlation of three-day returns with the market, 1260 days", 5 * YEAR + 2),
    ("rmax5_21d", "new", "average of the five largest daily returns, 21 days", MONTH),
    ("rmax5_rvol_21d", "new", "rmax5_21d over rvol_21d", MONTH),
    ("ni_be", "new", "trailing earnings over book equity", 1),
    ("ocf_at", "new", "operating cash flow over assets", 1),
    ("ocf_at_chg1", "new", "one-year change in ocf_at", 1),
    ("mispricing_perf", "new", "Stambaugh-Yuan performance mispricing score (percentile mean)", YEAR),
    ("mispricing_mgmt", "new", "Stambaugh-Yuan management mispricing score (percentile mean)", YEAR),
    ("qmj", "new", "quality minus junk composite", 5 * YEAR + 2),
    ("qmj_prof", "new", "quality minus junk: profitability", 1),
    ("qmj_growth", "new", "quality minus junk: five-year profitability growth", 1),
    ("qmj_safety", "new", "quality minus junk: safety", 5 * YEAR + 2),
)
JKP_NAMES = tuple(entry[0] for entry in JKP_CHARS)
# Outputs that later composites read again; every other output is released once published.
COMPOSITE_INPUTS = frozenset({"o_score", "z_score", "ret_12_1", "gp_at", "niq_at", "chcsho_12m", "oaccruals_at",
                              "noa_at", "at_gr1", "ppeinv_gr1a", "eqnpo_12m", "betabab_1260d"})


class JKP153Library(PanelFactorLibrary):
    """Jensen-Kelly-Pedersen characteristics, re-dated to daily point-in-time panels."""

    family = "jkp153"
    required_inputs = ("high", "low", "close", "amount", "cap")

    @property
    def specs(self) -> tuple[FactorSpec, ...]:
        return tuple(
            FactorSpec(
                canonical_name=f"jkp_{ordinal:03d}",
                source_name=name,
                family=self.family,
                ordinal=ordinal,
                formula=f"{theme}: {description}",
                max_lookback=lookback,
                implementation=f"rquant.factors.libraries.jkp153.JKPChars.{name}",
                catalog_version=1,
            )
            for ordinal, (name, theme, description, lookback) in enumerate(JKP_CHARS, 1)
        )

    def load_reference_inputs(self, canonical_root: Any, dates: Any, instruments: Any) -> tuple[dict[str, Any], dict]:
        from rquant.data.canonical import ts_code_to_qlib
        from rquant.data.reference import load_reference_manifest

        canonical_root = Path(canonical_root)
        root = canonical_root / "reference"
        manifest = load_reference_manifest(canonical_root)
        files = manifest["files"]
        if "fundamentals.parquet" not in files:
            raise DataContractError(
                "jkp153 needs financial statements; run `rquant data sync-fundamentals` and "
                "`rquant data build-reference` first"
            )
        dates = pd.DatetimeIndex(dates)
        years = set(dates.year)
        daily_files = sorted(
            name for name in files if name.startswith("daily_basic/") and int(name.split("=")[1][:4]) in years
        )
        for name in (*daily_files, "market_factors.parquet", "fundamentals.parquet"):
            if sha256_file(root / name) != files[name]:
                raise DataContractError(f"Reference file differs from its manifest: {root / name}")
        fundamentals = pd.read_parquet(root / "fundamentals.parquet")
        missing = sorted(set(STATEMENT_LINES) - set(fundamentals.columns))
        if missing:
            raise DataContractError(
                f"Fundamental reference lacks jkp153 statement items {missing[:5]}; run `rquant data "
                "sync-fundamentals` and `rquant data build-reference` again"
            )

        wanted = [("instrument", "in", list(instruments))]
        columns = ["instrument", "datetime", *JKP_DAILY_BASIC]
        daily = pd.concat(
            [pd.read_parquet(root / name, columns=columns, filters=wanted) for name in daily_files], ignore_index=True
        )
        inputs: dict[str, Any] = {}
        for field, scale in JKP_DAILY_BASIC.items():
            panel = daily.pivot(index="datetime", columns="instrument", values=field)
            inputs[f"basic_{field}"] = panel.reindex(index=dates, columns=instruments) * scale
        factors = pd.read_parquet(root / "market_factors.parquet").set_index("datetime").sort_index()
        for name in MARKET_FACTORS:
            inputs[name] = factors[name].reindex(dates)
        inputs["fundamentals"] = fundamentals[fundamentals["instrument"].isin(set(instruments))]

        basic = pd.read_parquet(root / "stock_basic.parquet", columns=["ts_code", "list_date"])
        stock_basic_hash = sha256_file(root / "stock_basic.parquet")
        basic = basic.assign(instrument=basic["ts_code"].map(ts_code_to_qlib)).drop_duplicates("instrument")
        listed = pd.to_datetime(basic.set_index("instrument")["list_date"], format="%Y%m%d", errors="coerce")
        inputs["list_date"] = listed.reindex(instruments)

        bar_files = sorted((canonical_root / "bars").glob("year=*/*.parquet"))
        bar_files = [path for path in bar_files if int(path.parent.name.split("=")[1]) in years]
        if not bar_files:
            raise DataContractError(f"Canonical bars do not exist under {canonical_root / 'bars'}")
        members = pd.concat(
            [pd.read_parquet(path, columns=["datetime", "instrument", "in_csi300"], filters=wanted)
             for path in bar_files],
            ignore_index=True,
        )
        members["datetime"] = pd.to_datetime(members["datetime"])
        membership = members.pivot(index="datetime", columns="instrument", values="in_csi300")
        inputs["universe"] = membership.reindex(index=dates, columns=instruments).fillna(False).astype(bool)
        hashes = {
            "jkp153_reference": manifest["fingerprint"],
            "jkp153_daily_basic": stable_hash({name: files[name] for name in daily_files}),
            "jkp153_market_factors": files["market_factors.parquet"],
            "jkp153_fundamentals": files["fundamentals.parquet"],
            "jkp153_stock_basic": stock_basic_hash,
        }
        return inputs, hashes

    def calculate(self, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        calculator = JKPChars(inputs)
        return {name: calculator.calculate(name) for name in JKP_NAMES}

    def calculate_into(self, inputs: Mapping[str, Any], output_buffers: Mapping[str, Any]) -> None:
        calculator = JKPChars(inputs)
        for spec in self.specs:
            output_buffers[spec.canonical_name][...] = calculator.calculate(spec.source_name).to_numpy(dtype=float)


# ---------------------------------------------------------------------------------------------- helpers
def prepare_fundamentals(frame: pd.DataFrame) -> pd.DataFrame:
    """Zero-fill statement lines a published statement leaves blank."""
    frame = frame.copy()
    for anchor, lines in ZERO_FILL.items():
        published = frame[anchor].notna()
        for line in lines:
            frame.loc[published & frame[line].isna(), line] = 0.0
    return frame


def positive(values: np.ndarray) -> np.ndarray:
    return np.where(values > 0, values, np.nan)


def growth(current: np.ndarray, previous: np.ndarray) -> np.ndarray:
    """Relative change, defined only for a positive base."""
    return ratio(current, positive(previous)) - 1.0


def window_count(values: np.ndarray, window: int) -> np.ndarray:
    """Finite values in every trailing ``window`` rows."""
    finite = np.concatenate([np.zeros((1, values.shape[1])), np.cumsum(np.isfinite(values), axis=0)])
    out = np.zeros(values.shape)
    out[window - 1 :] = finite[window:] - finite[:-window]
    out[: window - 1] = finite[1:window]
    return out


def step_sums(values: np.ndarray, step: int, count: int) -> tuple[np.ndarray, np.ndarray]:
    """Sum and count of the finite values on rows ``t, t - step, ..., t - step (count - 1)``."""
    finite = np.isfinite(values)
    chain = np.where(finite, values, 0.0)
    known = finite.astype(float)
    for row in range(step, len(values)):
        chain[row] += chain[row - step]
        known[row] += known[row - step]
    span = step * count
    total = chain.copy()
    seen = known.copy()
    total[span:] -= chain[:-span]
    seen[span:] -= known[:-span]
    return total, seen


def cross_section_rank(values: np.ndarray, universe: np.ndarray, *, z_score: bool) -> np.ndarray:
    """Per-day percentile rank (or rank z-score) among universe members; non-members are ranked into it."""
    out = np.full(values.shape, np.nan)
    for row in range(len(values)):
        members = universe[row] & np.isfinite(values[row])
        if members.sum() < 3:
            continue
        reference = np.sort(values[row, members])
        finite = np.isfinite(values[row])
        lower = np.searchsorted(reference, values[row, finite], side="left")
        upper = np.searchsorted(reference, values[row, finite], side="right")
        ranks = (lower + upper + 1) / 2.0
        if z_score:
            member_ranks = np.arange(1, len(reference) + 1)
            out[row, finite] = (ranks - member_ranks.mean()) / member_ranks.std(ddof=1)
        else:
            out[row, finite] = ranks / len(reference)
    return out


def composite(parts: list[np.ndarray], minimum: int) -> np.ndarray:
    mean, count = nan_mean(np.stack(parts, axis=2), 2)
    return np.where(count >= minimum, mean, np.nan)


def corwin_schultz(high: np.ndarray, low: np.ndarray) -> np.ndarray:
    """Daily Corwin-Schultz spread from today's and yesterday's high and low; negative estimates are zero."""
    with np.errstate(invalid="ignore", divide="ignore"):
        single = np.log(high / low) ** 2
        beta = single + np.concatenate([np.full((1, high.shape[1]), np.nan), single[:-1]])
        two_high = np.fmax(high, np.concatenate([np.full((1, high.shape[1]), np.nan), high[:-1]]))
        two_low = np.fmin(low, np.concatenate([np.full((1, low.shape[1]), np.nan), low[:-1]]))
        previous_known = np.concatenate([np.zeros((1, high.shape[1]), dtype=bool), np.isfinite(high[:-1])])
        gamma = np.where(previous_known, np.log(two_high / two_low) ** 2, np.nan)
        denominator = 3.0 - 2.0 * np.sqrt(2.0)
        alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / denominator - np.sqrt(gamma / denominator)
        spread = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    return np.where(np.isfinite(spread), np.maximum(spread, 0.0), np.nan)


class JKPChars:
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
        key = f"out:{name}"
        if key in self._cache:
            return self._cache[key]
        method: Callable[[], Any] = getattr(self, f"_c_{name}")
        result = method()
        if isinstance(result, np.ndarray):
            result = Panel(result, index=self.dates, columns=self.instruments)
        result = replace_inf(result.astype(float)).where(self.trading)
        if name in COMPOSITE_INPUTS:
            self._cache[key] = result
        return result

    def _input(self, key: str) -> Any:
        value = self.inputs.get(key)
        if value is None:
            raise DataContractError(f"jkp153 input is missing: {key}")
        return value

    def _series(self, key: str) -> np.ndarray:
        return np.asarray(pd.Series(self._input(key)).reindex(self.dates), dtype=float)

    def _universe(self) -> np.ndarray:
        universe = self.inputs.get("universe")
        if universe is None:
            return np.ones((len(self.dates), len(self.instruments)), dtype=bool)
        return np.asarray(universe.reindex(index=self.dates, columns=self.instruments).fillna(False), dtype=bool)

    def _memo(self, key: str, build: Callable[[], Any]) -> Any:
        if key not in self._cache:
            self._cache[key] = build()
        return self._cache[key]

    @property
    def grid(self) -> QuarterGrid:
        if self._grid is None:
            self._grid = QuarterGrid(prepare_fundamentals(self._input("fundamentals")), self.dates, self.instruments)
        return self._grid

    def _array(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=float)

    def _q(self, values: np.ndarray) -> Panel:
        return self.grid.panel(values)

    # ------------------------------------------------------------------ quarter items (instruments x quarters)
    def x(self, name: str) -> np.ndarray:
        """A Compustat-style item: balance-sheet value at the quarter end, or trailing-twelve-month flow."""
        return self._memo(f"x:{name}", lambda: self._item(name))

    def _item(self, name: str) -> np.ndarray:
        g = self.grid
        if name in ITEMS:
            values = sum(g.value(line) for line in ITEMS[name])
            return np.where(values >= 0, values, np.nan) if name == "at" else values
        if name in FLOWS:
            return sum(g.ttm(line) for line in FLOWS[name])
        builder = getattr(self, f"_x_{name}", None)
        if builder is None:
            raise DataContractError(f"jkp153 has no item {name}")
        return builder()

    def quarterly(self, line: str) -> np.ndarray:
        return self.grid.single(line)

    def _x_debt(self) -> np.ndarray:
        return self.x("dltt") + self.x("dlc")

    def _x_gp(self) -> np.ndarray:
        return self.x("sale") - self.x("cogs")

    def _x_opex(self) -> np.ndarray:
        return self.x("cogs") + self.x("xsga")

    def _x_ebitda(self) -> np.ndarray:
        return self.x("sale") - self.x("opex")

    def _x_ebit(self) -> np.ndarray:
        return self.x("ebitda") - self.x("dp")

    def _x_op(self) -> np.ndarray:
        return self.x("ebitda") + self.x("xrd")

    def _x_ope(self) -> np.ndarray:
        return self.x("ebitda") - self.x("xint")

    def _x_coa(self) -> np.ndarray:
        return self.x("act") - self.x("che")

    def _x_col(self) -> np.ndarray:
        return self.x("lct") - self.x("dlc")

    def _x_cowc(self) -> np.ndarray:
        return self.x("coa") - self.x("col")

    def _x_ncoa(self) -> np.ndarray:
        return self.x("at") - self.x("act") - self.x("ivao")

    def _x_ncol(self) -> np.ndarray:
        return self.x("lt") - self.x("lct") - self.x("dltt")

    def _x_nncoa(self) -> np.ndarray:
        return self.x("ncoa") - self.x("ncol")

    def _x_noa(self) -> np.ndarray:
        return (self.x("coa") + self.x("ncoa")) - (self.x("col") + self.x("ncol"))

    def _x_lnoa(self) -> np.ndarray:
        # Long-term operating assets less long-term operating liabilities, plus D&A as in JKP's lnoa.
        return (self.x("nca") - self.x("ivao")) - (self.x("ncl") - self.x("dltt")) + self.x("dp")

    def _x_fna(self) -> np.ndarray:
        return self.x("ivst") + self.x("ivao")

    def _x_fnl(self) -> np.ndarray:
        return self.x("debt") + self.x("pstk")

    def _x_nfna(self) -> np.ndarray:
        return self.x("fna") - self.x("fnl")

    def _x_be(self) -> np.ndarray:
        return positive(self.x("seq") + self.x("txditc") - self.x("pstk"))

    def _x_bev(self) -> np.ndarray:
        return positive(self.x("debt") - self.x("che") + self.x("seq") + self.x("mib"))

    def _x_ppeinv(self) -> np.ndarray:
        return self.x("ppegt") + self.x("invt")

    def _x_oacc(self) -> np.ndarray:
        return self.x("ni") - self.x("oancf")

    def _x_tacc(self) -> np.ndarray:
        nfna = self.x("nfna")
        return self.x("oacc") + nfna - lag(nfna, 4)

    def _x_cop(self) -> np.ndarray:
        return self.x("ebitda") + self.x("xrd") - self.x("oacc")

    def _x_fcf(self) -> np.ndarray:
        return self.x("oancf") - self.x("capx")

    def _x_dbnetis(self) -> np.ndarray:
        return self.x("dltis") - self.x("dltr")

    def _x_netis(self) -> np.ndarray:
        return self.x("sstk") + self.x("dbnetis")

    def _x_aliq(self) -> np.ndarray:
        return self.x("che") + 0.75 * self.x("coa") + 0.5 * (self.x("at") - self.x("act") - self.x("intan"))

    # ------------------------------------------------------------------ quarter-space operators
    def over_at(self, values: np.ndarray, *, lagged: bool = False) -> np.ndarray:
        assets = self.x("at")
        return ratio(values, positive(lag(assets, 4) if lagged else assets))

    def change_over_at(self, name: str) -> Panel:
        values = self.x(name)
        return self._q(ratio(values - lag(values, 4), positive(self.x("at"))))

    def change_to_expected(self, values: np.ndarray) -> np.ndarray:
        base = (lag(values, 4) + lag(values, 8)) / 2.0
        return ratio(values, positive(base)) - 1.0

    def standardized_unexpected(self, values: np.ndarray) -> np.ndarray:
        change = values - lag(values, 4)
        history = np.stack([lag(change, k) for k in range(1, 9)], axis=2)
        mean, count = nan_mean(history, 2)
        squares = np.where(np.isfinite(history), (history - mean[..., None]) ** 2, 0.0).sum(axis=2)
        with np.errstate(invalid="ignore", divide="ignore"):
            std = np.sqrt(squares / (count - 1))
        return np.where(count >= 6, ratio(change - mean, std), np.nan)

    # ------------------------------------------------------------------ daily building blocks
    def me(self) -> Panel:
        cap = self._input("cap")
        return cap.where(cap > 0)

    def dividends(self) -> Panel:
        """Trailing-twelve-month cash dividends in yuan; zero for non-payers with a daily_basic row."""
        yield_ = self._input("basic_dv_ttm")
        known = yield_.notna() | self._input("basic_turnover_rate").notna()
        return (yield_.fillna(0.0) * self.me()).where(known)

    def mev(self) -> Panel:
        other = self._q(self.x("debt") - self.x("che") + self.x("pstk") + self.x("mib"))
        value = self.me() + other
        return value.where(value > 0)

    def _momentum(self, start: int, end: int) -> Panel:
        """Return from ``start`` trading days ago to ``end`` trading days ago (start > end)."""
        return self.price.shift(end) / self.price.shift(start) - 1.0

    def _monthly_returns(self, months: int) -> np.ndarray:
        """Stacked 21-day returns ending 0, 21, ..., 21 (months - 1) rows ago."""
        price = self.price.to_numpy(dtype=float)
        out = np.full((months, *price.shape), np.nan)
        for k in range(months):
            end, start = MONTH * k, MONTH * (k + 1)
            if start < len(price):
                out[k, start:] = price[start - end : len(price) - end] / price[: len(price) - start] - 1.0
        return out

    def _regression(self, key: str, regressors: tuple[str, ...], window: int, minimum: int, **kwargs: Any) -> dict:
        def build() -> dict:
            x = np.column_stack([self._series(name) for name in regressors])
            return rolling_regression(self.returns.to_numpy(dtype=float), x, window, minimum, **kwargs)

        return self._memo(f"fit:{key}", build)

    def _rolling(self, panel: Panel, window: int, minimum: int) -> Any:
        return panel.rolling(window, min_periods=minimum)

    def _turnover(self) -> Panel:
        return self._input("basic_turnover_rate")

    # ------------------------------------------------------------------ momentum
    def _c_niq_su(self) -> Panel:
        return self._q(self.standardized_unexpected(self.quarterly("n_income_attr_p")))

    def _c_ret_6_1(self) -> Panel:
        return self._momentum(6 * MONTH, MONTH)

    def _c_ret_12_1(self) -> Panel:
        return self._momentum(YEAR, MONTH)

    def _c_saleq_su(self) -> Panel:
        return self._q(self.standardized_unexpected(self.quarterly("revenue")))

    def _c_tax_gr1a(self) -> Panel:
        return self.change_over_at("txt")

    def _c_ni_inc8q(self) -> Panel:
        earnings = self.quarterly("n_income_attr_p")
        increases = []
        for k in range(8):
            current, previous = lag(earnings, k), lag(earnings, k + 4)
            known = np.isfinite(current) & np.isfinite(previous)
            increases.append(np.where(known, (current > previous).astype(float), np.nan))
        stacked = np.stack(increases, axis=2)
        known = np.isfinite(stacked).all(axis=2)
        run = np.cumprod(np.nan_to_num(stacked, nan=0.0), axis=2).sum(axis=2)
        return self._q(np.where(known, run, np.nan))

    def _c_prc_highprc_252d(self) -> Panel:
        return safe_div(self.price, self.price.rolling(YEAR, min_periods=120).max())

    def _residual_momentum(self) -> dict:
        window = 756
        slices = {"im12": (window - YEAR, window - MONTH), "im6": (window - 6 * MONTH, window - MONTH)}
        return self._regression("ff3_momentum", MARKET_FACTORS, window, 378, residual_slices=slices)

    def _c_resff3_6_1(self) -> np.ndarray:
        return self._residual_momentum()["im6"]

    def _c_resff3_12_1(self) -> np.ndarray:
        return self._residual_momentum()["im12"]

    # ------------------------------------------------------------------ value
    def _to_me(self, values: np.ndarray) -> Panel:
        return safe_div(self._q(values), self.me())

    def _c_be_me(self) -> Panel:
        return self._to_me(self.x("be"))

    def _c_debt_me(self) -> Panel:
        return self._to_me(self.x("debt"))

    def _c_at_me(self) -> Panel:
        return self._to_me(self.x("at"))

    def _c_ret_60_12(self) -> Panel:
        return self._momentum(5 * YEAR, YEAR)

    def _c_ni_me(self) -> Panel:
        return self._to_me(self.x("ni"))

    def _c_fcf_me(self) -> Panel:
        return self._to_me(self.x("fcf"))

    def _c_div12m_me(self) -> Panel:
        return safe_div(self.dividends(), self.me())

    def _c_eqnpo_me(self) -> Panel:
        return safe_div(self.dividends() - self._q(self.x("sstk")), self.me())

    def _c_sale_gr3(self) -> Panel:
        sale = self.x("sale")
        return self._q(growth(sale, lag(sale, 12)))

    def _c_sale_gr1(self) -> Panel:
        sale = self.x("sale")
        return self._q(growth(sale, lag(sale, 4)))

    def _c_ebitda_mev(self) -> Panel:
        return safe_div(self._q(self.x("ebitda")), self.mev())

    def _c_sale_me(self) -> Panel:
        return self._to_me(self.x("sale"))

    def _c_ocf_me(self) -> Panel:
        return self._to_me(self.x("oancf"))

    def _c_ival_me(self) -> Panel:
        be = self._q(self.x("be"))
        lagged_be = self._q(lag(self.x("be"), 4))
        income = self._q(self.x("ni"))
        assets = self._q(self.x("at"))
        dividends = self.dividends()
        payout = safe_div(dividends, income.where(income > 0)).where(
            income > 0, safe_div(dividends, assets * 0.06)
        )
        roe = safe_div(income, ((be + lagged_be) / 2.0).where(be + lagged_be > 0))
        next_be = (1.0 + (1.0 - payout) * roe) * be
        excess = roe - DISCOUNT
        value = be + excess / (1.0 + DISCOUNT) * be + excess / ((1.0 + DISCOUNT) * DISCOUNT) * next_be
        return safe_div(value.where(value > 0), self.me())

    def _c_bev_mev(self) -> Panel:
        return safe_div(self._q(self.x("bev")), self.mev())

    def _c_netdebt_me(self) -> Panel:
        return self._to_me(self.x("debt") - self.x("che"))

    def _c_eq_dur(self) -> Panel:
        horizon, roe_mean, roe_ar1, g_mean, g_ar1 = 10, 0.12, 0.57, 0.06, 0.24
        be = self.x("be")
        sale = self.x("sale")
        roe = ratio(self.x("ni"), np.where(lag(be, 4) > 1, lag(be, 4), np.nan))
        g = ratio(sale, np.where(lag(sale, 4) > 1, lag(sale, 4), np.nan)) - 1.0
        book = be
        weighted = np.zeros(be.shape)
        plain = np.zeros(be.shape)
        negative = np.zeros(be.shape, dtype=bool)
        for t in range(1, horizon + 1):
            g = g_mean * (1 - g_ar1) + g_ar1 * g
            roe = roe_mean * (1 - roe_ar1) + roe_ar1 * roe
            dividend = book * (roe - g)
            book = book * (1 + g)
            negative |= book < 0
            weighted = weighted + t * dividend / (1 + DISCOUNT) ** t
            plain = plain + dividend / (1 + DISCOUNT) ** t
        constant = horizon + (1 + DISCOUNT) / DISCOUNT
        me = self.me()
        duration = safe_div(self._q(weighted), me) + constant * safe_div(me - self._q(plain), me)
        return duration.where(~self._q(np.where(np.isfinite(be), negative.astype(float), np.nan)).eq(1.0)).where(
            duration > 0
        )

    # ------------------------------------------------------------------ investment
    def _c_capex_abn(self) -> Panel:
        intensity = ratio(self.x("capx"), positive(self.x("sale")))
        base = (lag(intensity, 4) + lag(intensity, 8) + lag(intensity, 12)) / 3.0
        return self._q(ratio(intensity, base) - 1.0)

    def _c_at_gr1(self) -> Panel:
        assets = self.x("at")
        return self._q(growth(assets, lag(assets, 4)))

    def _c_ppeinv_gr1a(self) -> Panel:
        values = self.x("ppeinv")
        return self._q(self.over_at(values - lag(values, 4), lagged=True))

    def _c_noa_at(self) -> Panel:
        return self._q(self.over_at(self.x("noa"), lagged=True))

    def _c_noa_gr1a(self) -> Panel:
        values = self.x("noa")
        return self._q(self.over_at(values - lag(values, 4), lagged=True))

    def _c_lnoa_gr1a(self) -> Panel:
        values = self.x("lnoa")
        assets = self.x("at")
        return self._q(ratio(values - lag(values, 4), positive(assets + lag(assets, 4))))

    def _capex_growth(self, quarters: int) -> Panel:
        capex = self.x("capx")
        return self._q(growth(capex, lag(capex, quarters)))

    def _c_capx_gr1(self) -> Panel:
        return self._capex_growth(4)

    def _c_capx_gr2(self) -> Panel:
        return self._capex_growth(8)

    def _c_capx_gr3(self) -> Panel:
        return self._capex_growth(12)

    def _c_chcsho_12m(self) -> Panel:
        shares = self._input("basic_total_share").ffill()
        return safe_div(shares, shares.shift(YEAR).where(shares.shift(YEAR) > 0)) - 1.0

    def _c_eqnpo_12m(self) -> Panel:
        me = self.me().ffill()
        with np.errstate(invalid="ignore", divide="ignore"):
            price_growth = np.log(safe_div(self.price, self.price.shift(YEAR)))
            value_growth = np.log(safe_div(me, me.shift(YEAR)))
        return price_growth - value_growth

    def _c_debt_gr3(self) -> Panel:
        debt = self.x("debt")
        return self._q(growth(debt, lag(debt, 12)))

    def _c_inv_gr1(self) -> Panel:
        inventory = self.x("invt")
        return self._q(growth(inventory, lag(inventory, 4)))

    def _c_inv_gr1a(self) -> Panel:
        return self.change_over_at("invt")

    def _c_oaccruals_at(self) -> Panel:
        return self._q(self.over_at(self.x("oacc")))

    def _c_taccruals_at(self) -> Panel:
        return self._q(self.over_at(self.x("tacc")))

    def _c_cowc_gr1a(self) -> Panel:
        return self.change_over_at("cowc")

    def _c_coa_gr1a(self) -> Panel:
        return self.change_over_at("coa")

    def _c_col_gr1a(self) -> Panel:
        return self.change_over_at("col")

    def _c_nncoa_gr1a(self) -> Panel:
        return self.change_over_at("nncoa")

    def _c_ncoa_gr1a(self) -> Panel:
        return self.change_over_at("ncoa")

    def _c_ncol_gr1a(self) -> Panel:
        return self.change_over_at("ncol")

    def _c_nfna_gr1a(self) -> Panel:
        return self.change_over_at("nfna")

    def _c_sti_gr1a(self) -> Panel:
        return self.change_over_at("ivst")

    def _c_lti_gr1a(self) -> Panel:
        return self.change_over_at("ivao")

    def _c_fnl_gr1a(self) -> Panel:
        return self.change_over_at("fnl")

    def _c_be_gr1a(self) -> Panel:
        return self.change_over_at("be")

    def _c_oaccruals_ni(self) -> Panel:
        return self._q(ratio(self.x("oacc"), np.abs(self.x("ni"))))

    def _c_taccruals_ni(self) -> Panel:
        return self._q(ratio(self.x("tacc"), np.abs(self.x("ni"))))

    def _c_netis_at(self) -> Panel:
        return self._q(self.over_at(self.x("netis")))

    def _c_eqnetis_at(self) -> Panel:
        return self._q(self.over_at(self.x("sstk")))

    def _c_dbnetis_at(self) -> Panel:
        return self._q(self.over_at(self.x("dbnetis")))

    # ------------------------------------------------------------------ profitability
    def _niq_be(self) -> np.ndarray:
        return ratio(self.quarterly("n_income_attr_p"), positive(lag(self.x("be"), 1)))

    def _niq_at(self) -> np.ndarray:
        return ratio(self.quarterly("n_income_attr_p"), positive(lag(self.x("at"), 1)))

    def _c_niq_be(self) -> Panel:
        return self._q(self._niq_be())

    def _c_niq_be_chg1(self) -> Panel:
        values = self._niq_be()
        return self._q(values - lag(values, 4))

    def _c_niq_at(self) -> Panel:
        return self._q(self._niq_at())

    def _c_niq_at_chg1(self) -> Panel:
        values = self._niq_at()
        return self._q(values - lag(values, 4))

    def _c_ebit_bev(self) -> Panel:
        return self._q(ratio(self.x("ebit"), self.x("bev")))

    def _c_ebit_sale(self) -> Panel:
        return self._q(ratio(self.x("ebit"), positive(self.x("sale"))))

    def _c_sale_bev(self) -> Panel:
        return self._q(ratio(self.x("sale"), self.x("bev")))

    def _c_at_turnover(self) -> Panel:
        assets = self.x("at")
        return self._q(ratio(self.x("sale"), positive((assets + lag(assets, 4)) / 2.0)))

    def _c_gp_at(self) -> Panel:
        return self._q(self.over_at(self.x("gp")))

    def _c_gp_atl1(self) -> Panel:
        return self._q(self.over_at(self.x("gp"), lagged=True))

    def _c_ope_be(self) -> Panel:
        return self._q(ratio(self.x("ope"), self.x("be")))

    def _c_ope_bel1(self) -> Panel:
        return self._q(ratio(self.x("ope"), lag(self.x("be"), 4)))

    def _c_op_at(self) -> Panel:
        return self._q(self.over_at(self.x("op")))

    def _c_op_atl1(self) -> Panel:
        return self._q(self.over_at(self.x("op"), lagged=True))

    def _c_cop_at(self) -> Panel:
        return self._q(self.over_at(self.x("cop")))

    def _c_cop_atl1(self) -> Panel:
        return self._q(self.over_at(self.x("cop"), lagged=True))

    def _c_f_score(self) -> Panel:
        assets = self.x("at")
        lagged_assets = positive(lag(assets, 4))
        roa = ratio(self.x("ni"), lagged_assets)
        croa = ratio(self.x("oancf"), lagged_assets)
        leverage = ratio(self.x("dltt"), positive(assets))
        current = ratio(self.x("act"), positive(self.x("lct")))
        margin = ratio(self.x("gp"), positive(self.x("sale")))
        sale = self.x("sale")
        turnover = ratio(sale, lagged_assets) - ratio(lag(sale, 4), positive(lag(assets, 8)))
        signals = [
            roa > 0,
            croa > 0,
            roa - lag(roa, 4) > 0,
            croa - roa > 0,
            current - lag(current, 4) > 0,
            margin - lag(margin, 4) > 0,
            turnover > 0,
            leverage - lag(leverage, 4) < 0,
            np.nan_to_num(self.x("sstk"), nan=0.0) == 0,
        ]
        inputs = (roa, croa, lag(roa, 4), current, lag(current, 4), margin, lag(margin, 4), turnover, leverage,
                  lag(leverage, 4))
        known = np.all([np.isfinite(value) for value in inputs], axis=0)
        score = np.sum([signal.astype(float) for signal in signals], axis=0)
        return self._q(np.where(known, score, np.nan))

    def _o_score(self) -> np.ndarray:
        def build() -> np.ndarray:
            assets = self.x("at")
            liabilities = self.x("lt")
            income = self.x("ni")
            previous = lag(income, 4)
            with np.errstate(invalid="ignore", divide="ignore"):
                log_assets = np.log(positive(assets))
            negative_equity = np.where(np.isfinite(liabilities - assets), (liabilities > assets).astype(float), np.nan)
            negative_earnings = np.where(np.isfinite(income + previous), ((income < 0) & (previous < 0)).astype(float),
                                         np.nan)
            return (
                -1.32
                - 0.407 * log_assets
                + 6.03 * ratio(self.x("debt"), positive(assets))
                - 1.43 * ratio(self.x("act") - self.x("lct"), positive(assets))
                + 0.076 * ratio(self.x("lct"), positive(self.x("act")))
                - 1.72 * negative_equity
                - 2.37 * ratio(income, positive(assets))
                - 1.83 * ratio(self.x("pi") + self.x("dp"), positive(liabilities))
                + 0.285 * negative_earnings
                - 0.52 * ratio(income - previous, np.abs(income) + np.abs(previous))
            )

        return self._memo("o_score", build)

    def _c_o_score(self) -> Panel:
        return self._q(self._o_score())

    def _c_z_score(self) -> Panel:
        assets = positive(self.x("at"))
        quarterly = (
            1.2 * ratio(self.x("act") - self.x("lct"), assets)
            + 1.4 * ratio(self.x("re"), assets)
            + 3.3 * ratio(self.x("ebitda"), assets)
            + 1.0 * ratio(self.x("sale"), assets)
        )
        return self._q(quarterly) + 0.6 * safe_div(self.me(), self._q(positive(self.x("lt"))))

    def _c_pi_nix(self) -> Panel:
        return self._q(ratio(positive(self.x("pi")), positive(self.x("ni"))))

    def _c_at_be(self) -> Panel:
        return self._q(ratio(self.x("at"), self.x("be")))

    def _c_saleq_gr1(self) -> Panel:
        sales = self.quarterly("revenue")
        return self._q(growth(sales, lag(sales, 4)))

    # ------------------------------------------------------------------ intangibles
    def _c_rd_me(self) -> Panel:
        return self._to_me(self.x("xrd"))

    def _c_rd_sale(self) -> Panel:
        return self._q(ratio(self.x("xrd"), positive(self.x("sale"))))

    def _c_opex_at(self) -> Panel:
        return self._q(self.over_at(self.x("opex")))

    def _c_rd5_at(self) -> Panel:
        rd = self.x("xrd")
        capital = rd + 0.8 * lag(rd, 4) + 0.6 * lag(rd, 8) + 0.4 * lag(rd, 12) + 0.2 * lag(rd, 16)
        return self._q(self.over_at(capital))

    def _c_age(self) -> np.ndarray:
        listed = pd.Series(self._input("list_date")).reindex(self.instruments)
        days = self.dates.to_numpy()[:, None] - pd.to_datetime(listed).to_numpy()[None, :]
        months = days / np.timedelta64(1, "D") / 30.4375
        return np.where(months >= 0, months, np.nan)

    def _expected_gap(self, first: np.ndarray, second: np.ndarray) -> Panel:
        return self._q(self.change_to_expected(first) - self.change_to_expected(second))

    def _c_dsale_dinv(self) -> Panel:
        return self._expected_gap(self.x("sale"), self.x("invt"))

    def _c_dsale_drec(self) -> Panel:
        return self._expected_gap(self.x("sale"), self.x("rect"))

    def _c_dgp_dsale(self) -> Panel:
        return self._expected_gap(self.x("gp"), self.x("sale"))

    def _c_dsale_dsga(self) -> Panel:
        return self._expected_gap(self.x("sale"), self.x("xsga"))

    def _c_tangibility(self) -> Panel:
        tangible = self.x("che") + 0.715 * self.x("rect") + 0.547 * self.x("invt") + 0.535 * self.x("ppegt")
        return self._q(ratio(tangible, self.x("at")))

    def _c_kz_index(self) -> Panel:
        plant = self._q(positive(lag(self.x("ppent"), 4)))
        assets = self._q(positive(self.x("at")))
        debt = self.x("debt")
        cash_flow = safe_div(self._q(self.x("ni") + self.x("dp")), plant)
        dividends = safe_div(self.dividends(), plant)
        cash = safe_div(self._q(self.x("che")), plant)
        tobin = safe_div(assets + self.me() - self._q(self.x("be")), assets)
        leverage = self._q(ratio(debt, debt + self.x("seq")))
        return -1.002 * cash_flow + 0.283 * tobin + 3.139 * leverage - 39.368 * dividends - 1.315 * cash

    def _c_ocfq_saleq_std(self) -> Panel:
        sales = self.quarterly("revenue")
        values = ratio(self.quarterly("n_cashflow_act"), positive(sales))
        return self._q(rolling_quarter_std(values, 16, 8))

    def _c_cash_at(self) -> Panel:
        return self._q(self.over_at(self.x("che")))

    def _earnings_ar1(self) -> dict[str, np.ndarray]:
        def build() -> dict[str, np.ndarray]:
            roa = ratio(self.x("ni"), self.x("at"))
            y = np.stack([lag(roa, 4 * k) for k in range(5)], axis=2)
            x = np.stack([lag(roa, 4 * k + 4) for k in range(5)], axis=2)
            known = (np.isfinite(y) & np.isfinite(x)).all(axis=2)
            with np.errstate(invalid="ignore", divide="ignore"):
                mean_x, mean_y = x.mean(axis=2), y.mean(axis=2)
                slope = ((x - mean_x[..., None]) * (y - mean_y[..., None])).sum(axis=2) / (
                    (x - mean_x[..., None]) ** 2
                ).sum(axis=2)
                residual = y - (mean_y - slope * mean_x)[..., None] - slope[..., None] * x
                ivol = np.sqrt((residual**2).sum(axis=2) / 3.0)
            return {"ar1": np.where(known, slope, np.nan), "ivol": np.where(known, ivol, np.nan)}

        return self._memo("earnings_ar1", build)

    def _c_ni_ar1(self) -> Panel:
        return self._q(self._earnings_ar1()["ar1"])

    def _c_ni_ivol(self) -> Panel:
        return self._q(self._earnings_ar1()["ivol"])

    def _c_earnings_variability(self) -> Panel:
        assets = self.x("at")
        roa = np.stack([lag(ratio(self.x("ni"), assets), 4 * k) for k in range(5)], axis=2)
        croa = np.stack([lag(ratio(self.x("oancf"), assets), 4 * k) for k in range(5)], axis=2)
        known = (np.isfinite(roa) & np.isfinite(croa)).all(axis=2)
        with np.errstate(invalid="ignore"):
            result = ratio(roa.std(axis=2, ddof=1), croa.std(axis=2, ddof=1))
        return self._q(np.where(known, result, np.nan))

    def _c_aliq_at(self) -> Panel:
        return self._q(self.over_at(self.x("aliq"), lagged=True))

    def _c_aliq_mat(self) -> Panel:
        market_assets = self._q(self.x("at") - self.x("be")) + self.me()
        lagged = market_assets.shift(YEAR)
        return safe_div(self._q(self.x("aliq")), lagged.where(lagged > 0))

    def _seasonality(self, first_year: int, last_year: int) -> tuple[np.ndarray, np.ndarray]:
        """Mean 21-day return of the same month (and of the other months) ``first_year`` to ``last_year`` back.

        Month ``k`` is the 21-day block ending ``21 k`` rows ago; month 11 is the coming month one year earlier.
        Every month of the span must be known.
        """
        def build() -> tuple[np.ndarray, np.ndarray]:
            price = self.price.to_numpy(dtype=float)
            sums = {True: np.zeros(price.shape), False: np.zeros(price.shape)}
            counts = {True: 0, False: 0}
            for k in range((first_year - 1) * 12, last_year * 12):
                annual = k % 12 == 11
                block = np.full(price.shape, np.nan)
                end, start = MONTH * k, MONTH * (k + 1)
                if start < len(price):
                    block[start:] = price[start - end : len(price) - end] / price[: len(price) - start] - 1.0
                sums[annual] = sums[annual] + block
                counts[annual] += 1
            return sums[True] / counts[True], sums[False] / counts[False]

        return self._memo(f"seas:{first_year}:{last_year}", build)

    def _c_seas_1_1an(self) -> np.ndarray:
        return self._seasonality(1, 1)[0]

    def _c_seas_1_1na(self) -> np.ndarray:
        return self._seasonality(1, 1)[1]

    def _c_seas_2_5an(self) -> np.ndarray:
        return self._seasonality(2, 5)[0]

    def _c_seas_2_5na(self) -> np.ndarray:
        return self._seasonality(2, 5)[1]

    def _c_seas_6_10an(self) -> np.ndarray:
        return self._seasonality(6, 10)[0]

    def _c_seas_6_10na(self) -> np.ndarray:
        return self._seasonality(6, 10)[1]

    # ------------------------------------------------------------------ trading frictions
    def _c_market_equity(self) -> Panel:
        return self.me()

    def _capm(self, window: int, minimum: int) -> dict:
        return self._regression(f"capm{window}", ("mkt",), window, minimum)

    def _ff3(self) -> dict:
        return self._regression("ff3_21", MARKET_FACTORS, MONTH, 15)

    def _c_ivol_ff3_21d(self) -> np.ndarray:
        return self._ff3()["resid_std"]

    def _c_ivol_capm_252d(self) -> np.ndarray:
        return self._capm(YEAR, 120)["resid_std"]

    def _c_ivol_capm_21d(self) -> np.ndarray:
        return self._capm(MONTH, 15)["resid_std"]

    def _c_rvol_21d(self) -> Panel:
        return self._rolling(self.returns, MONTH, 15).std()

    def _c_beta_60m(self) -> np.ndarray:
        stock = self._monthly_returns(1)[0]
        market = np.log1p(self._series("mkt"))
        cumulative = np.concatenate([[0.0], np.nancumsum(market)])
        known = np.concatenate([[0], np.cumsum(np.isfinite(market))])
        monthly = np.full(len(market), np.nan)
        complete = (known[MONTH:] - known[:-MONTH]) == MONTH
        monthly[MONTH - 1 :] = np.where(complete, np.expm1(cumulative[MONTH:] - cumulative[:-MONTH]), np.nan)
        x = np.broadcast_to(monthly[:, None], stock.shape)
        both = np.isfinite(stock) & np.isfinite(x)
        y0, x0 = np.where(both, stock, np.nan), np.where(both, x, np.nan)
        sums = {key: step_sums(values, MONTH, 60)[0] for key, values in
                (("x", x0), ("y", y0), ("xx", x0 * x0), ("xy", x0 * y0))}
        count = step_sums(y0, MONTH, 60)[1]
        with np.errstate(invalid="ignore", divide="ignore"):
            covariance = sums["xy"] - sums["x"] * sums["y"] / count
            variance = sums["xx"] - sums["x"] ** 2 / count
            beta = covariance / variance
        return np.where((count >= 36) & (variance > 0), beta, np.nan)

    def _fp_parts(self) -> tuple[Panel, Panel]:
        def build() -> tuple[Panel, Panel]:
            log_returns = np.log1p(self.returns)
            market = np.log1p(self._series("mkt"))
            market_panel = Panel(np.repeat(market[:, None], len(self.instruments), axis=1), index=self.dates,
                                 columns=self.instruments)
            sigma_stock = log_returns.rolling(YEAR, min_periods=120).std()
            sigma_market = market_panel.where(log_returns.notna()).rolling(YEAR, min_periods=120).std()
            stock_3d = log_returns.rolling(3, min_periods=3).sum()
            market_3d = market_panel.rolling(3, min_periods=3).sum()
            correlation = stock_3d.rolling(5 * YEAR, min_periods=750).corr(market_3d).clip(-1.0, 1.0)
            return correlation, safe_div(sigma_stock, sigma_market)

        return self._memo("frazzini_pedersen", build)

    def _c_betabab_1260d(self) -> Panel:
        correlation, volatility = self._fp_parts()
        return correlation * volatility

    def _c_beta_dimson_21d(self) -> np.ndarray:
        market = self._series("mkt")
        lagged = np.concatenate([[np.nan], market[:-1]])
        lead = np.concatenate([market[1:], [np.nan]])
        fit = rolling_regression(self.returns.to_numpy(dtype=float), np.column_stack([lagged, market, lead]), MONTH, 15)
        # A window ending on day d uses the market return of d + 1, so its estimate is published on d + 1.
        shifted = np.full(fit["slopes"].shape, np.nan)
        shifted[1:] = fit["slopes"][:-1]
        return shifted

    def _variation(self, panel: Panel, window: int, minimum: int) -> Panel:
        rolling = self._rolling(panel, window, minimum)
        return safe_div(rolling.std(), rolling.mean())

    def _c_turnover_126d(self) -> Panel:
        return self._rolling(self._turnover(), 126, 60).mean()

    def _c_turnover_var_126d(self) -> Panel:
        return self._variation(self._turnover(), 126, 60)

    def _c_dolvol_126d(self) -> Panel:
        return self._rolling(self._input("amount"), 126, 60).mean()

    def _c_dolvol_var_126d(self) -> Panel:
        return self._variation(self._input("amount"), 126, 60)

    def _c_prc(self) -> Panel:
        return self._input("basic_close")

    def _c_ami_126d(self) -> Panel:
        amount = self._input("amount")
        return self._rolling(safe_div(self.returns.abs(), amount.where(amount > 0)), 126, 60).mean() * 1e6

    def _zero_trades(self, window: int) -> np.ndarray:
        close = self._input("close").to_numpy(dtype=float)
        quoted = np.isfinite(close)
        first = np.where(quoted.any(axis=0), quoted.argmax(axis=0), len(close))
        listed = np.arange(len(close))[:, None] - first[None, :] >= window - 1
        missing = window - window_count(close, window)
        turnover = self._turnover().rolling(window, min_periods=1).mean().to_numpy(dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            tiebreak = (1.0 / turnover) / 480_000.0
        result = (missing + tiebreak) * MONTH / window
        return np.where(listed & np.isfinite(turnover) & (turnover > 0), result, np.nan)

    def _c_zero_trades_21d(self) -> np.ndarray:
        return self._zero_trades(MONTH)

    def _c_zero_trades_126d(self) -> np.ndarray:
        return self._zero_trades(126)

    def _c_zero_trades_252d(self) -> np.ndarray:
        return self._zero_trades(YEAR)

    def _c_rmax1_21d(self) -> Panel:
        return self._rolling(self.returns, MONTH, 15).max()

    def _c_rskew_21d(self) -> Panel:
        return self._rolling(self.returns, MONTH, 15).skew()

    def _c_iskew_capm_21d(self) -> np.ndarray:
        return self._capm(MONTH, 15)["resid_skew"]

    def _c_iskew_ff3_21d(self) -> np.ndarray:
        return self._ff3()["resid_skew"]

    def _c_coskew_21d(self) -> np.ndarray:
        return self._capm(MONTH, 15)["coskew"]

    def _c_ret_1_0(self) -> Panel:
        return self._momentum(MONTH, 0)

    def _c_betadown_252d(self) -> np.ndarray:
        market = self._series("mkt")
        returns = self.returns.to_numpy(dtype=float)
        down = np.isfinite(market) & (market < 0)
        x = np.where(down, market, np.nan)[:, None]
        both = np.isfinite(returns) & np.isfinite(x)
        y0, x0 = np.where(both, returns, 0.0), np.where(both, x, 0.0)
        n = window_count(np.where(both, 1.0, np.nan), YEAR)

        def total(values: np.ndarray) -> np.ndarray:
            prefix = np.concatenate([np.zeros((1, values.shape[1])), np.cumsum(values, axis=0)])
            out = np.full(values.shape, np.nan)
            out[YEAR - 1 :] = prefix[YEAR:] - prefix[:-YEAR]
            return out

        sx, sy, sxx, sxy = total(x0), total(y0), total(x0 * x0), total(x0 * y0)
        with np.errstate(invalid="ignore", divide="ignore"):
            variance = sxx - sx**2 / n
            beta = (sxy - sx * sy / n) / variance
        return np.where((n >= 60) & (variance > 0), beta, np.nan)

    def _c_bidaskhl_21d(self) -> Panel:
        high = self._input("high").to_numpy(dtype=float)
        low = self._input("low").to_numpy(dtype=float)
        spread = Panel(corwin_schultz(high, low), index=self.dates, columns=self.instruments)
        return self._rolling(spread, MONTH, 15).mean()

    # ------------------------------------------------------------------ new
    def _c_ret_3_1(self) -> Panel:
        return self._momentum(3 * MONTH, MONTH)

    def _c_ret_9_1(self) -> Panel:
        return self._momentum(9 * MONTH, MONTH)

    def _c_ret_12_7(self) -> Panel:
        return self._momentum(YEAR, 6 * MONTH)

    def _c_corr_1260d(self) -> Panel:
        return self._fp_parts()[0]

    def _rmax5(self) -> np.ndarray:
        def build() -> np.ndarray:
            returns = self.returns.to_numpy(dtype=float)
            result = np.full(returns.shape, np.nan)
            if len(returns) < MONTH:
                return result
            for first in range(0, returns.shape[1], 64):
                block = np.lib.stride_tricks.sliding_window_view(returns[:, first : first + 64], MONTH, axis=0)
                count = np.isfinite(block).sum(axis=2)
                largest = -np.sort(np.where(np.isfinite(block), -block, np.inf), axis=2)[:, :, :5]
                mean, _ = nan_mean(largest, 2)
                result[MONTH - 1 :, first : first + 64] = np.where(count >= 15, mean, np.nan)
            return result

        return self._memo("rmax5", build)

    def _c_rmax5_21d(self) -> np.ndarray:
        return self._rmax5()

    def _c_rmax5_rvol_21d(self) -> Panel:
        return safe_div(Panel(self._rmax5(), index=self.dates, columns=self.instruments), self._c_rvol_21d())

    def _c_ni_be(self) -> Panel:
        return self._q(ratio(self.x("ni"), self.x("be")))

    def _ocf_at(self) -> np.ndarray:
        return self.over_at(self.x("oancf"))

    def _c_ocf_at(self) -> Panel:
        return self._q(self._ocf_at())

    def _c_ocf_at_chg1(self) -> Panel:
        values = self._ocf_at()
        return self._q(values - lag(values, 4))

    def _ranked(self, name: str, *, z_score: bool, sign: float = 1.0) -> np.ndarray:
        values = sign * self.calculate(name).to_numpy(dtype=float)
        return cross_section_rank(values, self._memo("universe", self._universe), z_score=z_score)

    def _c_mispricing_perf(self) -> np.ndarray:
        parts = [self._ranked("o_score", z_score=False, sign=-1.0), self._ranked("ret_12_1", z_score=False),
                 self._ranked("gp_at", z_score=False), self._ranked("niq_at", z_score=False)]
        return composite(parts, 3)

    def _c_mispricing_mgmt(self) -> np.ndarray:
        parts = [self._ranked(name, z_score=False, sign=-1.0)
                 for name in ("chcsho_12m", "oaccruals_at", "noa_at", "at_gr1", "ppeinv_gr1a")]
        parts.append(self._ranked("eqnpo_12m", z_score=False))
        return composite(parts, 3)

    def _quality_panel(self, values: np.ndarray) -> np.ndarray:
        panel = self._q(values).astype(float).where(self.trading)
        return cross_section_rank(panel.to_numpy(dtype=float), self._memo("universe", self._universe), z_score=True)

    def _qmj_parts(self) -> dict[str, np.ndarray]:
        def build() -> dict[str, np.ndarray]:
            universe = self._memo("universe", self._universe)
            assets, be, sale = self.x("at"), self.x("be"), self.x("sale")
            ratios = {
                "gpoa": self.over_at(self.x("gp")),
                "roe": ratio(self.x("ni"), be),
                "roa": self.over_at(self.x("ni")),
                "cfoa": self.over_at(self.x("oancf")),
                "gmar": ratio(self.x("gp"), positive(sale)),
            }
            profitability = [self._quality_panel(values) for name, values in ratios.items()]
            profitability.append(self._quality_panel(-self.over_at(self.x("oacc"))))
            growth_parts = [self._quality_panel(values - lag(values, 20)) for name, values in ratios.items()]
            roe_quarterly = ratio(self.quarterly("n_income_attr_p"), positive(be))
            safety = [
                cross_section_rank(-self.calculate("betabab_1260d").to_numpy(dtype=float), universe, z_score=True),
                self._quality_panel(-ratio(self.x("debt"), positive(assets))),
                self._quality_panel(-self._o_score()),
                cross_section_rank(self.calculate("z_score").to_numpy(dtype=float), universe, z_score=True),
                self._quality_panel(-rolling_quarter_std(roe_quarterly, 20, 12)),
            ]
            scores = {}
            for key, parts in (("prof", profitability), ("growth", growth_parts), ("safety", safety)):
                scores[key] = cross_section_rank(composite(parts, 3), universe, z_score=True)
            total = scores["prof"] + scores["growth"] + scores["safety"]
            scores["qmj"] = cross_section_rank(total, universe, z_score=True)
            return scores

        return self._memo("qmj", build)

    def _c_qmj(self) -> np.ndarray:
        return self._qmj_parts()["qmj"]

    def _c_qmj_prof(self) -> np.ndarray:
        return self._qmj_parts()["prof"]

    def _c_qmj_growth(self) -> np.ndarray:
        return self._qmj_parts()["growth"]

    def _c_qmj_safety(self) -> np.ndarray:
        return self._qmj_parts()["safety"]
