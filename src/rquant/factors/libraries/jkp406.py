"""Independent daily A-share implementation of the extended JKP characteristic library.

The source inventory and omissions are locked in jkp406_catalog. Accounting and
market conventions match jkp153 (TTM quarterly flows, actual first announcements,
21-day months). The 144 existing outputs retain their exact definitions. Added
long/short debt issuance uses the upstream-documented balance-change fallback;
annual dividend/buyback and extraordinary-income definitions are omitted instead
of inferred from dv_ttm or set to zero. No upstream code or data is embedded.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

from rquant.errors import DataContractError
from rquant.factors import statements
from rquant.factors.extensions import kunquant_oap as op
from rquant.factors.libraries import jkp153 as core
from rquant.factors.libraries import jkp406_catalog as inventory
from rquant.factors.libraries.base import FactorSpec
from rquant.factors.libraries.jkp153 import g, growth, lag, positive, ratio
from rquant.factors.libraries.jkp406_catalog import JKP406_NAMES
from rquant.io import sha256_file, stable_hash

# Local accounting aliases; no new fields or data permissions are required.
ITEM_ALIASES = {
    "ca": "act",
    "cl": "lct",
    "cash": "che",
    "inv": "invt",
    "rec": "rect",
    "ppeg": "ppegt",
    "ppen": "ppent",
    "lti": "ivao",
    "debtst": "dlc",
    "debtlt": "dltt",
    "sga": "xsga",
    "ocf": "oancf",
    "tax": "txt",
    "eqnetis": "sstk",
    "eqis": "sstk",
    "rd": "xrd",
}
# Explicit exceptions whose source abbreviations do not name their numerator/denominator.
RATIOS = {
    "int_debtlt": ("xint", "dltt"),
    "int_debt": ("xint", "debt"),
    "inv_act": ("invt", "act"),
    "rec_act": ("rect", "act"),
    "profit_cl": ("ebitda", "lct"),
    "ebit_int": ("ebit", "xint"),
    "caliq_cl": ("caliq", "lct"),
}
CORE_METADATA = {name: (theme, description, lookback) for name, theme, description, lookback in core.JKP_CHARS}
RAW_ITEMS = {"assets": "at", "sales": "sale", "book_equity": "be", "net_income": "ni"}
FIVE_YEAR = {
    "gpoa": ("gp", "at"),
    "roe": ("ni", "be"),
    "roa": ("ni", "at"),
    "cfoa": ("oancf", "at"),
    "gmar": ("gp", "sale"),
}


def metadata(name: str) -> tuple[str, int]:
    """Describe the actual local formula and its daily history requirement."""
    if name in CORE_METADATA:
        theme, description, window = CORE_METADATA[name]
        return f"{theme}: {description}", window
    if name in {"seas_11_15an", "seas_11_15na"}:
        period = "same-month" if name.endswith("an") else "other-month"
        return f"mean {period} return 11 to 15 years ago; month = 21 trading days", 15 * 252 + 1
    if name in RAW_ITEMS:
        return f"latest usable statement item {RAW_ITEMS[name]} (TTM for flows), in CNY", 1
    if name == "enterprise_value":
        return "market equity + statement debt - cash + preferred equity + minority interest, positive CNY", 1
    if name == "intrinsic_value":
        return "JKP residual-income intrinsic value, in CNY; identical projections to ival_me", 1
    if match := re.fullmatch(r"(.+)_gr([13])(a?)", name):
        item, years, scaled = match.groups()
        return (
            f"({item}(t) - {item}(t-{years}y)) / positive assets(t)"
            if scaled
            else f"{item}(t) / positive {item}(t-{years}y) - 1; year = 4 fiscal quarters"
        ), 1
    if match := re.fullmatch(r"ret_(\d+)_(\d+)", name):
        start, end = map(int, match.groups())
        return f"adjusted close(t-{21 * end}) / adjusted close(t-{21 * start}) - 1", 21 * start + 1
    if match := re.fullmatch(r"(chcsho|eqnpo)_(\d+)m", name):
        kind, months = match.groups()
        return (
            f"{kind}: {months} x 21-day share growth"
            if kind == "chcsho"
            else f"log adjusted-price return minus log market-equity growth over {months} x 21 days"
        ), 21 * int(months) + 1
    if name in {"beta_21d", "beta_252d"}:
        days = int(name.split("_")[1][:-1])
        return f"CAPM slope of stock returns on market returns; {days} trailing trading days", days
    if name == "ivol_capm_60m":
        return "sample std of CAPM residuals, 60 non-overlapping 21-day returns, minimum 36", 1261
    if name == "rvol_252d":
        return "sample std of daily close returns over 252 days, minimum 120 observations", 253
    if name == "rvolhl_21d":
        return "21-day mean of nonnegative Corwin-Schultz high-low volatility with overnight adjustment, minimum 15", 22
    if name in {"inv_days", "rec_days", "ap_days", "inv_turnover", "rec_turnover", "ap_turnover", "cash_conversion"}:
        return f"{name}: annual balance mean = (current + 4-quarter lag)/2; days use 365 calendar days", 1
    if name in {"niq_saleq_std", "roeq_be_std", "roe_be_std"}:
        return {
            "niq_saleq_std": "sample std of single-quarter NI/sales over 16 quarters, minimum 8",
            "roeq_be_std": "sample std of single-quarter NI/book equity over 20 quarters, minimum 12",
            "roe_be_std": "sample std of TTM NI/book equity at five annual quarter-grid lags",
        }[name], 1
    if name.endswith("_ch5"):
        return f"five-year change in {name[:-4]} profitability ratio (20 fiscal quarters)", 1
    first, second = RATIOS.get(name, tuple(name.rsplit("_", 1)))
    if second in {"me", "mev"}:
        return (
            f"latest usable {first} / positive current {'market equity' if second == 'me' else 'enterprise value'}",
            1,
        )
    return f"latest usable statement {first} / {second}; finite nonzero denominator", 1


class JKP406Graph(core.JKPGraph):
    """Compose the extended definitions into the existing quarter/daily graph."""

    def _q(self, values):
        key = f"q:{id(values)}"
        if key not in self._cache:
            name = f"q_jkpx_{len(self.statements):03d}"
            self.statements[name] = values
            self._cache[key] = (op.input_(name), values)
        return self._cache[key][0]

    def _item(self, name: str):
        name = ITEM_ALIASES.get(name, name)
        if name == "oa":
            return self.x("coa") + self.x("ncoa")
        if name == "ol":
            return self.x("col") + self.x("ncol")
        if name == "nwc":
            return self.x("act") - self.x("lct")
        if name == "caliq":
            return self.x("act") - self.x("invt")
        if name in {"dltnetis", "dstnetis"}:
            debt = self.x("dltt" if name == "dltnetis" else "dlc")
            return debt - lag(debt, 4)
        return super()._item(name)

    def calculate(self, name: str):
        if name in CORE_METADATA:
            return super().calculate(name)
        if name not in JKP406_NAMES:
            raise KeyError(f"Unsupported JKP extended characteristic: {name}")

        def build():
            result = self._extended(name)
            return g.where(g.isfinite(result) & self.trading, result, g.nan)

        return self._memo(f"out:{name}", build)

    def _extended(self, name: str):
        if name in {"seas_11_15an", "seas_11_15na"}:
            value = self._seasonality(11, 15, name.endswith("an"))
            if name.endswith("na"):
                value = g.where(g.isfinite(self.calculate("seas_11_15an")), value, g.nan)
            return g.where(g.isfinite(lag(self.price, 15 * 252)), value, g.nan)
        if name in RAW_ITEMS:
            return self._q(self.x(RAW_ITEMS[name]))
        if name == "enterprise_value":
            return self.mev()
        if name == "intrinsic_value":
            return self.calculate("ival_me") * self.me()
        if match := re.fullmatch(r"(.+)_gr([13])(a?)", name):
            item, years, scaled = match.groups()
            current = self.x(item)
            previous = lag(current, 4 * int(years))
            return self._q(ratio(current - previous, positive(self.x("at"))) if scaled else growth(current, previous))
        if match := re.fullmatch(r"ret_(\d+)_(\d+)", name):
            start, end = map(int, match.groups())
            return self._momentum(21 * start, 21 * end)
        if match := re.fullmatch(r"chcsho_(\d+)m", name):
            shares = self._input("basic_total_share")
            return growth(shares, lag(shares, 21 * int(match[1])))
        if match := re.fullmatch(r"eqnpo_(\d+)m", name):
            window = 21 * int(match[1])
            return g.log(positive(self.price / lag(self.price, window))) - g.log(self.me() / lag(self.me(), window))
        if name in {"beta_21d", "beta_252d"}:
            window = int(name.split("_")[1][:-1])
            return self._regression(("mkt",), window, 15 if window == 21 else 120, "kBeta")
        if name == "rvol_252d":
            return op.trailing(self.returns, 252, 120, "std")
        if name == "ivol_capm_60m":
            return self._monthly_ivol()
        if name == "rvolhl_21d":
            high, low = self._input("high"), self._input("low")
            previous = lag(self.close, 1)
            below, above = (previous > 0) & (previous < low), (previous > 0) & (previous > high)
            high_t = g.where(below, high - low + previous, g.where(above, previous, high))
            low_t = g.where(below, previous, g.where(above, low + previous - high, low))
            beta = g.log(positive(high_t / low_t)) ** 2 + lag(g.log(positive(high / low)) ** 2, 1)
            hi = g.maximum(high_t, lag(high, 1))
            lo = g.minimum(low_t, lag(low, 1))
            gamma = g.log(positive(hi / lo)) ** 2
            k2, const = (8.0 / 3.141592653589793) ** 0.5, 3 - 2 * 2**0.5
            sigma = (g.sqrt(beta / 2) - g.sqrt(beta)) / (k2 * const) + g.sqrt(gamma / (k2 * k2 * const))
            return op.trailing(g.maximum(sigma, 0.0), 21, 15, "mean")
        if name.endswith("_days") or name.endswith("_turnover") or name == "cash_conversion":
            return self._q(self._efficiency(name))
        if name == "niq_saleq_std":
            value = ratio(self.quarterly("n_income_attr_p"), positive(self.quarterly("revenue")))
            return self._q(op.trailing(value, 16, 8, "std"))
        if name == "roeq_be_std":
            value = ratio(self.quarterly("n_income_attr_p"), positive(self.x("be")))
            return self._q(op.trailing(value, 20, 12, "std"))
        if name == "roe_be_std":
            value = ratio(self.x("ni"), positive(self.x("be")))
            samples = [lag(value, 4 * k) for k in range(5)]
            mean = sum(samples) / 5.0
            return self._q(g.sqrt(sum((v - mean) ** 2 for v in samples) / 4.0))
        if name.endswith("_ch5"):
            numerator, denominator = FIVE_YEAR[name[:-4]]
            value = ratio(self.x(numerator), positive(self.x(denominator)))
            return self._q(value - lag(value, 20))
        numerator, denominator = RATIOS.get(name, tuple(name.rsplit("_", 1)))
        if denominator in {"me", "mev"}:
            return ratio(self._q(self.x(numerator)), self.me() if denominator == "me" else self.mev())
        lower = self.x(denominator)
        if denominator in {"at", "be", "bev", "sale", "cl"} or name in {"fcf_ocf", "sale_nwc", "tax_pi", "caliq_cl"}:
            lower = positive(lower)
        return self._q(ratio(self.x(numerator), lower))

    def _efficiency(self, name: str):
        def average(item):
            values = self.x(item)
            return (values + lag(values, 4)) / 2.0

        pairs = {"inv": ("invt", "cogs"), "rec": ("rect", "sale"), "ap": ("ap", "cogs")}
        if name == "cash_conversion":
            value = self._efficiency("inv_days") + self._efficiency("rec_days") - self._efficiency("ap_days")
            return g.where(value >= 0, value, g.nan)
        prefix, kind = name.split("_")
        stock, flow = pairs[prefix]
        if kind == "days":
            return 365.0 * ratio(average(stock), self.x(flow))
        numerator = self.x(flow)
        if prefix == "ap":
            inv = self.x("invt")
            numerator = numerator + inv - lag(inv, 4)
        return ratio(numerator, average(stock))

    def _monthly_ivol(self):
        stock = self._monthly_block(0)
        market_log = op.scalar("std::log1p(x0)", self._input("mkt"))
        market = op.scalar("std::expm1(x0)", op.trailing(market_log, 21, 21, "sum"))
        code = """
        s.y.push_back(x0); s.x.push_back(x1);
        const int t = (int)s.y.size() - 1;
        double sx=0, sy=0, sxx=0, sxy=0, count=0;
        for (int k=0; k<60; ++k) {
            const int i=t-21*k; if(i<0) break;
            if(std::isfinite(s.y[i]) && std::isfinite(s.x[i])) {
                sx+=s.x[i]; sy+=s.y[i]; sxx+=s.x[i]*s.x[i]; sxy+=s.x[i]*s.y[i]; ++count;
            }
        }
        const double variance=sxx-sx*sx/count;
        y=NAN;
        if(count>=36 && variance>0) {
            const double beta=(sxy-sx*sy/count)/variance, alpha=(sy-beta*sx)/count;
            double sse=0;
            for(int k=0; k<60; ++k) {
                const int i=t-21*k; if(i<0) break;
                if(std::isfinite(s.y[i]) && std::isfinite(s.x[i])) {
                    const double e=s.y[i]-alpha-beta*s.x[i]; sse+=e*e;
                }
            }
            y=std::sqrt(sse/(count-1));
        }"""
        return op.state([stock, market], code, "std::vector<double> x, y;")


@lru_cache(maxsize=1)
def _definitions() -> tuple[dict[str, Any], dict[str, Any]]:
    graph = JKP406Graph()
    outputs = {name: graph.calculate(name) for name in JKP406_NAMES}
    return outputs, dict(graph.statements)


class JKP406Library(core.JKP153Library):
    """Extended JKP source universe, supported daily Tushare subset in KunQuant."""

    family = "jkp406"

    @property
    def statement_outputs(self) -> tuple[str, ...]:
        return tuple(_definitions()[1])

    @property
    def specs(self) -> tuple[FactorSpec, ...]:
        return tuple(
            FactorSpec(
                canonical_name=f"jkpx_{ordinal:03d}",
                source_name=name,
                family=self.family,
                ordinal=ordinal,
                formula=metadata(name)[0],
                max_lookback=metadata(name)[1],
                implementation=f"rquant.factors.libraries.jkp406.JKP406Graph.{name}",
                catalog_version=2,
            )
            for ordinal, name in enumerate(JKP406_NAMES, 1)
        )

    @property
    def implementation_fingerprint(self) -> str:
        return stable_hash(
            {
                "module": super().implementation_fingerprint,
                "core": inspect.getsource(core),
                "inventory": inspect.getsource(inventory),
                "statements": inspect.getsource(statements),
                "operators": inspect.getsource(op),
                "kernel_operators": inspect.getsource(op.k),
            }
        )

    def load_reference_inputs(self, canonical_root: Any, dates: Any, instruments: Any) -> tuple[dict[str, Any], dict]:
        from rquant.data.reference import load_reference_manifest

        root = Path(canonical_root) / "reference"
        manifest = load_reference_manifest(canonical_root)
        expected = manifest["files"].get("stock_basic.parquet")
        # CanonicalBuilder historically publishes this file without listing it in the reference manifest.
        # Keep those manifests usable; the inherited loader records the actual consumed hash in every build.
        if expected is not None and sha256_file(root / "stock_basic.parquet") != expected:
            raise DataContractError("jkp406 stock_basic.parquet differs from its reference manifest; rebuild reference")
        inputs, hashes = super().load_reference_inputs(canonical_root, dates, instruments)
        return inputs, {key.replace("jkp153_", "jkp406_", 1): value for key, value in hashes.items()}

    def build_statements(self, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        memo: dict[int, Any] = {}
        return {name: op.lower(expr, inputs, memo) for name, expr in _definitions()[1].items()}

    def build(self, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        memo: dict[int, Any] = {}
        return {name: op.lower(expr, inputs, memo) for name, expr in _definitions()[0].items()}
