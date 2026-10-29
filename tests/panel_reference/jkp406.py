"""Independent Pandas/NumPy oracle for the JKP extended A-share definitions.

Only the source-name inventory is shared with production. No production graph,
lowering, formula dispatch or compiled helper is used by this reference.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from panel_reference.jkp153 import JKP_NAMES, JKPChars, positive
from rquant.factors.libraries.jkp406_catalog import JKP406_NAMES
from rquant.factors.panel_operators import replace_inf
from rquant.factors.statements import lag, ratio

ALIASES = {
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
EXCEPTIONS = {
    "int_debtlt": ("xint", "dltt"),
    "int_debt": ("xint", "debt"),
    "inv_act": ("invt", "act"),
    "rec_act": ("rect", "act"),
    "profit_cl": ("ebitda", "lct"),
    "ebit_int": ("ebit", "xint"),
    "caliq_cl": ("caliq", "lct"),
}


class JKP406Chars(JKPChars):
    def _item(self, name):
        name = ALIASES.get(name, name)
        if name == "oa":
            return self.x("coa") + self.x("ncoa")
        if name == "ol":
            return self.x("col") + self.x("ncol")
        if name == "nwc":
            return self.x("act") - self.x("lct")
        if name == "caliq":
            return self.x("act") - self.x("invt")
        if name in {"dltnetis", "dstnetis"}:
            values = self.x("dltt" if name == "dltnetis" else "dlc")
            return values - lag(values, 4)
        return super()._item(name)

    def calculate(self, name):
        if name in JKP_NAMES:
            return super().calculate(name)
        if name not in JKP406_NAMES:
            raise KeyError(name)
        with np.errstate(invalid="ignore", divide="ignore"):
            value = self.extended(name)
        if isinstance(value, np.ndarray):
            value = pd.DataFrame(value, index=self.dates, columns=self.instruments)
        return replace_inf(value.astype(float)).where(self.trading)

    def extended(self, name):
        if name in {"seas_11_15an", "seas_11_15na"}:
            annual, other = self._seasonality(11, 15)
            value = annual if name.endswith("an") else np.where(np.isfinite(annual), other, np.nan)
            return np.where(np.isfinite(self.price.shift(15 * 252)), value, np.nan)
        raw = {"assets": "at", "sales": "sale", "book_equity": "be", "net_income": "ni"}
        if name in raw:
            return self._q(self.x(raw[name]))
        if name == "enterprise_value":
            return self.mev()
        if name == "intrinsic_value":
            return self.calculate("ival_me") * self.me()
        if match := re.fullmatch(r"(.+)_gr([13])(a?)", name):
            item, years, scaled = match.groups()
            value = self.x(item)
            old = lag(value, 4 * int(years))
            return self._q(ratio(value - old, positive(self.x("at"))) if scaled else ratio(value, positive(old)) - 1)
        if match := re.fullmatch(r"ret_(\d+)_(\d+)", name):
            start, end = map(int, match.groups())
            return self.price.shift(end * 21) / self.price.shift(start * 21) - 1
        if match := re.fullmatch(r"chcsho_(\d+)m", name):
            shares = self._input("basic_total_share")
            return shares / shares.shift(21 * int(match[1])).where(lambda x: x > 0) - 1
        if match := re.fullmatch(r"eqnpo_(\d+)m", name):
            days = 21 * int(match[1])
            return np.log(self.price / self.price.shift(days)) - np.log(self.me() / self.me().shift(days))
        if name.startswith("beta_") and name.endswith("d"):
            window = int(name[5:-1])
            return self._capm(window, 15 if window == 21 else 120)["slopes"]
        if name == "rvol_252d":
            return self.returns.rolling(252, min_periods=120).std()
        if name == "ivol_capm_60m":
            return self.monthly_ivol()
        if name == "rvolhl_21d":
            high, low = self._input("high"), self._input("low")
            previous = self.inputs["close"].shift(1)
            gap = (low - previous).where((previous > 0) & (previous < low), 0.0)
            gap += (high - previous).where((previous > 0) & (previous > high), 0.0)
            high_t, low_t = high - gap, low - gap
            beta = np.log(high_t / low_t) ** 2 + np.log(high.shift(1) / low.shift(1)) ** 2
            hi = high_t.combine(high.shift(1), np.maximum)
            lo = low_t.combine(low.shift(1), np.minimum)
            gamma = np.log(hi / lo) ** 2
            k = np.sqrt(8 / np.pi)
            c = 3 - 2 * np.sqrt(2)
            sigma = ((np.sqrt(beta / 2) - np.sqrt(beta)) / (k * c) + np.sqrt(gamma / (k * k * c))).clip(lower=0)
            return sigma.rolling(21, min_periods=15).mean()
        if name in {
            "inv_days",
            "rec_days",
            "ap_days",
            "inv_turnover",
            "rec_turnover",
            "ap_turnover",
            "cash_conversion",
        }:
            return self._q(self.efficiency(name))
        if name in {"niq_saleq_std", "roeq_be_std"}:
            lower = positive(self.quarterly("revenue") if name == "niq_saleq_std" else self.x("be"))
            value = ratio(self.quarterly("n_income_attr_p"), lower)
            frame = pd.DataFrame(value.T)
            value = (
                frame.rolling(16 if name == "niq_saleq_std" else 20, min_periods=8 if name == "niq_saleq_std" else 12)
                .std()
                .to_numpy()
                .T
            )
            return self._q(value)
        if name == "roe_be_std":
            roe = ratio(self.x("ni"), positive(self.x("be")))
            sample = np.stack([lag(roe, i * 4) for i in range(5)])
            return self._q(np.std(sample, axis=0, ddof=1))
        if name.endswith("_ch5"):
            pair = {
                "gpoa": ("gp", "at"),
                "roe": ("ni", "be"),
                "roa": ("ni", "at"),
                "cfoa": ("oancf", "at"),
                "gmar": ("gp", "sale"),
            }[name[:-4]]
            value = ratio(self.x(pair[0]), positive(self.x(pair[1])))
            return self._q(value - lag(value, 20))
        top, bottom = EXCEPTIONS.get(name, tuple(name.rsplit("_", 1)))
        if bottom in {"me", "mev"}:
            return self._q(self.x(top)) / (self.me() if bottom == "me" else self.mev())
        denominator = self.x(bottom)
        if bottom in {"at", "be", "bev", "sale", "cl"} or name in {"fcf_ocf", "sale_nwc", "tax_pi", "caliq_cl"}:
            denominator = positive(denominator)
        return self._q(ratio(self.x(top), denominator))

    def efficiency(self, name):
        if name == "cash_conversion":
            value = self.efficiency("inv_days") + self.efficiency("rec_days") - self.efficiency("ap_days")
            return np.where(value >= 0, value, np.nan)
        prefix, kind = name.split("_")
        stock, flow = {"inv": ("invt", "cogs"), "rec": ("rect", "sale"), "ap": ("ap", "cogs")}[prefix]
        amount = self.x(stock)
        mean = (amount + lag(amount, 4)) / 2
        if kind == "days":
            return 365 * ratio(mean, self.x(flow))
        income = self.x(flow)
        if prefix == "ap":
            inv = self.x("invt")
            income = income + inv - lag(inv, 4)
        return ratio(income, mean)

    def monthly_ivol(self):
        stock = self._monthly_returns(1)[0]
        market = np.expm1(pd.Series(np.log1p(self._series("mkt"))).rolling(21, min_periods=21).sum()).to_numpy()
        result = np.full(stock.shape, np.nan)
        for end in range(len(stock)):
            rows = np.arange(end, max(-1, end - 1260), -21)
            for column in range(stock.shape[1]):
                x, y = market[rows], stock[rows, column]
                known = np.isfinite(x) & np.isfinite(y)
                if known.sum() < 36:
                    continue
                x, y = x[known], y[known]
                if np.var(x) <= 0:
                    continue
                design = np.column_stack([np.ones(len(x)), x])
                fitted = design @ np.linalg.lstsq(design, y, rcond=None)[0]
                result[end, column] = np.std(y - fitted, ddof=1)
        return result
