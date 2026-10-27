"""Stock characteristics and factor loadings of Liu, Stambaugh and Yuan's "Size and Value in China" (JFE 2019).

The paper's CH-3 model explains Chinese stock returns with market, size (SMB) and EP-based value (VMG) factors built
after removing the smallest 30% of stocks, and CH-4 adds a turnover factor (PMO, pessimistic minus optimistic). This
module is an independent implementation from the published definitions with Tushare inputs. It publishes, per stock
and trading day:

* the three sorting characteristics: size (log total market value), EP and abnormal turnover;
* rolling CH-3 and CH-4 loadings, idiosyncratic volatility, fit and alpha from daily regressions of the stock's return
  on the daily factor returns in ``reference/ch4_factors.parquet`` (see ``rquant.data.ch4_factors``);
* CH-4 residual reversal (last month) and residual momentum (months t-12..t-1).

Rows are trading dates and columns are instruments. Every regression window ends on the row it is published on and
uses factor returns of that day and earlier only. Every output is missing on days the stock has no quote.

The formulas run in KunQuant: the regressions are C++ kernels (``CPP_PRELUDE``) that read each stock's whole return
series and accumulate the window sums with prefix sums, as the pandas reference in ``tests/panel_reference`` does.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from rquant.errors import DataContractError
from rquant.factors.extensions import kunquant_panel as k
from rquant.factors.libraries.base import FactorLibrary, FactorSpec
from rquant.io import sha256_file, stable_hash

MONTH = 21
HALF_YEAR = 120
YEAR = 250
TURNOVER_DAYS = 20
CH3 = ("mkt", "smb", "vmg")
CH4 = ("mkt", "smb4", "vmg", "pmo")
# Regression fits: (key, regressors, window, minimum observations, residual slices inside the window).
FITS: dict[str, tuple[tuple[str, ...], int, int, dict[str, tuple[int, int]]]] = {
    "ch3_120": (CH3, HALF_YEAR, 60, {}),
    "ch4_120": (CH4, HALF_YEAR, 60, {"resrev": (HALF_YEAR - MONTH, HALF_YEAR)}),
    "ch4_250": (CH4, YEAR, 120, {"resmom": (0, YEAR - MONTH)}),
}

# (source name, description, maximum lookback in trading days, fit key and output, or None for a characteristic).
SIZE_VALUE_CHINA: tuple[tuple[str, str, int, tuple[str, str] | None], ...] = (
    ("size", "log total market value", 1, None),
    ("ep", "trailing earnings over total market value (1 / pe_ttm); missing for loss makers", 1, None),
    ("abturn", "20-day over 250-day average daily turnover of total shares", YEAR, None),
    ("ch3_beta_mkt", "CH-3 market loading, 120 days", HALF_YEAR, ("ch3_120", "beta_mkt")),
    ("ch3_beta_smb", "CH-3 SMB loading, 120 days", HALF_YEAR, ("ch3_120", "beta_smb")),
    ("ch3_beta_vmg", "CH-3 VMG loading, 120 days", HALF_YEAR, ("ch3_120", "beta_vmg")),
    ("ch3_ivol", "CH-3 residual volatility, 120 days", HALF_YEAR, ("ch3_120", "ivol")),
    ("ch4_beta_mkt", "CH-4 market loading, 120 days", HALF_YEAR, ("ch4_120", "beta_mkt")),
    ("ch4_beta_smb", "CH-4 SMB loading, 120 days", HALF_YEAR, ("ch4_120", "beta_smb4")),
    ("ch4_beta_vmg", "CH-4 VMG loading, 120 days", HALF_YEAR, ("ch4_120", "beta_vmg")),
    ("ch4_beta_pmo", "CH-4 PMO loading, 120 days", HALF_YEAR, ("ch4_120", "beta_pmo")),
    ("ch4_ivol", "CH-4 residual volatility, 120 days", HALF_YEAR, ("ch4_120", "ivol")),
    ("ch4_r2", "CH-4 R-squared, 120 days", HALF_YEAR, ("ch4_120", "r2")),
    ("ch4_resrev", "mean over std of CH-4 residuals in the last 21 days of a 120-day fit", HALF_YEAR,
     ("ch4_120", "resrev")),
    ("ch4_beta_mkt_y", "CH-4 market loading, 250 days", YEAR, ("ch4_250", "beta_mkt")),
    ("ch4_beta_smb_y", "CH-4 SMB loading, 250 days", YEAR, ("ch4_250", "beta_smb4")),
    ("ch4_beta_vmg_y", "CH-4 VMG loading, 250 days", YEAR, ("ch4_250", "beta_vmg")),
    ("ch4_beta_pmo_y", "CH-4 PMO loading, 250 days", YEAR, ("ch4_250", "beta_pmo")),
    ("ch4_ivol_y", "CH-4 residual volatility, 250 days", YEAR, ("ch4_250", "ivol")),
    ("ch4_alpha_y", "CH-4 daily alpha (intercept), 250 days", YEAR, ("ch4_250", "alpha")),
    ("ch4_resmom", "mean over std of CH-4 residuals in days t-249..t-21 of a 250-day fit", YEAR,
     ("ch4_250", "resmom")),
)
SIZE_VALUE_CHINA_NAMES = tuple(name for name, _, _, _ in SIZE_VALUE_CHINA)
DAILY_BASIC_FIELDS = ("total_mv", "pe_ttm", "turnover_total")
REFERENCE_INPUTS = (
    *(f"basic_{field}" for field in DAILY_BASIC_FIELDS),
    *(f"ch4_{name}" for name in sorted(set(CH3) | set(CH4))),
)

CPP_PRELUDE = r"""
namespace rq_svc {

// Mean of the finite values among the ``window`` rows ending at ``t``, missing with fewer than ``minimum`` of them.
static inline double rolling_mean(const double *values, int t, int window, int minimum) {
    double total = 0.0;
    int count = 0;
    for (int i = t - window + 1 < 0 ? 0 : t - window + 1; i <= t; ++i) {
        if (std::isfinite(values[i])) { total += values[i]; ++count; }
    }
    return count >= minimum ? total / count : std::numeric_limits<double>::quiet_NaN();
}

// Solve the k x k system ``a x = b`` by Gaussian elimination with partial pivoting; false when singular.
static inline bool solve(int k, std::vector<double> a, std::vector<double> b, double *x) {
    for (int c = 0; c < k; ++c) {
        int pivot = c;
        for (int r = c + 1; r < k; ++r) if (std::fabs(a[r * k + c]) > std::fabs(a[pivot * k + c])) pivot = r;
        if (a[pivot * k + c] == 0.0) return false;
        if (pivot != c) {
            for (int j = 0; j < k; ++j) std::swap(a[c * k + j], a[pivot * k + j]);
            std::swap(b[c], b[pivot]);
        }
        for (int r = c + 1; r < k; ++r) {
            const double factor = a[r * k + c] / a[c * k + c];
            for (int j = c; j < k; ++j) a[r * k + j] -= factor * a[c * k + j];
            b[r] -= factor * b[c];
        }
    }
    for (int r = k - 1; r >= 0; --r) {
        double value = b[r];
        for (int j = r + 1; j < k; ++j) value -= a[r * k + j] * x[j];
        x[r] = value / a[r * k + r];
    }
    return true;
}

// Rolling OLS of a stock's daily return on a constant and ``nx`` regressors over trailing ``window`` rows, dropping
// rows where the return or a regressor is missing. ``output`` selects the published statistic: 0 alpha, 1..nx the
// loadings, nx + 1 the residual standard deviation, nx + 2 R-squared, nx + 3 the mean over standard deviation of the
// full-window residuals in window positions [first, last).
static inline void rolling_ols(int n, const double *close, const double *const *x, int nx, int window, int minimum,
                               int output, int first, int last, double *out) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const int k = nx + 1;
    std::vector<double> design(static_cast<size_t>(n) * k, 0.0), weight(n, 0.0), value(n, 0.0);
    int observed = 0;
    for (int t = 0; t < n; ++t) {
        const double y = t == 0 ? nan : close[t] / close[t - 1] - 1.0;
        bool ok = true;
        for (int j = 0; j < nx; ++j) ok = ok && std::isfinite(x[j][t]);
        if (ok) {
            design[t * k] = 1.0;
            for (int j = 0; j < nx; ++j) design[t * k + 1 + j] = x[j][t];
        }
        if (ok && std::isfinite(y)) { weight[t] = 1.0; value[t] = y; ++observed; }
    }
    if (n < window || observed < minimum) return;
    // Prefix sums of the window statistics: count, y, y^2, x w, x y and x x' w.
    std::vector<double> pn(n + 1, 0.0), py(n + 1, 0.0), pyy(n + 1, 0.0);
    std::vector<double> px(static_cast<size_t>(n + 1) * k, 0.0), pxy(static_cast<size_t>(n + 1) * k, 0.0);
    std::vector<double> pxx(static_cast<size_t>(n + 1) * k * k, 0.0);
    for (int t = 0; t < n; ++t) {
        pn[t + 1] = pn[t] + weight[t];
        py[t + 1] = py[t] + value[t];
        pyy[t + 1] = pyy[t] + value[t] * value[t];
        for (int a = 0; a < k; ++a) {
            const double d = design[t * k + a];
            px[(t + 1) * k + a] = px[t * k + a] + d * weight[t];
            pxy[(t + 1) * k + a] = pxy[t * k + a] + d * value[t];
            for (int b = 0; b < k; ++b) {
                const size_t at = (static_cast<size_t>(t + 1) * k + a) * k + b;
                pxx[at] = pxx[at - k * k] + d * design[t * k + b] * weight[t];
            }
        }
    }
    std::vector<double> gram(k * k), sxy(k), beta(k);
    // Sum of x_a x_b w over rows [lo, hi).
    auto xx = [&](int lo, int hi, int a, int b) {
        return pxx[(static_cast<size_t>(hi) * k + a) * k + b] - pxx[(static_cast<size_t>(lo) * k + a) * k + b];
    };
    // beta' (sum of x x' w) beta and beta' (sum of v) over rows [lo, hi).
    auto quadratic = [&](int lo, int hi) {
        double total = 0.0;
        for (int a = 0; a < k; ++a) {
            double row = 0.0;
            for (int b = 0; b < k; ++b) row += xx(lo, hi, a, b) * beta[b];
            total += beta[a] * row;
        }
        return total;
    };
    auto linear = [&](const std::vector<double> &prefix, int lo, int hi) {
        double total = 0.0;
        for (int a = 0; a < k; ++a) total += beta[a] * (prefix[hi * k + a] - prefix[lo * k + a]);
        return total;
    };
    for (int t = window - 1; t < n; ++t) {
        const int lo = t + 1 - window, hi = t + 1;
        const double count = pn[hi] - pn[lo];
        if (!(count >= minimum)) continue;
        for (int a = 0; a < k; ++a) {
            sxy[a] = pxy[hi * k + a] - pxy[lo * k + a];
            for (int b = 0; b < k; ++b) gram[a * k + b] = xx(lo, hi, a, b);
        }
        if (!solve(k, gram, sxy, beta.data())) continue;
        double result = nan;
        if (output <= nx) {
            result = beta[output];
        } else {
            const double sy = py[hi] - py[lo], syy = pyy[hi] - pyy[lo];
            const double squared = syy - 2.0 * linear(pxy, lo, hi) + quadratic(lo, hi);
            const double total = syy - sy * sy / count;
            const bool keep = squared > 1e-14 * (syy > 1e-300 ? syy : 1e-300) && total > 0;
            if (output == nx + 1) {
                result = keep && count > k ? std::sqrt(squared / (count - k)) : nan;
            } else if (output == nx + 2) {
                result = keep ? 1.0 - squared / total : nan;
            } else {
                const int start = lo + first, stop = lo + last;
                const double part_n = pn[stop] - pn[start];
                const double part_e = py[stop] - py[start] - linear(px, start, stop);
                const double part_ee = pyy[stop] - pyy[start] - 2.0 * linear(pxy, start, stop) + quadratic(start, stop);
                const double mean = part_e / part_n;
                const double spread = std::sqrt((part_ee - part_n * (mean * mean)) / (part_n - 1));
                result = part_n >= (last - first) / 2 && spread > 0 ? mean / spread : nan;
            }
        }
        out[t] = std::isfinite(result) && std::isfinite(close[t]) ? result : nan;
    }
}

}  // namespace rq_svc
"""


class SizeValueChinaLibrary(FactorLibrary):
    """CH-3/CH-4 characteristics and loadings, evaluated as daily point-in-time panels in KunQuant."""

    family = "size_value_china"
    required_inputs = ("close",)
    reference_inputs = REFERENCE_INPUTS
    strict_float = True
    cpp_prelude = CPP_PRELUDE

    @property
    def specs(self) -> tuple[FactorSpec, ...]:
        return tuple(
            FactorSpec(
                canonical_name=f"svc_{ordinal:03d}",
                source_name=name,
                family=self.family,
                ordinal=ordinal,
                formula=description,
                max_lookback=lookback,
                implementation=f"rquant.factors.libraries.size_value_china.{name}",
                catalog_version=2,
            )
            for ordinal, (name, description, lookback, _) in enumerate(SIZE_VALUE_CHINA, 1)
        )

    def load_reference_inputs(self, canonical_root: Any, dates: Any, instruments: Any) -> tuple[dict[str, Any], dict]:
        from rquant.data.reference import load_reference_manifest

        root = Path(canonical_root) / "reference"
        manifest = load_reference_manifest(canonical_root)
        files = manifest["files"]
        if "ch4_factors.parquet" not in files:
            raise DataContractError("size_value_china needs CH-4 factor returns; run `rquant data build-reference`")
        dates = pd.DatetimeIndex(dates)
        years = set(dates.year)
        daily_files = sorted(
            name for name in files if name.startswith("daily_basic/") and int(name.split("=")[1][:4]) in years
        )
        for name in (*daily_files, "ch4_factors.parquet"):
            if sha256_file(root / name) != files[name]:
                raise DataContractError(f"Reference file differs from its manifest: {root / name}")

        columns = ["instrument", "datetime", *DAILY_BASIC_FIELDS]
        daily = pd.concat([pd.read_parquet(root / name, columns=columns) for name in daily_files], ignore_index=True)
        daily = daily[daily["instrument"].isin(set(instruments))]
        inputs: dict[str, Any] = {}
        for field in DAILY_BASIC_FIELDS:
            panel = daily.pivot(index="datetime", columns="instrument", values=field)
            inputs[f"basic_{field}"] = panel.reindex(index=dates, columns=instruments)
        factors = pd.read_parquet(root / "ch4_factors.parquet").set_index("datetime").sort_index()
        for name in sorted(set(CH3) | set(CH4)):
            inputs[f"ch4_{name}"] = factors[name].reindex(dates)
        hashes = {
            "size_value_china_reference": manifest["fingerprint"],
            "size_value_china_daily_basic": stable_hash({name: files[name] for name in daily_files}),
            "size_value_china_factors": files["ch4_factors.parquet"],
        }
        return inputs, hashes

    def build(self, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        close = inputs["close"]
        trading = k.notna(close)
        total_mv, pe = inputs["basic_total_mv"], inputs["basic_pe_ttm"]
        outputs = {
            "size": k.log(k.mask(total_mv, k.gt(total_mv, 0.0))),
            "ep": 1.0 / k.mask(pe, k.gt(pe, 0.0)),
            "abturn": k.series_kernel(
                [close, inputs["basic_turnover_total"]],
                f"""
                std::vector<double> turnover(n);
                for (int t = 0; t < n; ++t) turnover[t] = std::isfinite(x0[t]) ? x1[t] : NAN;
                for (int t = 0; t < n; ++t) {{
                    const double year = rq_svc::rolling_mean(turnover.data(), t, {YEAR}, 120);
                    out[t] = rq_svc::rolling_mean(turnover.data(), t, {TURNOVER_DAYS}, 15) / (year > 0 ? year : NAN);
                }}""",
            ),
        }
        for name, _, _, source in SIZE_VALUE_CHINA:
            if source is None:
                continue
            key, statistic = source
            regressors, window, minimum, slices = FITS[key]
            if statistic == "alpha":
                output, first, last = 0, 0, 0
            elif statistic.startswith("beta_"):
                output, first, last = regressors.index(statistic.removeprefix("beta_")) + 1, 0, 0
            elif statistic in ("ivol", "r2"):
                output, first, last = len(regressors) + (1 if statistic == "ivol" else 2), 0, 0
            else:
                output, (first, last) = len(regressors) + 3, slices[statistic]
            columns = ", ".join(f"x{index}" for index in range(1, len(regressors) + 1))
            outputs[name] = k.series_kernel(
                [close, *(inputs[f"ch4_{regressor}"] for regressor in regressors)],
                f"""
                const double *regressors[] = {{{columns}}};
                rq_svc::rolling_ols(n, x0, regressors, {len(regressors)}, {window}, {minimum}, {output}, {first},
                                    {last}, out);""",
            )
        return {name: k.mask(k.replace_inf(value), trading) for name, value in outputs.items()}
