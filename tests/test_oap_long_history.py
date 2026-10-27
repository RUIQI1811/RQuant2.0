from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from test_kunquant_oap import run_library

from rquant.factors.engine import cpp_compiler_available
from rquant.factors.extensions import kunquant_oap as op
from rquant.factors.libraries.oap import OAPSignals

pytestmark = pytest.mark.skipif(not cpp_compiler_available(), reason="C++ compiler required")


def seasonal(inputs):
    graph = OAPSignals()
    value = graph._s_MomSeason06YrPlus()
    return op.lower(op.where(graph.trading, value, np.nan), inputs)


def off_season(inputs):
    graph = OAPSignals()
    value = graph._s_MomOffSeason06YrPlus()
    return op.lower(op.where(graph.trading, value, np.nan), inputs)


def test_six_to_ten_year_momentum_includes_the_full_longest_lag():
    rng = np.random.default_rng(187)
    close = pd.DataFrame(20 * np.exp(np.cumsum(rng.normal(0, 0.01, (2650, 3)), axis=0)))
    close.iloc[:14, 1] = np.nan
    close.iloc[800:820, 2] = np.nan
    output = run_library({"season": seasonal, "off": off_season}, {"close": close.to_numpy()})
    price = close.ffill()
    monthly = price / price.shift(21) - 1
    for name, lags, minimum in (
        ("season", (72, 84, 96, 108, 120), 2),
        ("off", tuple(j for j in range(61, 121) if j % 12), 27),
    ):
        values = np.stack([monthly.shift(21 * (j - 1)).to_numpy() for j in lags])
        count = np.isfinite(values).sum(axis=0)
        total = np.where(np.isfinite(values), values, 0).sum(axis=0)
        expected = np.where((count >= minimum) & close.notna().to_numpy(), total / np.maximum(count, 1), np.nan)
        np.testing.assert_allclose(output[name], expected, rtol=1e-12, atol=1e-12, equal_nan=True)
        assert np.isfinite(output[name][-1]).all()
