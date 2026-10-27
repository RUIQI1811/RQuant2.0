from __future__ import annotations

import tempfile

import numpy as np
import pytest

from rquant.factors.engine import FactorBuildConfig, KunQuantFactorEngine, cpp_compiler_available
from rquant.factors.extensions import kunquant_oap as op
from rquant.factors.libraries.china_chars import CPP_PRELUDE as CNC_PRELUDE
from rquant.factors.libraries.custom import CustomFactor, CustomFactorLibrary
from rquant.factors.libraries.oap import OAPLibrary, OAPSignals
from rquant.factors.libraries.registry import FactorLibraryRegistry

pytestmark = pytest.mark.skipif(not cpp_compiler_available(), reason="C++ compiler required")


def grouped_mean(inputs):
    return op.lower(op.group_stat(op.input_("close"), op.as_expr(1), op.input_("volume") == 1, "mean"), inputs)


def grouped_median(inputs):
    return op.lower(op.group_stat(op.input_("close"), op.as_expr(1), op.input_("volume") == 1, "median"), inputs)


def grouped_rank(inputs):
    return op.lower(op.cross_rank(op.input_("close"), op.input_("volume") == 1), inputs)


def cross_residual(inputs):
    return op.lower(op.cross_residual(op.input_("close"), [op.input_("cap")], op.input_("volume") == 1), inputs)


def tail_index(inputs):
    return op.lower(OAPSignals().tail_index(), {"close": inputs["close"], "universe": inputs["volume"]})


def run_library(builders, arrays, prelude=None):
    library = CustomFactorLibrary(
        family="oap_ops",
        required_inputs=tuple(arrays),
        factors=tuple(
            CustomFactor(f"oap_ops_{i:03d}", name, name, 1, builder)
            for i, (name, builder) in enumerate(builders.items(), 1)
        ),
    )
    library.cpp_prelude = op.CPP_PRELUDE if prelude is None else prelude
    registry = FactorLibraryRegistry()
    registry.register_library(library)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temporary:
        engine = KunQuantFactorEngine(temporary, FactorBuildConfig(factor_set="oap_ops", workers=2), registry=registry)
        output = engine.run(arrays)
        return {spec.source_name: output[spec.canonical_name] for spec in library.specs}


def test_distinct_cross_kernels_with_aliased_inputs_and_padded_stocks():
    # Both group code and weight are the same constant. KunQuant binds their shared buffer only once.
    close = np.array([[1.0, 2.0, 2.0, 9.0, 100.0], [np.nan, 3.0, 6.0, 12.0, 100.0]])
    output = run_library(
        {"mean": grouped_mean, "median": grouped_median, "rank": grouped_rank},
        {"close": close, "volume": np.tile([1.0, 1.0, 1.0, 1.0, 0.0], (2, 1))},
    )
    np.testing.assert_allclose(output["mean"], np.tile([[3.5], [7.0]], (1, 5)))
    np.testing.assert_allclose(output["median"], np.tile([[2.0], [6.0]], (1, 5)))
    np.testing.assert_allclose(
        output["rank"], [[0.25, 0.625, 0.625, 1.0, np.nan], [np.nan, 1 / 3, 2 / 3, 1.0, np.nan]], equal_nan=True
    )


def test_shared_cpp_helpers_can_be_combined_with_china_chars():
    output = run_library(
        {"mean": grouped_mean},
        {"close": np.array([[1.0, 2.0, 3.0]]), "volume": np.ones((1, 3))},
        CNC_PRELUDE + OAPLibrary.cpp_prelude,
    )
    np.testing.assert_allclose(output["mean"], 2.0)


def test_cross_regression_keeps_small_and_rank_deficient_directions():
    rng = np.random.default_rng(741)
    x = np.stack([np.ones(41), 1.0 + 1e-8 * np.arange(41), 1e8 * rng.normal(size=41)])
    y = 0.7 * x + rng.normal(size=x.shape) * np.array([1, 1e-9, 1])[:, None]
    output = run_library({"residual": cross_residual}, {"close": y, "cap": x, "volume": np.ones_like(x)})
    for row in range(len(x)):
        design = np.column_stack([np.ones(41), x[row]])
        beta = np.linalg.lstsq(design, y[row], rcond=None)[0]
        expected = y[row] - design @ beta
        np.testing.assert_allclose(output["residual"][row], expected, rtol=1e-6, atol=1e-7)


def test_tail_index_requires_a_full_month_before_pooling_returns():
    rng = np.random.default_rng(991)
    close = 20 * np.exp(np.cumsum(rng.normal(0, 0.02, (25, 41)), axis=0))
    output = run_library({"tail": tail_index}, {"close": close, "volume": np.ones_like(close)})["tail"]
    assert np.isnan(output[:20]).all()
    returns = np.full_like(close, np.nan)
    returns[1:] = close[1:] / close[:-1] - 1
    for row in range(20, len(close)):
        pooled = returns[row - 20 : row + 1].ravel()
        pooled = pooled[np.isfinite(pooled)]
        threshold = np.quantile(pooled, 0.05)
        expected = np.log(pooled[pooled < threshold] / threshold).mean()
        np.testing.assert_allclose(output[row], expected, rtol=1e-12)
