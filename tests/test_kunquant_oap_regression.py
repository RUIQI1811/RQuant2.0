from __future__ import annotations

import tempfile
from functools import partial, update_wrapper

import numpy as np
import pytest

from rquant.factors.engine import FactorBuildConfig, KunQuantFactorEngine, cpp_compiler_available
from rquant.factors.extensions import kunquant_oap as op
from rquant.factors.extensions import kunquant_oap_regression as regression_op
from rquant.factors.libraries.china_chars import CPP_PRELUDE as CNC_PRELUDE
from rquant.factors.libraries.custom import CustomFactor, CustomFactorLibrary
from rquant.factors.libraries.registry import FactorLibraryRegistry
from rquant.factors.panel_operators import rolling_regression

pytestmark = pytest.mark.skipif(not cpp_compiler_available(), reason="C++ compiler required")


def regression_output(inputs, kind, two):
    regressors = [op.input_("cap")]
    if two:
        regressors.append(op.input_("volume"))
    return op.lower(regression_op.regression(op.input_("close"), regressors, 20, 10, kind, first=10, last=20), inputs)


def inverse_output(inputs, column, deficient, solve):
    diagonal = 2 if deficient else 3
    matrix = f"{{4,2,2,2,2,2,2,2,{diagonal}}}"
    code = f"std::vector<double> gram={matrix};"
    if solve:
        code += "std::vector<double> rhs={1,2,3}, result(3);"
        code += "rq_oap_regression::solve_or_pinv(3,gram,rhs,result.data());"
        code += f"y=result[{column}];"
    else:
        code += "auto result=rq_oap_regression::inverse_or_pinv(3,gram);"
        code += f"y=result[{column * 3 + column}];"
    return op.lower(op.state([op.input_("close")], code), inputs)


def run_library(builders, arrays, prelude):
    library = CustomFactorLibrary(
        family="oap_singular",
        required_inputs=tuple(arrays),
        factors=tuple(
            CustomFactor(f"oap_singular_{i:03d}", name, name, 1, builder)
            for i, (name, builder) in enumerate(builders.items(), 1)
        ),
    )
    library.cpp_prelude = prelude
    library.strict_float = True
    registry = FactorLibraryRegistry()
    registry.register_library(library)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temporary:
        engine = KunQuantFactorEngine(
            temporary, FactorBuildConfig(factor_set="oap_singular", workers=2), registry=registry
        )
        output = engine.run(arrays)
    return {spec.source_name: output[spec.canonical_name] for spec in library.specs}


@pytest.mark.parametrize("case", ["zero", "constant", "duplicated", "ordinary", "mixed"])
def test_rolling_regression_matches_independent_pseudoinverse_reference(case):
    rng = np.random.default_rng(81)
    target = rng.normal(0, 0.01, (42, 3))
    target[21, 1] = np.nan
    two = case == "duplicated"
    x = (np.arange(42) % 3).astype(float) if two else np.zeros(42)
    if case in ("ordinary", "mixed"):
        x = rng.normal(0, 0.01, 42)
        if case == "mixed":
            x[:22] = 0
    elif case == "constant":
        x[:] = 1
    regressors = np.column_stack([x, 2 * x]) if two else x[:, None]
    expected = rolling_regression(target, regressors, 20, 10, residual_slices={"momentum": (10, 20)})
    kinds = {
        "beta": "kBeta",
        "resid_std": "kResidualStd",
        "resid_skew": "kResidualSkew",
        "coskew": "kCoskew",
        "momentum": "kResidualMomentum",
    }
    arrays = {"close": target, "cap": np.broadcast_to(x[:, None], target.shape).copy()}
    if two:
        arrays["volume"] = 2 * arrays["cap"]
    output = run_library(
        {
            name: update_wrapper(partial(regression_output, kind=kind, two=two), regression_output)
            for name, kind in kinds.items()
        },
        arrays,
        CNC_PRELUDE + op.CPP_PRELUDE + regression_op.CPP_PRELUDE,
    )
    for name in kinds:
        np.testing.assert_allclose(output[name], expected[name], rtol=1e-7, atol=1e-9, equal_nan=True, err_msg=name)


@pytest.mark.parametrize("deficient", [False, True])
def test_pseudoinverse_diagonal_and_coefficients_match_numpy_for_price_delay(deficient):
    matrix = np.array([[4.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0 if deficient else 3.0]])
    inverse = np.linalg.pinv(matrix) if deficient else np.linalg.inv(matrix)
    coefficients = inverse @ np.array([1.0, 2.0, 3.0])
    builders = {}
    for solve in (False, True):
        for column in range(3):
            builders[f"{solve}_{column}"] = update_wrapper(
                partial(inverse_output, column=column, deficient=deficient, solve=solve), inverse_output
            )
    output = run_library(builders, {"close": np.ones((1, 3))}, CNC_PRELUDE + op.CPP_PRELUDE + regression_op.CPP_PRELUDE)
    for column in range(3):
        np.testing.assert_allclose(output[f"False_{column}"], inverse[column, column], rtol=1e-7, atol=1e-9)
        np.testing.assert_allclose(output[f"True_{column}"], coefficients[column], rtol=1e-7, atol=1e-9)
