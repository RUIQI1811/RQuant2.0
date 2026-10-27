"""KunQuant operators with RQuant's pandas panel semantics.

RQuant's panel factor libraries were first written against aligned pandas date-by-instrument panels
(``rquant.factors.panel_operators``). This module rebuilds those operators from KunQuant graph nodes so the same
formulas run inside a compiled KunQuant graph and publish the same values:

* rolling operators need a complete window (``min_periods == window``): one missing value in the window gives a
  missing result, and the first ``window - 1`` rows are missing;
* comparisons with a missing value are false, and element-wise minimum, maximum and clipping keep missing values
  missing instead of returning the other operand;
* divisions by a near-zero denominator and infinite results are missing (``safe_div``);
* a correlation over a window in which either side is constant is 0, and a constant window's standard deviation is
  exactly 0;
* logarithms, powers and trigonometric functions call the C library per element (``scalar_map``) instead of
  KunQuant's vectorised approximations, so they agree with NumPy to the last bits.

Operators KunQuant does not have are built from its stateful ``Accumulator`` or from ``CppStateOp``, a small
RQuant node that runs a C++ snippet per stock and time step with per-stock state. Every function must be called
inside a KunQuant ``Builder`` context.
"""

from __future__ import annotations

import itertools
import math
from functools import lru_cache
from types import SimpleNamespace
from typing import Any

from rquant.errors import DependencyError

Node = Any
Operand = Any  # a KunQuant node or a Python number

_ACCUMULATOR_NAMES = itertools.count()


@lru_cache(maxsize=1)
def _kq() -> SimpleNamespace:
    try:
        from KunQuant import Op, ops
    except ImportError as exc:
        raise DependencyError("KunQuant==0.1.11 is required to build factor graphs") from exc
    symbols = {name: getattr(ops, name) for name in dir(ops) if not name.startswith("_")}
    symbols.update({name: getattr(Op, name) for name in dir(Op) if not name.startswith("_")})
    symbols["CppStateOp"] = _cpp_state_op_class(Op)
    symbols["SeriesKernelOp"] = _series_kernel_op_class(Op)
    return SimpleNamespace(**symbols)


def _cpp_state_op_class(op_module: Any) -> type:
    class CppStateOp(op_module.OpBase, op_module.GloablStatefulOpTrait, op_module.MayRequireWholeTime):
        """Per-stock C++ step function with optional state carried from one time step to the next.

        ``members`` declares the per-stock state fields (``{T}`` is the element type), ``init`` runs once per stock
        with the state as ``s``, and ``step`` runs every time step with the inputs as ``x0, x1, ...`` and must
        assign the output ``y``. All three are C++ snippets formatted with ``T``.
        """

        def __init__(self, inputs: list[Any], members: str, init: str, step: str) -> None:
            super().__init__(list(inputs), [("members", members), ("init", init), ("step", step)])

        def is_whole_time_required(self) -> bool:
            return bool(self.attrs["members"])

        def generate_init_code(
            self, idx: str, elem_type: str, simd_lanes: int, inputs: list[str], aligned: bool
        ) -> str:
            vec = f"kun_simd::vec<{elem_type}, {simd_lanes}>"
            count = len(self.inputs)
            members = self.attrs["members"].replace("{T}", elem_type)
            init = self.attrs["init"].replace("{T}", elem_type)
            step = self.attrs["step"].replace("{T}", elem_type)
            parameters = ", ".join(f"{vec} a{k}" for k in range(count))
            stores = " ".join(
                f"alignas(64) {elem_type} b{k}[{simd_lanes}]; {vec}::store(a{k}, b{k});" for k in range(count)
            )
            scalars = " ".join(f"const {elem_type} x{k} = b{k}[lane];" for k in range(count))
            return (
                f"struct rq_state_{idx}_t {{ {members} int rq_unused; }}; "
                f"rq_state_{idx}_t rq_states_{idx}[{simd_lanes}] = {{}}; "
                f"for (int lane = 0; lane < {simd_lanes}; ++lane) {{ "
                f"auto& s = rq_states_{idx}[lane]; (void)s; {init} }} "
                f"auto rq_step_{idx} = [&rq_states_{idx}]({parameters}) {{ {stores} alignas(64) {elem_type} "
                f"rq_out[{simd_lanes}]; for (int lane = 0; lane < {simd_lanes}; ++lane) {{ auto& s = "
                f"rq_states_{idx}[lane]; (void)s; {scalars} {elem_type} y; {step} rq_out[lane] = y; }} "
                f"return {vec}::load(rq_out); }};"
            )

        def generate_step_code(self, idx: str, time_idx: str, inputs: list[str], **kwargs: Any) -> str:
            return f"auto v{idx} = rq_step_{idx}({', '.join(inputs)});"

    CppStateOp.__name__ = "RQuantCppStateOp"
    return CppStateOp


def _series_kernel_op_class(op_module: Any) -> type:
    class SeriesKernelOp(op_module.OpBase, op_module.GloablStatefulOpTrait, op_module.MayRequireWholeTime):
        """A C++ kernel that reads each stock's whole input series at once and writes its whole output series.

        The inputs must be graph inputs (their buffers hold every time step before the stage runs). ``kernel`` is a
        C++ block that sees ``n`` (the number of time steps), the stock's inputs as ``const T *x0, *x1, ...`` and
        fills ``T *out``, which starts all missing; the node then publishes ``out`` step by step.
        """

        def __init__(self, inputs: list[Any], kernel: str) -> None:
            for value in inputs:
                if not isinstance(value, op_module.Input):
                    raise TypeError("SeriesKernelOp reads whole input buffers and only accepts graph inputs")
            super().__init__(list(inputs), [("kernel", kernel)])

        def generate_init_code(
            self, idx: str, elem_type: str, simd_lanes: int, inputs: list[str], aligned: bool
        ) -> str:
            buffers = [f"buf_{value.attrs['name']}" for value in self.inputs]
            declare = " ".join(f"std::vector<T> rq_x{k}(__length);" for k in range(len(buffers)))
            gather = " ".join(
                f"rq_x{k}[t] = {buffer}.getPtr(t)[lane];" for k, buffer in enumerate(buffers)
            )
            pointers = " ".join(f"const T *x{k} = rq_x{k}.data();" for k in range(len(buffers)))
            return (
                f"std::vector<{elem_type}> rq_series_{idx}(__length * {simd_lanes}); "
                f"{{ using T = {elem_type}; {declare} std::vector<T> rq_out(__length); "
                f"for (int lane = 0; lane < {simd_lanes}; ++lane) {{ "
                f"for (size_t t = 0; t < __length; ++t) {{ {gather} }} "
                "std::fill(rq_out.begin(), rq_out.end(), std::numeric_limits<T>::quiet_NaN()); "
                f"{{ const int n = (int)__length; {pointers} T *out = rq_out.data(); (void)n; "
                f"{self.attrs['kernel']} }} "
                f"for (size_t t = 0; t < __length; ++t) rq_series_{idx}[t * {simd_lanes} + lane] = rq_out[t]; }} }} "
                f"auto rq_series_at_{idx} = [&rq_series_{idx}](size_t t) {{ "
                f"return kun_simd::vec<{elem_type}, {simd_lanes}>::load(rq_series_{idx}.data() + t * {simd_lanes}); }};"
            )

        def generate_step_code(self, idx: str, time_idx: str, inputs: list[str], **kwargs: Any) -> str:
            return f"auto v{idx} = rq_series_at_{idx}({time_idx});"

    SeriesKernelOp.__name__ = "RQuantSeriesKernelOp"
    return SeriesKernelOp


# ---------------------------------------------------------------------------------------------------- basics
def const(value: float | str) -> Node:
    return _kq().ConstantOp(value)


def nan() -> Node:
    return _kq().ConstantOp("nan")


def node(value: Operand) -> Node:
    return value if isinstance(value, _kq().OpBase) else const(float(value))


def always() -> Node:
    """A true mask."""
    k = _kq()
    return k.Equals(k.ConstantOp(0.0), k.ConstantOp(0.0))


def notna(value: Node) -> Node:
    return _kq().Equals(value, value)


def isna(value: Node) -> Node:
    return _kq().Not(notna(value))


def valid(*values: Node) -> Node:
    """True where every value is present."""
    k = _kq()
    result = notna(values[0])
    for value in values[1:]:
        result = k.And(result, notna(value))
    return result


def all_of(*conditions: Node) -> Node:
    k = _kq()
    result = conditions[0]
    for condition in conditions[1:]:
        result = k.And(result, condition)
    return result


def any_of(*conditions: Node) -> Node:
    k = _kq()
    result = conditions[0]
    for condition in conditions[1:]:
        result = k.Or(result, condition)
    return result


def select(condition: Node, when_true: Operand, when_false: Operand) -> Node:
    """``when_true`` where the condition holds, else ``when_false`` (a missing condition is false)."""
    return _kq().Select(condition, node(when_true), node(when_false))


def mask(value: Operand, condition: Node) -> Node:
    """pandas ``value.where(condition)``: missing where the condition does not hold."""
    return select(condition, value, nan())


def indicator(condition: Node) -> Node:
    """1.0 where the condition holds, else 0.0."""
    return select(condition, 1.0, 0.0)


def gt(left: Operand, right: Operand) -> Node:
    return _kq().GreaterThan(node(left), node(right))


def ge(left: Operand, right: Operand) -> Node:
    return _kq().GreaterEqual(node(left), node(right))


def lt(left: Operand, right: Operand) -> Node:
    return _kq().LessThan(node(left), node(right))


def le(left: Operand, right: Operand) -> Node:
    return _kq().LessEqual(node(left), node(right))


def eq(left: Operand, right: Operand) -> Node:
    return _kq().Equals(node(left), node(right))


def ne(left: Operand, right: Operand) -> Node:
    """Not equal; false when either side is missing, like pandas ``!=`` followed by a validity mask."""
    left, right = node(left), node(right)
    return all_of(valid(left, right), _kq().Not(_kq().Equals(left, right)))


def replace_inf(value: Node) -> Node:
    """Missing where the value is infinite."""
    return mask(value, lt(_kq().Abs(value), math.inf))


def safe_div(numerator: Operand, denominator: Operand) -> Node:
    """Division that is missing for a denominator within 1e-12 of zero and for infinite results."""
    numerator = node(numerator)
    if not isinstance(denominator, _kq().OpBase):
        if abs(float(denominator)) < 1e-12:
            return mask(numerator, lt(const(0.0), const(0.0)))
        return replace_inf(numerator / float(denominator))
    usable = select(lt(_kq().Abs(denominator), 1e-12), nan(), denominator)
    return replace_inf(numerator / usable)


def emin(left: Operand, right: Operand) -> Node:
    left, right = node(left), node(right)
    return mask(_kq().Min(left, right), valid(left, right))


def emax(left: Operand, right: Operand) -> Node:
    left, right = node(left), node(right)
    return mask(_kq().Max(left, right), valid(left, right))


def clip(value: Node, lower: float | None = None, upper: float | None = None) -> Node:
    k = _kq()
    result = value
    if lower is not None:
        result = k.Max(result, const(lower))
    if upper is not None:
        result = k.Min(result, const(upper))
    return mask(result, notna(value))


def absolute(value: Node) -> Node:
    return _kq().Abs(value)


def sign(value: Node) -> Node:
    return _kq().Sign(value)


def sqrt(value: Node) -> Node:
    return _kq().Sqrt(value)


# ---------------------------------------------------------------------------------------------------- C++ helpers
def cpp_state(inputs: list[Node], step: str, *, members: str = "", init: str = "") -> Node:
    """A per-stock C++ step function; see ``CppStateOp``."""
    return _kq().CppStateOp(inputs, members, init, step)


def series_kernel(inputs: list[Node], kernel: str) -> Node:
    """A C++ kernel over each stock's whole series of graph inputs; see ``SeriesKernelOp``."""
    return _kq().SeriesKernelOp(list(inputs), kernel)


def scalar_map(expression: str, *inputs: Operand) -> Node:
    """Apply a C++ expression of ``x0, x1, ...`` to every element, e.g. ``std::log(x0)``."""
    return cpp_state([node(value) for value in inputs], f"y = ({expression});")


def log(value: Node) -> Node:
    return scalar_map("std::log(x0)", value)


def exp(value: Node) -> Node:
    return scalar_map("std::exp(x0)", value)


def power(base: Operand, exponent: Operand) -> Node:
    """NumPy ``power``: ``pow`` of the C library, missing for a negative base with a fractional exponent."""
    return scalar_map("std::pow(x0, x1)", base, exponent)


# ---------------------------------------------------------------------------------------------------- cross-section
SNAP_BITS = 40
SNAP_ZERO = 1e-12


def snap(value: Node) -> Node:
    """Round to ``SNAP_BITS`` significant bits and flush ``|x| < SNAP_ZERO`` to zero.

    Ranks compare values that are often equal in exact arithmetic (statistics of ranks, zero covariances) but differ
    in the last bits depending on the summation order; snapping makes such values tie exactly, so ranks do not
    depend on the compiler or the CPU.
    """
    return cpp_state(
        [node(value)],
        f"int e = 0; {{T}} m = std::frexp(x0, &e); "
        f"y = !std::isfinite(x0) ? x0 : std::fabs(x0) < {SNAP_ZERO!r} ? 0 : "
        f"std::ldexp(std::nearbyint(std::ldexp(m, {SNAP_BITS})), e - {SNAP_BITS});",
    )


def rank(value: Node) -> Node:
    """Percentile rank among the stocks with a value on the same day (ties share their average rank)."""
    return _kq().Rank(snap(value))


def scale(value: Node) -> Node:
    """Divide by the day's sum of absolute values."""
    return _kq().Scale(value)


# ---------------------------------------------------------------------------------------------------- time series
def delay(value: Node, periods: int) -> Node:
    return value if periods == 0 else _kq().BackRef(value, int(periods))


def delta(value: Node, periods: int) -> Node:
    return value - delay(value, periods)


def ts_sum(value: Node, window: int) -> Node:
    return _kq().WindowedSum(value, int(window))


def mean(value: Node, window: int) -> Node:
    return _kq().WindowedAvg(value, int(window))


def ts_min(value: Node, window: int) -> Node:
    return value if window == 1 else _kq().WindowedMin(value, int(window))


def ts_max(value: Node, window: int) -> Node:
    return value if window == 1 else _kq().WindowedMax(value, int(window))


def _constant_window(value: Node, window: int) -> Node:
    """True where a complete window holds one repeated value."""
    return eq(ts_max(value, window), ts_min(value, window))


def stddev(value: Node, window: int) -> Node:
    """Sample standard deviation; exactly 0 over a constant window, as pandas reports it."""
    return select(_constant_window(value, window), 0.0, _kq().WindowedStddev(value, int(window)))


def covariance(left: Node, right: Node, window: int) -> Node:
    return _kq().WindowedCovariance(left, int(window), right)


def correlation(left: Node, right: Node, window: int) -> Node:
    """Pearson correlation; 0 where either side's window is constant, clipped to [-1, 1]."""
    raw = _kq().WindowedCorrelation(left, int(window), right)
    flat = any_of(_constant_window(left, window), _constant_window(right, window))
    return clip(replace_inf(select(flat, 0.0, raw)), -1.0, 1.0)


def ts_rank(value: Node, window: int) -> Node:
    """Percentile rank of the newest value inside its window (ties share their average rank)."""
    return _kq().TsRank(snap(value), int(window)) / float(window)


def decay_linear(value: Node, window: int) -> Node:
    """Weighted mean with weights 1..window, the newest value weighted most."""
    return value if window == 1 else _kq().DecayLinear(value, int(window))


def product(value: Node, window: int) -> Node:
    return value if window == 1 else _kq().WindowedProduct(value, int(window))


def weighted_sum(value: Node, weights: list[float]) -> Node:
    """``sum(weights[k] * value[t - k])``; ``weights[0]`` applies to the newest value."""
    total = None
    for lag, weight in enumerate(weights):
        term = delay(value, lag) * float(weight)
        total = term if total is None else total + term
    return total


def days_since_extreme(value: Node, window: int, *, maximum: bool) -> Node:
    """Days from the newest row back to the latest window maximum (or minimum); 0 when it is today."""
    extreme = ts_max(value, window) if maximum else ts_min(value, window)
    result = nan()
    for lag in range(window - 1, -1, -1):
        result = select(eq(delay(value, lag), extreme), float(lag), result)
    return result


def count(condition: Node, window: int, *, valid_rows: Node | None = None) -> Node:
    """Rolling count of rows where the condition holds; rows outside ``valid_rows`` make the window missing."""
    numeric = indicator(condition)
    if valid_rows is not None:
        numeric = mask(numeric, valid_rows)
    return ts_sum(numeric, window)


def sumif(value: Node, window: int, condition: Node, *, valid_rows: Node | None = None) -> Node:
    """Rolling sum of the values on rows where the condition holds."""
    selected = mask(select(condition, value, 0.0), notna(value))
    if valid_rows is not None:
        selected = mask(selected, valid_rows)
    return ts_sum(selected, window)


def nan_mean(*values: Node) -> Node:
    """Mean of the present values among ``values``; missing when none is present."""
    terms = [f"if (!std::isnan(x{index})) {{ total += x{index}; ++count; }}" for index in range(len(values))]
    body = " ".join(terms)
    return cpp_state(list(values), f"{{T}} total = 0; int count = 0; {body} y = count > 0 ? total / count : NAN;")


def rolling_nan_std(value: Node, window: int, minimum: int) -> Node:
    """Sample standard deviation of the present values among the newest ``window`` rows.

    Missing with fewer than ``max(minimum, 2)`` present values, like pandas ``rolling(window, min_periods)``.
    """
    return cpp_state(
        [value],
        f"s.ring[s.next] = x0; s.next = (s.next + 1) % {window}; "
        f"{{T}} total = 0; int count = 0; "
        f"for (int i = 0; i < {window}; ++i) if (!std::isnan(s.ring[i])) {{ total += s.ring[i]; ++count; }} "
        f"const {{T}} mean = total / count; {{T}} squares = 0; "
        f"for (int i = 0; i < {window}; ++i) if (!std::isnan(s.ring[i])) "
        "squares += (s.ring[i] - mean) * (s.ring[i] - mean); "
        f"y = count >= {max(minimum, 2)} ? std::sqrt(squares / (count - 1)) : NAN;",
        members=f"{{T}} ring[{window}]; int next;",
        init=f"for (int i = 0; i < {window}; ++i) s.ring[i] = NAN; s.next = 0;",
    )


def recursive(value: Node, update: Any, *, initial: float | str = "nan") -> Node:
    """Carry a per-stock state through time: ``state = update(value, previous_state)`` each row.

    The returned node is the new state of each row. ``update`` receives KunQuant nodes and builds the new state.
    """
    k = _kq()
    accumulator = k.Accumulator(value, f"rq_state_{next(_ACCUMULATOR_NAMES)}", is_whole_time_required=True,
                                init_val=initial)
    new_state = update(value, accumulator)
    stored = k.SetAccumulator(accumulator, always(), new_state)
    return k.ReturnFirstValue([new_state, stored])


def sma_cn(value: Node, periods: int, weight: int) -> Node:
    """Chinese SMA ``Y = (m X + (n - m) Y[-1]) / n``, restarted from ``X`` after every missing row."""
    n, m = float(periods), float(weight)

    def update(x: Node, previous: Node) -> Node:
        smoothed = select(notna(previous), (x * m + previous * (n - m)) / n, x)
        return mask(smoothed, notna(x))

    return recursive(value, update)
