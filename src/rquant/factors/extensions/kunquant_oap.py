"""Expression builders for OAP; every arithmetic, rolling and cross-sectional operation lowers to KunQuant."""

from __future__ import annotations

import contextlib
import hashlib
import re
from functools import lru_cache

from rquant.factors.extensions import kunquant_panel as k


class Expr:
    """A deferred graph expression, allowing statement and daily graphs to share readable formulas."""

    def __init__(self, kind, *args):
        self.kind, self.args = kind, args

    def binary(self, other, op):
        return Expr("scalar", f"x0 {op} x1", self, as_expr(other))

    def __add__(self, other):
        return self.binary(other, "+")

    def __radd__(self, other):
        return as_expr(other) + self

    def __sub__(self, other):
        return self.binary(other, "-")

    def __rsub__(self, other):
        return as_expr(other) - self

    def __mul__(self, other):
        return self.binary(other, "*")

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self.binary(other, "/")

    def __rtruediv__(self, other):
        return as_expr(other) / self

    def __pow__(self, other):
        return Expr("scalar", "std::pow(x0,x1)", self, as_expr(other))

    def __neg__(self):
        return Expr("scalar", "-x0", self)

    def __gt__(self, other):
        return self.binary(other, ">")

    def __ge__(self, other):
        return self.binary(other, ">=")

    def __lt__(self, other):
        return self.binary(other, "<")

    def __le__(self, other):
        return self.binary(other, "<=")

    def __eq__(self, other):
        return self.binary(other, "==")

    def __ne__(self, other):
        return self.binary(other, "!=")

    def __and__(self, other):
        return self.binary(other, "&&")

    def __or__(self, other):
        return self.binary(other, "||")

    def __invert__(self):
        return Expr("scalar", "!x0", self)

    def astype(self, *args, **kwargs):
        return self

    def copy(self):
        return self

    @property
    def shape(self):
        return ()


def as_expr(value):
    return value if isinstance(value, Expr) else Expr("constant", float(value))


def input_(name):
    return Expr("input", name)


def scalar(code, *args):
    return Expr("scalar", code, *(as_expr(x) for x in args))


def state(args, code, members="", init=""):
    return Expr("state", code, members, init, *(as_expr(x) for x in args))


def shift(value, periods):
    return value if periods == 0 else Expr("delay", as_expr(value), periods)


def finite(value):
    return scalar("std::isfinite(x0)", value)


def where(condition, yes, no):
    return scalar("x0 ? x1 : x2", condition, yes, no)


def ratio(a, b):
    return scalar("rq_cnc::safe_div(x0,x1)", a, b)


def ffill(value):
    return state([value], "if (std::isfinite(x0)) s.last=x0; y=s.last;", "{T} last;", "s.last=NAN;")


def trailing(value, window, minimum, how, stride=1):
    """Trailing calendar/residue-class windows, matching pandas min_periods and ddof=1."""
    statistic = {
        "sum": "y=total;",
        "mean": "y=mean;",
        "min": "y=*std::min_element(v.begin(),v.end());",
        "max": "y=*std::max_element(v.begin(),v.end());",
        "std": "if(count>1) y=std::sqrt(ss/(count-1));",
        "var": "if(count>1) y=ss/(count-1);",
        "skew": (
            "if(count>2) {if(*std::min_element(v.begin(),v.end())==*std::max_element(v.begin(),v.end())) y=0; "
            "else if(ss/count>1e-14) "
            "y=std::sqrt(count*(count-1))* (cube/count) / ((count-2)*std::pow(ss/count,1.5));}"
        ),
    }[how]
    code = f"""
    s.history.push_back(x0);
    const int t = (int)s.history.size()-1;
    std::vector<double> v;
    for(int i=std::max(0,t-{(window - 1) * stride});i<=t;++i)
        if((t-i)%{stride}==0 && std::isfinite(s.history[i])) v.push_back(s.history[i]);
    y=NAN;
    if(v.size()>={minimum} && !v.empty()) {{
        double total=0; for(double z:v) total+=z;
        double mean=total/v.size(), ss=0, cube=0;
        for(double z:v) {{double d=z-mean;ss+=d*d;cube+=d*d*d;}}
        const double count=(double)v.size();
        {statistic}
    }}
    """
    return state([value], code, "std::vector<double> history;")


def mean_lags(value, lags, minimum):
    values = [shift(value, j) for j in lags]
    count = sum(finite(v) for v in values)
    return where(count >= minimum, sum(where(finite(v), v, 0) for v in values) / count, float("nan"))


class GraphNumpy:
    nan = float("nan")

    @staticmethod
    def errstate(**kwargs):
        return contextlib.nullcontext()

    where = staticmethod(where)
    isfinite = staticmethod(finite)
    log = staticmethod(lambda x: scalar("std::log(x0)", x))
    log1p = staticmethod(lambda x: scalar("std::log1p(x0)", x))
    abs = staticmethod(lambda x: scalar("std::fabs(x0)", x))
    sqrt = staticmethod(lambda x: scalar("std::sqrt(x0)", x))
    exp = staticmethod(lambda x: scalar("std::exp(x0)", x))
    sign = staticmethod(lambda x: scalar("std::isnan(x0)?NAN:(x0>0?1.0:(x0<0?-1.0:0.0))", x))
    fmax = staticmethod(lambda x, y: scalar("std::fmax(x0,x1)", x, y))
    fmin = staticmethod(lambda x, y: scalar("std::fmin(x0,x1)", x, y))
    maximum = staticmethod(lambda x, y: scalar("std::isnan(x0)||std::isnan(x1)?NAN:std::max(x0,x1)", x, y))
    minimum = staticmethod(lambda x, y: scalar("std::isnan(x0)||std::isnan(x1)?NAN:std::min(x0,x1)", x, y))
    ceil = staticmethod(lambda x: scalar("std::ceil(x0)", x))
    clip = staticmethod(lambda x, a, b: scalar("std::isnan(x0)?NAN:std::min(x2,std::max(x1,x0))", x, a, b))
    zeros = staticmethod(lambda shape: as_expr(0))
    sum = staticmethod(lambda values, axis=None: sum(values))
    all = staticmethod(lambda values, axis=None: __import__("functools").reduce(lambda a, b: a & b, values))


g = GraphNumpy()


def lower(expr, inputs, memo=None):
    memo = {} if memo is None else memo
    expr = as_expr(expr)
    key = id(expr)
    if key in memo:
        return memo[key]
    kind, args = expr.kind, expr.args
    if kind == "input":
        out = inputs[args[0]]
    elif kind == "constant":
        out = k.nan() if args[0] != args[0] else k.const(args[0])
    elif kind == "scalar":
        out = k.scalar_map(args[0], *(lower(x, inputs, memo) for x in args[1:]))
    elif kind == "delay":
        out = k.delay(lower(args[0], inputs, memo), args[1])
    elif kind == "state":
        out = k.cpp_state([lower(x, inputs, memo) for x in args[3:]], args[0], members=args[1], init=args[2])
    elif kind == "cross":
        nodes = [lower(x, inputs, memo) for x in args[2:]]
        out = _cross_class(args[0], args[1], len(memo))(nodes)
    else:
        raise ValueError(kind)
    memo[key] = out
    return out


@lru_cache(None)
def _cross_class(body, head, tag):
    base = k._kq().GenericCrossSectionalOp

    class Cross(base):
        def __init__(self, inputs):
            super().__init__(inputs, [("body", body), ("head", head)])

        def generate_head(self):
            return self.attrs["head"]

        def generate_body(self):
            # A partition binds each distinct buffer once, even when this op reads it more than once.
            names = list(dict.fromkeys(value.attrs["name"] for value in self.inputs))
            slots = [names.index(value.attrs["name"]) for value in self.inputs]
            return re.sub(r"\binput_(\d+)\b", lambda match: f"input_{slots[int(match[1])]}", self.attrs["body"])

    digest = hashlib.sha256(f"{body}\n{head}\n{tag}".encode()).hexdigest()[:16]
    Cross.__name__ = "RQuantOAPCross" + digest
    return Cross


def cross(body, *inputs, head=""):
    return Expr("cross", body, head, *(as_expr(x) for x in inputs))


def group_stat(values, groups, universe, how, weights=None):
    weight = as_expr(1) if weights is None else weights
    stat = {
        "mean": "y=total/ws;",
        "std": (
            "if(v.size()>1) {double ss=0; for(double z:v) ss+=(z-total/ws)*(z-total/ws); y=std::sqrt(ss/(v.size()-1));}"
        ),
        "median": "std::sort(v.begin(),v.end()); y=(v[(v.size()-1)/2]+v[v.size()/2])/2;",
        "count": "y=(double)v.size();",
        "rank": (
            "double less=0,equal=0; for(double z:v) {less+=z<input_0[i];equal+=z==input_0[i];} "
            "if(input_2[i] && std::isfinite(input_0[i])) y=(less+(equal+1)/2)/v.size();"
        ),
    }[how]
    return cross(
        f"""
    for(size_t i=0;i<num_stocks;++i) {{
        double y=NAN,total=0,ws=0;std::vector<double> v;
        if(std::isfinite(input_1[i])) for(size_t j=0;j<num_stocks;++j)
            if(input_1[j]==input_1[i] && input_2[j]>0 && std::isfinite(input_0[j])
               && std::isfinite(input_3[j]) && input_3[j]>0) {{
                v.push_back(input_0[j]);total+=input_0[j]*input_3[j];ws+=input_3[j];
            }}
        if(!v.empty()) {{{stat}}}
        output_0[i]=y;
    }}""",
        values,
        groups,
        universe,
        weight,
    )


def cross_rank(values, universe):
    return group_stat(values, as_expr(1), universe, "rank")


def group_rank(values, groups, universe):
    return group_stat(values, groups, universe, "rank")


def cross_bucket(values, universe, buckets):
    return g.clip(g.ceil(cross_rank(values, universe) * buckets), 1, buckets)


CPP_PRELUDE = r"""
namespace rq_oap {
// One-sided Jacobi SVD: least-squares coefficients including the minimum-norm solution for deficient designs.
static inline std::vector<double> lstsq(std::vector<double> a, const std::vector<double>& y,
                                       int rows, int cols, double rcond=-1.0) {
    std::vector<double> v(cols*cols,0.0), b(cols,0.0), norm(cols,0.0);
    for(int j=0;j<cols;++j) v[j*cols+j]=1;
    for(int sweep=0;sweep<100;++sweep) {
        bool changed=false;
        for(int p=0;p<cols;++p) for(int q=p+1;q<cols;++q) {
            double aa=0,bb=0,ab=0;
            for(int i=0;i<rows;++i) {double x=a[i*cols+p],z=a[i*cols+q];aa+=x*x;bb+=z*z;ab+=x*z;}
            if(std::fabs(ab)<=2e-16*std::sqrt(aa*bb) || ab==0) continue;
            double tau=(bb-aa)/(2*ab), t=std::copysign(1.0,tau)/(std::fabs(tau)+std::sqrt(1+tau*tau));
            double c=1/std::sqrt(1+t*t),s=c*t;
            for(int i=0;i<rows;++i) {double x=a[i*cols+p],z=a[i*cols+q];a[i*cols+p]=c*x-s*z;a[i*cols+q]=s*x+c*z;}
            for(int i=0;i<cols;++i) {double x=v[i*cols+p],z=v[i*cols+q];v[i*cols+p]=c*x-s*z;v[i*cols+q]=s*x+c*z;}
            changed=true;
        }
        if(!changed) break;
    }
    for(int j=0;j<cols;++j) for(int i=0;i<rows;++i) norm[j]+=a[i*cols+j]*a[i*cols+j];
    const double tolerance=rcond<0?std::max(rows,cols)*2.220446049250313e-16:rcond;
    double cutoff=*std::max_element(norm.begin(),norm.end())*tolerance*tolerance;
    for(int j=0;j<cols;++j) if(norm[j]>cutoff) {
        double dot=0;for(int i=0;i<rows;++i) dot+=a[i*cols+j]*y[i];
        for(int c=0;c<cols;++c) b[c]+=v[c*cols+j]*dot/norm[j];
    }
    return b;
}
}
"""


def cross_fit(target, regressors, universe, groups=None, minimum=30, coefficient=None):
    count = len(regressors)
    group = as_expr(float("nan")) if groups is None else groups
    complete = " && ".join(f"std::isfinite(input_{j + 3}[i])" for j in range(count)) or "true"
    assignment = " ".join(f"a.push_back(input_{j + 3}[i]);" for j in range(count))
    prediction = "output_0[i]=y[r]-pred;" if coefficient is None else f"output_0[i]=beta[{coefficient}];"
    tail = "" if coefficient is None else f"for(size_t i=0;i<num_stocks;++i) output_0[i]=beta[{coefficient}];"
    return cross(
        f"""
    std::vector<size_t> rows;std::vector<double> codes,a,y;
    for(size_t i=0;i<num_stocks;++i) {{
        output_0[i]=NAN;
        if(input_1[i]>0 && std::isfinite(input_0[i]) && {complete}
           {"&& std::isfinite(input_2[i])" if groups is not None else ""}) {{
            rows.push_back(i); if({"true" if groups is not None else "false"}) codes.push_back(input_2[i]);
        }}
    }}
    std::sort(codes.begin(),codes.end());codes.erase(std::unique(codes.begin(),codes.end()),codes.end());
    const int cols=1+{count}+std::max(0,(int)codes.size()-1);
    if(rows.size()>={minimum} && rows.size()>=(size_t)(cols+10)) {{
        for(size_t i:rows) {{
            a.push_back(1);{assignment}
            for(size_t j=1;j<codes.size();++j) a.push_back(input_2[i]==codes[j]);
            y.push_back(input_0[i]);
        }}
        auto beta=rq_oap::lstsq(a,y,(int)rows.size(),cols);
        for(size_t r=0;r<rows.size();++r) {{
            size_t i=rows[r];double pred=0;
            for(int j=0;j<cols;++j) pred+=a[r*cols+j]*beta[j];
            {prediction}
        }}
        {tail}
    }}""",
        target,
        universe,
        group,
        *regressors,
    )


def cross_residual(target, regressors, universe, *, groups=None, minimum=30):
    return cross_fit(target, regressors, universe, groups, minimum)


def regression(target, regressors, window, minimum, kind, stride=1, first=0, last=0):
    count = len(regressors) + 1
    append = "".join(f"s.h{j}.push_back(x{j});" for j in range(count))
    gather = "".join(f"v{j}.push_back(s.h{j}[i]);" for j in range(count))
    vectors = "".join(f"std::vector<double> v{j};" for j in range(count))
    pointers = ",".join(f"v{j}.data()" for j in range(1, count))
    code = f"""{append} y=NAN;int t=(int)s.h0.size()-1;
    if(t>={(window - 1) * stride}) {{
        {vectors}
        for(int i=t-{(window - 1) * stride};i<=t;i+={stride}) {{{gather}}}
        std::vector<double> result({window},NAN);
        rq_oap_regression::rolling_regression({window},v0.data(),{{{pointers}}},{window},{minimum},
                                   rq_cnc::{kind},{first},{last},result.data());
        y=result.back();
    }}"""
    return state([target, *regressors], code, "".join(f"std::vector<double> h{j};" for j in range(count)))


def coskew(target, market, window, minimum, stride=1):
    return state(
        [target, market],
        f"""
    s.a.push_back(x0);s.b.push_back(x1);int t=(int)s.a.size()-1;y=NAN;
    if(t>={(window - 1) * stride}) {{
        double count=0,sy=0,sm=0;
        for(int i=t-{(window - 1) * stride};i<=t;i+={stride})
            if(std::isfinite(s.a[i])&&std::isfinite(s.b[i])) {{++count;sy+=s.a[i];sm+=s.b[i];}}
        if(count>={minimum}) {{double my=sy/count,mm=sm/count,yy=0,bb=0,co=0;
            for(int i=t-{(window - 1) * stride};i<=t;i+={stride})
                if(std::isfinite(s.a[i])&&std::isfinite(s.b[i])) {{
                    double dy=s.a[i]-my,dm=s.b[i]-mm;
                    yy+=dy*dy;bb+=dm*dm;co+=dy*dm*dm;
                }}
            if(yy>0&&bb>0) y=(co/count)/(std::sqrt(yy/count)*bb/count);
        }}
    }}""",
        "std::vector<double> a,b;",
    )


def correlation(a, b, window, minimum):
    return state(
        [a, b],
        f"""
    s.a.push_back(x0);s.b.push_back(x1);int t=(int)s.a.size()-1;double n=0,sa=0,sb=0;y=NAN;
    for(int i=std::max(0,t-{window - 1});i<=t;++i)
        if(std::isfinite(s.a[i])&&std::isfinite(s.b[i])) {{++n;sa+=s.a[i];sb+=s.b[i];}}
    if(n>={minimum}) {{
        double aa=0,bb=0,ab=0;
        for(int i=std::max(0,t-{window - 1});i<=t;++i)
            if(std::isfinite(s.a[i])&&std::isfinite(s.b[i])) {{
                double da=s.a[i]-sa/n,db=s.b[i]-sb/n;aa+=da*da;bb+=db*db;ab+=da*db;
            }}
        y=ab/std::sqrt(aa*bb);
    }}
    """,
        "std::vector<double> a,b;",
    )


def price_delay(target, market, window, minimum, which):
    xs = [target, *[shift(market, j) for j in range(5)]]
    append = "".join(f"s.h{j}.push_back(x{j});" for j in range(6))
    check = " && ".join(f"std::isfinite(s.h{j}[i])" for j in range(6))
    design = ",".join(["1.0", *[f"s.h{j}[i]" for j in range(1, 6)]])
    expression = {"rsq": "r2>0?1-r0/r2:NAN", "slope": "weighted/sum", "tstat": "wt/ts"}[which]
    return state(
        xs,
        f"""
    {append} int t=(int)s.h0.size()-1;y=NAN;
    if(t>={window - 1}) {{
        std::vector<double> gram(36,0),xy(6,0),beta(6);double count=0,yy=0;
        for(int i=t-{window - 1};i<=t;++i) if({check}) {{
            double a[6]={{{design}}};double z=s.h0[i];++count;yy+=z*z;
            for(int j=0;j<6;++j) {{xy[j]+=a[j]*z;for(int k=0;k<6;++k) gram[j*6+k]+=a[j]*a[k];}}
        }}
        if(count>={minimum}) {{
            const auto inv=rq_oap_regression::inverse_or_pinv(6,gram);
            for(int j=0;j<6;++j) for(int k=0;k<6;++k) beta[j]+=inv[j*6+k]*xy[k];
            double sse=yy;for(int j=0;j<6;++j) sse-=beta[j]*xy[j];
            double total=yy-xy[0]*xy[0]/count;
            if(total>0&&sse>1e-14*std::max(yy,1e-300)) {{
                double det=gram[0]*gram[7]-gram[1]*gram[6];
                double b0=(xy[0]*gram[7]-xy[1]*gram[1])/det,b1=(xy[1]*gram[0]-xy[0]*gram[6])/det;
                double r2=1-sse/total,r0=1-(yy-b0*xy[0]-b1*xy[1])/total;
                double weighted=0,sum=0,wt=0,ts=0;
                for(int j=1;j<6;++j) {{
                    sum+=beta[j];weighted+=(j-1)*beta[j];
                    double stat=beta[j]/std::sqrt(sse/(count-6)*inv[j*6+j]);ts+=stat;wt+=(j-1)*stat;
                }}
                y={expression};
            }}
        }}
    }}""",
        "".join(f"std::vector<double> h{j};" for j in range(6)),
    )
