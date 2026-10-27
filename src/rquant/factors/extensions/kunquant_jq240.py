"""Symbolic panel facade and C++ rolling kernels for JoinQuant definitions.

The facade builds KunQuant nodes only; it never evaluates production arrays.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from rquant.factors.extensions import kunquant_panel as k
from rquant.factors.extensions.kunquant import kunquant_symbols


def unwrap(x):
    if isinstance(x, Value):
        return x.node
    if isinstance(x, (int, float)) and not np.isfinite(x):
        return k.nan() if np.isnan(x) else k.const(float(x))
    return k.node(x)


class Value:
    __array_priority__ = 1000

    def __init__(self, node, trading=None):
        self.node, self.trading = node, trading

    def wrap(self, node):
        return Value(node, self.trading)

    def __add__(self, other):
        return self.wrap(self.node + unwrap(other))

    __radd__ = __add__

    def __sub__(self, other):
        return self.wrap(self.node - unwrap(other))

    def __rsub__(self, other):
        return self.wrap(unwrap(other) - self.node)

    def __mul__(self, other):
        return self.wrap(self.node * unwrap(other))

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self.wrap(self.node / unwrap(other))

    def __rtruediv__(self, other):
        return self.wrap(unwrap(other) / self.node)

    def __pow__(self, other):
        return self.wrap(k.power(self.node, unwrap(other)))

    def __neg__(self):
        return self.wrap(0.0 - self.node)

    def __gt__(self, other):
        return self.wrap(k.indicator(k.gt(self.node, unwrap(other))))

    def __lt__(self, other):
        return self.wrap(k.indicator(k.lt(self.node, unwrap(other))))

    def __ge__(self, other):
        return self.wrap(k.indicator(k.ge(self.node, unwrap(other))))

    def __le__(self, other):
        return self.wrap(k.indicator(k.le(self.node, unwrap(other))))

    def __eq__(self, other):
        return self.wrap(k.indicator(k.eq(self.node, unwrap(other))))

    def __and__(self, other):
        return self.wrap(k.indicator(k.all_of(k.gt(self.node, 0), k.gt(unwrap(other), 0))))

    def __or__(self, other):
        return self.wrap(k.indicator(k.any_of(k.gt(self.node, 0), k.gt(unwrap(other), 0))))

    def __invert__(self):
        return self.wrap(k.indicator(k.eq(self.node, 0)))

    def abs(self):
        return self.wrap(k.absolute(self.node))

    def notna(self):
        return self.wrap(k.indicator(k.notna(self.node)))

    def where(self, condition, other=np.nan):
        return self.wrap(k.select(k.gt(unwrap(condition), 0), self.node, unwrap(other)))

    def mask(self, condition):
        return self.where(~condition)

    def fillna(self, value):
        return self.wrap(k.select(k.notna(self.node), self.node, unwrap(value)))

    def astype(self, dtype):
        return self

    def clip(self, lower=None, upper=None):
        return self.wrap(k.clip(self.node, lower, upper))

    def shift(self, periods):
        if not periods:
            return self
        if self.trading is None:
            return self.wrap(k.delay(self.node, periods))
        return self.wrap(
            k.cpp_state(
                [self.node, self.trading],
                f"""y=std::numeric_limits<{{T}}>::quiet_NaN(); if(!std::isnan(x1)){{
                if(s.v.size()>={periods}) y=s.v[s.v.size()-{periods}];
                s.v.push_back(x0); if(s.v.size()>{periods}) s.v.erase(s.v.begin()); }}""",
                members="std::vector<{T}> v;",
            )
        )

    def rolling(self, window, min_periods=None):
        return Rolling(self, window, window if min_periods is None else min_periods)

    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        nodes = [unwrap(x) for x in args]
        functions = {
            np.absolute: k.absolute,
            np.log: k.log,
            np.sqrt: k.sqrt,
            np.maximum: k.emax,
            np.minimum: k.emin,
            np.log1p: lambda x: k.scalar_map("std::log1p(x0)", x),
        }
        return self.wrap(functions[ufunc](*nodes))


class Rolling:
    def __init__(self, value, window, minimum):
        self.value, self.window, self.minimum = value, window, minimum

    def reduce(self, statistic):
        value, w, m = self.value, self.window, self.minimum
        inputs = [value.node] + ([] if value.trading is None else [value.trading])
        guard = "true" if value.trading is None else "!std::isnan(x1)"
        body = f"""y=std::numeric_limits<{{T}}>::quiet_NaN(); if({guard}){{
        s.v.push_back(x0); if(s.v.size()>{w}) s.v.erase(s.v.begin());
        int count=0; double total=0,lo=INFINITY,hi=-INFINITY;
        for(double v:s.v) if(std::isfinite(v)){{++count;total+=v;lo=std::min(lo,v);hi=std::max(hi,v);}}
        if(count>={m}){{ double mean=total/count,ss=0,m3=0,m4=0,ad=0,slope=0;
        for(size_t i=0;i<s.v.size();++i) if(std::isfinite(s.v[i])){{
        double d=s.v[i]-mean;ss+=d*d;m3+=d*d*d;m4+=d*d*d*d;ad+=std::abs(d);
        slope+=(double(i)-(double(s.v.size())-1)/2)*s.v[i];}}
        {statistic}
        }} }}"""
        return value.wrap(k.cpp_state(inputs, body, members="std::vector<{T}> v;"))

    def sum(self):
        return self.reduce("y=total;")

    def mean(self):
        return self.reduce("y=mean;")

    def std(self):
        return self.reduce("if(count>1)y=std::sqrt(ss/(count-1));")

    def min(self):
        return self.reduce("y=lo;")

    def max(self):
        return self.reduce("y=hi;")

    def skew(self):
        return self.reduce(
            "if(count>=3){if(lo==hi)y=0;else if(ss/count>1e-14)"
            "y=count*std::sqrt(count-1.0)/(count-2)*m3/std::pow(ss,1.5);}"
        )

    def kurt(self):
        return self.reduce(
            "if(count>=4){if(lo==hi)y=-3;else if(ss/count>1e-14)"
            "y=((count*count-1.0)*count*m4/(ss*ss)-3*(count-1.0)*(count-1.0))/((count-2.0)*(count-3.0));}"
        )


def ema(value, span):
    inputs = [value.node] + ([] if value.trading is None else [value.trading])
    guard = "true" if value.trading is None else "!std::isnan(x1)"
    alpha = 2 / (span + 1)
    return value.wrap(
        k.cpp_state(
            inputs,
            f"""
    y=std::numeric_limits<{{T}}>::quiet_NaN(); if({guard}){{
    if(!std::isnan(s.mean)) s.weight*={1 - alpha!r};
    if(!std::isnan(x0)){{++s.count;if(std::isnan(s.mean))s.mean=x0;
    else s.mean=(s.weight*s.mean+{alpha!r}*x0)/(s.weight+{alpha!r});s.weight=1;}}
    if(s.count>={span})y=s.mean; }}""",
            members="double mean,weight; int count;",
            init="s.mean=NAN;s.weight=1;s.count=0;",
        )
    )


def ratio(a, b):
    return Value(k.safe_div(unwrap(a), unwrap(b)), getattr(a, "trading", getattr(b, "trading", None)))


def lag(value, periods):
    return value.shift(periods)


def replace_inf(value):
    return value.wrap(k.replace_inf(value.node))


@lru_cache(None)
def cross_class(kind, regressors=0):
    base = kunquant_symbols("GenericCrossSectionalOp")["GenericCrossSectionalOp"]

    class Cross(base):
        def __init__(self, inputs):
            super().__init__(inputs, [("kind", kind), ("regressors", regressors)])

        def generate_head(self):
            return ""

        def generate_body(self):
            if kind == "z":
                return """
            double sum=0;int count=0;for(size_t i=0;i<num_stocks;++i)
            if(input_1[i]==1 && std::isfinite(input_0[i])){sum+=input_0[i];++count;}
            double mean=sum/count,ss=0;for(size_t i=0;i<num_stocks;++i)
            if(input_1[i]==1 && std::isfinite(input_0[i]))ss+=(input_0[i]-mean)*(input_0[i]-mean);
            double sd=std::sqrt(ss/(count-1));for(size_t i=0;i<num_stocks;++i)
            output_0[i]=sd>0?std::max(T(-3),std::min(T(3),T((input_0[i]-mean)/sd))):NAN;
            for(size_t i=0;i<num_stocks;++i)if(!std::isfinite(input_0[i]))output_0[i]=NAN;
            """
            # One-sided Jacobi SVD works on the design itself, avoiding squared condition numbers.
            p = regressors + 1
            xs = "".join(
                f"x[{j + 1}]=input_{j + 2}[i];valid=valid&&std::isfinite(x[{j + 1}]);" for j in range(regressors)
            )
            return f"""
            const int p={p};std::vector<double> design, target;double v[p][p]={{}},coef[p]={{}};
            for(int j=0;j<p;++j)v[j][j]=1;
            for(size_t i=0;i<num_stocks;++i){{double x[p]={{1}};bool valid=std::isfinite(input_0[i]);{xs}
            if(!valid||input_1[i]!=1)continue;target.push_back(input_0[i]);for(int j=0;j<p;++j)design.push_back(x[j]);}}
            const int count=target.size();
            for(int it=0;it<100;++it){{bool changed=false;
            for(int h=0;h<p;++h)for(int l=h+1;l<p;++l){{double aa=0,bb=0,ab=0;
            for(int i=0;i<count;++i){{double a=design[i*p+h],b=design[i*p+l];aa+=a*a;bb+=b*b;ab+=a*b;}}
            if(std::abs(ab)<=2.22e-16*std::sqrt(aa*bb))continue;changed=true;
            double tau=(bb-aa)/(2*ab),t=(tau>=0?1.0:-1.0)/(std::abs(tau)+std::sqrt(1+tau*tau));
            double c=1/std::sqrt(1+t*t),s=t*c;
            for(int i=0;i<count;++i){{double a=design[i*p+h],b=design[i*p+l];
            design[i*p+h]=c*a-s*b;design[i*p+l]=s*a+c*b;}}
            for(int j=0;j<p;++j){{double a=v[j][h],b=v[j][l];v[j][h]=c*a-s*b;v[j][l]=s*a+c*b;}}
            }}if(!changed)break;}}
            double norm[p]={{}},largest=0;for(int j=0;j<p;++j){{
            for(int i=0;i<count;++i)norm[j]+=design[i*p+j]*design[i*p+j];largest=std::max(largest,norm[j]);}}
            double cutoff=2.220446049250313e-16*std::max(count,p);cutoff=cutoff*cutoff*largest;
            for(int j=0;j<p;++j)if(norm[j]>cutoff){{double dot=0;
            for(int i=0;i<count;++i)dot+=design[i*p+j]*target[i];
            for(int q=0;q<p;++q)coef[q]+=v[q][j]*dot/norm[j];}}
            for(size_t i=0;i<num_stocks;++i){{double x[p]={{1}};bool valid=std::isfinite(input_0[i]);{xs}
            double fit=0;for(int j=0;j<p;++j)fit+=coef[j]*x[j];
            output_0[i]=valid&&count>p+1?input_0[i]-fit:NAN;}}
            """

    Cross.__name__ = "JQCross" + kind + str(regressors)
    return Cross
