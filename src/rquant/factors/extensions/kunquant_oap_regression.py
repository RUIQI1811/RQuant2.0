"""OAP regression fallback for rank-deficient windows; ordinary windows retain the shared kernel."""

from __future__ import annotations

from rquant.factors.extensions import kunquant_oap as op

CPP_PRELUDE = r"""
namespace rq_oap_regression {
// NumPy's reference solves the Gram system, falling back to its Moore-Penrose inverse when singular.
static inline void solve_or_pinv(int k, const std::vector<double>& gram, const std::vector<double>& rhs,
                                 double *out) {
    if (rq_cnc::solve(k, gram, rhs, out)) return;
    const auto beta = rq_oap::lstsq(gram, rhs, k, k, 1e-15);
    std::copy(beta.begin(), beta.end(), out);
}

static inline std::vector<double> inverse_or_pinv(int k, const std::vector<double>& gram) {
    std::vector<double> result(k*k), rhs(k, 0.0), column(k);
    bool singular = false;
    for (int c=0; c<k; ++c) {
        std::fill(rhs.begin(), rhs.end(), 0.0); rhs[c]=1;
        if (!rq_cnc::solve(k, gram, rhs, column.data())) {singular=true; break;}
        for (int r=0; r<k; ++r) result[r*k+c]=column[r];
    }
    if (singular) for (int c=0; c<k; ++c) {
        std::fill(rhs.begin(), rhs.end(), 0.0); rhs[c]=1;
        column=rq_oap::lstsq(gram, rhs, k, k, 1e-15);
        for (int r=0; r<k; ++r) result[r*k+c]=column[r];
    }
    return result;
}

// The expression builder supplies exactly one trailing window (including the residue-class month sampling).
static inline void rolling_regression(int n, const double *y, const std::vector<const double *>& x, int window,
                                      int minimum, rq_cnc::Fit fit, int first, int last, double *out) {
    if (n<window) return;
    const int k=(int)x.size()+1, lo=n-window;
    std::vector<double> gram(k*k,0.0), rhs(k,0.0), beta(k), design(window*k,0.0), response(window,0.0);
    std::vector<bool> known(window,false);
    int count=0; double yy=0;
    for (int t=0; t<window; ++t) {
        const int row=lo+t;
        bool usable=std::isfinite(y[row]);
        for (int j=1; j<k; ++j) usable=usable&&std::isfinite(x[j-1][row]);
        if (!usable) continue;
        known[t]=true; ++count; response[t]=y[row]; yy+=y[row]*y[row];
        design[t*k]=1;
        for (int j=1; j<k; ++j) design[t*k+j]=x[j-1][row];
        for (int a=0; a<k; ++a) {
            rhs[a]+=design[t*k+a]*y[row];
            for (int b=0; b<k; ++b) gram[a*k+b]+=design[t*k+a]*design[t*k+b];
        }
    }
    if (count<minimum) return;
    if (rq_cnc::solve(k,gram,rhs,beta.data())) {
        rq_cnc::rolling_regression(n,y,x,window,minimum,fit,first,last,out);
        return;
    }
    beta=rq_oap::lstsq(gram,rhs,k,k,1e-15);
    if (fit==rq_cnc::kBeta) {out[n-1]=beta[1]; return;}
    if (fit==rq_cnc::kSlopes) {
        double total=0; for (int j=1; j<k; ++j) total+=beta[j]; out[n-1]=total; return;
    }
    double squared=0,cubed=0,part_sum=0,part_squared=0,market_sum=0;
    int part_count=0;
    std::vector<double> residual(window,0.0);
    for (int t=0; t<window; ++t) if (known[t]) {
        double e=response[t]; for (int j=0; j<k; ++j) e-=design[t*k+j]*beta[j];
        residual[t]=e; squared+=e*e; cubed+=e*e*e; market_sum+=design[t*k+1];
        if (t>=first && t<last) {++part_count; part_sum+=e; part_squared+=e*e;}
    }
    if (fit==rq_cnc::kResidualMomentum) {
        double mean=part_sum/part_count;
        double spread=std::sqrt((part_squared-part_count*mean*mean)/(part_count-1));
        if (part_count>=(last-first)/2 && spread>0) out[n-1]=mean/spread;
        return;
    }
    if (!(squared>1e-14*std::max(yy,1e-300))) return;
    if (fit==rq_cnc::kResidualStd) {
        if (count>k) out[n-1]=std::sqrt(squared/(count-k));
    } else if (fit==rq_cnc::kResidualSkew) {
        out[n-1]=(cubed/count)/std::pow(squared/count,1.5);
    } else if (fit==rq_cnc::kCoskew) {
        double market_mean=market_sum/count,market_squared=0,co=0;
        for (int t=0; t<window; ++t) if (known[t]) {
            double d=design[t*k+1]-market_mean; market_squared+=d*d; co+=residual[t]*d*d;
        }
        out[n-1]=(co/count)/(std::sqrt(squared/count)*market_squared/count);
    }
}
}
"""


def regression(target, regressors, window, minimum, kind, stride=1, first=0, last=0):
    """The OAP window builder with a singular-Gram fallback, preserving exclusive residual-slice endpoints."""
    expression = op.regression(target, regressors, window, minimum, kind, stride, first, last)
    code, members, init, *args = expression.args
    return op.Expr(
        "state",
        code.replace("rq_cnc::rolling_regression", "rq_oap_regression::rolling_regression"),
        members,
        init,
        *args,
    )
