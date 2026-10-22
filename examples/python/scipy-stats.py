# Source: the SciPy reference examples for scipy.stats distributions (norm; t with df=2.74; gamma with a=1.99; binom with
# n=5, p=0.4: stats(moments='mvsk'), ppf([0.001, 0.5, 0.999]), cdf, pdf/pmf), ttest_ind (two samples of 500 from N(5, 10^2),
# pooled and Welch), ttest_1samp (50 uniforms against popmean=0.5), pearsonr (x = [1..7], y = [10, 9, 2.5, 6, 4, 3, 2]) and
# linregress (x = rng.random(10), y = 1.6 x + rng.random(10); slope, intercept, rvalue, pvalue, stderr, intercept_stderr and
# the 95 % confidence half-widths from the t quantile).
#   https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html  (likewise t, gamma, binom)
#   https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html
#   https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_1samp.html
#   https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html
#   https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html
# The pages draw from unseeded np.random.default_rng(); here the seeds are 11 (ttest_ind), 12 (ttest_1samp) and 13
# (linregress), which lmpy's rng(seed) reproduces. Prints one JSON object; examples/python/scipy-stats.lms pins it.
import json
import numpy as np
from scipy import stats
from scipy.stats import binom, gamma, norm, t

out = {}
probs = [0.001, 0.5, 0.999]
for name, rv in [("norm", norm()), ("t", t(2.74)), ("gamma", gamma(1.99)), ("binom", binom(5, 0.4))]:
    mean, var, skew, kurt = rv.stats(moments='mvsk')
    out[name + "_mvs"] = [float(mean), float(var), float(skew)]
    if np.isinf(kurt):  # t(2.74) has infinite kurtosis; JSON has no inf, so record it as a flag
        out[name + "_kurtosis_is_inf"] = bool(np.isinf(kurt))
    else:
        out[name + "_kurtosis"] = float(kurt)
    vals = rv.ppf(probs)
    out[name + "_ppf"] = vals.tolist()
    out[name + "_cdf_of_ppf"] = rv.cdf(vals).tolist()
    if name == "binom":
        out[name + "_pmf"] = rv.pmf(np.arange(6)).tolist()
    else:
        out[name + "_pdf"] = rv.pdf(np.linspace(rv.ppf(0.01), rv.ppf(0.99), 7)).tolist()

rng = np.random.default_rng(11)
rvs1 = stats.norm.rvs(loc=5, scale=10, size=500, random_state=rng)
rvs2 = stats.norm.rvs(loc=5, scale=10, size=500, random_state=rng)
r = stats.ttest_ind(rvs1, rvs2)
out["ttest_ind"] = [float(r.statistic), float(r.pvalue)]
r = stats.ttest_ind(rvs1, rvs2, equal_var=False)
out["ttest_welch"] = [float(r.statistic), float(r.pvalue), float(r.df)]

rng = np.random.default_rng(12)
rvs = stats.uniform.rvs(size=50, random_state=rng)
r = stats.ttest_1samp(rvs, popmean=0.5)
out["ttest_1samp"] = [float(r.statistic), float(r.pvalue)]

x, y = [1, 2, 3, 4, 5, 6, 7], [10, 9, 2.5, 6, 4, 3, 2]
r = stats.pearsonr(x, y)
out["pearsonr"] = [float(r.statistic), float(r.pvalue)]

rng = np.random.default_rng(13)
x = rng.random(10)
y = 1.6 * x + rng.random(10)
res = stats.linregress(x, y)
tinv = lambda p, df: abs(t.ppf(p / 2, df))
ts = tinv(0.05, len(x) - 2)
out["linregress"] = [res.slope, res.intercept, res.rvalue, res.pvalue, res.stderr, res.intercept_stderr]
out["linregress_r_squared"] = res.rvalue**2
out["linregress_ci_halfwidths"] = [ts * res.stderr, ts * res.intercept_stderr]

print(json.dumps(out))
