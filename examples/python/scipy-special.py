# Source: the SciPy reference examples for scipy.special: gamma([0, 0.5, 1, 5]) and gamma(0.5)**2, erf on linspace(-3, 3),
# jv (order 0, 1, 1.5 at 1.0; order 0 and 1 at [-2, 0, 3]), jn_zeros(3, 4), ellipk, expit([-inf, -1.5, 0, 1.5, inf]) and
# logit(expit(...)), logit([0, 0.25, 0.5, 0.75, 1]), and logsumexp (arange(10), with weights b = arange(10, 0, -1), and the
# masked-array example through zero weights).
#   https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.gamma.html  (likewise erf, jv, jn_zeros, ellipk,
#   expit, logit, logsumexp)
# Prints the answers as one JSON object; examples/python/scipy-special.lms pins them.
import json
import numpy as np
from scipy.special import ellipk, erf, expit, gamma, jn_zeros, jv, logit, logsumexp

a = np.arange(10)
b = np.arange(10, 0, -1)
am = np.array([np.log(2), 2, np.log(3)])
bm = np.array([1, 0, 1])

print(json.dumps({
    "gamma": gamma([0.5, 1, 5]).tolist(),
    "gamma_0_is_inf": bool(np.isinf(gamma(0))),
    "gamma_half_squared": gamma(0.5)**2,
    "erf": erf(np.linspace(-3, 3)).tolist(),
    "jv_at_1": [jv(0, 1.), jv(1, 1.), jv(1.5, 1.)],
    "jv0_points": jv(0, np.array([-2., 0., 3.])).tolist(),
    "jv1_points": jv(1, np.array([-2., 0., 3.])).tolist(),
    "jn_zeros_3_4": jn_zeros(3, 4).tolist(),
    "ellipk": ellipk(np.linspace(0, 0.99, 12)).tolist(),
    "expit": expit([-np.inf, -1.5, 0, 1.5, np.inf]).tolist(),
    "logit_of_expit": logit(expit([-2.5, 0, 3.1, 5.0])).tolist(),
    "logit_inner": logit([0.25, 0.5, 0.75]).tolist(),
    "logit_0_is_minus_inf": bool(logit(0) == -np.inf),  # JSON has no inf, so the ends are recorded as flags
    "logit_1_is_plus_inf": bool(logit(1) == np.inf),
    "logsumexp": logsumexp(a),
    "logsumexp_weighted": logsumexp(a, b=b),
    "logsumexp_masked": logsumexp(am, b=bm),
}))
