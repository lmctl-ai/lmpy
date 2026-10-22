# Source: the SciPy reference examples for scipy.optimize.root_scalar (the cubic x^3 - 1 bracketed on [0, 3] with brentq)
# and scipy.optimize.fsolve (x0 cos(x1) = 4, x1 x0 - x1 = 5 from [1, 1]), and the SciPy optimization tutorial section
# "Root finding" (x + 2 cos(x) = 0 by root from 0.3, and the same system by root with method='lm' and its Jacobian).
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.root_scalar.html
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fsolve.html
# https://docs.scipy.org/doc/scipy/tutorial/optimize.html#root-finding
# Prints the answers as one JSON object; examples/python/scipy-root-finding.lms pins them.
import json
import numpy as np
from scipy import optimize
from scipy.optimize import fsolve, root


def f(x):
    return (x**3 - 1)  # only one real root at x = 1


sol = optimize.root_scalar(f, bracket=[0, 3], method='brentq')


def func(x):
    return [x[0] * np.cos(x[1]) - 4,
            x[1] * x[0] - x[1] - 5]


fs = fsolve(func, [1, 1])


def func1(x):
    return x + 2 * np.cos(x)


sol1 = root(func1, 0.3)


def func2(x):
    f = [x[0] * np.cos(x[1]) - 4,
         x[1]*x[0] - x[1] - 5]
    df = np.array([[np.cos(x[1]), -x[0] * np.sin(x[1])],
                   [x[1], x[0] - 1]])
    return f, df


sol2 = root(func2, [1, 1], jac=True, method='lm')

print(json.dumps({
    "cubic_root": sol.root, "cubic_iterations": sol.iterations, "cubic_function_calls": sol.function_calls,
    "fsolve_x": fs.tolist(),
    "root_x": sol1.x.tolist(),
    "lm_x": sol2.x.tolist(),
}))
