# Source: the SciPy optimization tutorial, "Unconstrained minimization of multivariate scalar functions" (Rosenbrock's
# function of five variables by Nelder-Mead with xatol 1e-8, then BFGS with the analytic gradient rosen_der) and
# "Sequential Least SQuares Programming (SLSQP)" (Rosenbrock in two variables with one equality, three inequality
# constraints and bounds). https://docs.scipy.org/doc/scipy/tutorial/optimize.html
# Prints the answers as one JSON object; examples/python/scipy-minimize-rosenbrock.lms pins them.
import json
import numpy as np
from scipy.optimize import minimize, Bounds


def rosen(x):
    """The Rosenbrock function"""
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)


def rosen_der(x):
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    der = np.zeros_like(x)
    der[1:-1] = 200*(xm-xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1-xm)
    der[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
    der[-1] = 200*(x[-1]-x[-2]**2)
    return der


x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
nm = minimize(rosen, x0, method='nelder-mead', options={'xatol': 1e-8})
bfgs = minimize(rosen, x0, method='BFGS', jac=rosen_der)

bounds = Bounds([0, -0.5], [1.0, 2.0])
ineq_cons = {'type': 'ineq',
             'fun': lambda x: np.array([1 - x[0] - 2*x[1],
                                        1 - x[0]**2 - x[1],
                                        1 - x[0]**2 + x[1]]),
             'jac': lambda x: np.array([[-1.0, -2.0],
                                        [-2*x[0], -1.0],
                                        [-2*x[0], 1.0]])}
eq_cons = {'type': 'eq',
           'fun': lambda x: np.array([2*x[0] + x[1] - 1]),
           'jac': lambda x: np.array([2.0, 1.0])}
slsqp = minimize(rosen, np.array([0.5, 0]), method='SLSQP', jac=rosen_der,
                 constraints=[eq_cons, ineq_cons], options={'ftol': 1e-9}, bounds=bounds)

print(json.dumps({
    "nelder_mead_x": nm.x.tolist(), "nelder_mead_fun": nm.fun,
    "bfgs_x": bfgs.x.tolist(), "bfgs_fun": bfgs.fun,
    "slsqp_x": slsqp.x.tolist(), "slsqp_fun": slsqp.fun,
}))
