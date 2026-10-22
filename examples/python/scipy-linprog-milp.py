# Source: the SciPy reference examples for scipy.optimize.linprog (min -x0 + 4 x1 with a free x0 and x1 >= -3, its
# marginals and slacks, and the right-hand-side perturbation by eps = 0.05) and scipy.optimize.milp (the Wikipedia integer
# programming example and its LP relaxation), and the SciPy optimization tutorial sections "Linear programming (linprog)"
# (four variables with inequality and equality constraints, infeasible with x1 <= 5 and optimal with x1 <= 6) and
# "Mixed integer linear programming" (the 0/1 knapsack).
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html
# https://docs.scipy.org/doc/scipy/tutorial/optimize.html#linear-programming-linprog
# Prints the answers as one JSON object; examples/python/scipy-linprog-milp.lms pins them.
import json
import numpy as np
from scipy import optimize
from scipy.optimize import linprog, milp, LinearConstraint

# linprog reference example
c = [-1, 4]
A = [[-3, 1], [1, 2]]
b = [6, 4]
x0_bounds = (None, None)
x1_bounds = (-3, None)
res = linprog(c, A_ub=A, b_ub=b, bounds=[x0_bounds, x1_bounds])
eps = 0.05
b[1] += eps
fun_eps = linprog(c, A_ub=A, b_ub=b, bounds=[x0_bounds, x1_bounds]).fun
b = [6, 4]  # reset to original values
b[0] -= 39
fun_minus39 = linprog(c, A_ub=A, b_ub=b, bounds=[x0_bounds, x1_bounds]).fun

# linprog tutorial
c = np.array([-29.0, -45.0, 0.0, 0.0])
A_ub = np.array([[1.0, -1.0, -3.0, 0.0],
                 [-2.0, 3.0, 7.0, -3.0]])
b_ub = np.array([5.0, -10.0])
A_eq = np.array([[2.0, 8.0, 1.0, 0.0],
                 [4.0, 4.0, 0.0, 1.0]])
b_eq = np.array([60.0, 60.0])
x0_bounds = (0, None)
x1_bounds = (0, 5.0)
x2_bounds = (-np.inf, 0.5)  # +/- np.inf can be used instead of None
x3_bounds = (-3.0, None)
bounds = [x0_bounds, x1_bounds, x2_bounds, x3_bounds]
infeasible = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
x1_bounds = (0, 6)
bounds = [x0_bounds, x1_bounds, x2_bounds, x3_bounds]
tut = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)

# milp reference example
c = -np.array([0, 1])
A = np.array([[-1, 1], [3, 2], [2, 3]])
b_u = np.array([1, 12, 12])
b_l = np.full_like(b_u, -np.inf, dtype=float)
constraints = LinearConstraint(A, b_l, b_u)
integrality = np.ones_like(c)
mi = milp(c=c, constraints=constraints, integrality=integrality)
relaxed = milp(c=c, constraints=constraints)

# milp tutorial: knapsack
sizes = np.array([21, 11, 15, 9, 34, 25, 41, 52])
values = np.array([22, 12, 16, 10, 35, 26, 42, 53])
kb = optimize.Bounds(0, 1)  # 0 <= x_i <= 1
integrality = np.full_like(values, True)  # x_i are integers
capacity = 100
constraints = optimize.LinearConstraint(A=sizes, lb=0, ub=capacity)
knap = milp(c=-values, constraints=constraints, integrality=integrality, bounds=kb)

print(json.dumps({
    "lp_x": res.x.tolist(), "lp_fun": res.fun,
    "lp_ineqlin_residual": res.ineqlin.residual.tolist(), "lp_ineqlin_marginals": res.ineqlin.marginals.tolist(),
    "lp_fun_eps": fun_eps, "lp_fun_minus39": fun_minus39,
    "tutorial_infeasible": bool(infeasible.status == 2),
    "tutorial_x": tut.x.tolist(), "tutorial_fun": tut.fun,
    "milp_x": mi.x.tolist(), "milp_fun": mi.fun, "relaxed_x": relaxed.x.tolist(), "relaxed_fun": relaxed.fun,
    "knapsack_x": knap.x.tolist(), "knapsack_value": -knap.fun,
}))
