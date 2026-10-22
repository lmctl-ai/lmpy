# Source: the SciPy reference examples for scipy.integrate.solve_ivp: exponential decay y' = -0.5 y from [2, 4, 8] (automatic
# steps, then t_eval=[0, 1, 2, 4, 10]), the cannon fired upward with a terminal hit_ground event and a non-terminal apex event
# read through dense output, and the Lotka-Volterra system with args=(1.5, 1, 3, 1) and dense_output evaluated on 300 points.
#   https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
# The page's complex-matrix examples (y' = A y with complex A) are omitted: lmpy's ODE state is real.
# Prints the answers as one JSON object; examples/python/scipy-solve-ivp.lms pins them.
import json
import numpy as np
from scipy.integrate import solve_ivp


def exponential_decay(t, y): return -0.5 * y


sol = solve_ivp(exponential_decay, [0, 10], [2, 4, 8])
decay_steps = len(sol.t) - 1
decay_final = sol.y[:, -1]
sol2 = solve_ivp(exponential_decay, [0, 10], [2, 4, 8], t_eval=[0, 1, 2, 4, 10])


def upward_cannon(t, y): return [y[1], -0.5]
def hit_ground(t, y): return y[0]


hit_ground.terminal = True
hit_ground.direction = -1
cannon = solve_ivp(upward_cannon, [0, 100], [0, 10], events=hit_ground)


def apex(t, y): return y[1]


cannon2 = solve_ivp(upward_cannon, [0, 100], [0, 10], events=(hit_ground, apex), dense_output=True)


def lotkavolterra(t, z, a, b, c, d):
    x, y = z
    return [a*x - b*x*y, -c*y + d*x*y]


lv = solve_ivp(lotkavolterra, [0, 15], [10, 5], args=(1.5, 1, 3, 1), dense_output=True)
t = np.linspace(0, 15, 300)
z = lv.sol(t)

print(json.dumps({
    "decay_steps": decay_steps,
    "decay_final": decay_final.tolist(),
    "decay_nfev": sol.nfev,
    "decay_t_eval": sol2.y.T.tolist(),
    "cannon_t_events": cannon.t_events[0].tolist(),
    "cannon_steps": len(cannon.t) - 1,
    "cannon_hit_t": cannon2.t_events[0].tolist(),
    "cannon_apex_t": cannon2.t_events[1].tolist(),
    "cannon_apex_state": cannon2.sol(cannon2.t_events[1][0]).tolist(),
    "cannon_apex_y_events": cannon2.y_events[1][0].tolist(),
    "lv_steps": len(lv.t) - 1,
    "lv_nfev": lv.nfev,
    "lv_final": lv.y[:, -1].tolist(),
    "lv_dense": z.T.tolist(),
}))
