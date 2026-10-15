"""Independent S3 references: the assembled worker (for builder-only tests), NIST Misra1a
(certified to 11 digits), textbook optima and plain-Python/numpy/scipy versions of every
expression function. Nothing here calls the worker's solver code."""
import importlib.util, math, pathlib, sys, tempfile
import numpy as np
from scipy import special

ROOT = pathlib.Path(__file__).resolve().parents[2]


def load_worker():
    """Same assembly order as CMakeLists.txt: frames.py, numerics.py, s4.py, sm.py, ml.py, s7.py, worker.py (one module namespace)."""
    text = ''.join((ROOT / 'py' / name).read_text() + '\n' for name in ('frames.py', 'numerics.py', 's4.py', 'sm.py', 'ml.py', 's7.py', 'worker.py'))
    directory = tempfile.mkdtemp()
    path = pathlib.Path(directory) / 'assembled_worker.py'
    path.write_text(text)
    spec = importlib.util.spec_from_file_location('assembled_worker', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# NIST StRD Misra1a: y = b1 * (1 - exp(-b2 * x)), 14 observations, higher difficulty.
MISRA1A_Y = [10.07, 14.73, 17.94, 23.93, 29.61, 35.18, 40.02, 44.82, 50.76, 55.05, 61.01, 66.40, 75.47, 81.78]
MISRA1A_X = [77.6, 114.9, 141.1, 190.8, 239.9, 289.0, 332.8, 378.4, 434.8, 477.3, 536.8, 593.1, 689.1, 760.0]
MISRA1A_CERTIFIED = {'b1': 2.3894212918e+02, 'b2': 5.5015643181e-04}
MISRA1A_SE = {'b1': 2.7070075241e+00, 'b2': 7.2668688436e-06}
MISRA1A_RSS = 1.2455138894e-01
MISRA1A_STARTS = ({'b1': 500.0, 'b2': 1e-4}, {'b1': 250.0, 'b2': 5e-4})


def lre(computed, certified):
    error = abs(computed - certified)
    return 15.0 if error == 0 else min(15.0, -math.log10(error / abs(certified)))


# Plain Python lambdas for the textbook problems (independent of the expression engine).
def rosenbrock(x, y):
    return 100.0 * (y - x * x) ** 2 + (1.0 - x) ** 2


def himmelblau(x, y):
    return (x * x + y - 11.0) ** 2 + (x + y * y - 7.0) ** 2


def himmelblau_grad(x, y):
    return (4 * x * (x * x + y - 11) + 2 * (x + y * y - 7), 2 * (x * x + y - 11) + 4 * y * (x + y * y - 7))


HIMMELBLAU_MINIMA = [(3.0, 2.0), (-2.805118086952745, 3.131312518250573), (-3.779310253377747, -3.283185991286170), (3.584428340330492, -1.848126526964404)]

# Reference implementations of every function in expr-spec.json, written independently of numpy's namesakes
# wherever Python's math/statistics modules can do it.
def _qnorm(p):
    import statistics
    return statistics.NormalDist().inv_cdf(p)

REFERENCE_1 = {
    'exp': math.exp, 'log': math.log, 'log1p': math.log1p, 'expm1': math.expm1, 'log2': math.log2, 'log10': math.log10,
    'sqrt': math.sqrt, 'abs': abs, 'sin': math.sin, 'cos': math.cos, 'tan': math.tan, 'asin': math.asin, 'acos': math.acos,
    'atan': math.atan, 'sinh': math.sinh, 'cosh': math.cosh, 'tanh': math.tanh, 'floor': math.floor, 'ceiling': math.ceil,
    'trunc': math.trunc, 'round': round,  # Python round is half-to-even like IEC 60559 / numpy
    'sign': lambda v: (v > 0) - (v < 0), 'gamma': math.gamma, 'lgamma': math.lgamma,
    'digamma': lambda v: float(special.psi(v)), 'pnorm': lambda v: 0.5 * math.erfc(-v / math.sqrt(2.0)),
    'dnorm': lambda v: math.exp(-0.5 * v * v) / math.sqrt(2.0 * math.pi), 'qnorm': _qnorm,
    'plogis': lambda v: 1.0 / (1.0 + math.exp(-v)),
}
REFERENCE_2 = {
    'atan2': math.atan2, 'pmin': min, 'pmax': max,
    'beta': lambda a, b: math.gamma(a) * math.gamma(b) / math.gamma(a + b),
    'lbeta': lambda a, b: math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b),
    'choose': lambda n, k: float(math.comb(int(n), int(k))),
    'lchoose': lambda n, k: math.log(math.comb(int(n), int(k))),
}
# Argument points that are valid for every function in the group.
POINTS_1 = {'log': [0.5, 2.0, 10.0], 'log1p': [0.5, 2.0], 'log2': [0.5, 8.0], 'log10': [0.5, 1000.0], 'sqrt': [0.25, 2.0, 9.0],
            'asin': [-0.5, 0.3], 'acos': [-0.5, 0.3], 'gamma': [0.5, 4.5, 6.0], 'lgamma': [0.5, 4.5, 20.0], 'digamma': [0.5, 3.0],
            'qnorm': [0.025, 0.5, 0.975], 'round': [0.5, 1.5, 2.5, -0.5, 2.4, 2.6], 'floor': [-1.5, 1.5, 2.0], 'ceiling': [-1.5, 1.5, 2.0],
            'trunc': [-1.5, 1.5], 'sign': [-3.0, 0.0, 2.0], 'plogis': [-3.0, 0.0, 2.5]}
DEFAULT_POINTS = [-1.25, 0.0, 0.3, 1.0, 2.5]

from semantics import *  # noqa: E402,F401  (grid, tokens, cell comparison)


def evaluate_semantics(worker, spec, check, expression):
    """Evaluate one expression over the grid with the real parser (check) and the real builder; returns result tokens."""
    names = ['x', 'y'] if uses_y(expression) else ['x']
    parsed = check({'expression': expression, 'names': names})
    assert parsed['accepted'], (expression, parsed)
    program = worker.compile_expression(parsed['ast'], set(names), {}, spec)
    out = []
    with np.errstate(all='ignore'):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for x in SEMANTIC_X:
                for y in (SEMANTIC_Y if 'y' in names else [None]):
                    env = {'x': np.float64(x)}
                    if y is not None:
                        env['y'] = np.float64(y)
                    try:
                        out.append(token(worker.evaluate(program, env)))
                    except Exception:
                        out.append('ERR')
    return out
