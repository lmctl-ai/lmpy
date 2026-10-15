"""S3: expression programs, minimisers, least squares, quadrature and roots. Embedded with the worker.

Caller text never reaches Python: the C++ parser emits a JSON AST, which compile_expression() re-validates
node by node and turns into a flat postfix program of function OBJECTS taken from the closed tables below.
There is no name-to-callable lookup other than those tables and no recursion over caller structure.
"""
EXPR_INSTRUCTIONS = ('push', 'var', 'idx', 'f1', 'f2', 'f3')


class SolverStop(Exception):
    status = 'STOPPED'

    def __init__(self, message=''):
        super().__init__(message)
        self.message = message


class EvalLimit(SolverStop):
    status = 'EVAL_LIMIT'


class TimeLimit(SolverStop):
    status = 'TIME_LIMIT'


class NonFinite(SolverStop):
    status = 'NON_FINITE'


def load_spec(p):
    import hashlib
    import json
    with open(p['spec_file'], encoding='utf-8') as stream:
        text = stream.read()
    return json.loads(text), hashlib.sha256(text.encode('utf-8')).hexdigest()


_WARNED = set()


def warn_once(key, message):
    """One NumericalWarning per kind per process: expression calls can run 10^6 times."""
    if key not in _WARNED:
        _WARNED.add(key)
        warnings.warn(message, NumericalWarning)


def function_table(np):
    """Spec function name -> numpy/scipy function object (objects, never names resolved from strings).

    Functions that rcli evaluates with R's own semantics (NaN/NA propagation in conditions, R's choose/lchoose/beta
    domains, R_pow special cases, gamma/digamma poles) are wrapped to match; docs/s3.md lists the grid that pins this.
    """
    from scipy import special
    nan, inf = np.nan, np.inf

    def dnorm(x):
        return np.exp(-0.5 * np.square(x)) / np.sqrt(2.0 * np.pi)

    def truth(x):
        x = np.asarray(x, dtype=float)
        return np.where(np.isnan(x), nan, (x != 0).astype(float))  # R: numeric -> logical, NaN -> NA

    def ifelse(condition, yes, no):
        test = truth(condition)
        return np.where(np.isnan(test), nan, np.where(test != 0, yes, no))  # an NA test yields NA, as in R

    def gamma(x):
        x = np.asarray(x, dtype=float)
        return np.where(x == 0, nan, special.gamma(x))  # R: gamma(0) is NaN

    def lgamma(x):
        x = np.asarray(x, dtype=float)
        return np.where(np.isneginf(x), inf, special.gammaln(x))  # R: lgamma(-Inf) is Inf

    def digamma(x):
        x = np.asarray(x, dtype=float)
        return np.where(x == 0, nan, special.digamma(x))

    def beta(a, b):
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        value = np.where(np.isinf(a) | np.isinf(b), 0.0, special.beta(a, b))
        value = np.where((a == 0) | (b == 0), inf, value)
        value = np.where((a < 0) | (b < 0), nan, value)  # R restricts beta to a, b >= 0
        return np.where(np.isnan(a) | np.isnan(b), nan, value)

    def lbeta(a, b):
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        low, high = np.minimum(a, b), np.maximum(a, b)
        value = np.where(np.isinf(high), -inf, special.betaln(a, b))
        value = np.where(low == 0, inf, value)
        value = np.where(low < 0, nan, value)
        return np.where(np.isnan(a) | np.isnan(b), nan, value)

    def integer_like(n):
        return np.isinf(n) | (np.isfinite(n) & (np.abs(n - np.rint(n)) <= 1e-7 * np.maximum(1.0, np.abs(n))))  # R_IS_INT counts +-Inf

    def round_k(k):
        rounded = np.rint(k)  # R_forceint: nearbyint, half to even
        if np.any(np.isfinite(k) & (np.abs(k - rounded) > 1e-7)):
            warn_once('choose_k', "'k' of choose/lchoose is not an integer and was rounded, as R does")
        return rounded

    def choose(n, k):
        n, k_in = np.asarray(n, dtype=float), np.asarray(k, dtype=float)
        k = round_k(k_in)
        whole = integer_like(n)
        n = np.where(whole, np.rint(n), n)
        negative = whole & (n < 0)
        effective = np.where(negative, -n + k - 1, n)
        value = special.binom(effective, k)
        value = np.where(np.isinf(effective), np.where(k == 0, 1.0, inf), value)
        value = np.where(negative & (np.mod(k, 2) == 1), -value, value)
        value = np.where(whole & (n >= 0) & (k > n), 0.0, value)
        value = np.where(k < 0, 0.0, value)
        value = np.where(whole & (np.abs(value) < 2.0 ** 53), np.rint(value), value)
        # R quirks reproduced on purpose: its large-k branch evaluates exp(lfastchoose), which is NaN for infinite n, and
        # k = +Inf is 0 only for a finite non-negative n (a negative n recurses into choose(Inf, Inf)).
        value = np.where(np.isposinf(k), np.where(np.isfinite(n) & (n >= 0), 0.0, nan), value)
        value = np.where(np.isneginf(k), 0.0, value)
        value = np.where(np.isinf(n) & (k >= 30) & np.isfinite(k), nan, value)
        return np.where(np.isnan(n) | np.isnan(k_in), nan, value)

    def lchoose(n, k):
        n, k_in = np.asarray(n, dtype=float), np.asarray(k, dtype=float)
        k = round_k(k_in)
        original = n
        whole = integer_like(n)
        n = np.where(whole, np.rint(n), n)
        effective = np.where(whole & (n < 0), -n + k - 1, n)
        with np.errstate(all='ignore'):
            # -log(n+1) - lbeta(n-k+1, k+1) as in R's lfastchoose where the arguments are positive (accurate for huge n, k);
            # log-gamma differences elsewhere. Neither ever forms the coefficient, so large results do not overflow.
            positive = (effective + 1 > 0) & (effective - k + 1 > 0) & (k + 1 > 0)
            by_beta = -np.log(effective + 1) - special.betaln(np.where(positive, effective - k + 1, 1.0), np.where(positive, k + 1, 1.0))
            by_gamma = special.gammaln(effective + 1) - special.gammaln(k + 1) - special.gammaln(effective - k + 1)
            value = np.where(positive, by_beta, by_gamma)
        value = np.where(np.isinf(effective), np.where(k == 0, 0.0, inf), value)
        value = np.where(whole & (n >= 0) & (k > n), -inf, value)
        value = np.where(k == 1, np.log(np.abs(original)), value)
        value = np.where(k == 0, 0.0, value)
        value = np.where(np.isinf(n) & (k >= 2) & np.isfinite(k), nan, value)  # R: lfastchoose(Inf, k) is NaN
        value = np.where(np.isposinf(k), np.where(np.isfinite(n) & (n >= 0), -inf, nan), value)
        value = np.where(k < 0, -inf, value)
        return np.where(np.isnan(n) | np.isnan(k_in), nan, value)

    return {
        'exp': np.exp, 'log': np.log, 'log1p': np.log1p, 'expm1': np.expm1, 'log2': np.log2, 'log10': np.log10,
        'sqrt': np.sqrt, 'abs': np.abs, 'sin': np.sin, 'cos': np.cos, 'tan': np.tan, 'asin': np.arcsin,
        'acos': np.arccos, 'atan': np.arctan, 'sinh': np.sinh, 'cosh': np.cosh, 'tanh': np.tanh,
        'floor': np.floor, 'ceiling': np.ceil, 'trunc': np.trunc, 'round': np.round, 'sign': np.sign,
        'gamma': gamma, 'lgamma': lgamma, 'digamma': digamma,
        'pnorm': special.ndtr, 'dnorm': dnorm, 'qnorm': special.ndtri, 'plogis': special.expit,
        'atan2': np.arctan2, 'pmin': np.minimum, 'pmax': np.maximum, 'beta': beta,
        'lbeta': lbeta, 'choose': choose, 'lchoose': lchoose, 'ifelse': ifelse,
        'sum': np.sum, 'mean': np.mean, 'prod': np.prod, 'min': np.min, 'max': np.max,
    }


def operator_table(np):
    """Operators with R's three-valued logic: comparisons and & | return 1.0 / 0.0 / NaN (NA); numbers are truthy if nonzero."""
    nan = np.nan

    def power(x, y):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        value = np.power(x, y)
        # R_pow: NaN for a negative base with an infinite exponent and for -Inf with a non-integer/infinite exponent
        undefined = (np.isinf(y) & (x < 0)) | (np.isneginf(x) & ~(np.isfinite(y) & (y == np.floor(y))))
        return np.where(undefined & ~((x == 1) | (y == 0)), nan, value)

    def comparison(function):
        def compare(x, y):
            x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
            return np.where(np.isnan(x) | np.isnan(y), nan, function(x, y).astype(float))
        return compare

    def truth(x):
        x = np.asarray(x, dtype=float)
        return np.where(np.isnan(x), nan, (x != 0).astype(float))

    def conjunction(x, y):
        a, b = truth(x), truth(y)
        return np.where((a == 0) | (b == 0), 0.0, np.where(np.isnan(a) | np.isnan(b), nan, 1.0))  # NA & FALSE is FALSE

    def disjunction(x, y):
        a, b = truth(x), truth(y)
        return np.where((a == 1) | (b == 1), 1.0, np.where(np.isnan(a) | np.isnan(b), nan, 0.0))  # NA | TRUE is TRUE

    return {'+': np.add, '-': np.subtract, '*': np.multiply, '/': np.divide, '^': power,
            '<': comparison(np.less), '<=': comparison(np.less_equal), '>': comparison(np.greater), '>=': comparison(np.greater_equal),
            '==': comparison(np.equal), '!=': comparison(np.not_equal), '&': conjunction, '|': disjunction}


def check_bindings(spec, table):
    """A spec function without a Python binding is a build failure, not a runtime surprise."""
    missing = sorted(set(spec['functions']) - set(table))
    if missing:
        raise ExprMismatch('expr-spec.json names functions without a pycli binding: ' + ', '.join(missing))


def _int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def compile_expression(ast, names, dims, spec, tables=None):
    """AST -> postfix program [(kind, payload, payload2)]. Iterative: no recursion over caller structure."""
    import math
    import numpy as np
    functions, operators = tables or (function_table(np), operator_table(np))
    caps, arity, constants = spec['caps'], {k: v['arity'] for k, v in spec['functions'].items()}, spec['constants']
    program, count = [], 0
    stack = [(ast, False)]
    while stack:
        node, done = stack.pop()
        if not isinstance(node, dict):
            raise ExprMismatch('AST node is not an object')
        kind = node.get('t')
        if done:
            if kind == 'neg':
                program.append(('f1', np.negative, None))
            elif kind == 'op':
                program.append(('f2', operators[node['o']], None))
            elif kind == 'ifelse':
                program.append(('f3', functions['ifelse'], None))
            else:
                program.append(('f%d' % arity[node['f']], functions[node['f']], None))
            continue
        count += 1
        if count > caps['nodes']:
            raise ExprMismatch('AST exceeds the node cap')
        if kind == 'num':
            value = node.get('v')
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ExprMismatch('bad number node')
            program.append(('push', np.float64(value), None))
        elif kind == 'const':
            name = node.get('n')
            if not isinstance(name, str) or name not in constants:
                raise ExprMismatch('unknown constant')
            program.append(('push', np.float64(constants[name]), None))
        elif kind == 'var':
            name = node.get('n')
            if not isinstance(name, str) or name not in names or name in dims:
                raise ExprMismatch('undeclared or unindexed variable')
            program.append(('var', name, None))
        elif kind == 'idx':
            name, index = node.get('n'), node.get('i')
            if not isinstance(name, str) or name not in dims or not _int(index) or not 0 <= index < dims[name]:
                raise ExprMismatch('bad vector index')
            program.append(('idx', name, index))
        else:
            if kind == 'neg':
                children = node.get('a')
                ok = isinstance(children, list) and len(children) == 1
            elif kind == 'op':
                children = node.get('a')
                ok = isinstance(node.get('o'), str) and node['o'] in operators and isinstance(children, list) and len(children) == 2
            elif kind == 'call':
                children = node.get('a')
                name = node.get('f')
                ok = (isinstance(name, str) and name in functions and name != 'ifelse' and name in arity
                      and isinstance(children, list) and len(children) == arity[name])
            elif kind == 'ifelse':
                children = [node.get('c'), node.get('a'), node.get('b')]
                ok = True
            else:
                raise ExprMismatch('unknown node kind')
            if not ok:
                raise ExprMismatch('node violates the spec')
            stack.append((node, True))
            for child in reversed(children):
                stack.append((child, False))
    return program


def evaluate(program, env):
    """Run a postfix program. env maps declared names to float64 scalars or arrays."""
    values = []
    for kind, a, b in program:
        if kind == 'push':
            values.append(a)
        elif kind == 'var':
            values.append(env[a])
        elif kind == 'idx':
            values.append(env[a][b])
        elif kind == 'f1':
            values[-1] = a(values[-1])
        elif kind == 'f2':
            right = values.pop()
            values[-1] = a(values[-1], right)
        else:
            third = values.pop()
            second = values.pop()
            values[-1] = a(values[-1], second, third)
    return values[0]


class Guard:
    """Evaluation counter and wall-clock budget shared by every expression call of one operation."""

    def __init__(self, max_evals, budget_seconds):
        import time
        self.count = 0
        self.max_evals = max_evals
        self.deadline = time.monotonic() + budget_seconds
        self.nonfinite = 0

    def tick(self):
        import time
        # Checked before the evaluation it admits, so `count` never exceeds max_evals and no call starts after the deadline.
        if self.count >= self.max_evals:
            raise EvalLimit('expression evaluation cap of %d reached' % self.max_evals)
        if time.monotonic() > self.deadline:
            raise TimeLimit('wall-clock budget exhausted after %d evaluations' % self.count)
        self.count += 1


class Context:
    """Everything an operation needs: numpy, compiled programs, variable layout and data columns."""

    def __init__(self, p):
        import numpy as np
        self.np = np
        self.spec, self.digest = load_spec(p)
        self.tables = (function_table(np), operator_table(np))
        check_bindings(self.spec, self.tables[0])
        self.layout = [(v['n'], int(v['len'])) for v in p['variables']]
        self.dims = {n: length for n, length in self.layout if length}
        self.x0 = np.array(p['x0_flat'], dtype=float)
        self.static = {}
        for column in p.get('data', []):
            self.static[column['n']] = np.load(p[column['key'] + '_file'], allow_pickle=False)
        self.names = {n for n, _ in self.layout} | set(self.static)
        self.guard = Guard(p['max_evals'], p['time_budget_s'])

    def program(self, ast):
        return compile_expression(ast, self.names, self.dims, self.spec, self.tables)

    def bind(self, xv):
        env = dict(self.static)
        at = 0
        for name, length in self.layout:
            if length:
                env[name] = xv[at:at + length]
                at += length
            else:
                env[name] = xv[at]
                at += 1
        return env

    def unpack(self, xv):
        """Flat vector -> {name: float | [floats]} in layout order."""
        out, at = {}, 0
        for name, length in self.layout:
            if length:
                out[name] = [float(v) for v in xv[at:at + length]]
                at += length
            else:
                out[name] = float(xv[at])
                at += 1
        return out

    def bound_arrays(self, p):
        np = self.np
        if 'lb' not in p:
            return None
        lb = np.array([-np.inf if v is None else v for v in p['lb']], dtype=float)
        ub = np.array([np.inf if v is None else v for v in p['ub']], dtype=float)
        return lb, ub


def warn_nonfinite(guard, what):
    if guard.nonfinite:
        warnings.warn('%d evaluations of the %s were non-finite' % (guard.nonfinite, what), NumericalWarning)


def scalar_value(np, value, what):
    if np.ndim(value) != 0:
        raise BadData(what + ' must evaluate to a single number; wrap data columns in sum(), mean(), ...', 'EXPR_NOT_SCALAR')
    return float(value)


def opt_minimize(p):
    ctx = Context(p)
    np = ctx.np
    from scipy import optimize
    guard = ctx.guard
    objective = ctx.program(p['objective_ast'])
    constraint_programs = [(c['type'], ctx.program(c['ast'])) for c in p.get('constraints', [])]
    gradient_programs = [ctx.program(a) for a in p.get('gradient_asts', [])]
    policy_error = p['nonfinite'] == 'error'
    best = {'fun': np.inf, 'x': ctx.x0.copy()}
    state = {'at_start': True}

    def f(xv):
        guard.tick()
        value = scalar_value(np, evaluate(objective, ctx.bind(xv)), 'objective')
        if not math.isfinite(value):
            if state['at_start']:
                raise BadData('objective is not finite at x0', 'NONFINITE_OBJECTIVE')
            guard.nonfinite += 1
            if policy_error:
                raise NonFinite('non-finite objective value; last parameters kept in par')
            return math.inf
        if value < best['fun']:
            best['fun'], best['x'] = value, np.array(xv, dtype=float)
        return value

    def gradient_values(xv):
        guard.tick()
        env = ctx.bind(xv)
        return np.array([scalar_value(np, evaluate(g, env), 'gradient') for g in gradient_programs], dtype=float)

    def gradient(xv):
        out = gradient_values(xv)
        if not np.isfinite(out).all():
            guard.nonfinite += 1
            raise NonFinite('non-finite gradient value')
        return out

    def constraint_function(program):
        def g(xv):
            guard.tick()
            return scalar_value(np, evaluate(program, ctx.bind(xv)), 'constraint')
        return g

    def check_gradient_at_start(f0):
        """A wrong analytic gradient is the usual reason a gradient-based run silently fails: compare with forward
        differences at x0. Every evaluation is counted and budgeted like the solver's own."""
        analytic = gradient_values(ctx.x0)
        if not np.isfinite(analytic).all():
            raise BadData('a gradient expression is not finite at x0', 'NONFINITE_GRADIENT')
        numeric = np.full(ctx.x0.size, np.nan)
        for i in range(ctx.x0.size):
            up = ctx.x0.copy()
            h = 1.4901161193847656e-8 * max(1.0, abs(up[i]))
            up[i] += h
            guard.tick()
            value = scalar_value(np, evaluate(objective, ctx.bind(up)), 'objective')
            if math.isfinite(value):
                numeric[i] = (value - f0) / h
        usable = np.isfinite(numeric)
        if usable.any():
            scale = max(1.0, float(np.max(np.abs(numeric[usable]))))
            worst = float(np.max(np.abs(analytic[usable] - numeric[usable])))
            if not worst <= 1e-4 * scale:
                warnings.warn('gradient expressions disagree with finite differences at x0 (max difference %.3g)' % worst, NumericalWarning)

    method = p['method']
    stop, res, violation = None, None, None
    xv, fun, message, nit, njev, status = best['x'], math.nan, '', None, None, 'OK'
    with np.errstate(all='ignore'):
        try:
            f0 = f(ctx.x0)  # validates scalar-ness and finiteness at the start before any solver runs
            state['at_start'] = False
            if gradient_programs:
                check_gradient_at_start(f0)
            kwargs = {'method': method, 'options': {'maxiter': p['maxiter']}}
            if 'tol' in p:
                kwargs['tol'] = p['tol']
            if gradient_programs:
                kwargs['jac'] = gradient
            bounds = ctx.bound_arrays(p)
            if bounds is not None:
                kwargs['bounds'] = optimize.Bounds(bounds[0], bounds[1])
            if constraint_programs:
                kwargs['constraints'] = [{'type': kind, 'fun': constraint_function(program)} for kind, program in constraint_programs]
            res = optimize.minimize(f, ctx.x0, **kwargs)
            xv, fun = np.asarray(res.x, dtype=float), float(res.fun)
            message, nit, njev = str(res.get('message', '')), res.get('nit'), res.get('njev')
            status = 'OK' if res.get('success') else 'NO_CONVERGENCE'
        except SolverStop as exc:  # the start point, the gradient check and the solver all end here
            stop = exc
            xv, fun, message, status = best['x'], best['fun'], exc.message, exc.status
        if not stop and not (np.isfinite(xv).all() and math.isfinite(fun)):
            status, message = 'NUMERICAL_FAILURE', 'the solver returned a non-finite point'
        if constraint_programs:
            # Reporting the violation is bookkeeping after the stop, not part of the run: it bypasses the budget.
            env = ctx.bind(xv)
            values = [(kind, scalar_value(np, evaluate(program, env), 'constraint')) for kind, program in constraint_programs]
            violation = max([abs(v) if kind == 'eq' else max(0.0, -v) for kind, v in values] or [0.0])
            if status == 'OK' and not violation <= 1e-6:
                status, message = 'NO_CONVERGENCE', 'constraint violation %.3g exceeds 1e-6; %s' % (violation, message)
    warn_nonfinite(guard, 'objective')
    result = {'status': status, 'message': message, 'method': method, 'par': ctx.unpack(xv), 'fun': fun,
              'nit': None if nit is None else int(nit), 'nfev': int(res.get('nfev', guard.count)) if res is not None else guard.count,
              'evals': guard.count, 'nonfinite_evals': guard.nonfinite}
    if njev is not None:
        result['njev'] = int(njev)
    if violation is not None:
        result['constraint_violation'] = float(violation)
    return result


def adaptive_jacobian(np, residual, xv, r_ref, scipy_jac, scale, bounds):
    """Per-parameter central differences with a step ladder (factors 2^-2 .. 2^6) and the step whose neighbours agree best.
    The base step is eps^(1/3) * max(|x_j|, 1e-4 * rho/||J_j||) where J_j is the solver's own Jacobian column and rho the
    residual scale: a parameter that is ~0 at the solution still gets a step that moves the residuals well above rounding noise
    (scipy's absolute floor and the earlier 1e-8 floor were both wrong in opposite directions). One-sided at bounds.
    Returns (jacobian, columns_disagreeing_with_scipy)."""
    n = xv.size
    lo, hi = bounds if bounds is not None else (np.full(n, -np.inf), np.full(n, np.inf))
    columns, disagree = [], 0
    for j in range(n):
        norm = float(np.linalg.norm(scipy_jac[:, j]))
        natural = scale / norm if norm > 0 else 1.0
        base = 6.0554544523933395e-6 * max(abs(xv[j]), 1e-4 * natural)
        estimates = []
        for factor in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0):
            h = base * factor
            up, down = xv.copy(), xv.copy()
            up[j] += h
            down[j] -= h
            if up[j] > hi[j]:
                estimates.append((r_ref - residual(down)) / h)
            elif down[j] < lo[j]:
                estimates.append((residual(up) - r_ref) / h)
            else:
                estimates.append((residual(up) - residual(down)) / (2.0 * h))
        errors = [float(np.linalg.norm(estimates[i] - estimates[i + 1])) for i in range(len(estimates) - 1)]
        column = estimates[int(np.argmin(errors))]
        columns.append(column)
        if float(np.linalg.norm(column - scipy_jac[:, j])) > 1e-3 * max(norm, float(np.linalg.norm(column))):
            disagree += 1
    return np.column_stack(columns), disagree


def covariance(np, jac, cost, m, n, absolute):
    """curve_fit's convention: pinv(J'J) via SVD with a rank tolerance; scaled by 2*cost/(m-n) unless absolute.
    Returns (cov or None, status, condition number of the column-equilibrated Jacobian)."""
    # Column equilibration first: rank decisions must not depend on parameter units. The relative threshold 1e-8 is
    # well above the finite-difference noise of the Jacobian (~1e-10) so exactly collinear models are detected;
    # legitimately worse-conditioned fits (high-degree polynomials) are reported as SINGULAR_JACOBIAN with the condition number.
    norms = np.linalg.norm(jac, axis=0)
    if not np.all(norms > 0):
        return None, 'SINGULAR_JACOBIAN', math.inf
    _, s, vt = np.linalg.svd(jac / norms, full_matrices=False)
    condition = float(s[0] / s[-1]) if s[-1] > 0 else math.inf
    keep = s > 1e-8 * s[0]
    if keep.sum() < n:
        return None, 'SINGULAR_JACOBIAN', condition
    cov = (vt[keep].T / s[keep] ** 2) @ vt[keep]
    cov = cov / np.outer(norms, norms)
    if not absolute:
        if m <= n:
            return None, 'NO_DOF', condition
        cov = cov * (2.0 * cost / (m - n))
    return cov, 'OK', condition


def std_errors(ctx, errors):
    if errors is None:
        return {name: ('NA' if not length else ['NA'] * length) for name, length in ctx.layout}
    return ctx.unpack(errors)


MAX_JACOBIAN_ENTRIES = 20_000_000  # m * n doubles: 160 MB per copy; scipy keeps several


def opt_least_squares(p):
    ctx = Context(p)
    np = ctx.np
    from scipy import optimize
    guard = ctx.guard
    n = ctx.x0.size
    if 'model_ast' in p:
        model = ctx.program(p['model_ast'])
        response = ctx.static[p['response']]
        sigma = ctx.static[p['sigma_key']] if 'sigma_key' in p else p.get('sigma_value')
        programs = None
    else:
        programs = [ctx.program(a) for a in p['residual_asts']]
    state = {'at_start': True, 'rows': None}
    best = {'cost': np.inf, 'x': ctx.x0.copy()}

    def residual(xv):
        guard.tick()
        env = ctx.bind(xv)
        if programs is None:
            r = np.asarray(evaluate(model, env) - response, dtype=float)
            if sigma is not None:
                r = r / sigma
        else:
            r = np.concatenate([np.atleast_1d(np.asarray(evaluate(g, env), dtype=float)).ravel() for g in programs])
        r = np.ravel(r)
        if state['rows'] is None:
            state['rows'] = r.size
            if r.size * n > MAX_JACOBIAN_ENTRIES:
                raise BadData('%d residuals x %d unknowns exceeds the Jacobian cap of %d entries' % (r.size, n, MAX_JACOBIAN_ENTRIES), 'INPUT_TOO_LARGE')
        elif r.size != state['rows']:
            raise BadData('residual length changed between evaluations', 'BAD_DATA')
        if not np.isfinite(r).all():
            if state['at_start']:
                raise BadData('residuals are not finite at x0', 'NONFINITE_RESIDUAL')
            guard.nonfinite += 1
            raise NonFinite('non-finite residual; best parameters so far are reported')
        cost = 0.5 * float(np.dot(r, r))
        if cost < best['cost']:
            best['cost'], best['x'] = cost, np.array(xv, dtype=float)
        return r

    method = p['method']
    stop, res, final, m = None, None, None, 0
    xv, message, status = best['x'], '', 'OK'
    bounds = ctx.bound_arrays(p)
    with np.errstate(all='ignore'):
        try:
            r0 = residual(ctx.x0)
            state['at_start'] = False
            m = r0.size
            if method == 'lm' and m < n:
                raise BadData('method lm needs at least as many residuals as unknowns', 'BAD_DATA')
            kwargs = {'method': method, 'loss': p['loss'], 'f_scale': p['f_scale'], 'ftol': p['tol'], 'xtol': p['tol'],
                      'gtol': p['tol'], 'max_nfev': min(p['max_nfev'], p['max_evals']), 'x_scale': 'jac'}
            if bounds is not None:
                kwargs['bounds'] = bounds
            res = optimize.least_squares(residual, ctx.x0, **kwargs)
            xv, message = np.asarray(res.x, dtype=float), str(res.message)
            status = 'OK' if res.status > 0 else 'NO_CONVERGENCE'
        except SolverStop as exc:
            stop = exc
            xv, message, status = best['x'], exc.message, exc.status
        finite = bool(np.isfinite(xv).all())
        if not finite:
            status, message = 'NUMERICAL_FAILURE', 'the solver returned a non-finite point'
        result = {'status': status, 'message': message, 'method': method, 'loss': p['loss'], 'par': ctx.unpack(xv),
                  'n_residuals': int(m), 'n_parameters': int(n)}
        cov_status, condition, errors = 'UNAVAILABLE', None, None
        if res is not None and finite:
            cost = float(res.cost)
            try:
                final = residual(xv)  # every post-solver evaluation is counted and budgeted too
                jac = res.jac
                if p['loss'] == 'linear':  # robust losses keep scipy's loss-modified Jacobian (documented)
                    scale = max(float(np.linalg.norm(r0)), float(np.linalg.norm(final)), 1e-300)
                    jac, disagree = adaptive_jacobian(np, residual, xv, final, res.jac, scale, bounds)
                    if not np.isfinite(jac).all():
                        jac, disagree = res.jac, 0
                else:
                    disagree = 0
                cov, cov_status, condition = covariance(np, jac, cost, m, n, bool(p['absolute_sigma']))
                if cov is not None and disagree:
                    cov_status = 'JACOBIAN_UNCERTAIN'
                    warnings.warn('the finite-difference Jacobian disagrees with the solver\'s on %d column(s); standard errors are uncertain' % disagree, NumericalWarning)
                if cov is not None and condition > 1e6:
                    warnings.warn('the column-equilibrated Jacobian is ill-conditioned (condition number %.3g): standard errors may be inaccurate' % condition, NumericalWarning)
                if status != 'OK':
                    warnings.warn('standard errors come from a non-converged iterate (%s) and are unreliable' % status, NumericalWarning)
                errors = None if cov is None else np.sqrt(np.diag(cov))
                result['rss'] = float(np.dot(final, final))
            except SolverStop as exc:
                cov_status = 'UNAVAILABLE_BUDGET'
                warnings.warn('standard errors skipped: %s' % exc.message, NumericalWarning)
                result['rss'] = float(np.dot(res.fun, res.fun))
            result.update({'cost': cost, 'nfev': int(res.nfev), 'optimality': float(res.optimality)})
            if res.njev is not None:
                result['njev'] = int(res.njev)
        else:
            result.update({'cost': best['cost'], 'rss': 2.0 * best['cost'], 'nfev': guard.count})
    result.update({'std_err': std_errors(ctx, errors), 'cov_status': cov_status, 'evals': guard.count, 'nonfinite_evals': guard.nonfinite})
    if condition is not None:
        result['jacobian_condition'] = condition
    warn_nonfinite(guard, 'residual')
    return result


def single_variable_function(ctx, program, hint=''):
    np = ctx.np
    guard = ctx.guard

    def f(x):
        guard.tick()
        value = evaluate(program, {'x': np.float64(x)})
        if np.ndim(value) != 0:
            raise BadData('f must evaluate to a single number for scalar x', 'EXPR_NOT_SCALAR')
        value = float(value)
        if not math.isfinite(value):
            guard.nonfinite += 1
            raise NonFinite('f(%r) is not finite%s' % (x, hint))
        return value
    return f


class ScalarContext(Context):
    """Context for the one-variable ops: variables fixed to x, no data."""

    def __init__(self, p):
        import numpy as np
        self.np = np
        self.spec, self.digest = load_spec(p)
        self.tables = (function_table(np), operator_table(np))
        check_bindings(self.spec, self.tables[0])
        self.layout, self.dims, self.static = [('x', 0)], {}, {}
        self.names = {'x'}
        self.guard = Guard(p['max_evals'], p['time_budget_s'])


def num_quad(p):
    ctx = ScalarContext(p)
    np = ctx.np
    from scipy import integrate
    hint = ('; a singularity sitting on a Gauss-Kronrod node (for example the midpoint of a symmetric interval) is evaluated exactly: '
            'pass points to isolate it')
    f = single_variable_function(ctx, ctx.program(p['expression_ast']), hint)
    ends = {'Inf': np.inf, '-Inf': -np.inf}
    a, b = [ends[v] if isinstance(v, str) else float(v) for v in (p['a'], p['b'])]
    kwargs = {'epsabs': p['epsabs'], 'epsrel': p['epsrel'], 'limit': p['limit'], 'full_output': 1}
    if p.get('points'):
        kwargs['points'] = p['points']
    with np.errstate(all='ignore'):
        try:
            out = integrate.quad(f, a, b, **kwargs)
            value, abserr, info = out[0], out[1], out[2]
            status, message = ('OK', '') if len(out) == 3 else ('NO_CONVERGENCE', str(out[3]).split('\n')[0])
        except SolverStop as exc:
            return {'status': exc.status, 'message': exc.message, 'evals': ctx.guard.count}
    return {'status': status, 'message': message, 'value': float(value), 'abserr': float(abserr),
            'neval': int(info['neval']), 'subdivisions': int(info['last']), 'evals': ctx.guard.count}


def num_root(p):
    ctx = ScalarContext(p)
    np = ctx.np
    from scipy import optimize
    f = single_variable_function(ctx, ctx.program(p['expression_ast']))
    lower, upper = float(p['lower']), float(p['upper'])
    with np.errstate(all='ignore'):
        try:
            try:
                fa, fb = f(lower), f(upper)
            except NonFinite:
                raise BadData('f is not finite at a bracket end', 'NONFINITE_BRACKET')
            for value, point in ((fa, lower), (fb, upper)):
                if value == 0.0:  # an end is an exact root: scipy would return before filling in its iteration count
                    return {'status': 'OK', 'message': '', 'method': p['method'], 'root': point, 'f_root': 0.0, 'iterations': 0,
                            'function_calls': 2, 'evals': ctx.guard.count}
            if math.copysign(1.0, fa) == math.copysign(1.0, fb):  # sign test, not fa*fb (which underflows for tiny values)
                raise BadData('f(lower) and f(upper) must have opposite signs (or one must be zero)', 'BAD_INTERVAL')
            solver = {'brentq': optimize.brentq, 'bisect': optimize.bisect, 'ridder': optimize.ridder, 'toms748': optimize.toms748}[p['method']]
            root, info = solver(f, lower, upper, xtol=p['xtol'], rtol=p['rtol'], maxiter=p['maxiter'], full_output=True, disp=False)
        except SolverStop as exc:
            return {'status': exc.status, 'message': exc.message, 'method': p['method'], 'evals': ctx.guard.count}
    return {'status': 'OK' if info.converged else 'NO_CONVERGENCE', 'message': '' if info.converged else str(info.flag),
            'method': p['method'], 'root': float(root), 'f_root': f_safe(f, root), 'iterations': max(0, int(info.iterations)),
            'function_calls': max(0, int(info.function_calls)), 'evals': ctx.guard.count}


def f_safe(f, x):
    try:
        return f(x)
    except SolverStop:
        return math.nan
