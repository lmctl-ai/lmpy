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
        self.count += 1
        if self.count > self.max_evals:
            raise EvalLimit('expression evaluation cap of %d reached' % self.max_evals)
        if time.monotonic() > self.deadline:
            raise TimeLimit('wall-clock budget exhausted after %d evaluations' % self.count)


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

    def gradient(xv):
        guard.tick()
        env = ctx.bind(xv)
        out = np.array([scalar_value(np, evaluate(g, env), 'gradient') for g in gradient_programs], dtype=float)
        if not np.isfinite(out).all():
            guard.nonfinite += 1
            raise NonFinite('non-finite gradient value')
        return out

    def constraint_function(program):
        def g(xv):
            guard.tick()
            return scalar_value(np, evaluate(program, ctx.bind(xv)), 'constraint')
        return g

    with np.errstate(all='ignore'):
        f(ctx.x0)  # validates scalar-ness and finiteness at the start before any solver runs
        state['at_start'] = False
        if gradient_programs:
            # A wrong analytic gradient is the usual reason a gradient-based run silently fails: compare at x0.
            analytic = gradient(ctx.x0)
            numeric = optimize.approx_fprime(ctx.x0, lambda v: evaluate(objective, ctx.bind(v)), 1.4901161193847656e-8)
            scale = max(1.0, float(np.max(np.abs(numeric))))
            if not float(np.max(np.abs(analytic - numeric))) <= 1e-4 * scale:
                warnings.warn('gradient expressions disagree with finite differences at x0 (max difference %.3g)'
                              % float(np.max(np.abs(analytic - numeric))), NumericalWarning)
        method = p['method']
        options = {'maxiter': p['maxiter']}
        kwargs = {'method': method, 'options': options}
        if 'tol' in p:
            kwargs['tol'] = p['tol']
        if gradient_programs:
            kwargs['jac'] = gradient
        bounds = ctx.bound_arrays(p)
        if bounds is not None:
            kwargs['bounds'] = optimize.Bounds(bounds[0], bounds[1])
        if constraint_programs:
            kwargs['constraints'] = [{'type': kind, 'fun': constraint_function(program)} for kind, program in constraint_programs]
        stop = None
        try:
            res = optimize.minimize(f, ctx.x0, **kwargs)
            xv, fun = np.asarray(res.x, dtype=float), float(res.fun)
            message, nit, njev = str(res.get('message', '')), res.get('nit'), res.get('njev')
            status = 'OK' if res.get('success') else 'NO_CONVERGENCE'
        except SolverStop as exc:
            stop = exc
            xv, fun, message, nit, njev = best['x'], best['fun'], exc.message, None, None
            status = exc.status
        if not stop and not (np.isfinite(xv).all() and math.isfinite(fun)):
            status, message = 'NUMERICAL_FAILURE', 'the solver returned a non-finite point'
        violation = None
        if constraint_programs:
            env = ctx.bind(xv)
            values = [(kind, scalar_value(np, evaluate(program, env), 'constraint')) for kind, program in constraint_programs]
            violation = max([abs(v) if kind == 'eq' else max(0.0, -v) for kind, v in values] or [0.0])
            if status == 'OK' and not violation <= 1e-6:
                status, message = 'NO_CONVERGENCE', 'constraint violation %.3g exceeds 1e-6; %s' % (violation, message)
    warn_nonfinite(guard, 'objective')
    result = {'status': status, 'message': message, 'method': method, 'par': ctx.unpack(xv), 'fun': fun,
              'nit': None if nit is None else int(nit), 'nfev': int(res.get('nfev', guard.count)) if not stop else guard.count,
              'evals': guard.count, 'nonfinite_evals': guard.nonfinite}
    if njev is not None:
        result['njev'] = int(njev)
    if violation is not None:
        result['constraint_violation'] = float(violation)
    return result


def central_jacobian(np, residual, xv, r0, bounds):
    """Central differences (error O(h^2)) at the solution; scipy's own 2-point Jacobian is only O(h) with an absolute
    step, which costs ~5 digits of the standard errors for small parameters (NIST Misra1a b2 ~ 5e-4). One-sided at bounds."""
    n = xv.size
    lo, hi = (bounds if bounds is not None else (np.full(n, -np.inf), np.full(n, np.inf)))
    columns = []
    for i in range(n):
        h = 6.0554544523933395e-6 * max(abs(xv[i]), 1e-8)  # eps**(1/3)
        up, down = xv.copy(), xv.copy()
        up[i] += h
        down[i] -= h
        if up[i] > hi[i]:
            columns.append((r0 - residual(down)) / h)
        elif down[i] < lo[i]:
            columns.append((residual(up) - r0) / h)
        else:
            columns.append((residual(up) - residual(down)) / (2.0 * h))
    return np.column_stack(columns)


def covariance(np, jac, cost, m, n, absolute):
    """curve_fit's convention: pinv(J'J) via SVD with a rank tolerance; scaled by 2*cost/(m-n) unless absolute."""
    # Column equilibration first: rank decisions must not depend on parameter units. The relative threshold 1e-8 is
    # well above the finite-difference noise of the Jacobian (~1e-10) so exactly collinear models are detected.
    norms = np.linalg.norm(jac, axis=0)
    if not np.all(norms > 0):
        return None, 'SINGULAR_JACOBIAN'
    _, s, vt = np.linalg.svd(jac / norms, full_matrices=False)
    keep = s > 1e-8 * s[0]
    if keep.sum() < n:
        return None, 'SINGULAR_JACOBIAN'
    cov = (vt[keep].T / s[keep] ** 2) @ vt[keep]
    cov = cov / np.outer(norms, norms)
    if not absolute:
        if m <= n:
            return None, 'NO_DOF'
        cov = cov * (2.0 * cost / (m - n))
    return cov, 'OK'


def std_errors(ctx, errors):
    if errors is None:
        return {name: ('NA' if not length else ['NA'] * length) for name, length in ctx.layout}
    return ctx.unpack(errors)


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
    with np.errstate(all='ignore'):
        r0 = residual(ctx.x0)
        state['at_start'] = False
        m = r0.size
        if method == 'lm' and m < n:
            raise BadData('method lm needs at least as many residuals as unknowns', 'BAD_DATA')
        kwargs = {'method': method, 'loss': p['loss'], 'f_scale': p['f_scale'], 'ftol': p['tol'], 'xtol': p['tol'],
                  'gtol': p['tol'], 'max_nfev': min(p['max_nfev'], p['max_evals']), 'x_scale': 'jac'}
        bounds = ctx.bound_arrays(p)
        if bounds is not None:
            kwargs['bounds'] = bounds
        stop = None
        res = None
        try:
            res = optimize.least_squares(residual, ctx.x0, **kwargs)
            xv, message = np.asarray(res.x, dtype=float), str(res.message)
            status = 'OK' if res.status > 0 else 'NO_CONVERGENCE'
        except SolverStop as exc:
            stop = exc
            xv, message, status = best['x'], exc.message, exc.status
        finite = bool(np.isfinite(xv).all())
        if not finite:
            status, message = 'NUMERICAL_FAILURE', 'the solver returned a non-finite point'
        final = residual(xv) if finite and not stop else None
    result = {'status': status, 'message': message, 'method': method, 'loss': p['loss'], 'par': ctx.unpack(xv),
              'n_residuals': int(m), 'n_parameters': int(n), 'evals': guard.count, 'nonfinite_evals': guard.nonfinite}
    if res is not None and not stop:
        cost = float(res.cost)
        jac = res.jac
        if p['loss'] == 'linear':  # robust losses keep scipy's loss-modified Jacobian (documented)
            try:
                with np.errstate(all='ignore'):
                    jac = central_jacobian(np, residual, xv, final, bounds)
                if not np.isfinite(jac).all():
                    jac = res.jac
            except SolverStop:
                jac = res.jac
        cov, cov_status = covariance(np, jac, cost, m, n, bool(p['absolute_sigma']))
        if status != 'OK':
            warnings.warn('standard errors come from a non-converged iterate (%s) and are unreliable' % status, NumericalWarning)
        result['std_err'] = std_errors(ctx, None if cov is None else np.sqrt(np.diag(cov)))
        result.update({'cost': cost, 'rss': float(np.dot(final, final)), 'cov_status': cov_status, 'nfev': int(res.nfev),
                       'optimality': float(res.optimality)})
        if res.njev is not None:
            result['njev'] = int(res.njev)
    else:
        result.update({'std_err': std_errors(ctx, None), 'cost': best['cost'], 'rss': 2.0 * best['cost'], 'cov_status': 'UNAVAILABLE', 'nfev': guard.count})
    warn_nonfinite(guard, 'residual')
    return result


def single_variable_function(ctx, program):
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
            raise NonFinite('f(%r) is not finite' % x)
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
    f = single_variable_function(ctx, ctx.program(p['expression_ast']))
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
            if fa * fb > 0:
                raise BadData('f(lower) and f(upper) must have opposite signs (or one must be zero)', 'BAD_INTERVAL')
            solver = {'brentq': optimize.brentq, 'bisect': optimize.bisect, 'ridder': optimize.ridder, 'toms748': optimize.toms748}[p['method']]
            root, info = solver(f, lower, upper, xtol=p['xtol'], rtol=p['rtol'], maxiter=p['maxiter'], full_output=True, disp=False)
        except SolverStop as exc:
            return {'status': exc.status, 'message': exc.message, 'method': p['method'], 'evals': ctx.guard.count}
    return {'status': 'OK' if info.converged else 'NO_CONVERGENCE', 'message': '' if info.converged else str(info.flag),
            'method': p['method'], 'root': float(root), 'f_root': f_safe(f, root), 'iterations': int(info.iterations),
            'function_calls': int(info.function_calls), 'evals': ctx.guard.count}


def f_safe(f, x):
    try:
        return f(x)
    except SolverStop:
        return math.nan
