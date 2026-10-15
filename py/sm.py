"""S5: statsmodels operations. Embedded with the worker (after numerics.py), never imported from caller paths.

No formula strings anywhere: C++ validates a JSON term list, sm_design() below builds the design matrix, and statsmodels is
driven through its array API (OLS/WLS/GLM/SARIMAX/acf/pacf). patsy and formulaic are never imported: statsmodels' package
__init__ imports patsy.util unconditionally, so sm_import() satisfies that single import with an empty stub that is removed
again, and the real packages are on the worker's blocked-import list.
"""
SM_ROWS = 200_000
SM_COLUMNS = 256
SM_CELLS = 10_000_000
SM_FORBIDDEN = ('patsy', 'formulaic', 'statsmodels.formula.api')


class _SmAlarm(BaseException):
    """Raised from the SIGALRM handler; BaseException so that library `except Exception` blocks cannot swallow it."""


def sm_import():
    """Trusted import phase (runs before the import guard). Returns the statsmodels module namespace used by the ops."""
    import sys
    import types
    stub = types.ModuleType('patsy')
    stub.__path__ = []
    util = types.ModuleType('patsy.util')

    class PatsyError(Exception):
        pass

    class Unavailable:
        """Stand-in for patsy classes statsmodels uses in isinstance() checks or as base classes; it can never be instantiated."""
        def __init__(self, *args, **kwargs):
            raise ImportError('patsy is not available: pycli never evaluates formulas')

    def unavailable(*args, **kwargs):
        raise ImportError('patsy is not available: pycli never evaluates formulas')
    classes = {'PatsyError': PatsyError}
    for name in ('NAAction', 'DesignMatrix', 'DesignInfo', 'EvalEnvironment'):
        classes[name] = type(name, (Unavailable,), {})

    def lookup(name):
        if name in classes:
            return classes[name]
        if name.startswith('__'):
            raise AttributeError(name)
        return unavailable
    stub.__getattr__ = lookup
    stub.util = util
    util.__getattr__ = lookup
    sys.modules['patsy'] = stub
    sys.modules['patsy.util'] = util
    try:
        import statsmodels
        import statsmodels.regression.linear_model
        import statsmodels.genmod.generalized_linear_model
        import statsmodels.genmod.families
        import statsmodels.genmod.families.links
        import statsmodels.tsa.statespace.sarimax
        import statsmodels.tsa.stattools
        import statsmodels.stats.stattools
        import statsmodels.tools.sm_exceptions
    finally:
        sys.modules.pop('patsy', None)
        sys.modules.pop('patsy.util', None)
    return statsmodels


class sm_time_budget:
    """Cooperative wall-clock budget for library loops we cannot instrument (IRLS, the SARIMAX optimizer)."""

    def __init__(self, seconds):
        self.seconds = float(seconds)

    def __enter__(self):
        import signal

        def handler(signum, frame):
            raise _SmAlarm()
        self.previous = signal.signal(signal.SIGALRM, handler)
        signal.setitimer(signal.ITIMER_REAL, self.seconds)
        return self

    def __exit__(self, kind, value, trace):
        import signal
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, self.previous)
        return False


def sm_numeric(np, df, meta, masks, name):
    if name not in df:
        raise BadData('unknown column: ' + name, 'UNKNOWN_COLUMN')
    kind = meta[name]['dtype']
    if kind not in ('int64', 'float64', 'bool'):
        raise BadData('column %s is %s, not numeric (use a cat term for categorical columns)' % (name, kind), 'NOT_NUMERIC')
    values = df[name].to_numpy(dtype=float, na_value=np.nan)
    return values


def sm_missing(np, df, meta, masks, names):
    mask = np.zeros(len(df), dtype=bool)
    for name in names:
        if name not in df:
            raise BadData('unknown column: ' + name, 'UNKNOWN_COLUMN')
        mask |= masks[name]
        if meta[name]['dtype'] == 'float64':
            mask |= np.isnan(df[name].to_numpy(dtype=float, na_value=np.nan))
    return mask


def sm_rows(np, df, meta, masks, names, action):
    """Rows to keep for the used columns, with the explicit missing-value policy. Infinities are never silently dropped."""
    if len(df) > SM_ROWS:
        raise BadData('more than %d rows' % SM_ROWS, 'INPUT_TOO_LARGE')
    for name in names:
        if name in df and meta[name]['dtype'] == 'float64' and np.isinf(df[name].to_numpy(dtype=float, na_value=np.nan)).any():
            raise BadData('column %s contains infinite values' % name, 'NONFINITE_DATA')
    missing = sm_missing(np, df, meta, masks, names)
    dropped = int(missing.sum())
    if dropped and action != 'drop':
        raise BadData('%d row(s) have missing values in the used columns; pass na_action "drop" to drop them (n_dropped is reported)' % dropped, 'MISSING_VALUES')
    return ~missing, dropped


def sm_levels(np, df, meta, name, keep):
    """Observed levels of a categorical column (after row selection) and their codes: category dtype keeps its declared order,
    strings are sorted by code point, integers numerically, booleans False before True."""
    kind = meta[name]['dtype']
    column = df[name]
    if kind == 'category':
        declared = list(column.cat.categories)
        values = column.astype(object).to_numpy()[keep]
        seen = set(values.tolist())
        levels = [v for v in declared if v in seen]
        text = [str(v) for v in levels]
    elif kind == 'string':
        values = column.astype(object).to_numpy()[keep]
        levels = sorted(set(values.tolist()))
        text = [str(v) for v in levels]
    elif kind in ('int64', 'bool'):
        values = column.astype(object).to_numpy()[keep]
        levels = sorted(set(values.tolist()))
        text = [str(bool(v)) if kind == 'bool' else str(int(v)) for v in levels]
    else:
        raise BadData('column %s is %s: categorical terms need string, category, integer or boolean columns' % (name, kind), 'NOT_CATEGORICAL')
    return values, levels, text


def sm_design(np, df, meta, masks, p, extra_columns, action):
    """Build (X, names, kept-row mask, coding, dropped) from the validated term list. Pure numpy; no formula machinery."""
    terms = p['terms']
    used = set(extra_columns)
    has_const = any('const' in t for t in terms)
    for t in terms:
        for key in ('var', 'cat'):
            if key in t:
                used.add(t[key])
        if 'poly' in t:
            used.add(t['poly'][0])
        if 'interact' in t:
            used.update(t['interact'])
    keep, dropped = sm_rows(np, df, meta, masks, sorted(used), action)
    n = int(keep.sum())
    if n == 0:
        raise BadData('no rows left after dropping missing values', 'NO_DATA')
    cats = {t['cat']: t['ref'] for t in terms if 'cat' in t}
    # Column plan first, so the column cap is enforced before any matrix is allocated.
    plan = {}
    coding = {}
    width = 0
    for t in terms:
        if 'cat' in t:
            values, levels, text = sm_levels(np, df, meta, t['cat'], keep)
            if t['ref'] not in text:
                shown = ', '.join(text[:20]) + (', ...' if len(text) > 20 else '')
                raise BadData('reference level %r of %s is not among its levels: %s' % (t['ref'], t['cat'], shown), 'BAD_REFERENCE')
            plan[t['cat']] = (values, levels, text)
            coding[t['cat']] = {'reference': t['ref'], 'levels': text}
    def widths(name):
        if name in cats:
            return len(plan[name][1]) - 1
        return 1
    for t in terms:
        if 'const' in t:
            width += 1 if t['const'] else 0
        elif 'var' in t:
            width += 1
        elif 'cat' in t:
            width += widths(t['cat'])
        elif 'poly' in t:
            width += t['poly'][1]
        else:
            w = 1
            for name in t['interact']:
                w *= widths(name)
            width += w
        if width > SM_COLUMNS:
            raise BadData('the design would have more than %d columns' % SM_COLUMNS, 'TOO_MANY_COLUMNS')
    if not has_const:
        width += 1
    if width > SM_COLUMNS:
        raise BadData('the design would have more than %d columns' % SM_COLUMNS, 'TOO_MANY_COLUMNS')
    if width * n > SM_CELLS:
        raise BadData('design of %d rows x %d columns exceeds %d cells' % (n, width, SM_CELLS), 'INPUT_TOO_LARGE')

    def numeric(name):
        return sm_numeric(np, df, meta, masks, name)[keep]

    def cat_columns(name):
        values, levels, text = plan[name]
        out = []
        for level, label in zip(levels, text):
            if label != cats[name]:
                out.append(('%s[T.%s]' % (name, label), (values == level).astype(float)))
        return out

    def factor_columns(name):
        if name in cats:
            return cat_columns(name)
        if meta[name]['dtype'] not in ('int64', 'float64', 'bool'):
            raise BadData('interaction factor %s is categorical: give it a cat term first' % name, 'INTERACT_NEEDS_CAT_TERM')
        return [(name, numeric(name))]
    columns = []
    if not has_const:
        columns.append(('const', np.ones(n)))
    for t in terms:
        if 'const' in t:
            if t['const']:
                columns.append(('const', np.ones(n)))
        elif 'var' in t:
            columns.append((t['var'], numeric(t['var'])))
        elif 'cat' in t:
            columns.extend(cat_columns(t['cat']))
        elif 'poly' in t:
            x = numeric(t['poly'][0])
            for k in range(1, t['poly'][1] + 1):
                columns.append((t['poly'][0] if k == 1 else '%s^%d' % (t['poly'][0], k), x ** k))
        else:
            combos = [[]]
            for name in t['interact']:
                combos = [c + [f] for c in combos for f in factor_columns(name)]
            for combo in combos:
                values = combo[0][1].copy()
                for _, other in combo[1:]:
                    values = values * other
                columns.append((':'.join(label for label, _ in combo), values))
    names = [c[0] for c in columns]
    if len(set(names)) != len(names):
        raise BadData('duplicate design columns: ' + ', '.join(sorted({x for x in names if names.count(x) > 1})), 'DUPLICATE_COLUMN')
    X = np.column_stack([c[1] for c in columns])
    if not np.isfinite(X).all():
        raise BadData('the design matrix contains non-finite values (overflow in a poly or interact term?)', 'NONFINITE_DATA')
    return X, names, keep, coding, dropped


def sm_table(np, names, params, bse, stat, pvalues, ci, usable):
    """Coefficient table as parallel arrays; undefined standard errors are the string NA."""
    if usable:
        return {'names': names, 'estimate': [float(v) for v in params], 'std_err': [float(v) for v in bse], 'statistic': [float(v) for v in stat],
                'p_value': [float(v) for v in pvalues], 'ci_lower': [float(v) for v in ci[:, 0]], 'ci_upper': [float(v) for v in ci[:, 1]]}
    na = ['NA'] * len(names)
    return {'names': names, 'estimate': [float(v) for v in params], 'std_err': list(na), 'statistic': list(na), 'p_value': list(na), 'ci_lower': list(na), 'ci_upper': list(na)}


def sm_fit_guard(p, body):
    """Run body() under the cooperative time budget; returns its result or a TIME_LIMIT shell."""
    try:
        with sm_time_budget(p['time_budget_s']):
            return body()
    except _SmAlarm:
        return None


def sm_inline_check(coefficients_count, extras, what):
    cells = 7 * coefficients_count + extras
    if cells > 900:
        raise OutputTooLarge('the %s result would hold about %d inline values (limit 1000); use %s_to' % (what, cells, what))


def sm_regression_output(np, p, names, res, usable, extras, fitted, resid, keep, dropped, coding, result):
    twin = bool(p.get('output_prefix'))
    count = len(names)
    ci = res.conf_int(alpha=p['alpha']) if usable else None
    table = sm_table(np, names, res.params, res.bse, res.tvalues, res.pvalues, ci, usable)
    if not twin or count <= 100:
        if not twin:
            sm_inline_check(count, sum(len(c['levels']) for c in coding.values()) + len(p.get('include', [])) * len(fitted), p['op_name'])
        result['coefficients'] = table
    else:
        result['coefficients_in_artifact'] = True
        result['coefficient_names'] = names
    if twin:
        columns = [res.params, res.bse, res.tvalues, res.pvalues, ci[:, 0], ci[:, 1]] if usable else [res.params] + [np.full(count, math.nan)] * 5
        items = [('coef', np.column_stack([np.asarray(c, dtype=float) for c in columns])), ('fitted', fitted), ('residuals', resid)]
        if dropped:
            items.append(('row_index', np.flatnonzero(keep).astype(np.int64)))
        result['outputs'] = save_outputs(p, items)
    else:
        for key in p.get('include', []):
            result[key] = [float(v) for v in (fitted if key == 'fitted' else resid)]
    return result


def sm_load(p):
    import numpy as np
    df, meta, masks, inferred = frame_load(p)
    return np, df, meta, masks


def sm_robust_f(np, res):
    """Overall F test for robust covariances. statsmodels' own res.fvalue goes through wald_test, which builds the constraint with
    patsy.DesignInfo; this is the same computation (identity restrictions on every non-constant coefficient, F reference distribution
    with df_resid) written on the array API so patsy stays unloaded. The tests compare it with statsmodels' value."""
    from scipy import stats
    k = len(res.params)
    constant = res.model.data.const_idx
    keep = [i for i in range(k) if i != constant]
    if not keep:
        return math.nan, math.nan
    beta = np.asarray(res.params, dtype=float)[keep]
    cov = np.asarray(res.cov_params(), dtype=float)[np.ix_(keep, keep)]
    statistic = float(beta @ np.linalg.solve(cov, beta)) / len(keep)
    return statistic, float(stats.f.sf(statistic, len(keep), res.df_resid))


def sm_ols(p):
    np, df, meta, masks = sm_load(p)
    p['op_name'] = 'sm_ols'
    from statsmodels.regression.linear_model import OLS, WLS
    from statsmodels.stats.stattools import durbin_watson
    extra = [p['y']] + [p[k] for k in ('weights', 'cluster') if k in p]
    X, names, keep, coding, dropped = sm_design(np, df, meta, masks, p, extra, p['na_action'])
    y = sm_numeric(np, df, meta, masks, p['y'])[keep]
    n, k = X.shape
    weights = None
    if 'weights' in p:
        weights = sm_numeric(np, df, meta, masks, p['weights'])[keep]
        if not (np.isfinite(weights).all() and (weights > 0).all()):
            raise BadData('weights must be positive and finite', 'BAD_WEIGHTS')
    cov_type = p['cov_type']
    use_t = p.get('use_t')
    if use_t is None:
        use_t = cov_type == 'nonrobust'
    kwargs = {'use_t': bool(use_t)}
    if cov_type == 'cluster':
        if p['cluster'] not in df:
            raise BadData('unknown column: ' + p['cluster'], 'UNKNOWN_COLUMN')
        groups = np.unique(df[p['cluster']].astype(object).to_numpy()[keep].astype(str), return_inverse=True)[1]
        if len(set(groups.tolist())) < 2:
            raise BadData('cluster needs at least two groups', 'BAD_DATA')
        kwargs.update(cov_type='cluster', cov_kwds={'groups': groups})
    elif cov_type != 'nonrobust':
        kwargs['cov_type'] = cov_type
    rank = int(np.linalg.matrix_rank(X))
    result = {'status': 'OK', 'n_obs': n, 'n_dropped': dropped, 'rank': rank, 'cov_type': cov_type, 'use_t': bool(use_t), 'alpha': p['alpha'], 'coding': coding}

    def body():
        model = OLS(y, X) if weights is None else WLS(y, X, weights=weights)
        with np.errstate(all='ignore'):
            return model.fit(**kwargs)
    res = sm_fit_guard(p, body)
    if res is None:
        result['status'] = 'TIME_LIMIT'
        return result
    result['df_model'] = int(res.df_model)
    result['df_resid'] = int(res.df_resid)
    no_dof = res.df_resid <= 0
    usable = rank == k and not no_dof
    if rank < k:
        result['status'] = 'RANK_DEFICIENT'
    if no_dof:
        result['no_dof'] = True
    with np.errstate(all='ignore'):
        f_statistic, f_p_value = (float(res.fvalue), float(res.f_pvalue)) if cov_type == 'nonrobust' else sm_robust_f(np, res)
        result.update({'r2': float(res.rsquared), 'r2_adj': float(res.rsquared_adj), 'f_statistic': f_statistic, 'f_p_value': f_p_value, 'k_params': k,
                       'llf': float(res.llf), 'aic': float(res.aic), 'bic': float(res.bic), 'ssr': float(res.ssr),
                       'sigma': float(math.sqrt(res.mse_resid)) if res.df_resid > 0 else math.nan,
                       'condition_number': float(res.condition_number), 'durbin_watson': float(durbin_watson(res.resid))})
    fitted, resid = np.asarray(res.fittedvalues, dtype=float), np.asarray(res.resid, dtype=float)
    return sm_regression_output(np, p, names, res, usable, 0, fitted, resid, keep, dropped, coding, result)




def sm_glm(p):
    np, df, meta, masks = sm_load(p)
    p['op_name'] = 'sm_glm'
    from statsmodels.genmod.generalized_linear_model import GLM
    from statsmodels.genmod import families
    from statsmodels.genmod.families import links
    from statsmodels.tools import sm_exceptions
    extra = [p['y']] + [p[k] for k in ('weights', 'cluster', 'offset', 'exposure') if k in p]
    X, names, keep, coding, dropped = sm_design(np, df, meta, masks, p, extra, p['na_action'])
    y = sm_numeric(np, df, meta, masks, p['y'])[keep]
    n, k = X.shape
    family_name = p['family']
    if family_name == 'poisson' and not ((y >= 0).all() and (y == np.floor(y)).all()):
        raise BadData('poisson responses must be non-negative integers', 'BAD_RESPONSE')
    if family_name == 'gamma' and not (y > 0).all():
        raise BadData('gamma responses must be positive', 'BAD_RESPONSE')
    if family_name == 'binomial' and not ((y >= 0).all() and (y <= 1).all()):
        raise BadData('binomial responses must lie in [0, 1]', 'BAD_RESPONSE')
    link = {'identity': links.Identity, 'log': links.Log, 'inverse_power': links.InversePower, 'logit': links.Logit, 'probit': links.Probit,
            'cloglog': links.CLogLog, 'sqrt': links.Sqrt}[p['link']]()
    family = {'gaussian': families.Gaussian, 'poisson': families.Poisson, 'binomial': families.Binomial, 'gamma': families.Gamma}[family_name](link=link)
    options = {}
    for key, option in (('offset', 'offset'), ('exposure', 'exposure'), ('weights', 'var_weights')):
        if key in p:
            values = sm_numeric(np, df, meta, masks, p[key])[keep]
            if key != 'offset' and not (np.isfinite(values).all() and (values > 0).all()):
                raise BadData('%s must be positive and finite' % key, 'BAD_WEIGHTS')
            options[option] = values
    cov_type = p['cov_type']
    use_t = bool(p.get('use_t', False))
    kwargs = {'maxiter': p['maxiter'], 'tol': p['tol'], 'use_t': use_t}
    if cov_type == 'cluster':
        groups = np.unique(df[p['cluster']].astype(object).to_numpy()[keep].astype(str), return_inverse=True)[1]
        if len(set(groups.tolist())) < 2:
            raise BadData('cluster needs at least two groups', 'BAD_DATA')
        kwargs.update(cov_type='cluster', cov_kwds={'groups': groups})
    elif cov_type != 'nonrobust':
        kwargs['cov_type'] = cov_type
    rank = int(np.linalg.matrix_rank(X))
    result = {'status': 'OK', 'family': family_name, 'link': p['link'], 'n_obs': n, 'n_dropped': dropped, 'rank': rank, 'cov_type': cov_type, 'use_t': use_t, 'alpha': p['alpha'], 'coding': coding}
    separation = []

    def body():
        import warnings
        model = GLM(y, X, family=family, **options)
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter('always')
            with np.errstate(all='ignore'):
                try:
                    res = model.fit(**kwargs)
                except sm_exceptions.PerfectSeparationError:
                    separation.append(True)
                    return None
        separation.extend(w for w in seen if issubclass(w.category, sm_exceptions.PerfectSeparationWarning))
        return res
    res = sm_fit_guard(p, body)
    if res is None and not separation:
        result['status'] = 'TIME_LIMIT'
        return result
    if res is None:
        result['status'] = 'PERFECT_SEPARATION'
        result['message'] = 'perfect separation detected; the maximum likelihood estimate does not exist'
        return result
    converged = bool(res.converged)
    result['converged'] = converged
    result['n_iter'] = int(res.fit_history['iteration']) if 'iteration' in res.fit_history else None
    if separation:
        result['status'] = 'PERFECT_SEPARATION'
    elif not converged:
        result['status'] = 'NO_CONVERGENCE'
    if rank < k:
        result['status'] = 'RANK_DEFICIENT'
    params = np.asarray(res.params, dtype=float)
    if not np.isfinite(params).all():
        result['status'] = 'NUMERICAL_FAILURE'
    parameters = rank  # statsmodels' convention: the scale of gaussian/gamma is not counted (R adds 1; see docs/oracle-findings.md)
    llf = float(res.llf)
    with np.errstate(all='ignore'):
        result.update({'df_model': int(res.df_model), 'df_resid': int(res.df_resid), 'deviance': float(res.deviance), 'null_deviance': float(res.null_deviance),
                       'pearson_chi2': float(res.pearson_chi2), 'scale': float(res.scale), 'llf': llf, 'k_params': parameters, 'aic': -2 * llf + 2 * parameters,
                       'bic': -2 * llf + math.log(n) * parameters})
    usable = rank == k and np.isfinite(np.asarray(res.bse, dtype=float)).all()
    fitted, resid = np.asarray(res.fittedvalues, dtype=float), np.asarray(res.resid_response, dtype=float)
    return sm_regression_output(np, p, names, res, usable, 0, fitted, resid, keep, dropped, coding, result)


def sm_series(np, p, df, meta, masks, name, allow_missing):
    values = sm_numeric(np, df, meta, masks, name)
    if len(df) > SM_ROWS:
        raise BadData('more than %d rows' % SM_ROWS, 'INPUT_TOO_LARGE')
    if np.isinf(values).any():
        raise BadData('the series contains infinite values', 'NONFINITE_DATA')
    missing = np.isnan(values)
    if missing.any() and not allow_missing:
        raise BadData('%d missing value(s) in the series' % int(missing.sum()), 'MISSING_VALUES')
    return values, int(missing.sum())


def sm_arima(p):
    np, df, meta, masks = sm_load(p)
    import warnings
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    y, n_missing = sm_series(np, p, df, meta, masks, p['series'], True)
    n = len(y)
    if n < 10 or n - n_missing < 5:
        raise BadData('the series needs at least 10 observations (5 non-missing)', 'BAD_DATA')
    if n > 20000:
        raise BadData('SARIMAX series are limited to 20000 observations', 'INPUT_TOO_LARGE')
    exog_names = list(p.get('exog', []))
    exog = None
    if exog_names:
        exog_columns = []
        for name in exog_names:
            column, absent = sm_series(np, p, df, meta, masks, name, False)
            exog_columns.append(column)
        exog = np.column_stack(exog_columns)
    seasonal = tuple(p['seasonal_order']) if p.get('seasonal_order') else (0, 0, 0, 0)
    steps = p['forecast_steps']
    result = {'status': 'OK', 'cov_type': p['cov_type'], 'nobs': n, 'n_missing': n_missing, 'order': list(p['order']), 'seasonal_order': list(seasonal), 'trend': p['trend'], 'alpha': p['alpha']}
    model = SARIMAX(y, exog=exog, order=tuple(p['order']), seasonal_order=seasonal, trend=None if p['trend'] == 'n' else p['trend'],
                    enforce_stationarity=p['enforce_stationarity'], enforce_invertibility=p['enforce_invertibility'])
    names = list(model.param_names)
    if exog_names:
        first = [i for i, name in enumerate(names) if name.startswith('x') and name[1:].isdigit()]
        for i, new in zip(first, exog_names):
            names[i] = new
    kwargs = {'method': p['method'], 'maxiter': p['maxiter'], 'disp': False, 'cov_type': p['cov_type']}
    if 'start_params' in p:
        if len(p['start_params']) != len(names):
            raise BadData('start_params has %d value(s) but this model has %d parameters (%s)' % (len(p['start_params']), len(names), ', '.join(names)), 'BAD_START_PARAMS')
        kwargs['start_params'] = np.asarray(p['start_params'], dtype=float)

    def body():
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # statsmodels' own convergence warnings: the verdict is read from mle_retvals
            with np.errstate(all='ignore'):
                return model.fit(**kwargs)
    res = sm_fit_guard(p, body)
    if res is None:
        result['status'] = 'TIME_LIMIT'
        return result
    retvals = res.mle_retvals or {}
    converged = bool(retvals.get('converged', True))
    result['converged'] = converged
    if not converged:
        result['status'] = 'NO_CONVERGENCE'
    llf = float(res.llf)
    if not math.isfinite(llf) or not np.isfinite(np.asarray(res.params, dtype=float)).all():
        result['status'] = 'NUMERICAL_FAILURE'
    bse = np.asarray(res.bse, dtype=float)
    with np.errstate(all='ignore'):
        table = sm_table(np, names, res.params, bse, res.tvalues, res.pvalues, res.conf_int(alpha=p['alpha']), True)
    for i, value in enumerate(bse):
        if not math.isfinite(value):  # an undefined standard error is NA, never a number
            for key in ('std_err', 'statistic', 'p_value', 'ci_lower', 'ci_upper'):
                table[key][i] = 'NA'
    result['coefficients'] = table
    sigma2 = float(res.params[-1]) if names[-1] == 'sigma2' else math.nan
    effective = int(res.nobs - res.loglikelihood_burn)
    result.update({'sigma2': sigma2, 'llf': llf, 'aic': float(res.aic), 'bic': float(res.bic), 'hqic': float(res.hqic), 'nobs_effective': effective})
    items = []
    if steps > 0 and result['status'] in ('OK', 'NO_CONVERGENCE'):
        future = None
        if exog_names:
            future = np.column_stack([np.asarray(p['exog_future'][name], dtype=float) for name in exog_names])
        with np.errstate(all='ignore'):
            frame = res.get_forecast(steps, exog=future).summary_frame(alpha=p['alpha'])
        mean, lower, upper = frame['mean'].to_numpy(), frame['mean_ci_lower'].to_numpy(), frame['mean_ci_upper'].to_numpy()
        if p.get('output_prefix'):
            items.append(('forecast', np.vstack([mean, lower, upper])))
        else:
            result['forecast'] = {'mean': [float(v) for v in mean], 'lower': [float(v) for v in lower], 'upper': [float(v) for v in upper]}
    if p.get('output_prefix'):
        items.append(('residuals', np.asarray(res.resid, dtype=float)))
        result['outputs'] = save_outputs(p, items)
    return result


def sm_tsa_acf(p):
    np, df, meta, masks = sm_load(p)
    from statsmodels.tsa.stattools import acf, pacf
    y, _ = sm_series(np, p, df, meta, masks, p['series'], False)
    n = len(y)
    nlags = p['nlags']
    if n < 3:
        raise BadData('the series needs at least 3 observations', 'BAD_DATA')
    if float(np.ptp(y)) == 0.0:
        raise BadData('the series is constant: its autocorrelation is undefined', 'CONSTANT_SERIES')
    limit = n // 2 if p['kind'] == 'pacf' else n
    if nlags >= limit:
        raise BadData('nlags %d must be below %d for a series of length %d' % (nlags, limit, n), 'BAD_PARAM')
    with np.errstate(all='ignore'):
        if p['kind'] == 'acf':
            values, ci = acf(y, nlags=nlags, adjusted=p['adjusted'], alpha=p['alpha'], fft=False)
        else:
            values, ci = pacf(y, nlags=nlags, method=p['method'], alpha=p['alpha'])
    values = np.asarray(values, dtype=float)
    ci = np.asarray(ci, dtype=float)
    return {'status': 'OK', 'kind': p['kind'], 'n': n, 'lags': list(range(nlags + 1)), 'values': [float(v) for v in values],
            'ci_lower': [float(v) for v in ci[:, 0]], 'ci_upper': [float(v) for v in ci[:, 1]], 'alpha': p['alpha']}


def sm_operation(name):
    return {'sm_ols': sm_ols, 'sm_ols_to': sm_ols, 'sm_glm': sm_glm, 'sm_glm_to': sm_glm, 'sm_arima': sm_arima, 'sm_arima_to': sm_arima, 'sm_tsa_acf': sm_tsa_acf}[name]
