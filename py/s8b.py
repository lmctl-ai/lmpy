"""S8b: matrix functions (la_matrix_fn), sparse linear algebra (la_sparse) and special functions (special_fn) on fixed scipy.linalg /
scipy.sparse.linalg / scipy.special paths. Assembled after s8.py (shares array, save_outputs, s7_emit, s7_guard, s8_read, BadData).
The C++ front end (src/s8b.hpp) has validated every enum, name, shape and cap before this runs; the worker repeats what a corrupted request could break."""
S8B_OPS = ('la_matrix_fn', 'la_matrix_fn_to', 'la_sparse', 'la_sparse_to', 'special_fn', 'special_fn_to')
S8B_MATRIX_RESIDUAL = 1e-6


def s8b_fro(np, x):
    return float(np.linalg.norm(x, 'fro'))


def s8b_complex_items(np, name, y):
    """(items, is_complex): a complex result is stored as <name>_re / <name>_im; an exactly real one as <name>."""
    y = np.asarray(y)
    if np.iscomplexobj(y):
        if np.all(y.imag == 0):
            return [(name, np.ascontiguousarray(y.real))], False
        return [(name + '_re', np.ascontiguousarray(y.real)), (name + '_im', np.ascontiguousarray(y.imag))], True
    return [(name, y)], False


# ------------------------------------------------------------------------------------------------------- la_matrix_fn
def la_matrix_fn(p):
    import numpy as np
    import scipy.linalg as sla
    a = np.asarray(array(p, 'a'), dtype=float)
    fn = p['fn']
    s7_cap(a.ndim == 2 and a.size <= 200000 and (fn == 'pinv' or a.shape[0] == a.shape[1]), 'matrix shape')
    n = a.shape[0]
    result = {'status': 'OK', 'fn': fn, 'shape': list(a.shape)}
    eps = np.finfo(float).eps
    norm_a = max(s8b_fro(np, a), 1e-300)

    def body():
        with np.errstate(all='ignore'):
            if fn == 'inv':
                cond = float(np.linalg.cond(a))
                result['condition_number'] = cond
                if not np.isfinite(cond) or cond >= 1.0 / eps:
                    return 'SINGULAR', 'the matrix is singular to working precision (condition number %.3g)' % cond, None
                y = sla.inv(a)
                residual = s8b_fro(np, a @ y - np.eye(n)) / np.sqrt(n)
            elif fn == 'pinv':
                y, rank = sla.pinv(a, rtol=p.get('rtol'), return_rank=True)
                result['rank'] = int(rank)
                residual = s8b_fro(np, a @ y @ a - a) / norm_a
            elif fn == 'expm':
                y = sla.expm(a)
                residual = None
            elif fn == 'logm':
                if n and float(np.min(np.abs(np.linalg.eigvals(a)))) == 0.0:
                    return 'SINGULAR', 'the matrix has a zero eigenvalue: its logarithm does not exist', None
                y = sla.logm(a)
                residual = s8b_fro(np, sla.expm(y) - a) / norm_a if np.isfinite(y).all() else None
            else:
                y = sla.sqrtm(a)
                residual = s8b_fro(np, y @ y - a) / norm_a if np.isfinite(y).all() else None
        return 'OK', '', (y, residual)
    got = s7_guard(p, body)
    if got is None:
        return {'status': 'TIME_LIMIT', 'fn': fn}
    status, message, payload = got
    if status != 'OK':
        return dict(result, status=status, message=message)
    y, residual = payload
    if not np.isfinite(y).all():
        return dict(result, status='NUMERICAL_FAILURE', message='the result is not finite (overflow, or no principal value exists)')
    if residual is not None:
        result['residual'] = residual
        if residual > S8B_MATRIX_RESIDUAL:
            return dict(result, status='NUMERICAL_FAILURE', message='the computed %s misses its defining identity by %.3g (relative); the matrix is too ill-conditioned or defective' % (fn, residual))
    items, is_complex = s8b_complex_items(np, 'y', y)
    result['complex'] = is_complex
    return s7_emit(np, p, result, items, 'la_matrix_fn')


# ---------------------------------------------------------------------------------------------------------- la_sparse
def s8b_coo(np, p):
    from scipy import sparse
    m, n = p['shape']
    if 'frame_file' in p:
        names, cmeta, rows, raws, masks = s8_read(p['frame_file'])
        index = {c: i for i, c in enumerate(names)}
        picked = []
        for key in ('row', 'col', 'val'):
            i = index[p[key]]
            if masks[i].any():
                raise BadData('column %s holds masked missing values' % p[key], 'NA_NOT_ALLOWED')
            dtype = cmeta[p[key]]['dtype']
            if key != 'val' and dtype != 'int64':
                raise BadData('column %s must be int64 (a row/column index)' % p[key], 'BAD_DTYPE')
            if key == 'val' and dtype not in ('int64', 'float64', 'bool'):
                raise BadData('column %s must be numeric' % p[key], 'BAD_DTYPE')
            picked.append(raws[i])
        row, col, val = picked[0].astype(np.int64), picked[1].astype(np.int64), picked[2].astype(float)
    else:
        row_f, col_f, val = array(p, 'row'), array(p, 'col'), np.asarray(array(p, 'val'), dtype=float)
        if not (np.all(row_f == np.floor(row_f)) and np.all(col_f == np.floor(col_f))):
            raise BadData('row and col must hold integer indices', 'BAD_DATA')
        row, col = row_f.astype(np.int64), col_f.astype(np.int64)
    s7_cap(len(val) <= 200000 and m <= 100000 and n <= 100000, 'sparse size')
    if len(row) != len(col) or len(row) != len(val) or len(val) == 0:
        raise BadData('row, col and val must be nonempty vectors of one length', 'BAD_DATA')
    if row.min() < 0 or col.min() < 0 or row.max() >= m or col.max() >= n:
        raise BadData('a row/col index lies outside the %d x %d matrix' % (m, n), 'BAD_INDEX')
    if not np.isfinite(val).all():
        raise BadData('val must be finite', 'BAD_DATA')
    return sparse.coo_matrix((val, (row, col)), shape=(m, n)).tocsr()


def s8b_symmetric(np, A):
    d = (A - A.T)
    scale = max(float(abs(A).max()), 1e-300)
    return d.nnz == 0 or float(abs(d).max()) <= 1e-12 * scale


def la_sparse(p):
    import numpy as np
    from scipy import sparse
    import scipy.sparse.linalg as spla
    task = p['task']
    A = s8b_coo(np, p)
    m, n = A.shape
    result = {'status': 'OK', 'task': task, 'shape': [m, n], 'nnz': int(A.nnz)}
    if task in ('cg', 'eigsh') and not s8b_symmetric(np, A):
        raise BadData('%s needs a symmetric matrix' % task, 'NOT_SYMMETRIC')
    rng = np.random.Generator(np.random.PCG64(p['seed'])) if 'seed' in p else None
    items = []
    started = {}

    def residual_vector(x, b):
        return float(np.linalg.norm(A @ x - b)) / max(float(np.linalg.norm(b)), 1e-300)

    def body():
        import warnings
        with np.errstate(all='ignore'):
            if task in ('spsolve', 'cg', 'gmres'):
                b = np.asarray(array(p, 'b'), dtype=float)
                if task == 'spsolve':
                    with warnings.catch_warnings():
                        warnings.simplefilter('error', spla.MatrixRankWarning)
                        try:
                            x = spla.spsolve(A.tocsc(), b)
                        except spla.MatrixRankWarning:
                            return 'SINGULAR', 'the matrix is singular (SuperLU found an exactly zero pivot)', {}
                    x = np.asarray(x, dtype=float)
                    if not np.isfinite(x).all():
                        return 'SINGULAR', 'the solve produced non-finite values: the matrix is singular to working precision', {}
                    rel = residual_vector(x, b)
                    result['relative_residual'] = rel
                    if rel > 1e-6:
                        return 'ILL_CONDITIONED', 'the direct solve leaves a relative residual of %.3g: the matrix is too ill-conditioned' % rel, {}
                    return 'OK', '', {'x': x}
                count = [0]

                def tick(_):
                    count[0] += 1
                if task == 'cg':
                    x, info = spla.cg(A, b, rtol=p['rtol'], maxiter=p['maxiter'], callback=tick)
                else:
                    x, info = spla.gmres(A, b, rtol=p['rtol'], restart=p['restart'], maxiter=p['maxiter'], callback=tick, callback_type='pr_norm')
                x = np.asarray(x, dtype=float)
                result['iterations'] = count[0]
                rel = residual_vector(x, b) if np.isfinite(x).all() else float('inf')
                result['relative_residual'] = rel
                # the residual a double-precision iteration can reach is bounded below by ~eps ||A|| ||x|| / ||b||; a tolerance tighter than that is not attainable
                floor = 1e3 * np.finfo(float).eps * float(abs(A).sum(axis=0).max()) * float(np.linalg.norm(x)) / max(float(np.linalg.norm(b)), 1e-300) if np.isfinite(x).all() else 0.0
                if info != 0 or not rel <= max(10.0 * p['rtol'], floor, 1e-14):
                    why = 'the iteration did not reach the tolerance within maxiter' if info > 0 else 'illegal input or breakdown (info %d)' % info if info < 0 else 'the true residual %.3g exceeds the tolerance despite the iteration reporting convergence' % rel
                    return 'NO_CONVERGENCE', why, {}
                return 'OK', '', {'x': x}
            k = p['k']
            kwargs = {'k': k, 'which': p['which'], 'maxiter': p.get('maxiter'), 'tol': p['rtol']}
            if p.get('ncv'):
                kwargs['ncv'] = p['ncv']
            try:
                if task == 'svds':
                    v0 = rng.uniform(-1.0, 1.0, min(m, n))
                    u, s, vt = spla.svds(A, v0=v0, **{key: val for key, val in kwargs.items() if key != 'maxiter' or val is not None})
                    order = np.argsort(-s)
                    s, u, vt = s[order], u[:, order], vt[order, :]
                    res = max(float(np.linalg.norm(A @ vt[i] - s[i] * u[:, i])) for i in range(len(s)))
                    result['max_residual'] = res
                    return 'OK', '', {'s': s, 'u': u, 'vt': vt}
                v0 = rng.uniform(-1.0, 1.0, n)
                if 'sigma' in p:
                    kwargs['sigma'] = p['sigma']
                kwargs = {key: val for key, val in kwargs.items() if key != 'maxiter' or val is not None}
                if task == 'eigsh':
                    w, v = spla.eigsh(A, v0=v0, **kwargs)
                    order = np.argsort(w)
                else:
                    w, v = spla.eigs(A, v0=v0, **kwargs)
                    order = np.lexsort((w.imag, w.real))
                w, v = w[order], v[:, order]
                res = max(float(np.linalg.norm(A @ v[:, i] - w[i] * v[:, i])) / max(abs(w[i]), 1.0) for i in range(len(w)))
                result['max_residual'] = res
                return 'OK', '', {'w': w, 'v': v}
            except spla.ArpackNoConvergence as exc:
                result['n_converged'] = int(len(exc.eigenvalues))
                return 'NO_CONVERGENCE', 'ARPACK converged to %d of %d requested eigenvalues within maxiter' % (len(exc.eigenvalues), k), {}
            except spla.ArpackError as exc:
                return 'NUMERICAL_FAILURE', 'ARPACK failed: %s' % str(exc)[:200], {}
    got = s7_guard(p, body)
    if got is None:
        return {'status': 'TIME_LIMIT', 'task': task}
    status, message, payload = got
    if status != 'OK':
        return dict(result, status=status, message=message)
    if task != 'svds' and task not in ('spsolve', 'cg', 'gmres') and result.get('max_residual', 0) > 1e-6:
        return dict(result, status='NUMERICAL_FAILURE', message='the eigenpairs leave a residual of %.3g' % result['max_residual'])
    if 'x' in payload:
        items.append(('x', payload['x']))
    elif task == 'svds':
        items += [('s', payload['s'])]
        if p['vectors']:
            items += [('u', payload['u']), ('vt', payload['vt'])]
    else:
        w_items, is_complex = s8b_complex_items(np, 'w', payload['w'])
        items += w_items
        result['complex'] = is_complex
        if p['vectors']:
            items += s8b_complex_items(np, 'v', payload['v'])[0]
    result['k'] = int(p['k']) if 'k' in p else None
    return s7_emit(np, p, result, items, 'la_sparse')


# -------------------------------------------------------------------------------------------------------- special_fn
def s8b_table():
    from scipy import special as sp
    import numpy as np

    def real_lambertw(z, k):
        w = sp.lambertw(z, k.astype(int))
        return np.where(w.imag == 0, w.real, np.nan)
    return {
        'erf': sp.erf, 'erfc': sp.erfc, 'erfcx': sp.erfcx, 'erfinv': sp.erfinv, 'erfcinv': sp.erfcinv,
        'gammainc': sp.gammainc, 'gammaincc': sp.gammaincc, 'gammaincinv': sp.gammaincinv, 'gammainccinv': sp.gammainccinv,
        'betainc': sp.betainc, 'betaincinv': sp.betaincinv, 'zeta': sp.zeta, 'polygamma': lambda n, x: sp.polygamma(n.astype(int), x),
        'lambertw': real_lambertw, 'expit': sp.expit, 'logit': sp.logit, 'log_ndtr': sp.log_ndtr, 'ndtri': sp.ndtri,
        'ellipk': sp.ellipk, 'ellipe': sp.ellipe, 'hyp2f1': sp.hyp2f1, 'jv': sp.jv, 'yv': sp.yv, 'iv': sp.iv, 'kv': sp.kv,
        'spherical_jn': lambda n, z: sp.spherical_jn(n.astype(int), z), 'struve': sp.struve, 'owens_t': sp.owens_t,
        'boxcox': sp.boxcox, 'inv_boxcox': sp.inv_boxcox, 'xlogy': sp.xlogy, 'xlog1py': sp.xlog1py}


def special_fn(p):
    import numpy as np
    from scipy import special as sp
    fn, names = p['fn'], p['arg_names']
    args = [np.asarray(array(p, 'arg_' + name), dtype=float) for name in names]
    s7_cap(all(a.size <= 200000 for a in args), 'argument length')
    sizes = [a.size for a, name in zip(args, names) if p['arg_vector'][name]]
    n = sizes[0] if sizes else 1
    if any(s != n for s in sizes):
        raise BadData('vector arguments must have equal length (scalars broadcast)', 'BAD_DATA')
    args = [np.broadcast_to(a.reshape(-1), (n,)).astype(float) if p['arg_vector'][name] else np.full(n, float(a.reshape(-1)[0])) for a, name in zip(args, names)]
    result = {'status': 'OK', 'fn': fn, 'n': int(n)}

    def body():
        with np.errstate(all='ignore'):
            if fn == 'logsumexp':
                return np.asarray([sp.logsumexp(args[0])])
            if fn == 'airy':
                return sp.airy(args[0])
            return np.asarray(s8b_table()[fn](*args), dtype=float)
    out = s7_guard(p, body)
    if out is None:
        return {'status': 'TIME_LIMIT', 'fn': fn}
    outputs = list(out) if fn == 'airy' else [out]
    keys = ['ai', 'aip', 'bi', 'bip'] if fn == 'airy' else ['y']
    bad = np.zeros(n, dtype=bool)
    nan = np.zeros(n, dtype=bool)
    for o in outputs:
        bad |= ~np.isfinite(o)
        nan |= np.isnan(o)
    if bad.any():
        first = int(np.flatnonzero(bad)[0])
        return dict(result, status='DOMAIN_ERROR' if nan.any() else 'NON_FINITE', n_bad=int(bad.sum()), n_nan=int(nan.sum()), first_index=first,
                    first_args={name: float(a[first]) for name, a in zip(names, args)},
                    message='%d of %d results are %s (first at index %d); special-function poles, overflow and arguments outside the domain are statuses, never values' % (
                        int(bad.sum()), n, 'undefined (NaN)' if nan.any() else 'infinite', first))
    return s7_emit(np, p, result, list(zip(keys, outputs)), 'special_fn')


def s8b_operation(name):
    base = name[:-3] if name.endswith('_to') else name
    return {'la_matrix_fn': la_matrix_fn, 'la_sparse': la_sparse, 'special_fn': special_fn}[base]
