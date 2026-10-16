"""S8b: matrix functions (la_matrix_fn), sparse linear algebra (la_sparse) and special functions (special_fn) on fixed scipy.linalg /
scipy.sparse.linalg / scipy.special paths. Assembled after s8.py (shares array, save_outputs, s7_emit, s7_guard, s8_read, BadData).
The C++ front end (src/s8b.hpp) has validated every enum, name, shape and cap before this runs; the worker repeats what a corrupted request could break."""
S8B_OPS = ('la_matrix_fn', 'la_matrix_fn_to', 'la_sparse', 'la_sparse_to', 'special_fn', 'special_fn_to')
S8B_MATRIX_RESIDUAL = 1e-6


def s8b_fro(np, x):
    scale = float(np.max(np.abs(x))) if np.size(x) else 0.0
    return np.longdouble(0) if scale == 0 else np.sqrt(np.sum(np.abs(np.asarray(x, dtype=np.clongdouble) / scale) ** 2)) * np.longdouble(scale)


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
    norm_a = s8b_fro(np, a) or np.longdouble(1)

    def body():
        with np.errstate(all='ignore'):
            if fn == 'inv':
                cond = float(np.linalg.cond(a))
                result['condition_number'] = cond
                if not np.isfinite(cond) or cond >= 1.0 / eps:
                    return 'SINGULAR', 'the matrix is singular to working precision (condition number %.3g)' % cond, None
                y = sla.inv(a)
                identity = s8b_fro(np, a @ y - np.eye(n))
                result['identity_residual'] = float(identity / np.sqrt(n))
                residual = float(np.longdouble(identity) / (np.longdouble(norm_a) * s8b_fro(np, y) + np.sqrt(n)))
                if cond > 1 / np.sqrt(eps):
                    warnings.warn('inverse is ill-conditioned (condition number %.3g); forward digits are limited by conditioning' % cond, NumericalWarning)
            elif fn == 'pinv':
                y, rank = sla.pinv(a, rtol=p.get('rtol'), return_rank=True)
                result['rank'] = int(rank)
                result['truncation'] = float(s8b_fro(np, a @ y @ a - a) / norm_a)
                ay, ya = a @ y, y @ a
                residual = max(s8b_fro(np, y @ a @ y - y) / max(s8b_fro(np, y), 1e-300), s8b_fro(np, ay - ay.T) / max(s8b_fro(np, ay), 1e-300), s8b_fro(np, ya - ya.T) / max(s8b_fro(np, ya), 1e-300))
                if rank == min(a.shape):
                    residual = max(residual, result['truncation'])
            elif fn == 'expm':
                y = sla.expm(a)
                residual = None
            elif fn == 'logm':
                cond = float(np.linalg.cond(a))
                result['condition_number'] = cond
                if not np.isfinite(cond) or cond >= 1.0 / eps:
                    return 'SINGULAR', 'the matrix is singular to working precision: a finite logarithm cannot be certified', None
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
        result['residual'] = float(residual)
        if not np.isfinite(residual) or residual > S8B_MATRIX_RESIDUAL:
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
        row, col = picked[0].astype(np.int64), picked[1].astype(np.int64)
        i = index[p['val']]
        val = s8_float_column(np, cmeta[p['val']], raws[i], masks[i], p['val'])[0]
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
    a_scale = float(abs(A).max())
    unit = A / a_scale if a_scale else A.copy()
    norm_unit = float(abs(unit).sum(axis=0).max())
    eigen_scale = a_scale or 1.0
    if a_scale and ((A.data != 0) & (unit.data == 0)).any():
        return dict(result, status='NUMERICAL_FAILURE', message='matrix dynamic range loses nonzero entries during normalisation')
    if task == 'spsolve' or 'sigma' in p:
        s7_cap(n <= 2000 and n * A.nnz <= 20_000_000, 'SuperLU factorisation work; use cg or gmres')

    def norm(v):
        scale = float(np.max(np.abs(v))) if np.size(v) else 0.0
        return float(np.linalg.norm(v / scale) * scale) if scale else 0.0

    def restored(z, bscale):
        return np.asarray(np.asarray(z, dtype=np.longdouble) * np.longdouble(bscale) / np.longdouble(a_scale), dtype=float)

    def residual_vector(x, bunit, bscale):
        # Recheck the returned binary64 solution, including any rounding during unit restoration.
        z = np.asarray(np.asarray(x, dtype=np.longdouble) * np.longdouble(a_scale) / np.longdouble(bscale), dtype=float)
        if not np.isfinite(z).all():
            return float('inf'), z
        r = norm(unit @ z - bunit)
        return (r / norm(bunit) if np.any(bunit) else r), z

    def body():
        import warnings
        with np.errstate(all='ignore'):
            if task in ('spsolve', 'cg', 'gmres'):
                b = np.asarray(array(p, 'b'), dtype=float)
                if not a_scale:
                    return 'SINGULAR', 'all matrix entries are zero', {}
                bscale = float(np.max(np.abs(b))) or 1.0
                bunit = b / bscale
                if task == 'spsolve':
                    with warnings.catch_warnings():
                        warnings.simplefilter('error', spla.MatrixRankWarning)
                        try:
                            x = restored(spla.spsolve(unit.tocsc(), bunit), bscale)
                        except spla.MatrixRankWarning:
                            return 'SINGULAR', 'the matrix is singular (SuperLU found an exactly zero pivot)', {}
                    x = np.asarray(x, dtype=float)
                    if not np.isfinite(x).all():
                        return 'SINGULAR', 'the solve produced non-finite values: the matrix is singular to working precision', {}
                    rel, _ = residual_vector(x, bunit, bscale)
                    result['relative_residual'] = rel
                    if rel > 1e-6:
                        return 'ILL_CONDITIONED', 'the direct solve leaves a relative residual of %.3g: the matrix is too ill-conditioned' % rel, {}
                    return 'OK', '', {'x': x}
                count = [0]

                def tick(_):
                    count[0] += 1
                if task == 'cg':
                    x, info = spla.cg(unit, bunit, rtol=p['rtol'], maxiter=p['maxiter'], callback=tick)
                else:
                    x, info = spla.gmres(unit, bunit, rtol=p['rtol'], restart=p['restart'], maxiter=p['maxiter'], callback=tick, callback_type='pr_norm')
                x = restored(x, bscale)
                result['iterations'] = count[0]
                rel, z = residual_vector(x, bunit, bscale) if np.isfinite(x).all() else (float('inf'), x)
                result['relative_residual'] = rel
                floor = 1e3 * np.finfo(float).eps * norm_unit * norm(z) / (norm(bunit) or 1.0) if np.isfinite(z).all() else float('inf')
                result['attainable_floor'] = floor
                result['tolerance_met'] = bool(rel <= 10.0 * p['rtol'])
                if info != 0 or not result['tolerance_met']:
                    why = 'the true residual %.3g exceeds 10*rtol %.3g (solver info %d)' % (rel, 10*p['rtol'], info)
                    status = 'TOLERANCE_UNATTAINABLE' if info == 0 and rel <= floor else 'NO_CONVERGENCE'
                    return status, why, {}
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
                    res = max(max(norm(unit @ vt[i] - (s[i] / eigen_scale) * u[:, i]), norm(unit.T @ u[:, i] - (s[i] / eigen_scale) * vt[i])) / max(s[i] / eigen_scale, norm_unit, 1e-300) for i in range(len(s)))
                    res = max(res, norm(u.T @ u - np.eye(len(s))), norm(vt @ vt.T - np.eye(len(s))))
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
                res = max(norm(unit @ v[:, i] - (w[i] / a_scale) * v[:, i]) / max(abs(w[i]) / a_scale, norm_unit, 1e-300) for i in range(len(w))) if a_scale else 0.0
                result['max_residual'] = res
                return 'OK', '', {'w': w, 'v': v}
            except spla.ArpackNoConvergence as exc:
                result['n_converged'] = int(len(exc.eigenvalues))
                return 'NO_CONVERGENCE', 'ARPACK converged to %d of %d requested eigenvalues within maxiter' % (len(exc.eigenvalues), k), {}
            except spla.ArpackError as exc:
                return 'NUMERICAL_FAILURE', 'ARPACK failed: %s' % str(exc)[:200], {}
            except RuntimeError as exc:
                if 'sigma' in p and 'singular' in str(exc).lower():
                    return 'SINGULAR', 'A - sigma*I is singular; sigma coincides with an eigenvalue, choose a different shift', {}
                raise
    got = s7_guard(p, body)
    if got is None:
        return {'status': 'TIME_LIMIT', 'task': task}
    status, message, payload = got
    if status != 'OK':
        return dict(result, status=status, message=message)
    if task not in ('spsolve', 'cg', 'gmres') and (not np.isfinite(result.get('max_residual', 0)) or result.get('max_residual', 0) > max(1e-6, 10 * p['rtol']) or any(not np.isfinite(v).all() for v in payload.values())):
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
    if 'k' in p:
        result['k'] = int(p['k'])
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
    n_output = int(np.size(outputs[0]))
    result['n_output'] = n_output
    bad = np.zeros(n_output, dtype=bool)
    nan = np.zeros(n_output, dtype=bool)
    for o in outputs:
        bad |= ~np.isfinite(o)
        nan |= np.isnan(o)
    if bad.any():
        first = int(np.flatnonzero(nan if nan.any() else bad)[0])
        n_nan, n_inf = int(nan.sum()), int((bad & ~nan).sum())
        return dict(result, status='DOMAIN_ERROR' if nan.any() else 'NON_FINITE', n_bad=int(bad.sum()), n_nan=n_nan, n_inf=n_inf, first_index=first,
                    first_args={name: float(a[first]) for name, a in zip(names, args)},
                    message='%d of %d outputs are non-finite (%d NaN, %d infinite); first %s output at index %d' % (int(bad.sum()), n_output, n_nan, n_inf, 'NaN' if nan.any() else 'infinite', first))
    return s7_emit(np, p, result, list(zip(keys, outputs)), 'special_fn')


def s8b_operation(name):
    base = name[:-3] if name.endswith('_to') else name
    return {'la_matrix_fn': la_matrix_fn, 'la_sparse': la_sparse, 'special_fn': special_fn}[base]
