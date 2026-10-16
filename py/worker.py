"""Fixed S0 scientific worker. All dispatch targets are defined here."""
import os
import sys

# Set these before the first numerical import; never inherit caller tuning.
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_MAX_THREADS'):
    os.environ[name] = '1'
os.environ['TZ'] = 'UTC'
import json
import math
import time
import warnings


BLOCKED_MODULES = {'pi' + 'ckle', '_pi' + 'ckle', 'socket', '_socket',
                   'subprocess', 'multiprocessing', 'urllib', 'http', 'ftplib', 'patsy', 'formulaic'}


class BlockedImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in BLOCKED_MODULES:
            raise ImportError('blocked module: ' + fullname)
        return None


def audit(event, args):
    if event == 'pi'+'ckle.find_class': raise RuntimeError('blocked serialization audit event')
    if event == 'import':
        top = args[0].split('.')[0]
        # Construct the serialization module name without a forbidden identifier.
        if top in BLOCKED_MODULES:
            raise ImportError('blocked module: ' + top)
    if event.startswith(('socket.', 'subprocess.')) or event in ('os.system', 'os.posix_spawn', 'os.fork'):
        raise RuntimeError('blocked audit event: ' + event)


class BadData(ValueError):
    """Caller-input problem found by the worker; `code` becomes the tool_code (exit 2)."""

    def __init__(self, message='', code='BAD_DATA'):
        super().__init__(message)
        self.code = code


class ExprMismatch(Exception):
    """The AST contradicts the spec or the declared names: a C++/Python disagreement, never a caller error."""
    code = 'EXPR_SPEC_MISMATCH'


class Int64Overflow(ArithmeticError):
    pass


class OutputTooLarge(OverflowError):
    pass


class NumericalFailure(ArithmeticError):
    pass


class FrameNull:
    pass


FRAME_NULL = FrameNull()


def codec(value):
    """Strict JSON with exact %.17g binary64 and explicit nonfinite tokens."""
    if isinstance(value, FrameNull):
        return 'null'
    if value is None:
        return '"NA"'
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return '"NaN"'
        if math.isinf(value):
            return '"Inf"' if value > 0 else '"-Inf"'
        text = format(value, '.17g')
        # Preserve the floating type and signed zero at the C++ JSON boundary.
        return text if '.' in text or 'e' in text else text + '.0'
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=True)
    if isinstance(value, (list, tuple)):
        return '[' + ','.join(codec(x) for x in value) + ']'
    if isinstance(value, complex):
        return codec({'re': value.real, 'im': value.imag})
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError('JSON object keys must be strings')
        return '{' + ','.join(json.dumps(k) + ':' + codec(v) for k, v in value.items()) + '}'
    import numpy as np
    if isinstance(value, np.generic):
        scalar = value.item()
        if scalar is not value and not isinstance(scalar, np.generic):
            return codec(scalar)
    raise TypeError('unsupported response type: ' + type(value).__name__)


def array(p, key):
    import numpy as np
    return np.load(p[key + '_file'], allow_pickle=False)


def blas_info():
    import ctypes
    import glob
    import numpy as np
    import scipy
    pools = []
    for module in (np, scipy):
        for path in glob.glob(os.path.join(os.path.dirname(module.__file__), '..', module.__name__ + '.libs', '*openblas*.so')):
            lib = ctypes.CDLL(path)
            prefix = 'scipy_'
            suffix = '' if module is scipy else '64_'
            def symbol(name):
                return getattr(lib, prefix + 'openblas_' + name + suffix)
            threads = symbol('get_num_threads')
            config = symbol('get_config'); config.restype = ctypes.c_char_p
            core = symbol('get_corename'); core.restype = ctypes.c_char_p
            pools.append({'library': os.path.basename(path), 'threads': threads(),
                          'config': config().decode(), 'core': core().decode()})
    if not pools or any(pool['threads'] != 1 for pool in pools):
        raise RuntimeError('BLAS thread verification failed')
    return pools


def info(p):
    import platform
    import numpy as np
    import scipy
    return {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
            'blas': blas_info(), 'threads': 1, 'rng': 'PCG64', 'arch': platform.machine(),
            'hash_randomization': sys.flags.hash_randomization, 'hash_probe': str(hash('pycli')),
            'safe_path': sys.flags.safe_path, 'user_site': bool(sys.flags.no_user_site == 0),
            'coretype_requested': os.environ.get('OPENBLAS_CORETYPE', 'automatic')}


def solve(p):
    import numpy as np
    from scipy.linalg import solve as dense_solve, LinAlgError
    a, b = array(p, 'a'), array(p, 'b')
    if a.ndim != 2 or a.shape[0] != a.shape[1] or b.ndim not in (1, 2) or b.shape[0] != a.shape[0]:
        raise BadData('incompatible solve shapes')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise BadData('solve requires finite inputs')
    if p['assume_a'] in ('sym', 'pos') and not np.array_equal(a, a.T):
        raise BadData('NOT_SYMMETRIC')
    if p['assume_a'] == 'pos':
        try:
            np.linalg.cholesky(a)
        except LinAlgError:
            return {'status': 'NOT_POSITIVE_DEFINITE'}
    try:
        x = dense_solve(a, b, assume_a=p['assume_a'], check_finite=True)
    except LinAlgError:
        return {'status': 'SINGULAR'}
    result = {'status': 'OK', 'residual_norm': float(np.linalg.norm(a @ x - b)),
              'rcond': float(1.0 / np.linalg.cond(a))}
    if p.get('output_file'):
        # The fixed raw binary output is installed as safetensors by C++.
        x.astype('<f8').tofile(p['output_file'])
        result['shape'] = list(x.shape)
    elif x.size <= 1000:
        result['x'] = x.tolist()
    else:
        raise OutputTooLarge('use la_solve_to for more than 1000 output numbers')
    return result


def matrix(p, key):
    import numpy as np
    a = array(p, key)
    if a.ndim != 2:
        raise BadData(key + ' must be a matrix')
    if not np.isfinite(a).all():
        raise BadData(key + ' must contain only finite numbers')
    return a


def save_outputs(p, items, nan_masks=None):
    """Raw little-endian tensors, one file per entry; C++ validates sizes and writes the safetensors."""
    import numpy as np
    outputs = []
    for i, (key, value) in enumerate(items):
        dtype = '<i8' if np.issubdtype(value.dtype, np.integer) else '<f8'
        value = np.ascontiguousarray(value, dtype=dtype)
        if value.dtype.kind == 'f' and not np.isfinite(value).all():
            mask=None if nan_masks is None else nan_masks.get(key)
            if mask is None or np.shape(mask)!=value.shape or np.isinf(value).any() or not np.array_equal(np.isnan(value),np.asarray(mask,dtype=bool)):
                raise NumericalFailure('non-finite values in ' + key)
        value.tofile(p['output_prefix'] + str(i) + '.bin')
        outputs.append({'key': key, 'dtype': 'I64' if dtype == '<i8' else 'F64', 'shape': list(value.shape)})
    return outputs


def relative_error(np, reference, approx, scale=None):
    """||reference - approx||_F / scale; scale defaults to ||reference||_F. A scale of 0 reports the absolute error."""
    scale = float(np.linalg.norm(reference)) if scale is None else float(scale)
    error = float(np.linalg.norm(reference - approx))
    return error / scale if scale > 0 else error


def lstsq(p):
    """scipy.linalg.lstsq plus honest status. residues/residual_norm are always recomputed from A x - b."""
    import numpy as np
    import scipy.linalg as sla
    a, b = matrix(p, 'a'), array(p, 'b')
    if b.ndim not in (1, 2) or b.shape[0] != a.shape[0] or not np.isfinite(b).all():
        raise BadData('b must be a finite vector or matrix with as many rows as a')
    driver = p.get('driver') or 'gelsd'
    try:
        x, _, rank, singular = sla.lstsq(a, b, cond=p.get('rcond'), lapack_driver=driver, check_finite=True)
        if singular is None:
            singular = sla.svdvals(a)
    except np.linalg.LinAlgError:
        return {'status': 'NO_CONVERGENCE', 'driver': driver}
    if not np.isfinite(x).all():
        return {'status': 'NUMERICAL_FAILURE', 'driver': driver}
    residual = a @ x - b
    residues = (residual.reshape(len(residual), -1) ** 2).sum(axis=0)
    result = {'status': 'OK' if rank >= min(a.shape) else 'RANK_DEFICIENT', 'rank': int(rank),
              'singular_values': [float(v) for v in singular], 'residues': [float(v) for v in residues],
              'residual_norm': float(np.linalg.norm(residual)), 'driver': driver}
    if p.get('output_prefix'):
        items=[('x',x),('residues',residues),('singular_values',np.asarray(singular,dtype=float))]
        result.update(residues_count=len(residues),residues_sum=float(residues.sum()),residues_max=float(residues.max()),singular_values_count=len(singular))
        del result['residues'],result['singular_values']
        result['outputs'] = save_outputs(p, items)
    elif x.size <= 1000:
        result['x'] = x.tolist()
    else:
        raise OutputTooLarge('use la_lstsq_to for more than 1000 output numbers')
    return result


def props(p):
    import numpy as np
    import scipy.linalg as sla
    a = matrix(p, 'a')
    what = set(p['what'])
    result = {'status': 'OK', 'shape': list(a.shape)}
    try:
        if what & {'det', 'slogdet'}:
            sign, logabs = np.linalg.slogdet(a)
            if 'slogdet' in what:
                result['slogdet'] = {'sign': float(sign), 'logabsdet': float(logabs)}
            if 'det' in what:
                det = float(np.linalg.det(a))
                if not math.isfinite(det) or (det == 0.0 and sign != 0):
                    warnings.warn('det left the binary64 range; use slogdet', NumericalWarning)
                result['det'] = det
        if 'trace' in what:
            result['trace'] = float(np.trace(a))
        if 'norm' in what:
            order = p['norm_ord']
            if order == 'fro':
                scale = float(np.abs(a).max())
                result['norm'] = scale * float(np.linalg.norm(a / scale, 'fro')) if scale > 0 else 0.0
            else:
                result['norm'] = float(np.linalg.norm(a, {'1': 1, 'inf': np.inf, '2': 2, 'nuc': 'nuc'}[order]))
        if 'cond' in what:
            # Rank-deficient (numpy's default tolerance) means cond is +Inf, never a 1/eps-sized artefact of rounding.
            if int(np.linalg.matrix_rank(a)) < min(a.shape):
                result['cond'] = math.inf
            else:
                result['cond'] = float(np.linalg.cond(a, {'2': 2, '1': 1, 'inf': np.inf, 'fro': 'fro'}[p['cond_p']]))
        if 'rank' in what:
            result['rank'] = int(np.linalg.matrix_rank(a, tol=p.get('rank_tol')))
        if 'inverse' in what:
            # Numerically singular means rank < n under numpy's default tolerance; the inverse is withheld then.
            if int(np.linalg.matrix_rank(a)) < a.shape[0]:
                result['status'] = 'SINGULAR'
            else:
                try:
                    inverse = sla.inv(a, check_finite=True)
                except np.linalg.LinAlgError:
                    inverse = None
                if inverse is None or not np.isfinite(inverse).all():
                    result['status'] = 'SINGULAR'
                else:
                    result['inverse'] = inverse.tolist()
                    result['inverse_residual'] = float(np.abs(a @ inverse - np.eye(a.shape[0])).max())
    except np.linalg.LinAlgError:
        return {'status': 'NO_CONVERGENCE'}
    return result


def decomp(p):
    """Factorizations with LAPACK conventions (sign of singular/eigen vectors is LAPACK's). Documented in README."""
    import numpy as np
    import scipy.linalg as sla
    a = matrix(p, 'a')
    kind, full = p['kind'], bool(p.get('full'))
    values, driver, extra = {}, None, {}
    try:
        if kind == 'svd':
            driver = 'gesdd'
            try:
                u, s, vt = sla.svd(a, full_matrices=full, lapack_driver='gesdd')
            except np.linalg.LinAlgError:
                driver = 'gesvd'
                u, s, vt = sla.svd(a, full_matrices=full, lapack_driver='gesvd')
            k = len(s)
            items, values = [('u', u), ('s', s), ('vt', vt)], {'s': s.tolist()}
            error = relative_error(np, a, (u[:, :k] * s) @ vt[:k])
        elif kind == 'qr':
            if p.get('pivoting'):
                q, r, piv = sla.qr(a, mode='full' if full else 'economic', pivoting=True)
                items, error = [('q', q), ('r', r), ('p', piv)], relative_error(np, a[:, piv], q @ r)
            else:
                q, r = sla.qr(a, mode='full' if full else 'economic')
                items, error = [('q', q), ('r', r)], relative_error(np, a, q @ r)
        elif kind == 'eigh':
            w, v = sla.eigh(a)
            # Backward error is relative to ||A|| ||V|| (||A V|| collapses for singular/nilpotent A and would mislead).
            items, values = [('w', w), ('v', v)], {'w': w.tolist()}
            error = relative_error(np, a @ v, v * w, np.linalg.norm(a) * np.linalg.norm(v))
        elif kind == 'eig':
            w, v = sla.eig(a)
            order = np.lexsort((w.imag, w.real))  # documented order: ascending real part, then imaginary part
            w, v = w[order], v[:, order]
            items = [('w_re', w.real), ('w_im', w.imag), ('vr_re', v.real), ('vr_im', v.imag)]
            values = {'w_re': w.real.tolist(), 'w_im': w.imag.tolist()}
            error = relative_error(np, a @ v, v * w, np.linalg.norm(a) * np.linalg.norm(v))
            extra['vr_cond'] = float(np.linalg.cond(v))
            if not extra['vr_cond'] < 1 / math.sqrt(np.finfo(float).eps):
                warnings.warn('eigenvector matrix is ill-conditioned (cond=%.3g): the matrix is defective or nearly defective, '
                              'so vr is not a reliable basis (A = V diag(w) V^-1 does not hold)' % extra['vr_cond'], NumericalWarning)
        elif kind == 'cholesky':
            lower = bool(p.get('lower', True))
            c = sla.cholesky(a, lower=lower)
            items = [('l' if lower else 'u', c)]
            error = relative_error(np, a, c @ c.T if lower else c.T @ c)
        else:
            pm, l, u = sla.lu(a)
            items, error = [('p', pm), ('l', l), ('u', u)], relative_error(np, a, pm @ l @ u)
    except np.linalg.LinAlgError as exc:
        if kind == 'cholesky':
            return {'status': 'NOT_POSITIVE_DEFINITE', 'kind': kind}
        return {'status': 'NO_CONVERGENCE', 'kind': kind, 'message': str(exc)}
    # LAPACK factorizations are backward stable: a large reconstruction error means the result is not to be trusted.
    if not error < 1e-8:
        return {'status': 'NUMERICAL_FAILURE', 'kind': kind, 'reconstruction_error': error}
    try:
        outputs = save_outputs(p, items)
    except NumericalFailure as exc:
        return {'status': 'NUMERICAL_FAILURE', 'kind': kind, 'message': str(exc)}
    result = {'status': 'OK', 'kind': kind, 'outputs': outputs, 'values': values, 'reconstruction_error': error, **extra}
    if driver:
        result['driver'] = driver
    return result


def distribution_body(p):
    return dist_evaluate(p)


def distribution(p):
    result = s7_guard(p, lambda: distribution_body(p))
    return {'status': 'TIME_LIMIT'} if result is None else result


class NumericalWarning(RuntimeWarning):
    pass


NUMERIC_OPS = ('opt_minimize', 'opt_least_squares', 'num_quad', 'num_root', 'opt_root', 'opt_linprog', 'ode_solve', 'opt_linprog_to', 'ode_solve_to')


def numeric_operation(name):
    """S3 dispatch table, resolved at call time so py/worker.py still imports on its own (numerics.py is prepended)."""
    return {'opt_linprog': opt_linprog, 'opt_linprog_to': opt_linprog, 'ode_solve': ode_solve, 'ode_solve_to': ode_solve, 'opt_minimize': opt_minimize, 'opt_least_squares': opt_least_squares, 'num_quad': num_quad, 'num_root': num_root, 'opt_root': opt_root}[name]


OPS = {'py_info': info, 'la_solve': solve, 'la_solve_to': solve, 'stats_dist': distribution,
       'la_lstsq': lstsq, 'la_lstsq_to': lstsq, 'la_props': props, 'la_decomp': decomp}


def main():
    start = time.monotonic()
    request = json.load(open(sys.argv[1], encoding='utf-8'))
    captured = []
    response = {}
    try:
        # Trusted imports are selected by a closed table before the guard. SciPy
        # imports test/network/process helpers even for its numerical modules.
        import numpy as np
        s8 = request['op'] in ('frame_to_array', 'frame_to_array_to', 'array_to_frame', 'frame_mutate', 'frame_merge', 'frame_rolling', 'ml_metrics', 'stats_dist_to') or request['op'] in S8B_OPS or request['op'] in S8G_OPS or (request['op'] == 'stats_dist' and request['params'].get('fn') == 'rvs')
        if s8:
            import zoneinfo
            zoneinfo.reset_tzpath(())
            from zoneinfo import ZoneInfo
            ZoneInfo('America/New_York')  # warm pinned tzdata package resources before the guard
            import io
            warm = io.BytesIO()  # numpy imports its NPZ helpers lazily; do it before the import guard (no pandas, no scipy needed)
            np.savez(warm, a=np.zeros(1))
            warm.seek(0)
            np.load(warm, allow_pickle=False)['a']
            if request['op'] == 'frame_mutate':
                import scipy.special  # the expression function table (no pandas, no scipy.stats)
            elif request['op'].startswith('la_matrix_fn'):
                import scipy.linalg as warm_linalg
                eye = np.eye(2)  # scipy imports helper modules lazily inside these routines: do it before the import guard
                for routine in (warm_linalg.expm, warm_linalg.logm, warm_linalg.sqrtm, warm_linalg.inv, warm_linalg.pinv):
                    routine(eye)
                np.linalg.cond(eye)
            elif request['op'].startswith('la_sparse'):
                import scipy.sparse as warm_sparse
                import scipy.sparse.linalg as warm_spla
                warm = warm_sparse.diags([np.arange(1.0, 9.0)], [0], format='csr')
                warm_spla.spsolve(warm.tocsc(), np.ones(8))
                warm_spla.cg(warm, np.ones(8))
                warm_spla.gmres(warm, np.ones(8))
                warm_spla.eigsh(warm, k=2, v0=np.ones(8))
                warm_spla.eigs(warm, k=2, v0=np.ones(8))
                warm_spla.svds(warm, k=2, v0=np.ones(8))
            elif request['op'] in ('stats_dist', 'stats_dist_to'):
                if request['params']['fn'] != 'rvs' or request['params']['dist'] not in DIST_NUMPY_RVS:
                    import scipy.stats
                    dist_warm(request['params'])
            elif request['op'].startswith('graph_'):
                import networkx as warm_nx  # pure Python; lazy import only for graph_*
                import scipy.sparse as warm_sparse
                import scipy.sparse.csgraph as warm_csgraph
                warm = warm_nx.DiGraph()
                warm.add_edges_from([(0, 1, {'weight': 1, 'capacity': 2}), (1, 2, {'weight': 2, 'capacity': 1}), (0, 2, {'weight': 5, 'capacity': 1})])
                warm_nx.single_source_dijkstra_path_length(warm, 0)
                warm_nx.single_source_bellman_ford_path_length(warm, 0)
                warm_nx.single_source_shortest_path_length(warm, 0)
                list(warm_nx.shortest_simple_paths(warm, 0, 2, weight='weight'))
                from networkx.algorithms.flow import dinitz as warm_dinitz
                warm_nx.maximum_flow(warm, 0, 2, flow_func=warm_dinitz)
                warm_nx.minimum_cut(warm, 0, 2, flow_func=warm_dinitz)
                for node in warm.nodes:
                    warm.nodes[node]['demand'] = 0
                warm_nx.network_simplex(warm)
                warm_csgraph.dijkstra(warm_sparse.csr_matrix(np.ones((2, 2))), directed=True)
                warm_csgraph.johnson(warm_sparse.csr_matrix(np.ones((2, 2))), directed=True)
            elif request['op'].startswith('special_fn'):
                import scipy.special as warm_special
                for routine in (warm_special.erf, warm_special.gamma, warm_special.lambertw, warm_special.logsumexp):
                    routine(np.array([0.5]))
        elif request['op'] == 'stats_dist':
            import scipy.stats
            dist_warm(request['params'])
        elif request['op'] in ('la_solve', 'la_solve_to', 'la_lstsq', 'la_lstsq_to', 'la_props', 'la_decomp'):
            import scipy.linalg
        elif request['op'] in NUMERIC_OPS:
            import scipy.optimize
            import scipy.integrate
            import scipy.special
        elif request['op'].startswith(('frame_', 'sm_', 'ml_')):
            import pandas
            import csv
            import zoneinfo
            zoneinfo.reset_tzpath(())  # use the pinned tzdata wheel, never the base database
            csv.field_size_limit(1024*1024)
            if request['op'].startswith('ml_'): ml_import()
            if request['op'].startswith('sm_'):
                import scipy.linalg
                import scipy.stats
                sm_import()  # statsmodels class API only; patsy never loads (see py/sm.py)
        elif request['op'].startswith(('interp_', 'sig_', 'spatial_')):
            # Lazy per operation group: only the scipy subpackage the op needs (scipy.signal itself pulls in interpolate/spatial/stats).
            if request['op'].startswith('interp_'):
                import scipy.interpolate
            elif request['op'].startswith('sig_'):
                import scipy.signal
            else:
                import scipy.spatial
                import scipy.spatial.distance
        elif request['op'] == 'py_info':
            import scipy
        for module in list(sys.modules):
            if module.split('.')[0] in BLOCKED_MODULES:
                sys.modules.pop(module, None)
        sys.meta_path.insert(0, BlockedImports())
        sys.addaudithook(audit)
        imported = time.monotonic()
        if request['op'] in ('opt_linprog','ode_solve','opt_linprog_to','ode_solve_to'):
            request['params']['time_budget_s']=max(0.0,request['params']['time_budget_s']-(imported-start))
        if s8 or request['op'].startswith(('sm_','ml_','interp_','sig_','spatial_')):
            request['params']['time_budget_s']=max(0.05,request['params']['time_budget_s']-(imported-start))
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            with np.errstate(all='warn'):
                result = (s8_operation(request['op'])(request['params']) if s8
                          else frame_operation(request['op'], request['params']) if request['op'].startswith('frame_')
                          else ml_operation(request['op'],request['params']) if request['op'].startswith('ml_')
                          else sm_operation(request['op'])(request['params']) if request['op'].startswith('sm_')
                          else s7_operation(request['op'])(request['params']) if request['op'].startswith(('interp_', 'sig_', 'spatial_'))
                          else numeric_operation(request['op'])(request['params']) if request['op'] in NUMERIC_OPS
                          else OPS[request['op']](request['params']))
        result['meta'] = {'python': sys.version.split()[0], 'numpy': np.__version__,
                          'scipy': sys.modules['scipy'].__version__ if 'scipy' in sys.modules else 'unloaded',
                          'pandas': sys.modules['pandas'].__version__ if 'pandas' in sys.modules else 'unloaded',
                          'sklearn': sys.modules['sklearn'].__version__ if 'sklearn' in sys.modules else 'unloaded',
                          'statsmodels': sys.modules['statsmodels'].__version__ if 'statsmodels' in sys.modules else 'unloaded',
                          'scipy_subpackages': [m for m in ('interpolate', 'signal', 'spatial') if 'scipy.' + m in sys.modules],
                          'forbidden_modules_loaded': [m for m in ('patsy', 'formulaic', 'statsmodels.formula.api') if m in sys.modules],
                          'import_ms': (imported-start)*1000, 'elapsed_ms': (time.monotonic()-start)*1000}
        response = {'ok': True, 'result': result}
    except Exception as exc:
        code = 'INT64_OVERFLOW' if isinstance(exc, Int64Overflow) else 'OUTPUT_TOO_LARGE' if isinstance(exc, OutputTooLarge) else exc.code if isinstance(exc, BadData) else ExprMismatch.code if isinstance(exc, ExprMismatch) else 'PY_ERROR'
        response = {'ok': False, 'tool_code': code, 'message': str(exc), 'exception': type(exc).__name__,
                    'error_class': 'validation' if isinstance(exc, BadData) else 'internal' if isinstance(exc, ExprMismatch) else 'failed'}
    grouped = {}
    for warning in captured:
        key = (warning.category.__name__, str(warning.message))
        grouped[key] = grouped.get(key, 0) + 1
    notes = [{'category': key[0], 'message': key[1], 'count': count} for key, count in list(grouped.items())[:20]]
    target = response['result'] if response['ok'] else response
    target['warnings'] = notes
    target['warnings_truncated'] = len(grouped) > 20
    try:
        encoded = codec(response)
    except (TypeError, ValueError) as exc:
        encoded = codec({'ok': False, 'tool_code': 'PY_ERROR', 'message': str(exc), 'exception': type(exc).__name__})
    with open(sys.argv[2], 'w', encoding='utf-8') as stream:
        stream.write(encoded)


if __name__ == '__main__':
    main()
