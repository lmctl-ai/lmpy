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
                   'subprocess', 'multiprocessing', 'urllib', 'http', 'ftplib'}


class BlockedImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in BLOCKED_MODULES:
            raise ImportError('blocked module: ' + fullname)
        return None


def audit(event, args):
    if event == 'import':
        top = args[0].split('.')[0]
        # Construct the serialization module name without a forbidden identifier.
        if top in BLOCKED_MODULES:
            raise ImportError('blocked module: ' + top)
    if event.startswith(('socket.', 'subprocess.')) or event in ('os.system', 'os.posix_spawn', 'os.fork'):
        raise RuntimeError('blocked audit event: ' + event)


class BadData(ValueError):
    pass


class OutputTooLarge(OverflowError):
    pass


def codec(value):
    """Strict JSON with exact %.17g binary64 and explicit nonfinite tokens."""
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


def distribution(p):
    from scipy.stats import norm, t
    families = {'norm': norm, 't': t}
    family = families[p['dist']]
    params = p['params']
    frozen = family(**params)
    functions = {'pdf': frozen.pdf, 'cdf': frozen.cdf, 'ppf': frozen.ppf, 'sf': frozen.sf}
    values = functions[p['fn']](array(p, 'x')).tolist()
    for i, (x, value) in enumerate(zip(array(p, 'x'), values)):
        if not math.isnan(float(x)) and math.isnan(value):
            warnings.warn('SciPy returned NaN for a finite/domain-valid distribution input', NumericalWarning)
    for i in p.get('missing', []):
        values[i] = 'NA'
    return {'values': values}


class NumericalWarning(RuntimeWarning):
    pass


OPS = {'py_info': info, 'la_solve': solve, 'la_solve_to': solve, 'stats_dist': distribution}


def main():
    start = time.monotonic()
    request = json.load(open(sys.argv[1], encoding='utf-8'))
    captured = []
    response = {}
    try:
        # Trusted imports are selected by a closed table before the guard. SciPy
        # imports test/network/process helpers even for its numerical modules.
        import numpy as np
        if request['op'] == 'stats_dist':
            import scipy.stats
        elif request['op'] in ('la_solve', 'la_solve_to'):
            import scipy.linalg
        elif request['op'] == 'py_info':
            import scipy
        for module in list(sys.modules):
            if module.split('.')[0] in BLOCKED_MODULES:
                sys.modules.pop(module, None)
        sys.meta_path.insert(0, BlockedImports())
        sys.addaudithook(audit)
        imported = time.monotonic()
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            with np.errstate(all='warn'):
                result = OPS[request['op']](request['params'])
        result['meta'] = {'python': sys.version.split()[0], 'numpy': np.__version__,
                          'scipy': sys.modules['scipy'].__version__ if 'scipy' in sys.modules else 'unloaded',
                          'import_ms': (imported-start)*1000, 'elapsed_ms': (time.monotonic()-start)*1000}
        response = {'ok': True, 'result': result}
    except Exception as exc:
        code = 'OUTPUT_TOO_LARGE' if isinstance(exc, OutputTooLarge) else 'BAD_DATA' if isinstance(exc, BadData) else 'PY_ERROR'
        response = {'ok': False, 'tool_code': code, 'message': str(exc), 'exception': type(exc).__name__}
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
