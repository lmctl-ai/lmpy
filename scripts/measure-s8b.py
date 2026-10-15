"""S8b timings through the production shim: wall time plus the worker's own import/elapsed milliseconds (meta), one small call per operation.
Usage: python3 scripts/measure-s8b.py pycli:runtime > docs/measurement-s8b-shim.json"""
import json, math, pathlib, statistics, subprocess, sys, tempfile, time
root = pathlib.Path(__file__).resolve().parents[1]
image = sys.argv[1]


def summary(values):
    values = sorted(values)
    return {'p50_ms': round(statistics.median(values), 3), 'p95_ms': round(values[min(len(values) - 1, int(.95 * len(values)))], 3), 'n': len(values)}


results = {'image_bytes': int(subprocess.check_output(['docker', 'image', 'inspect', image, '--format', '{{.Size}}']))}
n = 200
series = []
state = 7
for _ in range(n):
    state = (75 * state + 74) % 65537
    series.append(state / 65537 - 0.5)
with tempfile.TemporaryDirectory(dir='.') as directory:
    work = pathlib.Path(directory)
    shim = work / 'pycli'
    shim.write_text(pathlib.Path('bin/pycli').read_text().replace('pycli:runtime', image))
    shim.chmod(0o755)
    rows, cols, vals = [], [], []
    for i in range(100):
        rows.append(i); cols.append(i); vals.append(2.0)
        if i + 1 < 100:
            rows += [i, i + 1]; cols += [i + 1, i]; vals += [-1.0, -1.0]
    cases = [('py_info', 'py_info', {}),
             ('la_matrix_fn_expm_8x8', 'la_matrix_fn', {'a': [[(i * j) % 5 / 10.0 for j in range(8)] for i in range(8)], 'fn': 'expm'}),
             ('la_sparse_cg_100', 'la_sparse', {'task': 'cg', 'shape': [100, 100], 'row': rows, 'col': cols, 'val': vals, 'b': [1.0] * 100}),
             ('la_sparse_eigsh_100', 'la_sparse', {'task': 'eigsh', 'shape': [100, 100], 'row': rows, 'col': cols, 'val': vals, 'k': 3, 'which': 'LA', 'seed': 1}),
             ('special_fn_gammainc_100', 'special_fn', {'fn': 'gammainc', 'args': {'a': 2.5, 'x': [i / 10.0 for i in range(1, 101)]}}),
             ('opt_root_circle_line', 'opt_root', {'equations': ['x^2 + y^2 - 4', 'x - y'], 'start': {'x': 1.0, 'y': 1.0}}),
             ('sm_tsa_tools_adfuller_200', 'sm_tsa_tools', {'columns': {'y': series}, 'series': 'y', 'test': 'adfuller'}),
             ('sm_tsa_tools_ljung_box_200', 'sm_tsa_tools', {'columns': {'y': series}, 'series': 'y', 'test': 'ljung_box', 'lags': [5, 10]})]
    for name, op, params in cases:
        wall, imports, elapsed = [], [], []
        for _ in range(10):
            start = time.monotonic()
            out = json.loads(subprocess.check_output([str(shim), '--input-json', json.dumps({'op': op, 'params': params})]))
            wall.append((time.monotonic() - start) * 1000)
            assert 'error' not in out, out
            imports.append(out['meta']['import_ms'])
            elapsed.append(out['meta']['elapsed_ms'])
        results[name] = {'wall': summary(wall), 'worker_import_ms': summary(imports), 'worker_elapsed_ms': summary(elapsed)}
print(json.dumps(results, indent=2))
