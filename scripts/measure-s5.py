"""S5 timings through the production shim: wall time plus the worker's own import/elapsed milliseconds (meta).
Usage: python3 scripts/measure-s5.py pycli:runtime > docs/measurement-s5-shim.json"""
import json, pathlib, statistics, subprocess, sys, tempfile, time
root = pathlib.Path(__file__).resolve().parents[1]
data = json.loads((root / 'tests/fixtures/s5-r-golden.json').read_text())['data']
image = sys.argv[1]


def summary(values):
    values = sorted(values)
    return {'p50_ms': round(statistics.median(values), 3), 'p95_ms': round(values[min(len(values) - 1, int(.95 * len(values)))], 3), 'n': len(values)}


columns = {k: data[k] for k in ('y', 'x1', 'x2', 'g')}
terms = [{'const': True}, {'var': 'x1'}, {'var': 'x2'}, {'cat': 'g', 'ref': 'b'}, {'interact': ['x1', 'x2']}]
results = {'image_bytes': int(subprocess.check_output(['docker', 'image', 'inspect', image, '--format', '{{.Size}}']))}
with tempfile.TemporaryDirectory(dir='.') as directory:
    work = pathlib.Path(directory)
    shim = work / 'pycli'
    shim.write_text(pathlib.Path('bin/pycli').read_text().replace('pycli:runtime', image))
    shim.chmod(0o755)
    counter = [0]

    def unique():
        counter[0] += 1
        return str(work / f'out{counter[0]}.safetensors')
    cases = [('py_info', 'py_info', {}),
             ('sm_ols_40x6', 'sm_ols', {'columns': columns, 'y': 'y', 'terms': terms}),
             ('sm_ols_to_40x6', 'sm_ols_to', lambda: {'columns': columns, 'y': 'y', 'terms': terms, 'out': unique()}),
             ('sm_glm_poisson_40x3', 'sm_glm', {'columns': {k: data[k] for k in ('count', 'x1', 'x2')}, 'y': 'count', 'terms': [{'var': 'x1'}, {'var': 'x2'}], 'family': 'poisson'}),
             ('sm_arima_arma11_120', 'sm_arima', {'columns': {'s': data['series']}, 'series': 's', 'order': [1, 0, 1], 'forecast_steps': 5}),
             ('sm_tsa_acf_12_lags', 'sm_tsa_acf', {'columns': {'s': data['series']}, 'series': 's', 'kind': 'acf', 'nlags': 12})]
    for name, op, params in cases:
        wall, imports, elapsed = [], [], []
        for _ in range(10):
            call = params() if callable(params) else params
            start = time.monotonic()
            out = json.loads(subprocess.check_output([str(shim), '--input-json', json.dumps({'op': op, 'params': call})]))
            wall.append((time.monotonic() - start) * 1000)
            assert 'error' not in out, out
            imports.append(out['meta']['import_ms'])
            elapsed.append(out['meta']['elapsed_ms'])
        results[name] = {'wall': summary(wall), 'worker_import_ms': summary(imports), 'worker_elapsed_ms': summary(elapsed)}
print(json.dumps(results, indent=2))
