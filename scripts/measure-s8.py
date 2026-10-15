"""S8a timings through the production shim: wall time plus the worker's own import/elapsed milliseconds (meta), one small call per operation.
Usage: python3 scripts/measure-s8.py pycli:runtime > docs/measurement-s8-shim.json"""
import json, math, pathlib, statistics, subprocess, sys, tempfile, time
root = pathlib.Path(__file__).resolve().parents[1]
image = sys.argv[1]


def summary(values):
    values = sorted(values)
    return {'p50_ms': round(statistics.median(values), 3), 'p95_ms': round(values[min(len(values) - 1, int(.95 * len(values)))], 3), 'n': len(values)}


results = {'image_bytes': int(subprocess.check_output(['docker', 'image', 'inspect', image, '--format', '{{.Size}}']))}
with tempfile.TemporaryDirectory(dir='.') as directory:
    work = pathlib.Path(directory)
    shim = work / 'pycli'
    shim.write_text(pathlib.Path('bin/pycli').read_text().replace('pycli:runtime', image))
    shim.chmod(0o755)

    def invoke(op, params):
        out = json.loads(subprocess.check_output([str(shim), '--input-json', json.dumps({'op': op, 'params': params})]))
        assert 'error' not in out, out
        return out
    n = 200
    table = lambda name, columns, **extra: invoke('frame_import', dict(columns=columns, out=str(work / name), **extra))
    table('t.npz', {'a': [float(i) for i in range(n)], 'b': [i % 7 for i in range(n)], 'k': [i % 20 for i in range(n)]}, dtypes={'k': 'int64'})
    table('l.npz', {'k': list(range(40)), 'v': [float(i) for i in range(40)]})
    table('r.npz', {'k': list(range(20, 60)), 'w': [float(i) for i in range(40)]})
    counter = [0]

    def unique(ext):
        counter[0] += 1
        return str(work / ('out%d.%s' % (counter[0], ext)))
    cases = [('py_info', 'py_info', {}),
             ('frame_to_array_200x3', 'frame_to_array', lambda: {'frame': str(work / 't.npz'), 'columns': ['a', 'b'], 'limit': 100}),
             ('frame_to_array_to_200x3', 'frame_to_array_to', lambda: {'frame': str(work / 't.npz'), 'columns': ['a', 'b', 'k'], 'out': unique('safetensors')}),
             ('array_to_frame_100x2', 'array_to_frame', lambda: {'ref': [[float(i), float(2 * i)] for i in range(100)], 'columns': ['p', 'q'], 'out': unique('npz')}),
             ('frame_mutate_2_expressions', 'frame_mutate', lambda: {'frame': str(work / 't.npz'), 'assign': [{'name': 'z', 'expr': 'a * 2 + b'}, {'name': 'w', 'expr': 'sqrt(abs(z)) - k'}], 'out': unique('npz')}),
             ('frame_merge_outer_40x40', 'frame_merge', lambda: {'left': str(work / 'l.npz'), 'right': str(work / 'r.npz'), 'on': ['k'], 'how': 'outer', 'out': unique('npz')}),
             ('frame_rolling_window10_5_functions', 'frame_rolling', lambda: {'frame': str(work / 't.npz'), 'columns': ['a'], 'window': 10, 'functions': ['mean', 'std', 'min', 'max', 'median'], 'out': unique('npz')}),
             ('ml_metrics_binary_200', 'ml_metrics', {'y_true': [i % 2 for i in range(200)], 'y_pred': [(i // 3) % 2 for i in range(200)], 'y_score': [((i * 37) % 100) / 100 for i in range(200)], 'pos_label': 1, 'metrics': ['accuracy', 'f1', 'roc_auc', 'log_loss', 'confusion_matrix']}),
             ('stats_dist_rvs_1000', 'stats_dist', {'dist': 'gamma', 'fn': 'rvs', 'params': {'shape': 3.0, 'scale': 2.0}, 'n': 1000, 'seed': 7})]
    for name, op, params in cases:
        wall, imports, elapsed = [], [], []
        for _ in range(10):
            call = params() if callable(params) else params
            start = time.monotonic()
            out = invoke(op, call)
            wall.append((time.monotonic() - start) * 1000)
            imports.append(out['meta']['import_ms'])
            elapsed.append(out['meta']['elapsed_ms'])
        results[name] = {'wall': summary(wall), 'worker_import_ms': summary(imports), 'worker_elapsed_ms': summary(elapsed)}
print(json.dumps(results, indent=2))
