"""S7 timings through the production shim: wall time plus the worker's own import/elapsed milliseconds (meta), one small call per op group.
Usage: python3 scripts/measure-s7.py pycli:runtime > docs/measurement-s7-shim.json"""
import json, math, pathlib, statistics, subprocess, sys, tempfile, time
root = pathlib.Path(__file__).resolve().parents[1]
image = sys.argv[1]


def summary(values):
    values = sorted(values)
    return {'p50_ms': round(statistics.median(values), 3), 'p95_ms': round(values[min(len(values) - 1, int(.95 * len(values)))], 3), 'n': len(values)}


results = {'image_bytes': int(subprocess.check_output(['docker', 'image', 'inspect', image, '--format', '{{.Size}}']))}
x = [float(i) for i in range(40)]
y = [math.sin(v / 5) for v in x]
signal = [math.sin(0.3 * i) + 0.2 * math.sin(2.1 * i) for i in range(400)]
points = [[(7 * i % 31) / 31, (11 * i % 37) / 37] for i in range(60)]
with tempfile.TemporaryDirectory(dir='.') as directory:
    work = pathlib.Path(directory)
    shim = work / 'pycli'
    shim.write_text(pathlib.Path('bin/pycli').read_text().replace('pycli:runtime', image))
    shim.chmod(0o755)
    cases = [('py_info', 'py_info', {}),
             ('interp_eval_pchip_40', 'interp_eval', {'kind': 'pchip', 'x': x, 'y': y, 'x_new': [0.5 + i for i in range(39)]}),
             ('interp_eval_rbf_40', 'interp_eval', {'kind': 'rbf', 'x': [[v, v * v % 7] for v in x], 'y': y, 'x_new': [[1.5, 2.0], [10.5, 3.0]], 'extrapolate': 'extend'}),
             ('sig_filter_butter8_400', 'sig_filter', {'family': 'butter', 'order': 8, 'btype': 'lowpass', 'wn': 0.2, 'x': signal, 'mode': 'sosfiltfilt', 'freqs': [0.1, 0.2, 0.5]}),
             ('sig_spectrum_welch_400', 'sig_spectrum', {'x': signal, 'method': 'welch', 'nperseg': 64}),
             ('sig_peaks_400', 'sig_peaks', {'x': signal}),
             ('spatial_knn_60', 'spatial_knn', {'points': points, 'k': 5}),
             ('spatial_hull_delaunay_60', 'spatial_hull', {'points': points[:30], 'kind': 'delaunay'}),
             ('spatial_dist_60', 'spatial_dist', {'a': points[:20], 'b': points[20:40], 'metric': 'euclidean'})]
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
