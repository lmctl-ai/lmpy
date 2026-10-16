"""S8b-2 timings through the production shim: wall time plus the worker's own import/elapsed milliseconds (meta), one small call per operation, so the
networkx import cost of the graph operations is visible next to a numpy-only operation (frame_mutate) and a scipy one (special_fn).
Usage: python3 scripts/measure-s8b2.py pycli:runtime > docs/measurement-s8b2-shim.json"""
import json, pathlib, statistics, subprocess, sys, tempfile, time
image = sys.argv[1]


def summary(values):
    values = sorted(values)
    return {'p50_ms': round(statistics.median(values), 3), 'p95_ms': round(values[min(len(values) - 1, int(.95 * len(values)))], 3), 'n': len(values)}


results = {'image_bytes': int(subprocess.check_output(['docker', 'image', 'inspect', image, '--format', '{{.Size}}']))}
edges = [{'u': a, 'v': b, 'w': w, 'cap': w} for a, b, w in [(0, 1, 4), (0, 2, 2), (1, 2, 1), (1, 3, 5), (2, 3, 8), (2, 4, 10), (3, 4, 2), (3, 5, 6), (4, 5, 3)]]
base = dict(records=edges, from_column='u', to_column='v')
cases = [('py_info', 'py_info', {}),
         ('special_fn_erf_100', 'special_fn', {'fn': 'erf', 'args': {'x': [i / 50.0 for i in range(100)]}}),
         ('graph_paths_dijkstra_6', 'graph_paths', dict(base, weight_column='w', method='dijkstra', source=0, target=5, directed=True)),
         ('graph_paths_k3_6', 'graph_paths', dict(base, weight_column='w', method='dijkstra', source=0, target=5, k_shortest=3, directed=True)),
         ('graph_flow_max_flow_6', 'graph_flow', dict(base, capacity_column='cap', kind='max_flow', source=0, sink=5)),
         ('graph_flow_min_cut_6', 'graph_flow', dict(base, capacity_column='cap', kind='min_cut', source=0, sink=5)),
         ('stats_dist_genextreme_cdf', 'stats_dist', {'dist': 'genextreme', 'fn': 'cdf', 'params': {'c': 0.2}, 'x': [0.0, 0.5, 1.0]}),
         ('stats_dist_nct_cdf', 'stats_dist', {'dist': 'nct', 'fn': 'cdf', 'params': {'df': 7.0, 'nc': 1.0}, 'x': [0.0, 0.5, 1.0]}),
         ('stats_dist_levy_stable_cdf', 'stats_dist', {'dist': 'levy_stable', 'fn': 'cdf', 'params': {'alpha': 1.5, 'beta': 0.5}, 'x': [0.0, 0.5, 1.0]}),
         ('stats_dist_norm_rvs', 'stats_dist', {'dist': 'norm', 'fn': 'rvs', 'params': {}, 'n': 100, 'seed': 1})]
with tempfile.TemporaryDirectory(dir='.') as directory:
    shim = pathlib.Path(directory) / 'pycli'
    shim.write_text(pathlib.Path('bin/pycli').read_text().replace('pycli:runtime', image))
    shim.chmod(0o755)
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
