"""S3 timings through the production shim: wall time per call plus the worker's own import/elapsed milliseconds.
Usage: python3 scripts/measure-numerics.py pycli:runtime > docs/measurement-s3-shim.json"""
import json, pathlib, statistics, struct, subprocess, sys, tempfile, time, random
def summary(values):
    values = sorted(values)
    return {'p50_ms': round(statistics.median(values), 3), 'p95_ms': round(values[min(len(values) - 1, int(.95 * len(values)))], 3), 'n': len(values)}
image = sys.argv[1]
results = {'image_bytes': int(subprocess.check_output(['docker', 'image', 'inspect', image, '--format', '{{.Size}}']))}
with tempfile.TemporaryDirectory(dir='.') as directory:
    work = pathlib.Path(directory)
    shim = work / 'pycli'
    shim.write_text(pathlib.Path('bin/pycli').read_text().replace('pycli:runtime', image)); shim.chmod(0o755)
    rng = random.Random(7)
    xs = [i * 0.05 for i in range(200)]
    ys = [3.0 * 2.718281828459045 ** (-0.8 * x) + 0.2 + rng.gauss(0, 0.01) for x in xs]
    big = 200000
    header = ("{'descr': '<f8', 'fortran_order': False, 'shape': (%d,), }" % big).encode()
    header += b' ' * ((-(10 + len(header) + 1)) % 64) + b'\n'
    (work / 't.npy').write_bytes(b'\x93NUMPY\x01\x00' + struct.pack('<H', len(header)) + header + struct.pack('<%dd' % big, *(i * 1e-5 for i in range(big))))
    rosen = '100*(y-x^2)^2 + (1-x)^2'
    cases = [('py_info', 'py_info', {}),
             ('opt_minimize_rosenbrock_bfgs', 'opt_minimize', {'objective': rosen, 'x0': {'x': -1.2, 'y': 1}, 'method': 'BFGS', 'tol': 1e-6}),
             ('opt_minimize_slsqp_constrained', 'opt_minimize', {'objective': 'x^2+y^2', 'x0': {'x': 2, 'y': -1}, 'method': 'SLSQP', 'constraints': [{'type': 'eq', 'expr': 'x+y-1'}]}),
             ('opt_minimize_200000_rows', 'opt_minimize', {'objective': 'sum((t*a - 1)^2)', 'x0': {'a': 0.0}, 'data': {'t': str(work / 't.npy')}, 'method': 'BFGS'}),
             ('opt_least_squares_200_rows', 'opt_least_squares', {'model': 'c + a*exp(-b*t)', 'response': 'y', 'data': {'t': xs[:100], 'y': ys[:100]}, 'x0': {'a': 1, 'b': 0.5, 'c': 0}}),
             ('num_quad_gauss_infinite', 'num_quad', {'f': 'exp(-x^2)', 'a': '-Inf', 'b': 'Inf'}),
             ('num_root_brentq', 'num_root', {'f': 'cos(x)-x', 'lower': 0, 'upper': 1})]
    for name, op, params in cases:
        wall, imports, elapsed = [], [], []
        for _ in range(10):
            start = time.monotonic()
            out = json.loads(subprocess.check_output([str(shim), '--input-json', json.dumps({'op': op, 'params': params})]))
            wall.append((time.monotonic() - start) * 1000)
            assert 'error' not in out, out
            imports.append(out['meta']['import_ms']); elapsed.append(out['meta']['elapsed_ms'])
        results[name] = {'wall': summary(wall), 'worker_import_ms': summary(imports), 'worker_elapsed_ms': summary(elapsed)}
print(json.dumps(results, indent=2))
