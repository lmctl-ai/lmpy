"""Inside the hardened runtime: statsmodels ops at their design caps (200000 rows x 50 design columns = 10M cells; ARIMA at its
state/parameter caps). Each case reports wall time, response size and the cgroup peak memory seen so far.
Usage: docker run --rm --pull=never --network=none --cpus=2 --memory=4g --pids-limit=128 --read-only --cap-drop=ALL
  --security-opt=no-new-privileges --tmpfs /tmp:rw,noexec,nosuid,nodev,size=512m -v $PWD/scripts:/s:ro --entrypoint python3 pycli:runtime -I /s/measure-s5-caps.py"""
import json, pathlib, subprocess, tempfile, time
import numpy as np


def peak():
    return next((int(p.read_text()) for p in [pathlib.Path('/sys/fs/cgroup/memory.peak'), pathlib.Path('/sys/fs/cgroup/memory/memory.max_usage_in_bytes')] if p.exists()), None)


def run(root, op, params):
    started = time.monotonic()
    r = subprocess.run(['/usr/local/bin/pycli', op, '--input-json', json.dumps(params)], cwd=root, capture_output=True, text=True)
    elapsed = time.monotonic() - started
    j = json.loads(r.stdout)
    return {'exit': r.returncode, 'seconds': round(elapsed, 2), 'response_bytes': len(r.stdout.encode()), 'status': j.get('status'), 'result_status': (j.get('result') or j).get('status') if isinstance(j, dict) else None, 'peak_bytes_so_far': peak(), 'error': j.get('error')}


rng = np.random.default_rng(5)
cases = {}
with tempfile.TemporaryDirectory(prefix='pycli-s5-caps-', dir='/tmp') as d:
    root = pathlib.Path(d)
    n = 200000
    level = rng.integers(0, 50, n)
    x = rng.normal(size=n)
    y = 0.01 * level + rng.normal(size=n)
    (root / 'big.csv').write_text('y,g\n' + '\n'.join(f'{a:.6f},L{b}' for a, b in zip(y, level)) + '\n')
    terms = [{'const': True}, {'cat': 'g', 'ref': 'L0'}]
    cases['sm_ols_200000x50'] = run(root, 'sm_ols', {'csv': 'big.csv', 'y': 'y', 'terms': terms})
    cases['sm_ols_hc3_200000x50'] = run(root, 'sm_ols', {'csv': 'big.csv', 'y': 'y', 'terms': terms, 'cov_type': 'HC3'})
    (root / 'bin.csv').write_text('y,g\n' + '\n'.join(f'{int(a > 0.2)},L{b}' for a, b in zip(y, level)) + '\n')
    cases['sm_glm_binomial_200000x50'] = run(root, 'sm_glm', {'csv': 'bin.csv', 'y': 'y', 'terms': terms, 'family': 'binomial'})
    cases['sm_glm_poisson_to_200000x50'] = run(root, 'sm_glm_to', {'csv': 'bin.csv', 'y': 'y', 'terms': terms, 'family': 'poisson', 'out': 'g.safetensors'})
    # ARIMA: seasonal state-dimension cap (p=5,q=5,P=2,Q=2,s=12 -> 64 states is the cap) and a long series
    s = np.zeros(2000)
    e = rng.normal(size=2000)
    for t in range(2, 2000):
        s[t] = 0.5 * s[t - 1] - 0.2 * s[t - 2] + e[t] + 0.3 * e[t - 1]
    (root / 'ts.csv').write_text('v\n' + '\n'.join(f'{v:.6f}' for v in s) + '\n')
    cases['sm_arima_2_0_1_n2000'] = run(root, 'sm_arima', {'csv': 'ts.csv', 'series': 'v', 'order': [2, 0, 1], 'forecast_steps': 100})
    cases['sm_arima_seasonal_1_1_1_1_1_1_12_n2000'] = run(root, 'sm_arima', {'csv': 'ts.csv', 'series': 'v', 'order': [1, 1, 1], 'seasonal_order': [1, 1, 1, 12], 'forecast_steps': 24})
    cases['sm_arima_cap_5_0_5_2_0_2_12_n2000'] = run(root, 'sm_arima', {'csv': 'ts.csv', 'series': 'v', 'order': [5, 0, 5], 'seasonal_order': [2, 0, 2, 12], 'maxiter': 50})
    cases['sm_tsa_acf_100_lags'] = run(root, 'sm_tsa_acf', {'csv': 'ts.csv', 'series': 'v', 'kind': 'acf', 'nlags': 100})
print(json.dumps(cases, indent=1))
