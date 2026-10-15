"""Inside the hardened runtime: S8a operations at their caps (tables of 200000 rows, joins at the row cap, the rolling work cap, 1M-value sampling).
Usage: docker run --rm --pull=never --network=none --cpus=2 --memory=4g --pids-limit=128 --read-only --cap-drop=ALL --security-opt=no-new-privileges
  --tmpfs /tmp:rw,noexec,nosuid,nodev,size=512m -v $PWD/scripts:/s:ro --entrypoint python3 pycli:runtime -I /s/measure-s8-caps.py
(PYCLI_BIN overrides /usr/local/bin/pycli.) Each case prints the operation, exit code, status or error code, wall seconds and the cgroup peak so far."""
import json, os, pathlib, subprocess, tempfile, time
import numpy as np
BIN = os.environ.get('PYCLI_BIN', '/usr/local/bin/pycli')


def peak():
    for f in ('/sys/fs/cgroup/memory.peak', '/sys/fs/cgroup/memory/memory.max_usage_in_bytes'):
        if pathlib.Path(f).exists():
            return int(pathlib.Path(f).read_text())


d = pathlib.Path(tempfile.mkdtemp(prefix='pycli-s8-caps-', dir='/tmp'))


def run(tag, op, params):
    t = time.time()
    r = subprocess.run([BIN, '--input-json', json.dumps({'op': op, 'params': params})], cwd=d, capture_output=True, text=True)
    try:
        j = json.loads(r.stdout)
    except Exception:
        j = {'raw': r.stdout[:200]}
    status = j.get('status') or (j.get('error') or {}).get('data', {}).get('tool_code') or j.get('raw')
    print('%-46s rc %d %-18s %6.1f s  peak %.2f GB' % (tag, r.returncode, status, time.time() - t, (peak() or 0) / 1e9), flush=True)
    return j


rng = np.random.default_rng(8)
n = 200000
(d / 'a.csv').write_text('k,x,y\n' + ''.join('%d,%.6f,%.6f\n' % (i % 50000, rng.normal(), rng.normal()) for i in range(n)))
run('frame_import 200000 x 3 (600000 cells -> refused)', 'frame_import', dict(csv='a.csv', out='big.npz'))
(d / 'c.csv').write_text('x\n' + ''.join('%.6f\n' % rng.normal() for i in range(n)))
run('frame_import 200000 x 1', 'frame_import', dict(csv='c.csv', out='c.npz'))
run('frame_mutate 512-node expression x 200000 rows', 'frame_mutate', dict(frame='c.npz', assign=[{'name': 'w', 'expr': ' + '.join(['x*x'] * 100)}], out='m2.npz'))
run('frame_mutate transcendental x 200000 rows', 'frame_mutate', dict(frame='c.npz', assign=[{'name': 'z', 'expr': 'sqrt(x*x + 1) * 3 + sin(x) - cos(x) + exp(-abs(x))'}], out='m.npz'))
(d / 'f.csv').write_text('x\n' + ''.join('%.6f\n' % rng.normal() for i in range(15000)))
run('import 15000 x 1', 'frame_import', dict(csv='f.csv', out='f.npz'))
run('frame_mutate 32 assignments x 15000 rows (495000 cells)', 'frame_mutate', dict(frame='f.npz', assign=[{'name': 'z%d' % i, 'expr': 'x * %d + sin(x)' % (i + 1)} for i in range(32)], out='m32.npz'))
run('frame_to_array_to 200000 values', 'frame_to_array_to', dict(frame='c.npz', columns=['x'], squeeze=True, out='x.safetensors'))
np.save(d / 'v.npy', rng.normal(size=(100000, 2)))
run('array_to_frame 100000 x 2', 'array_to_frame', dict(ref='v.npy', columns=['p', 'q'], out='v.npz'))
(d / 'u1.csv').write_text('k,u\n' + ''.join('%d,%d\n' % (i, i) for i in range(100000)))
(d / 'u2.csv').write_text('k,w\n' + ''.join('%d,%d\n' % (i + 50000, i) for i in range(100000)))
run('import left', 'frame_import', dict(csv='u1.csv', out='u1.npz'))
run('import right', 'frame_import', dict(csv='u2.csv', out='u2.npz'))
for how in ('inner', 'outer'):
    run('frame_merge %s 100000 x 100000, 50000 matches' % how, 'frame_merge', dict(left='u1.npz', right='u2.npz', on=['k'], how=how, validate='one_to_one', out='j_%s.npz' % how))
(d / 'dl.csv').write_text('g,a\n' + ''.join('%d,%d\n' % (i % 400, i) for i in range(2000)))
(d / 'dr.csv').write_text('g,b\n' + ''.join('%d,%d\n' % (i % 400, i) for i in range(2000)))
run('import dup left', 'frame_import', dict(csv='dl.csv', out='dl.npz'))
run('import dup right', 'frame_import', dict(csv='dr.csv', out='dr.npz'))
run('frame_merge many-to-many 2000 x 2000 -> 10000 rows', 'frame_merge', dict(left='dl.npz', right='dr.npz', on=['g'], out='mm.npz', max_rows=200000))
run('frame_merge many-to-many 2000 x 2000 -> row cap', 'frame_merge', dict(left='dl.npz', right='dr.npz', on=['g'], out='mm2.npz', max_rows=9999))
(d / 'e1.csv').write_text('g,a\n' + ''.join('1,%d\n' % i for i in range(5000)))
run('import same-key table', 'frame_import', dict(csv='e1.csv', out='e1.npz'))
run('frame_merge 5000 x 5000 same key (25M rows refused)', 'frame_merge', dict(left='e1.npz', right='e1.npz', on=['g'], out='boom.npz'))
run('frame_rolling 200000 rows window 500 (work 1e8)', 'frame_rolling', dict(frame='c.npz', columns=['x'], window=500, functions=['mean'], out='r1.npz'))
run('frame_rolling 200000 rows window 500 median (cells ok, work 1e8)', 'frame_rolling', dict(frame='c.npz', columns=['x'], window=500, functions=['median'], out='r2.npz'))
(d / 'h.csv').write_text('x\n' + ''.join('%.6f\n' % rng.normal() for i in range(80000)))
run('import 80000 x 1', 'frame_import', dict(csv='h.csv', out='h.npz'))
run('frame_rolling 80000 rows window 100 x 5 functions', 'frame_rolling', dict(frame='h.npz', columns=['x'], window=100, functions=['mean', 'std', 'min', 'max', 'sum'], out='r3.npz'))
run('frame_rolling 80000 rows window 1000 mean (work 8e7)', 'frame_rolling', dict(frame='h.npz', columns=['x'], window=1000, functions=['mean'], out='r5.npz'))
run('frame_rolling 80000 rows window 1001 x 2 (refused work)', 'frame_rolling', dict(frame='h.npz', columns=['x'], window=1001, functions=['mean', 'sum'], out='r6.npz'))
(d / 't.csv').write_text('t,v\n' + ''.join('2026-01-01T%02d:%02d:%02dZ,%d\n' % (i // 3600 % 24, i // 60 % 60, i % 60, i % 7) for i in range(80000)))
run('import timestamps 80000', 'frame_import', dict(csv='t.csv', dtypes={'t': {'dtype': 'datetime64[ns]', 'tz': 'UTC'}}, out='t.npz'))
run('frame_rolling time window 600s over 80000 rows', 'frame_rolling', dict(frame='t.npz', columns=['v'], time_window='600s', on='t', functions=['mean', 'count'], out='r4.npz'))
labels = rng.integers(0, 30, 200000)
np.save(d / 'yt.npy', labels)
np.save(d / 'yp.npy', np.where(rng.random(200000) < 0.7, labels, rng.integers(0, 30, 200000)))
np.save(d / 'ys.npy', rng.random(200000))
run('ml_metrics 200000 rows, 30 classes, confusion', 'ml_metrics', dict(y_true='yt.npy', y_pred='yp.npy', metrics=['accuracy', 'balanced_accuracy', 'f1', 'confusion_matrix'], average='macro'))
np.save(d / 'yb.npy', rng.integers(0, 2, 200000))
run('ml_metrics roc_auc/log_loss/brier 200000', 'ml_metrics', dict(y_true='yb.npy', y_score='ys.npy', metrics=['roc_auc', 'log_loss', 'brier'], pos_label=1))
for dist, params in (('norm', {}), ('gamma', {'shape': 3.0, 'scale': 2.0}), ('binom', {'n': 100, 'p': 0.3}), ('poisson', {'lam': 1e6})):
    run('stats_dist_to rvs 1,000,000 %s' % dist, 'stats_dist_to', dict(dist=dist, fn='rvs', params=params, n=1000000, seed=5, out='%s.safetensors' % dist))
