"""Inside the hardened runtime: S8b operations at their caps (dense matrix functions at the 200000-number input cap, sparse solves and ARPACK at the
200000-entry cap, 200000-value special functions, a 16-equation root system).
Usage: docker run --rm --pull=never --network=none --cpus=2 --memory=4g --pids-limit=128 --read-only --cap-drop=ALL --security-opt=no-new-privileges
  --tmpfs /tmp:rw,noexec,nosuid,nodev,size=512m -v $PWD/scripts:/s:ro --entrypoint python3 pycli:runtime -I /s/measure-s8b-caps.py
(PYCLI_BIN overrides /usr/local/bin/pycli.) Each case prints the operation, exit code, status or error code, wall seconds and the cgroup peak so far."""
import json, os, pathlib, subprocess, tempfile, time
import numpy as np
BIN = os.environ.get('PYCLI_BIN', '/usr/local/bin/pycli')


def peak():
    for f in ('/sys/fs/cgroup/memory.peak', '/sys/fs/cgroup/memory/memory.max_usage_in_bytes'):
        if pathlib.Path(f).exists():
            return int(pathlib.Path(f).read_text())


d = pathlib.Path(tempfile.mkdtemp(prefix='pycli-s8b-caps-', dir='/tmp'))
counter = [0]


def run(tag, op, params):
    t = time.time()
    r = subprocess.run([BIN, '--input-json', json.dumps({'op': op, 'params': params})], cwd=d, capture_output=True, text=True)
    try:
        j = json.loads(r.stdout)
    except Exception:
        j = {'raw': r.stdout[:200]}
    status = j.get('status') or (j.get('error') or {}).get('data', {}).get('tool_code') or j.get('raw')
    print('%-62s rc %d %-18s %6.1f s  peak %.2f GB' % (tag, r.returncode, status, time.time() - t, (peak() or 0) / 1e9), flush=True)
    return j


def out():
    counter[0] += 1
    return 'o%d.safetensors' % counter[0]


def save(name, array):
    np.save(d / name, np.asarray(array))
    return name


rng = np.random.default_rng(9)
# dense matrix functions at n = 447 (the 200000-number input cap)
n = 447
a = rng.normal(size=(n, n)) / np.sqrt(n)
spd = a @ a.T + np.eye(n)
save('a.npy', a)
save('spd.npy', spd)
for fn, name in (('inv', 'a.npy'), ('expm', 'a.npy'), ('logm', 'spd.npy'), ('sqrtm', 'spd.npy'), ('sqrtm', 'a.npy'), ('logm', 'a.npy')):
    run('la_matrix_fn_to %s n=447 (%s)' % (fn, 'spd' if name == 'spd.npy' else 'general'), 'la_matrix_fn_to', dict(a=name, fn=fn, out=out()))
save('wide.npy', rng.normal(size=(400, 500)))
run('la_matrix_fn_to pinv 400 x 500', 'la_matrix_fn_to', dict(a='wide.npy', fn='pinv', out=out()))
# sparse: 1-D Poisson at the entry cap and a 2-D Poisson (fill-in for SuperLU)
m = 66000
r1 = np.concatenate([np.arange(m), np.arange(m - 1), np.arange(1, m)])
c1 = np.concatenate([np.arange(m), np.arange(1, m), np.arange(m - 1)])
v1 = np.concatenate([np.full(m, 2.0), np.full(m - 1, -1.0), np.full(m - 1, -1.0)])
for k, v in (('r1', r1), ('c1', c1), ('v1', v1)):
    save(k + '.npy', v)
save('b1.npy', np.ones(m))
run('la_sparse_to spsolve 1-D n=20000', 'la_sparse_to', dict(task='spsolve', shape=[20000, 20000], row=save('rs.npy', r1[(r1 < 20000) & (c1 < 20000)]), col=save('cs.npy', c1[(r1 < 20000) & (c1 < 20000)]), val=save('vs.npy', v1[(r1 < 20000) & (c1 < 20000)]), b=save('bs.npy', np.ones(20000)), out=out()))
run('la_sparse_to cg 1-D n=66000, 200000 entries', 'la_sparse_to', dict(task='cg', shape=[m, m], row='r1.npy', col='c1.npy', val='v1.npy', b='b1.npy', rtol=1e-6, maxiter=100000, out=out()))
run('la_sparse_to gmres 1-D n=66000', 'la_sparse_to', dict(task='gmres', shape=[m, m], row='r1.npy', col='c1.npy', val='v1.npy', b='b1.npy', rtol=1e-6, restart=30, maxiter=100000, out=out()))
run('la_sparse_to eigsh LM sigma=0 k=5 n=66000 (shift-invert)', 'la_sparse_to', dict(task='eigsh', shape=[m, m], row='r1.npy', col='c1.npy', val='v1.npy', k=5, which='LM', sigma=0.0, seed=1, vectors=True, out=out()))
run('la_sparse_to eigsh LA k=5 n=66000', 'la_sparse_to', dict(task='eigsh', shape=[m, m], row='r1.npy', col='c1.npy', val='v1.npy', k=5, which='LA', seed=1, out=out()))
run('la_sparse_to eigsh SA k=5 n=66000 (clustered end: slow)', 'la_sparse_to', dict(task='eigsh', shape=[m, m], row='r1.npy', col='c1.npy', val='v1.npy', k=5, which='SA', seed=1, out=out()))
g = 141
idx = np.arange(g * g).reshape(g, g)
rows, cols, vals = [idx.ravel()], [idx.ravel()], [np.full(g * g, 4.0)]
for a_, b_ in ((idx[:, :-1], idx[:, 1:]), (idx[:-1, :], idx[1:, :])):
    rows += [a_.ravel(), b_.ravel()]; cols += [b_.ravel(), a_.ravel()]; vals += [np.full(a_.size, -1.0)] * 2
r2, c2, v2 = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
for k, v in (('r2', r2), ('c2', c2), ('v2', v2)):
    save(k + '.npy', v)
run('la_sparse_to spsolve 2-D Poisson n=19881 (%d entries)' % len(v2), 'la_sparse_to', dict(task='spsolve', shape=[g * g, g * g], row='r2.npy', col='c2.npy', val='v2.npy', b=save('b2.npy', np.ones(g * g)), out=out()))
run('la_sparse_to svds k=5 2-D Poisson', 'la_sparse_to', dict(task='svds', shape=[g * g, g * g], row='r2.npy', col='c2.npy', val='v2.npy', k=5, which='LM', seed=2, out=out()))
rr = rng.integers(0, 100000, 200000)
cc = rng.integers(0, 100000, 200000)
run('la_sparse_to eigs LM k=3 random 100000 x 100000, 200000 entries', 'la_sparse_to', dict(task='eigs', shape=[100000, 100000], row=save('rr.npy', rr), col=save('cc.npy', cc), val=save('vr.npy', rng.normal(size=200000)), k=3, which='LM', seed=3, out=out()))
# special functions at 200000 values
xs = np.linspace(0.05, 30.0, 200000)
save('xs.npy', xs)
for fn, args in (('jv', {'v': 2.5, 'z': 'xs.npy'}), ('hyp2f1', {'a': 1.5, 'b': 2.0, 'c': 3.5, 'z': save('zs.npy', np.linspace(-0.9, 0.9, 200000))}), ('gammaincinv', {'a': 3.0, 'y': save('ys.npy', np.linspace(0.001, 0.999, 200000))}), ('airy', {'x': save('as.npy', np.linspace(-10, 10, 200000))}), ('struve', {'v': 1.5, 'x': 'xs.npy'}), ('owens_t', {'h': 'xs.npy', 'a': 0.7})):
    run('special_fn_to %s 200000 values' % fn, 'special_fn_to', dict(fn=fn, args=args, out=out()))
# a 16-equation system: x_i - cos-free chain x_i^2 = i
eq = ['x%d^2 - %d' % (i, i + 1) for i in range(16)]
run('opt_root 16 equations hybr', 'opt_root', dict(equations=eq, start={'x%d' % i: 1.0 for i in range(16)}))
run('opt_root 16 weakly coupled equations lm', 'opt_root', dict(equations=['x%d - 0.01*x%d^2 - %d' % (i, (i + 1) % 16, i) for i in range(16)], start={'x%d' % i: 0.0 for i in range(16)}, method='lm'))
