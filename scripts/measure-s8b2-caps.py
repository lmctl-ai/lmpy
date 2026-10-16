"""Inside the hardened runtime: S8b-2 graph operations and distribution functions at their caps and on adversarial shapes, to show that the cooperative
budget answers (OK or TIME_LIMIT) long before the 50 s hard kill.
Usage: docker run --rm --pull=never --network=none --cpus=2 --memory=4g --pids-limit=128 --read-only --cap-drop=ALL --security-opt=no-new-privileges
  --tmpfs /tmp:rw,noexec,nosuid,nodev,size=512m -v $PWD/scripts:/s:ro --entrypoint python3 pycli:runtime -I /s/measure-s8b2-caps.py
(PYCLI_BIN overrides /usr/local/bin/pycli.) Each case prints the operation, exit code, status or error code, wall seconds and the cgroup peak so far."""
import json, os, pathlib, random, subprocess, tempfile, time
import numpy as np
BIN = os.environ.get('PYCLI_BIN', '/usr/local/bin/pycli')


def peak():
    for f in ('/sys/fs/cgroup/memory.peak', '/sys/fs/cgroup/memory/memory.max_usage_in_bytes'):
        if pathlib.Path(f).exists():
            return int(pathlib.Path(f).read_text())


d = pathlib.Path(tempfile.mkdtemp(prefix='pycli-s8b2-caps-', dir='/tmp'))
counter = [0]
worst = [0.0]


def run(tag, op, params):
    t = time.time()
    r = subprocess.run([BIN, '--input-json', json.dumps({'op': op, 'params': params})], cwd=d, capture_output=True, text=True)
    try:
        j = json.loads(r.stdout)
    except Exception:
        j = {'raw': r.stdout[:200]}
    status = j.get('status') or (j.get('error') or {}).get('data', {}).get('tool_code') or j.get('raw')
    wall = time.time() - t
    worst[0] = max(worst[0], wall)
    print('%-70s rc %d %-18s %6.1f s  peak %.2f GB' % (tag, r.returncode, status, wall, (peak() or 0) / 1e9), flush=True)
    return j


def table(name, rows, header=('u', 'v', 'w')):
    path = d / (name + '.csv')
    path.write_text(','.join(header) + '\n' + ''.join(','.join(str(c) for c in row) + '\n' for row in rows))
    j = run('  frame_import %s (%d rows)' % (name, len(rows)), 'frame_import', dict(csv=str(path), out=str(d / (name + '.npz'))))
    assert 'error' not in j, j
    return str(d / (name + '.npz'))


def out(ext):
    counter[0] += 1
    return str(d / ('o%d.%s' % (counter[0], ext)))


def paths(frame, **extra):
    return dict(frame=frame, from_column='u', to_column='v', weight_column='w', **extra)


def flow(frame, **extra):
    return dict(frame=frame, from_column='u', to_column='v', capacity_column='w', **extra)


rng = random.Random(5)
# ---- single-source paths at 100000 edges
chain = table('chain', [(i, i + 1, rng.randint(1, 9)) for i in range(99999)])
rand = table('rand', [(rng.randrange(50000), rng.randrange(50000), rng.randint(1, 99)) for _ in range(100000)])
side = 223
grid_rows = []
for r in range(side):
    for c in range(side):
        if c + 1 < side:
            grid_rows.append((r * side + c, r * side + c + 1, rng.randint(1, 9)))
        if r + 1 < side:
            grid_rows.append((r * side + c, (r + 1) * side + c, rng.randint(1, 9)))
grid = table('grid', grid_rows)
for method in ('dijkstra', 'bellman_ford', 'bfs'):
    extra = {} if method != 'bfs' else {}
    for name, f, src in (('chain 100000 edges', chain, 0), ('random 50000 nodes 100000 edges', rand, rng.randrange(50000)), ('grid %dx%d undirected' % (side, side), grid, 0)):
        p = paths(f, method=method, source=src, directed=name.startswith('grid') is False)
        if method == 'bfs':
            p.pop('weight_column')
        run('graph_paths_to %s %s' % (method, name), 'graph_paths_to', dict(p, out=out('npz')))
run('graph_paths k_shortest=10 grid corner to corner', 'graph_paths', paths(grid, method='dijkstra', source=0, target=side * side - 1, k_shortest=10, directed=False))
run('graph_paths k_shortest=10 random', 'graph_paths', paths(rand, method='dijkstra', source=rng.randrange(50000), target=rng.randrange(50000), k_shortest=10))
# bellman_ford on a long chain listed against the edge direction with negative weights (DAG: no cycle)
neg = table('neg', [(i + 1, i, -rng.randint(1, 9)) for i in range(99999)][::-1])
run('graph_paths_to bellman_ford negative chain reversed order', 'graph_paths_to', dict(paths(neg, method='bellman_ford', source=99999, directed=True), out=out('npz')))
# ---- all pairs at the dense bound (n = 1377)
n = 1377
dense = table('dense', [(rng.randrange(n), rng.randrange(n), rng.randint(1, 99)) for _ in range(100000)])
for method in ('dijkstra', 'bellman_ford', 'bfs'):
    p = paths(dense, method=method, all_pairs=True, directed=True)
    if method == 'bfs':
        p.pop('weight_column')
    run('graph_paths_to all pairs n=1377 %s 100000 edges' % method, 'graph_paths_to', dict(p, out=out('safetensors')))
run('graph_paths_to all pairs n=1378 (over the matrix cap)', 'graph_paths_to', dict(paths(table('over', [(i, (i + 1) % 1378, 1) for i in range(1378)]), method='dijkstra', all_pairs=True, directed=True), out=out('safetensors')))
# ---- flows at 20000 edges
lay = []
for layer in range(99):
    for k in range(100):
        for j in rng.sample(range(100), 2):
            lay.append((layer * 100 + k, (layer + 1) * 100 + j, rng.randint(1, 20)))
lay = lay[:20000]
layered = table('layered', lay)
run('graph_flow_to max_flow layered 100x99 20000 edges (source/sink via supernodes)', 'graph_flow_to', dict(flow(layered, kind='max_flow', source=0, sink=lay[-1][1]), out=out('npz')))
gs = 70
grid_cap = []
for r in range(gs):
    for c in range(gs):
        if c + 1 < gs:
            grid_cap.append((r * gs + c, r * gs + c + 1, rng.randint(1, 5)))
            grid_cap.append((r * gs + c + 1, r * gs + c, rng.randint(1, 5)))
        if r + 1 < gs:
            grid_cap.append((r * gs + c, (r + 1) * gs + c, rng.randint(1, 5)))
            grid_cap.append(((r + 1) * gs + c, r * gs + c, rng.randint(1, 5)))
grid_cap = grid_cap[:20000]
gridf = table('gridf', grid_cap)
for kind in ('max_flow', 'min_cut'):
    run('graph_flow_to %s grid 70x70 bidirectional 19320 edges' % kind, 'graph_flow_to', dict(flow(gridf, kind=kind, source=0, sink=gs * gs - 1), out=out('npz')))
unit = table('unit', [(i, i + 1, 1) for i in range(19999)] + [])
run('graph_flow_to max_flow unit chain 19999 edges', 'graph_flow_to', dict(flow(unit, kind='max_flow', source=0, sink=19999), out=out('npz')))
# adversarial for augmenting-path methods: the classic diamond ladder with a unit cross edge
ladder = []
big = 10 ** 9
for i in range(0, 6000, 2):
    ladder += [(i, i + 2, big), (i, i + 3, big), (i + 1, i + 2, big), (i + 1, i + 3, big), (i + 2, i + 3, 1)]
ladderf = table('ladder', ladder[:20000])
run('graph_flow_to max_flow diamond ladder large capacities', 'graph_flow_to', dict(flow(ladderf, kind='max_flow', source=0, sink=5999), out=out('npz')))
mcf_rows = [(u, v, c, rng.randint(1, 50)) for (u, v, c) in lay[:20000]]
mcf = table('mcf', mcf_rows, header=('u', 'v', 'w', 'cost'))
run('graph_flow_to min_cost_flow layered 20000 edges value 5', 'graph_flow_to', dict(flow(mcf, kind='min_cost_flow', cost_column='cost', source=0, sink=lay[-1][1], flow_value=5), out=out('npz')))
mcf2 = table('mcf2', [(i, i + 1, 10 ** 6, rng.randint(1, 50)) for i in range(19999)], header=('u', 'v', 'w', 'cost'))
run('graph_flow_to min_cost_flow chain 19999 edges', 'graph_flow_to', dict(flow(mcf2, kind='min_cost_flow', cost_column='cost', source=0, sink=19999, flow_value=10 ** 5), out=out('npz')))
# ---- distribution functions at their caps
x1000 = list(np.linspace(-5, 5, 1000))
x200 = list(np.linspace(-5, 5, 200))
save = lambda name, a: (np.save(d / name, np.asarray(a)), name)[1]
big_x = save('bigx.npy', np.linspace(-5, 5, 200000))
for dist, params in (('genextreme', {'c': 0.2}), ('johnsonsu', {'a': 0.5, 'b': 1.5}), ('skewnorm', {'a': 3.0}), ('nct', {'df': 7.0, 'nc': 2.0}), ('nct', {'df': 3.0, 'nc': 1000.0}), ('ncf', {'dfn': 5.0, 'dfd': 9.0, 'nc': 1000.0}), ('ncf', {'dfn': 5.0, 'dfd': 9.0, 'nc': 20.0}), ('betabinom', {'n': 5000, 'a': 2.0, 'b': 3.0}), ('binom', {'n': 10 ** 9, 'p': 0.3}), ('nbinom', {'n': 1000.0, 'p': 0.001}), ('poisson', {'lam': 1e9})):
    for fn in ('cdf', 'ppf'):
        x = x1000 if fn == 'cdf' else list(np.linspace(0.001, 0.999, 1000))
        run('stats_dist %s %s 1000 points' % (dist, fn), 'stats_dist', dict(dist=dist, fn=fn, params=params, x=x))
for fn in ('pdf', 'cdf', 'ppf'):
    x = x200 if fn != 'ppf' else list(np.linspace(0.01, 0.99, 200))
    run('stats_dist levy_stable alpha 1.3 beta 0.5 %s 200 points' % fn, 'stats_dist', dict(dist='levy_stable', fn=fn, params={'alpha': 1.3, 'beta': 0.5}, x=x))
for dist, params in (('vonmises', {'kappa': 5.0}), ('vonmises', {'kappa': 1e5})):
    for fn in ('cdf', 'ppf'):
        run('stats_dist %s %s kappa %g 200 points' % (dist, fn, params['kappa']), 'stats_dist', dict(dist=dist, fn=fn, params=params, x=x200 if fn == 'cdf' else list(np.linspace(0.01, 0.99, 200))))
for a in (1.05, 1.5, 4.0):
    run('stats_dist zipf a=%g cdf 1000 points up to 1e6' % a, 'stats_dist', dict(dist='zipf', fn='cdf', params={'a': a}, x=list(np.linspace(1, 1e6, 1000))))
    run('stats_dist zipf a=%g ppf 1000 points' % a, 'stats_dist', dict(dist='zipf', fn='ppf', params={'a': a}, x=list(np.linspace(0.001, 0.999, 1000))))
run('stats_dist levy_stable alpha 0.1 cdf 200 points (hard)', 'stats_dist', dict(dist='levy_stable', fn='cdf', params={'alpha': 0.1, 'beta': 0.9}, x=x200))
for dist, params in (('genextreme', {'c': 0.2}), ('nct', {'df': 7.0, 'nc': 2.0}), ('skewnorm', {'a': 3.0}), ('zipf', {'a': 1.05}), ('levy_stable', {'alpha': 1.3, 'beta': 0.5}), ('vonmises', {'kappa': 5.0}), ('truncnorm', {'a': 5.0, 'b': 8.0})):
    run('stats_dist_to rvs %s n=1000000' % dist, 'stats_dist_to', dict(dist=dist, fn='rvs', params=params, n=1000000, seed=3, out=out('safetensors')))
for dist, params in (('genextreme', {'c': 0.2}), ('skewnorm', {'a': 3.0}), ('johnsonsu', {'a': 0.5, 'b': 1.5}), ('weibull_min', {'c': 1.5})):
    run('stats_dist_to cdf %s 200000 points' % dist, 'stats_dist_to', dict(dist=dist, fn='cdf', params=params, x=big_x, out=out('safetensors')))
run('stats_dist_to ppf nct 200000 points', 'stats_dist_to', dict(dist='nct', fn='ppf', params={'df': 7.0, 'nc': 2.0}, x=save('bigp.npy', np.linspace(0.0001, 0.9999, 200000)), out=out('safetensors')))
run('stats_dist_to cdf nct 200000 points', 'stats_dist_to', dict(dist='nct', fn='cdf', params={'df': 7.0, 'nc': 2.0}, x=big_x, out=out('safetensors')))
run('stats_dist_to cdf ncf 200000 points', 'stats_dist_to', dict(dist='ncf', fn='cdf', params={'dfn': 5.0, 'dfd': 9.0, 'nc': 20.0}, x=save('bigpos.npy', np.linspace(0.01, 30, 200000)), out=out('safetensors')))
print('slowest call %.1f s (hard kill 50 s)' % worst[0])
