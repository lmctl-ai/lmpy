"""S8b-2 review regressions (Opus, Fable, FableFuzz, Argon and Kimi against 52f0267; docs/s8b2-review-notes.md maps every finding to a test here).
Graph flows against the opt_linprog LP oracle, random graphs against the pure-Python algorithms of tests/oracle/s8b2.py (with an independent
tight-edge hop count), every distribution family against mpmath closed forms or quadrature, the warm-up/budget contract, and Kimi's five
mutation-detecting assertions. Every pre-spawn rejection runs with a missing interpreter."""
import json, math, os, pathlib, random, subprocess, sys, tempfile, time, unittest
import numpy as np
import mpmath as mp
BIN = str(pathlib.Path(sys.argv.pop(1)).resolve())
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests/oracle'))
from s1 import read_safetensors
import s8
import s8b2 as ref

mp.mp.dps = 30
NO_PYTHON = dict(os.environ, PYCLI_PYTHON='/no/such/python')
INF = float('inf')


def call(op, params, rc=0, env=None):
    p = subprocess.run([BIN, '--input-json', json.dumps({'op': op, 'params': params})], capture_output=True, text=True, env=env)
    assert p.returncode == rc, (op, rc, p.returncode, p.stdout[:1500], p.stderr[:600])
    return json.loads(p.stdout)


def rejected(op, params, code=None, rc=2):
    r = call(op, params, rc, NO_PYTHON)
    if code is not None:
        assert r['error']['data']['tool_code'] == code, (code, r)
    return r


def rec(edges, w='w'):
    return [{'u': a, 'v': b, w: c} for a, b, c in edges]


G = dict(from_column='u', to_column='v')
F = dict(from_column='u', to_column='v', capacity_column='c')


def values(dist, fn, x=None, params=None, **extra):
    q = dict(dist=dist, fn=fn, params=params or {}, **extra)
    if x is not None:
        q['x'] = x
    return call('stats_dist', q)['values']


def close(testcase, got, want, rel, what='', atol=0.0):
    for g, w in zip(got, want):
        w = float(w)
        testcase.assertLessEqual(abs(g - w), rel * abs(w) + atol + 1e-300, (what, g, w))


def tight_hops(adj, dist, source):
    """Fewest edges among the shortest paths: breadth-first over the tight edges dist[u] + w == dist[v] (independent of the worker's walk)."""
    hops, queue = {source: 0}, [source]
    for u in queue:
        for v, w in adj[u].items():
            if v in dist and v not in hops and abs(dist[u] + w - dist[v]) <= 1e-9 * max(1.0, abs(dist[v])):
                hops[v] = hops[u] + 1
                queue.append(v)
    return hops


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir='.')
        self.dir = pathlib.Path(self.tmp.name).resolve()
        self.addCleanup(self.tmp.cleanup)
        self.count = 0

    def p(self, name):
        return str(self.dir / name)

    def edges(self, rows, header=('u', 'v', 'w'), name=None):
        self.count += 1
        name = name or 'e%d.npz' % self.count
        import csv, io
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator='\n')
        writer.writerow(header)
        writer.writerows(rows)
        (self.dir / (name + '.csv')).write_text(buffer.getvalue())
        call('frame_import', dict(csv=self.p(name + '.csv'), out=self.p(name)))
        return self.p(name)


# =================================================================================================================== graphs
class GraphReview(Base):
    def test_float_tolerance_never_drops_flow_edges(self):
        """Argon B1: the verification tolerance follows the flow value, not the largest capacity; branch flows of 0.6 behind a 1e9 link survive."""
        edges = [('s', 'u', 1e9), ('u', 'a', 0.6), ('a', 't', 0.6), ('u', 'b', 0.6), ('b', 't', 0.6)]
        r = call('graph_flow', dict(records=rec(edges, 'c'), kind='max_flow', source='s', sink='t', **F))
        self.assertEqual(r['status'], 'OK', r)
        self.assertAlmostEqual(r['value'], 1.2, places=12)
        self.assertEqual(r['n_flow_edges'], 5)
        self.assertAlmostEqual(r['tolerance'], 1.2e-9, places=18)
        got = {(a, b): f for a, b, f in zip(r['flow_from'], r['flow_to'], r['flow'])}
        self.assertAlmostEqual(got[('u', 'a')], 0.6, places=12)
        self.assertAlmostEqual(got[('b', 't')], 0.6, places=12)
        r = call('graph_flow', dict(records=rec([('s', 'u', 1e9), ('u', 't', 0.5)], 'c'), kind='max_flow', source='s', sink='t', **F))
        self.assertEqual((r['status'], r['n_flow_edges']), ('OK', 2))
        self.assertEqual(sorted(r['flow']), [0.5, 0.5])
        c = call('graph_flow', dict(records=rec(edges, 'c'), kind='min_cut', source='s', sink='t', **F))
        self.assertAlmostEqual(c['value'], 1.2, places=12)
        self.assertAlmostEqual(sum(c['cut_capacity']), 1.2, places=12)
        tiny = call('graph_flow', dict(records=rec([(1, 2, 1e-12), (2, 3, 1e-12)], 'c'), kind='max_flow', source=1, sink=3, **F))
        self.assertEqual((tiny['value'], tiny['n_flow_edges']), (1e-12, 2))  # a tiny flow is still listed

    def test_capacity_weight_and_cost_caps_before_spawn(self):
        """Kimi 1 and Fable m5: capacities <= 1e15 for every kind (min_cost_flow included), costs <= 1e12, path weights <= 1e15, all checked before any spawn."""
        mcf = dict(kind='min_cost_flow', cost_column='k', **F)
        r = call('graph_flow', dict(mcf, records=[{'u': 1, 'v': 2, 'c': 10 ** 13, 'k': 1}], source=1, sink=2, flow_value=1))
        self.assertEqual((r['status'], r['cost']), ('OK', 1))
        r = call('graph_flow', dict(mcf, records=[{'u': 1, 'v': 2, 'c': 10 ** 15, 'k': 10 ** 12}], supplies=[[1, 10 ** 15], [2, -10 ** 15]]))
        self.assertEqual((r['status'], r['cost_exact'], r['flow']), ('OK', str(10 ** 27), [10 ** 15]))
        rejected('graph_flow', dict(mcf, records=[{'u': 1, 'v': 2, 'c': 1, 'k': 10 ** 12 + 1}], source=1, sink=2, flow_value=1), 'BAD_DATA')
        rejected('graph_flow', dict(mcf, records=[{'u': 1, 'v': 2, 'c': 10 ** 15 + 1, 'k': 1}], source=1, sink=2, flow_value=1), 'BAD_DATA')
        rejected('graph_flow', dict(records=rec([(1, 2, 2e15)], 'c'), kind='max_flow', source=1, sink=2, **F), 'BAD_DATA')
        rejected('graph_flow', dict(records=rec([(1, 2, -1)], 'c'), kind='min_cut', source=1, sink=2, **F), 'BAD_DATA')
        f = self.edges([(1, 2, 5, 10 ** 12 + 1)], header=('u', 'v', 'c', 'k'))
        rejected('graph_flow', dict(mcf, frame=f, source=1, sink=2, flow_value=1), 'BAD_DATA')
        rejected('graph_paths', dict(records=rec([(1, 2, 1e308), (2, 3, 1e308)]), weight_column='w', source=1, **G), 'BAD_DATA')
        rejected('graph_paths', dict(records=rec([(1, 2, -2e15)]), weight_column='w', source=1, method='bellman_ford', **G), 'BAD_DATA')
        big = self.edges([(1, 2, 2e15)])
        rejected('graph_paths', dict(frame=big, weight_column='w', source=1, **G), 'BAD_DATA')
        ok = call('graph_paths', dict(records=rec([(1, 2, 1e15), (2, 3, 1e15)]), weight_column='w', source=1, target=3, **G))
        self.assertEqual(ok['length'], 2e15)

    def test_negative_self_loops_are_negative_cycles(self):
        """Fable F2 and Argon M2: a negative self loop is a negative cycle for bellman_ford (reachable) and all pairs (anywhere); non-negative
        self loops are ignored everywhere without BAD_PARAM or DUPLICATE_EDGE surprises; min_cost_flow saturates negative-cost loops."""
        loop = rec([(1, 1, -2), (1, 2, 1)])
        r = call('graph_paths', dict(records=loop, weight_column='w', source=1, method='bellman_ford', **G))
        self.assertEqual((r['status'], r['cycle']), ('NEGATIVE_CYCLE', [1]))
        self.assertNotIn('distance', r)
        away = call('graph_paths', dict(records=rec([(1, 2, 1), (3, 3, -2)]), weight_column='w', source=1, method='bellman_ford', **G))
        self.assertEqual((away['status'], away['distance']), ('OK', [0, 1]))  # the loop is not reachable from 1
        r = call('graph_paths_to', dict(records=loop, weight_column='w', method='bellman_ford', all_pairs=True, out=self.p('ap.safetensors'), **G), rc=1)
        self.assertEqual((r['error']['data']['tool_code'], r['error']['data']['cycle']), ('NEGATIVE_CYCLE', [1]))
        self.assertFalse(os.path.exists(self.p('ap.safetensors')))
        rejected('graph_paths', dict(records=loop, weight_column='w', source=1, **G), 'BAD_PARAM')  # dijkstra: a negative weight, as documented
        frame = self.edges([(1, 1, -2), (1, 2, 1)])
        rejected('graph_paths', dict(frame=frame, weight_column='w', source=1, target=2, k_shortest=2, **G), 'BAD_PARAM')
        harmless = rec([('a', 'a', 0), ('a', 'a', 3), ('a', 'b', 2), ('b', 'b', 1)])
        for method in ('dijkstra', 'bellman_ford'):
            r = call('graph_paths', dict(records=harmless, weight_column='w', source='a', target='b', method=method, **G))
            self.assertEqual((r['status'], r['length'], r['path'], r['n_edges']), ('OK', 2, ['a', 'b'], 1), method)
        k = call('graph_paths', dict(records=harmless, weight_column='w', source='a', target='b', k_shortest=3, **G))
        self.assertEqual(k['k_found'], 1)
        mcf = dict(kind='min_cost_flow', cost_column='k', **F)
        r = call('graph_flow', dict(mcf, records=[{'u': 1, 'v': 1, 'c': 5, 'k': -1}, {'u': 1, 'v': 2, 'c': 5, 'k': 1}], supplies=[[1, 1], [2, -1]]))
        self.assertEqual((r['status'], r['cost']), ('OK', -4))  # 5 units around the loop at -1 each, 1 unit to node 2
        self.assertEqual(sorted(zip(r['flow_from'], r['flow_to'], r['flow'])), [(1, 1, 5), (1, 2, 1)])
        two = [{'u': 'a', 'v': 'a', 'c': 5, 'k': 1}, {'u': 'a', 'v': 'a', 'c': 5, 'k': 2}, {'u': 'a', 'v': 'b', 'c': 5, 'k': 1}]
        r = call('graph_flow', dict(mcf, records=two, source='a', sink='b', flow_value=1))
        self.assertEqual((r['status'], r['cost'], r['flow']), ('OK', 1, [1]))  # two non-negative self loops: ignored, not DUPLICATE_EDGE
        twoneg = [{'u': 'a', 'v': 'a', 'c': 5, 'k': -1}, {'u': 'a', 'v': 'a', 'c': 2, 'k': -3}, {'u': 'a', 'v': 'b', 'c': 5, 'k': 1}]
        r = call('graph_flow', dict(mcf, records=twoneg, source='a', sink='b', flow_value=1))
        self.assertEqual((r['status'], r['cost'], sorted(zip(r['flow_from'], r['flow_to'], r['flow']))), ('OK', -10, [('a', 'a', 7), ('a', 'b', 1)]))
        out = self.p('loops.npz')
        t = call('graph_flow_to', dict(mcf, records=twoneg, source='a', sink='b', flow_value=1, out=out))
        self.assertEqual((t['cost'], t['n_flow_edges']), (-10, 2))
        self.assertEqual(sorted(zip(*[s8.read_table(out)['data'][c] for c in ('from', 'to', 'flow')])), [('a', 'a', 7), ('a', 'b', 1)])
        for kind in ('max_flow', 'min_cut'):
            r = call('graph_flow', dict(records=rec([('a', 'a', 5), ('a', 'a', 7), ('a', 'b', 2)], 'c'), kind=kind, source='a', sink='b', **F))
            self.assertEqual((r['status'], r['value']), ('OK', 2), kind)
        self.assertEqual(call('graph_flow', dict(records=rec([(1, 1, 5), (1, 2, 3)], 'c'), kind='max_flow', source=1, sink=2, **F))['flow'], [3])

    def test_all_pairs_negative_cycle_carries_a_witness(self):
        """Kimi 2: the all-pairs NEGATIVE_CYCLE names a cycle of negative total weight."""
        edges = [(1, 2, 1), (2, 3, -3), (3, 1, 1), (4, 5, 2)]
        r = call('graph_paths_to', dict(records=rec(edges), weight_column='w', method='bellman_ford', all_pairs=True, out=self.p('nc.safetensors'), **G), rc=1)
        data = r['error']['data']
        self.assertEqual(data['tool_code'], 'NEGATIVE_CYCLE')
        cycle = data['cycle']
        weight = {(a, b): w for a, b, w in edges}
        self.assertGreaterEqual(len(cycle), 2)
        self.assertLess(sum(weight[(a, b)] for a, b in zip(cycle, cycle[1:] + cycle[:1])), 0)
        self.assertFalse(os.path.exists(self.p('nc.safetensors')))

    def test_inline_ids_beyond_int64_nul_and_length(self):
        """Opus 1, Opus 2 and Argon M3: unsigned ids beyond int64 are BAD_DTYPE before spawn, NUL in a string id is BAD_DATA, ids are at most 256 bytes."""
        huge = 2 ** 63
        rejected('graph_paths', dict(records=[{'u': huge, 'v': 1}, {'u': 1, 'v': 2}], source=1, **G), 'BAD_DTYPE')
        rejected('graph_paths', dict(records=[{'u': 2 ** 64 - 1, 'v': 1}], source=1, **G), 'BAD_DTYPE')
        rejected('graph_paths_to', dict(records=[{'u': huge, 'v': 1}], source=1, out=self.p('u.npz'), **G), 'BAD_DTYPE')
        rejected('graph_paths', dict(records=rec([(1, 2, 1)]), weight_column='w', source=huge, **G), 'BAD_PARAM')
        rejected('graph_paths', dict(records=rec([(1, 2, 1)]), weight_column='w', source=1, target=huge, **G), 'BAD_PARAM')
        rejected('graph_flow', dict(records=rec([(1, 2, 1)], 'c'), kind='max_flow', source=1, sink=huge, **F), 'BAD_PARAM')
        mcf = dict(kind='min_cost_flow', cost_column='k', records=[{'u': 1, 'v': 2, 'c': 5, 'k': 1}], **F)
        rejected('graph_flow', dict(mcf, supplies=[[1, huge], [2, -1]]), 'BAD_PARAM')
        rejected('graph_flow', dict(mcf, source=1, sink=2, flow_value=huge), 'BAD_PARAM')
        rejected('graph_paths', dict(records=[{'u': 'x\u0000y', 'v': 'a'}], source='a', **G), 'BAD_DATA')
        rejected('graph_paths', dict(records=[{'u': 'a', 'v': 'b'}], source='x\u0000y', **G), 'UNKNOWN_NODE')
        long_id = 'n' * 257
        rejected('graph_paths', dict(records=[{'u': long_id, 'v': 'a'}], source='a', **G), 'INPUT_TOO_LARGE')
        rejected('graph_flow', dict(records=[{'u': long_id, 'v': 'a', 'c': 1}], kind='max_flow', source=long_id, sink='a', **F), 'INPUT_TOO_LARGE')
        frame = self.edges([(long_id, 'a', 1)])
        rejected('graph_paths', dict(frame=frame, weight_column='w', source='a', **G), 'INPUT_TOO_LARGE')
        edge_id = 'm' * 256
        out = self.p('ids.npz')
        r = call('graph_paths_to', dict(records=[{'u': edge_id, 'v': 'a'}], source=edge_id, out=out, **G))
        self.assertEqual(r['status'], 'OK')
        self.assertEqual(sorted(s8.read_table(out)['data']['node']), sorted(['a', edge_id]))
        ok = call('graph_paths', dict(records=[{'u': 2 ** 63 - 1, 'v': -2 ** 63}], source=2 ** 63 - 1, target=-2 ** 63, **G))
        self.assertEqual(ok['path'], [2 ** 63 - 1, -2 ** 63])

    def test_output_too_large_is_command_failed_with_a_usable_hint(self):
        """Argon M1: an inline result that does not fit is OUTPUT_TOO_LARGE with exit code 1, and the k_shortest hint does not point at a twin that rejects it."""
        chain = [{'u': i, 'v': i + 1} for i in range(305)] + [{'u': i, 'v': i + 2} for i in range(3)]
        r = call('graph_paths', dict(records=chain, source=0, target=305, k_shortest=4, **G), rc=1)
        self.assertEqual((r['error']['code'], r['error']['data']['tool_code']), ('COMMAND_FAILED', 'OUTPUT_TOO_LARGE'))
        self.assertIn('k_shortest', r['error']['message'])
        self.assertNotIn('_to', r['error']['message'])
        wide = self.edges([(i, i + 1, 1) for i in range(400)])
        r = call('graph_flow', dict(frame=wide, capacity_column='w', kind='max_flow', source=0, sink=400, **G), rc=1)
        self.assertEqual((r['error']['code'], r['error']['data']['tool_code']), ('COMMAND_FAILED', 'OUTPUT_TOO_LARGE'))
        self.assertIn('graph_flow_to', r['error']['message'])
        listing = self.edges([(i, i + 1, 1) for i in range(450)])  # 451 nodes: known before spawn
        rejected('graph_paths', dict(frame=listing, weight_column='w', source=0, **G), 'OUTPUT_TOO_LARGE', rc=1)

    def test_min_cost_flow_cost_beyond_int64_is_exact(self):
        """Fable m4: a total cost outside int64 is not rounded through a double silently; cost_exact carries the integer."""
        mcf = dict(kind='min_cost_flow', cost_column='k', **F)
        r = call('graph_flow', dict(mcf, records=[{'u': 1, 'v': 2, 'c': 10 ** 12, 'k': -10 ** 12}, {'u': 2, 'v': 1, 'c': 10 ** 12, 'k': 0}], supplies=[[1, 0], [2, 0]]))
        self.assertEqual((r['status'], r['cost_exact']), ('OK', str(-10 ** 24)))
        self.assertEqual(r['cost'], float(-10 ** 24))
        self.assertTrue(any('cost_exact' in w['message'] for w in r['warnings']), r['warnings'])
        r = call('graph_flow', dict(mcf, records=[{'u': 1, 'v': 2, 'c': 10 ** 7, 'k': 10 ** 12}], supplies=[[1, 10 ** 7], [2, -10 ** 7]]))
        self.assertEqual((r['cost'], r['cost_exact']), (1e19, str(10 ** 19)))
        small = call('graph_flow', dict(mcf, records=[{'u': 1, 'v': 2, 'c': 3, 'k': 2}], supplies=[[1, 3], [2, -3]]))
        self.assertEqual(small['cost'], 6)
        self.assertNotIn('cost_exact', small)

    def test_twin_hops_are_exact_and_zero_row_tables_round_trip(self):
        """Kimi 4.1 and 4.2: the node table's hop counts equal the independent tight-edge BFS; zero-flow and zero-cut tables are written and readable."""
        rng = random.Random(90)
        nodes = ['n%02d' % i for i in range(20)]
        edges = ref.random_graph(rng, 20, 50, True, nodes=nodes, weights=(1, 4))
        edges += [(nodes[0], nodes[1], 1), (nodes[1], nodes[2], 1), (nodes[0], nodes[2], 2)]  # a tie: one hop against two
        f = self.edges(edges)
        out = self.p('hops.npz')
        call('graph_paths_to', dict(frame=f, weight_column='w', source=nodes[0], out=out, **G))
        t = s8.read_table(out)
        adj = ref.adjacency(edges, True)
        dist = ref.dijkstra(adj, nodes[0])
        hops = tight_hops(adj, dist, nodes[0])
        for k, node in enumerate(t['data']['node']):
            if node in dist:
                self.assertEqual(t['data']['hops'][k], hops[node], node)
        self.assertEqual(t['data']['hops'][t['data']['node'].index(nodes[2])], 1)
        disconnected = self.edges([(1, 2, 3), (3, 4, 5)])
        for kind, columns in (('max_flow', ['from', 'to', 'flow']), ('min_cut', ['from', 'to', 'capacity'])):
            out = self.p('zero-%s.npz' % kind)
            r = call('graph_flow_to', dict(frame=disconnected, capacity_column='w', kind=kind, source=1, sink=4, out=out, **G))
            self.assertEqual((r['status'], r['value'], r['shape']), ('OK', 0, [0, 3]))
            table = s8.read_table(out)
            self.assertEqual((table['columns'], table['n'], table['data']['from']), (columns, 0, []))


class GraphFuzz(Base):
    """Fable's fuzzers as fixed-seed regressions: random graphs against pure-Python algorithms, random networks against the LP oracle."""

    def test_random_graphs_against_pure_python(self):
        rng = random.Random(2024)
        for case in range(24):
            n = rng.randint(2, 7)
            strid = case % 3 == 2
            directed = case % 4 != 3
            wkind = ('int', 'float', 'neg', 'none')[case % 4] if case % 5 else 'int'
            ids = [('n%d' % i if strid else i * rng.choice([1, 3, 7])) for i in range(n)]
            m = rng.randint(0, int(n * (n - 1) * rng.choice([0.5, 1.0, 1.4])))
            edges = []
            for _ in range(m):
                u = rng.choice(ids)
                v = rng.choice(ids) if rng.random() > 0.1 else u
                w = {'int': rng.randint(0, 5), 'float': round(rng.uniform(0, 3), 2), 'neg': rng.randint(-3, 5), 'none': 1}[wkind]
                edges.append((u, v, w))
            if not edges:
                continue
            records = rec(edges)
            base = dict(records=records, directed=directed, **G)
            if wkind != 'none':
                base['weight_column'] = 'w'
            adj = ref.adjacency(edges, directed)
            nodes = sorted(adj)
            loops = {u for u, v, w in edges if u == v and w < 0}
            s, t = rng.choice(nodes), rng.choice(nodes)
            negative = any(w < 0 for _, _, w in edges)
            oracle, cyc = ref.bellman_ford(adj, s)
            reachable_loop = any(u in oracle for u in loops)
            if negative or loops:
                r = call('graph_paths', dict(base, source=s, target=t, method='bellman_ford'))
                if cyc or reachable_loop:
                    self.assertEqual(r['status'], 'NEGATIVE_CYCLE', (case, r))
                    cycle = r['cycle']
                    total = sum(adj[a][b] if a != b else min(w for u, v, w in edges if u == v == a) for a, b in zip(cycle, cycle[1:] + cycle[:1]))
                    self.assertLess(total, 0, (case, cycle))
                    continue
            for method in (('bellman_ford',) if negative else ('dijkstra', 'bellman_ford')):
                r = call('graph_paths', dict(base, source=s, method=method))
                self.assertEqual(r['status'], 'OK', (case, method, r))
                got = dict(zip(r['nodes'], r['distance']))
                self.assertEqual(set(got), set(oracle), (case, method))
                for k in oracle:
                    self.assertAlmostEqual(got[k], oracle[k], places=9, msg=(case, method, k))
                self.assertEqual(r['n_unreachable'], len(nodes) - len(oracle))
                r = call('graph_paths', dict(base, source=s, target=t, method=method))
                if t not in oracle:
                    self.assertEqual(r['status'], 'UNREACHABLE', (case, r))
                else:
                    path = r['path']
                    self.assertEqual((path[0], path[-1], len(set(path))), (s, t, len(path)), (case, path))
                    total = 0
                    for a, b in zip(path, path[1:]):
                        self.assertIn(b, adj[a], (case, path))
                        total += adj[a][b]
                    self.assertAlmostEqual(total, oracle[t], places=9, msg=case)
                    self.assertAlmostEqual(r['length'], oracle[t], places=9, msg=case)
                    self.assertEqual(r['hops'], tight_hops(adj, oracle, s)[t], (case, path))
            bfs = call('graph_paths', dict({k: v for k, v in base.items() if k != 'weight_column'}, source=s, method='bfs'))
            self.assertEqual(dict(zip(bfs['nodes'], bfs['distance'])), ref.bfs_hops(adj, s), case)
            if not negative and s != t:
                k = rng.randint(1, 5)
                every = ref.simple_paths(adj, s, t)
                r = call('graph_paths', dict(base, source=s, target=t, k_shortest=k))
                if not every:
                    self.assertEqual(r['status'], 'UNREACHABLE', case)
                else:
                    self.assertEqual(r['k_found'], min(k, len(every)), case)
                    self.assertEqual([round(q['length'], 9) for q in r['paths']], [round(e[0], 9) for e in every[:k]], case)
                    for q in r['paths']:
                        self.assertEqual((q['nodes'][0], q['nodes'][-1], len(set(q['nodes']))), (s, t, len(q['nodes'])), case)
                        self.assertAlmostEqual(sum(adj[a][b] for a, b in zip(q['nodes'], q['nodes'][1:])), q['length'], places=9)
                    self.assertEqual(len({tuple(q['nodes']) for q in r['paths']}), len(r['paths']), case)
            if not strid and wkind != 'none' and case % 2 == 0:
                out = self.p('ap%d.safetensors' % case)
                nodes_fw, want, neg_any = ref.floyd_warshall(adj)
                r = call('graph_paths_to', dict(base, method='bellman_ford' if negative else 'dijkstra', all_pairs=True, out=out), rc=1 if (neg_any or loops) else 0)
                if neg_any or loops:
                    self.assertEqual(r['error']['data']['tool_code'], 'NEGATIVE_CYCLE', case)
                    continue
                z = read_safetensors(out)
                self.assertEqual(z['nodes'].tolist(), nodes_fw)
                for i, u in enumerate(nodes_fw):
                    for j, v in enumerate(nodes_fw):
                        w = want[u][v]
                        self.assertTrue(math.isnan(z['distance'][i, j]) if w == INF else abs(z['distance'][i, j] - w) < 1e-9, (case, u, v))

    def lp(self, c, bounds, a_eq=None, b_eq=None, maximize=False):
        q = dict(c=c, bounds=bounds, maximize=maximize)
        if a_eq:
            q.update(a_eq=a_eq, b_eq=b_eq)
        return call('opt_linprog', q)

    def test_random_networks_against_linprog(self):
        rng = random.Random(77)
        for case in range(14):
            n = rng.randint(3, 7)
            strid = case % 4 == 1
            floats = case % 3 == 2
            ids = [('v%d' % i if strid else i) for i in range(n)]
            s, t = ids[0], ids[-1]
            edges = []
            for _ in range(rng.randint(2, int(n * (n - 1) * 1.2))):
                u = rng.choice(ids)
                v = rng.choice(ids) if rng.random() > 0.08 else u
                edges.append((u, v, round(rng.uniform(0, 6), 2) if floats else rng.randint(0, 6)))
            merged = {}
            for u, v, c in edges:
                if u != v:
                    merged[(u, v)] = merged.get((u, v), 0) + c
            nodes = sorted({x for e in edges for x in e[:2]}, key=str)
            if s not in nodes or t not in nodes:
                continue
            E = list(merged)
            obj = [(1 if u == s else 0) - (1 if v == s else 0) for u, v in E]
            rows, rhs = [], []
            for node in nodes:
                if node in (s, t):
                    continue
                row = [(1 if u == node else 0) - (1 if v == node else 0) for u, v in E]
                if any(row):
                    rows.append(row)
                    rhs.append(0)
            lp = self.lp(obj, [[0, float(merged[e])] for e in E], rows, rhs, maximize=True)
            self.assertEqual(lp['status'], 'OK', (case, lp))
            opt = lp['objective']
            tol = 1e-7 * max(1.0, opt)
            r = call('graph_flow', dict(records=rec(edges, 'c'), kind='max_flow', source=s, sink=t, **F))
            self.assertEqual(r['status'], 'OK', (case, r))
            self.assertLessEqual(abs(r['value'] - opt), tol, case)
            self.assertEqual(r['integral'], not floats)
            net = {}
            for u, v, f in zip(r['flow_from'], r['flow_to'], r['flow']):
                self.assertGreater(f, 0)
                self.assertLessEqual(f, merged[(u, v)] + 1e-9, (case, u, v))
                net[u] = net.get(u, 0) - f
                net[v] = net.get(v, 0) + f
            for node in nodes:
                want = -r['value'] if node == s else r['value'] if node == t else 0
                self.assertLessEqual(abs(net.get(node, 0) - want), 1e-9, (case, node))
            c = call('graph_flow', dict(records=rec(edges, 'c'), kind='min_cut', source=s, sink=t, **F))
            side = set(c['source_side'])
            self.assertIn(s, side)
            self.assertNotIn(t, side)
            crossing = {(u, v): x for (u, v), x in merged.items() if u in side and v not in side and x > 0}
            self.assertEqual(set(zip(c['cut_from'], c['cut_to'])), set(crossing), case)
            self.assertLessEqual(abs(sum(crossing.values()) - opt), tol, case)
            if floats:
                continue
            dedup = {}
            for u, v, cc in edges:
                if u != v and (u, v) not in dedup:
                    dedup[(u, v)] = cc
            costs = {e: rng.randint(-3, 5) for e in dedup}
            records = [{'u': u, 'v': v, 'c': cc, 'k': costs[(u, v)]} for (u, v), cc in dedup.items()]
            graph_nodes = sorted({x for e in dedup for x in e}, key=str)
            k = rng.randint(1, min(3, len(graph_nodes) - 1))
            chosen = rng.sample(graph_nodes, k + 1)
            amounts = [rng.randint(1, 4) for _ in range(k)]
            supply = {node: a for node, a in zip(chosen[:k], amounts)}
            supply[chosen[k]] = -sum(amounts)
            E2 = list(dedup)
            rows = [[(1 if u == node else 0) - (1 if v == node else 0) for u, v in E2] for node in graph_nodes]
            lp = self.lp([float(costs[e]) for e in E2], [[0, float(dedup[e])] for e in E2], rows, [float(supply.get(node, 0)) for node in graph_nodes])
            m = call('graph_flow', dict(records=records, kind='min_cost_flow', cost_column='k', supplies=[[node, a] for node, a in supply.items()], **F))
            if lp['status'] == 'INFEASIBLE':
                self.assertEqual(m['status'], 'INFEASIBLE', (case, m))
                continue
            self.assertEqual((lp['status'], m['status']), ('OK', 'OK'), (case, lp['status'], m))
            self.assertLessEqual(abs(m['cost'] - lp['objective']), 1e-6 * max(1.0, abs(lp['objective'])), case)
            flow = dict(zip(zip(m['flow_from'], m['flow_to']), m['flow']))
            for (u, v), f in flow.items():
                self.assertTrue(0 < f <= dedup[(u, v)], (case, u, v, f))
            for node in graph_nodes:
                self.assertEqual(sum(f for (u, v), f in flow.items() if u == node) - sum(f for (u, v), f in flow.items() if v == node), supply.get(node, 0), (case, node))
            self.assertEqual(sum(costs[e] * f for e, f in flow.items()), m['cost'], case)


# ============================================================================================================ distributions
def nct_pdf(x, df, nc):
    """f(x) = integral over s of s phi(x s - nc) f_S(s), S = sqrt(V / df): the integrand peaks sharply at s = nc / x for large nc, so break there."""
    fS = lambda s: 2 * s * df * (s * s * df) ** (df / 2 - 1) * mp.exp(-s * s * df / 2) / (2 ** (df / 2) * mp.gamma(df / 2))
    peak = nc / x if x and nc / x > 0 else 1.0
    width = 6.0 / max(abs(x), 1.0)
    points = sorted({0, max(peak - width, 1e-9), peak, peak + width, max(2 * peak, 2.0)})
    return float(mp.quad(lambda s: s * mp.npdf(x * s - nc) * fS(s), points + [mp.inf]))


class DistReview(unittest.TestCase):
    def test_warm_up_is_lazy_and_inside_the_budget(self):
        """Fable F1 and Argon B2.2: the requested function alone is evaluated, under the budget; the worker never aborts or hangs on validated parameters."""
        start = time.monotonic()
        r = call('stats_dist', dict(dist='zipf', fn='pdf', params={'a': 1.01}, x=[1, 2], time_limit=1.0))
        self.assertEqual(r['status'], 'OK')
        close(self, r['values'], [float(1 / mp.zeta(1.01)), float(2 ** -1.01 / mp.zeta(1.01))], 1e-12)
        self.assertLess(time.monotonic() - start, 10.0)
        r = call('stats_dist', dict(dist='zipf', fn='rvs', params={'a': 1.01}, n=4, seed=1, time_limit=1.0))
        self.assertEqual(r['status'], 'OK')
        r = call('stats_dist', dict(dist='nbinom', fn='cdf', params={'n': 3, 'p': 1e-14}, x=[5], time_limit=1.0))
        close(self, r['values'], [float(sum(mp.binomial(k + 2, k) * mp.mpf('1e-42') * (1 - mp.mpf('1e-14')) ** k for k in range(6)))], 1e-9)
        rejected('stats_dist', dict(dist='nbinom', fn='ppf', params={'n': 3, 'p': 1e-300}, x=[0.5]), 'BAD_PARAM')  # scipy's C quantile search never ends there
        rejected('stats_dist', dict(dist='geom', fn='rvs', params={'p': 1e-16}, n=2, seed=1), 'BAD_PARAM')
        r = call('stats_dist', dict(dist='zipf', fn='pdf', params={'a': 3.0, 'loc': -5000000}, x=[-4999999], time_limit=0.5))
        self.assertEqual(r['status'], 'OK')
        self.assertLess(r['meta']['elapsed_ms'], 3000)
        rejected('stats_dist', dict(dist='nct', fn='cdf', params={'df': 1e-20, 'nc': 1}, x=[1.0]), 'BAD_PARAM')  # Boost aborts the process below the floor
        rejected('stats_dist', dict(dist='nct', fn='pdf', params={'df': 1e-9, 'nc': 1}, x=[1.0]), 'BAD_PARAM')
        ok = call('stats_dist', dict(dist='nct', fn='pdf', params={'df': 1e-8, 'nc': 1}, x=[1.0]))
        self.assertEqual(ok['status'], 'OK')
        slow = call('stats_dist', dict(dist='betabinom', fn='ppf', params={'n': 5000, 'a': 2.0, 'b': 3.0}, x=[0.001 * k for k in range(1, 999)], time_limit=0.3))
        self.assertEqual(slow['status'], 'TIME_LIMIT')  # a long quantile search is the cooperative budget answering, not the kill

    def test_time_budget_covers_evaluations(self):
        """Argon m1: stats_dist evaluations receive the process budget like every S8 operation (PYCLI_TIMEOUT_S shrinks it below time_limit)."""
        env = dict(os.environ, PYCLI_TIMEOUT_S='4')
        start = time.monotonic()
        r = call('stats_dist', dict(dist='betabinom', fn='ppf', params={'n': 5000, 'a': 2.0, 'b': 3.0}, x=[0.001 * k for k in range(1, 999)], time_limit=10.0), env=env)
        self.assertEqual(r['status'], 'TIME_LIMIT', r)  # the process budget (4 - 10 s, floored at 1 s) wins over time_limit: no 4 s kill
        self.assertLess(time.monotonic() - start, 3.9)

    def test_zipf_loc_in_caps_and_per_value_out_of_range(self):
        """Argon B2.1 and Fable m3: the 1e5 summation cap counts terms from loc; a quantile beyond it leaves the other values intact."""
        r = call('stats_dist', dict(dist='zipf', fn='cdf', params={'a': 3.0, 'loc': 200000}, x=[200001, 200003]))
        close(self, r['values'], [float(1 / mp.zeta(3)), float((1 + mp.mpf(2) ** -3 + mp.mpf(3) ** -3) / mp.zeta(3))], 1e-12)
        r = call('stats_dist', dict(dist='zipf', fn='ppf', params={'a': 3.0, 'loc': 200000}, x=[0.5, 0.9]))
        self.assertEqual((r['status'], r['values']), ('OK', [200001.0, 200002.0]))
        rejected('stats_dist', dict(dist='zipf', fn='cdf', params={'a': 3.0, 'loc': 200000}, x=[300001]), 'INPUT_TOO_LARGE')
        rejected('stats_dist', dict(dist='zipf', fn='sf', params={'a': 3.0, 'loc': -100000000}, x=[0]), 'INPUT_TOO_LARGE')
        ok = call('stats_dist', dict(dist='zipf', fn='cdf', params={'a': 3.0, 'loc': -100000000}, x=[-99999999, -100000001]))
        close(self, ok['values'], [float(1 / mp.zeta(3)), 0.0], 1e-12)
        r = call('stats_dist', dict(dist='zipf', fn='ppf', params={'a': 1.2}, x=[0.5, 0.999, 'NA']))
        self.assertEqual((r['status'], r['n_out_of_range']), ('OUT_OF_RANGE', 1))
        self.assertEqual((r['values'][1], r['values'][2]), ('NA', 'NA'))
        self.assertGreaterEqual(r['values'][0], 1)
        r = call('stats_dist', dict(dist='zipf', fn='isf', params={'a': 1.2}, x=[0.5, 1e-3]))
        self.assertEqual((r['status'], r['values'][1]), ('OUT_OF_RANGE', 'NA'))
        with tempfile.TemporaryDirectory(dir='.') as d:
            out = str(pathlib.Path(d).resolve() / 'q.safetensors')
            r = call('stats_dist_to', dict(dist='zipf', fn='ppf', params={'a': 1.2}, x=[0.5, 0.999], out=out), rc=1)
            self.assertEqual(r['error']['data']['tool_code'], 'OUT_OF_RANGE')
            self.assertFalse(os.path.exists(out))

    def test_nct_density_inside_the_cap_does_not_underflow(self):
        """Fable M1: the noncentral t density near its body is never 0 / -Inf inside |nc| <= 1000."""
        r = call('stats_dist', dict(dist='nct', fn='pdf', params={'df': 5, 'nc': 1000}, x=[1000.0, 900.0]))
        c = values('nct', 'cdf', [999.0, 1001.0, 899.0, 901.0], {'df': 5, 'nc': 1000})
        self.assertAlmostEqual(r['values'][0], (c[1] - c[0]) / 2, delta=2e-6 * (c[1] - c[0]))
        self.assertAlmostEqual(r['values'][1], (c[3] - c[2]) / 2, delta=2e-6 * (c[3] - c[2]))
        lp = values('nct', 'logpdf', [1000.0], {'df': 5, 'nc': 1000})
        self.assertAlmostEqual(lp[0], math.log(r['values'][0]), places=9)
        close(self, values('nct', 'pdf', [700.0, 650.0], {'df': 5, 'nc': 700}), [nct_pdf(700, 5, 700), nct_pdf(650, 5, 700)], 1e-6)
        close(self, values('nct', 'pdf', [0.5, 2.5], {'df': 7, 'nc': 1.3}), [nct_pdf(0.5, 7, 1.3), nct_pdf(2.5, 7, 1.3)], 1e-7)  # scipy's own path is untouched
        self.assertEqual(values('nct', 'pdf', [-50.0], {'df': 5, 'nc': 1000}), [0.0])  # a true far-tail zero stays zero

    def test_vonmises_lives_on_the_circle(self):
        """Fable F3: cdf/sf/logcdf/logsf wrap x into loc + scale * [-pi, pi]; ppf(0) and ppf(1) are the circle ends."""
        pi = math.pi
        inside = values('vonmises', 'cdf', [10 - 4 * pi, 4 - 2 * pi], {'kappa': 1.0})
        outside = values('vonmises', 'cdf', [10.0, 4.0], {'kappa': 1.0})
        close(self, outside, inside, 1e-12)
        self.assertTrue(all(0 <= v <= 1 for v in outside))
        sf = values('vonmises', 'sf', [10.0, -10.0], {'kappa': 1.0})
        close(self, sf, [1 - outside[0], 1 - values('vonmises', 'cdf', [-10 + 4 * pi], {'kappa': 1.0})[0]], 1e-9)
        self.assertLessEqual(values('vonmises', 'logcdf', [10.0], {'kappa': 1.0})[0], 0.0)
        self.assertLessEqual(values('vonmises', 'logsf', [10.0], {'kappa': 1.0})[0], 0.0)
        self.assertEqual(values('vonmises', 'cdf', [-pi, pi], {'kappa': 2.0}), [0.0, 1.0])
        shifted = values('vonmises', 'cdf', [-3.0, 3.5], {'kappa': 1.0, 'loc': 3.0})
        close(self, shifted, values('vonmises', 'cdf', [-3.0 + 2 * pi, 3.5], {'kappa': 1.0, 'loc': 3.0}), 1e-12)
        self.assertTrue(0 <= shifted[0] <= 1)
        close(self, values('vonmises', 'ppf', [0.0, 1.0, 0.5], {'kappa': 1.0}), [-pi, pi, 0.0], 1e-12, atol=1e-15)
        close(self, values('vonmises', 'ppf', [0.0, 1.0], {'kappa': 1.0, 'loc': 0.5, 'scale': 2.0}), [0.5 - 2 * pi, 0.5 + 2 * pi], 1e-12)
        close(self, values('vonmises', 'isf', [0.0, 1.0], {'kappa': 1.0}), [pi, -pi], 1e-12)
        close(self, values('vonmises', 'pdf', [0.4, 0.4 + 2 * pi], {'kappa': 1.0}), [values('vonmises', 'pdf', [0.4], {'kappa': 1.0})[0]] * 2, 1e-12)

    def test_levy_stable_alpha_one_single_point(self):
        """Fable F4: alpha = 1 with beta != 0 at one point is a value, not PY_ERROR."""
        r = call('stats_dist', dict(dist='levy_stable', fn='cdf', params={'alpha': 1.0, 'beta': 0.5}, x=[0.0]))
        self.assertEqual(r['status'], 'OK')
        two = values('levy_stable', 'cdf', [0.0, 1.0], {'alpha': 1.0, 'beta': 0.5})
        self.assertAlmostEqual(r['values'][0], two[0], places=12)
        self.assertAlmostEqual(values('levy_stable', 'pdf', [0.0], {'alpha': 1.0, 'beta': -0.5})[0], values('levy_stable', 'pdf', [0.0, 2.0], {'alpha': 1.0, 'beta': -0.5})[0], places=12)
        with tempfile.TemporaryDirectory(dir='.') as d:
            out = str(pathlib.Path(d).resolve() / 'l.safetensors')
            r = call('stats_dist_to', dict(dist='levy_stable', fn='cdf', params={'alpha': 1.0, 'beta': 0.5}, x=[0.0], out=out))
            self.assertEqual((r['status'], r['n_values']), ('OK', 1))

    def test_lognorm_mu_and_nct_df_caps(self):
        """Fable m1: lognorm mu is capped where exp(mu) overflows (before spawn); the nct df floor is in the table too."""
        rejected('stats_dist', dict(dist='lognorm', fn='cdf', params={'mu': 710.0, 'sigma': 1.0}, x=[1.0]), 'BAD_PARAM')
        rejected('stats_dist', dict(dist='lognorm', fn='rvs', params={'mu': -701.0, 'sigma': 1.0}, n=2, seed=1), 'BAD_PARAM')
        r = call('stats_dist', dict(dist='lognorm', fn='cdf', params={'mu': 700.0, 'sigma': 1.0}, x=[math.exp(700.0)]))
        self.assertAlmostEqual(r['values'][0], 0.5, places=9)

    def test_weibull_moments_for_sharp_shapes(self):
        """Fable m2: weibull_min moments keep their digits for large c (Gumbel-min limits: skew -1.1395, excess kurtosis 2.4)."""
        m = call('stats_dist', dict(dist='weibull_min', fn='moments', params={'c': 1e6}))['moments']
        self.assertAlmostEqual(m['skew'], float(-12 * mp.sqrt(6) * mp.zeta(3) / mp.pi ** 3), delta=2e-5)
        self.assertAlmostEqual(m['kurtosis'], 2.4, delta=2e-4)
        self.assertAlmostEqual(m['var'], math.pi ** 2 / 6 / 1e12, delta=1e-16)
        for c in (0.7, 1.7, 100.0):
            m = call('stats_dist', dict(dist='weibull_min', fn='moments', params={'c': c, 'loc': 1.0, 'scale': 2.0}))['moments']
            g = [mp.gamma(1 + k / mp.mpf(c)) for k in range(1, 5)]
            mu, var = g[0], g[1] - g[0] ** 2
            skew = (g[2] - 3 * g[0] * g[1] + 2 * g[0] ** 3) / var ** 1.5
            kurt = (g[3] - 4 * g[0] * g[2] + 6 * g[0] ** 2 * g[1] - 3 * g[0] ** 4) / var ** 2 - 3
            close(self, [m['mean'], m['var'], m['skew'], m['kurtosis']], [1 + 2 * mu, 4 * var, skew, kurt], 1e-8, c)

    def test_geom_and_nbinom_rvs_saturation_is_a_status(self):
        """Fable m3: numpy's integer samplers saturate at 2^63 - 1 for tiny p; that is NUMERICAL_FAILURE, never a value."""
        rejected('stats_dist', dict(dist='geom', fn='rvs', params={'p': 1e-19}, n=4, seed=1), 'BAD_PARAM')  # p >= 1e-15 before spawn
        rejected('stats_dist', dict(dist='nbinom', fn='cdf', params={'n': 3, 'p': 1e-300}, x=[1]), 'BAD_PARAM')
        r = call('stats_dist', dict(dist='nbinom', fn='rvs', params={'n': 1e12, 'p': 1e-15}, n=3, seed=1))  # mean 1e27: numpy cannot draw it
        self.assertEqual(r['status'], 'NUMERICAL_FAILURE')
        self.assertNotIn('values', r)
        ok = call('stats_dist', dict(dist='geom', fn='rvs', params={'p': 1e-15}, n=4, seed=1))
        self.assertEqual(ok['status'], 'OK')
        self.assertTrue(all(1 <= v < 2 ** 63 - 1 for v in ok['values']))
        ok = call('stats_dist', dict(dist='geom', fn='rvs', params={'p': 1e-6}, n=1000, seed=2))
        self.assertEqual(ok['status'], 'OK')
        self.assertLess(abs(sum(ok['values']) / 1000 - 1e6), 2e5)

    def test_rvs_prefix_property_for_every_family(self):
        """Fable M2: a shorter n is a prefix of a longer one for all 31 families (two PCG64 streams for the compound samplers)."""
        families = [('norm', {}), ('t', {'df': 5}), ('uniform', {'low': 0, 'high': 2}), ('expon', {'scale': 2}), ('gamma', {'shape': 2, 'scale': 1}), ('beta', {'a': 2, 'b': 3}),
                    ('lognorm', {'mu': 0, 'sigma': 1}), ('chi2', {'df': 3}), ('poisson', {'lam': 2}), ('binom', {'n': 5, 'p': 0.3}), ('genextreme', {'c': 0.2}), ('genpareto', {'c': 0.2}),
                    ('skewnorm', {'a': 3}), ('levy_stable', {'alpha': 1.5, 'beta': 0.3}), ('levy_stable', {'alpha': 1.0, 'beta': 0.5}), ('johnsonsu', {'a': 1, 'b': 2}), ('nct', {'df': 5, 'nc': 1}),
                    ('ncf', {'dfn': 3, 'dfd': 7, 'nc': 1}), ('vonmises', {'kappa': 2}), ('truncnorm', {'a': -1, 'b': 2}), ('weibull_min', {'c': 2}), ('cauchy', {}), ('laplace', {}),
                    ('logistic', {}), ('gumbel_r', {}), ('pareto', {'b': 3}), ('rayleigh', {}), ('f', {'dfn': 3, 'dfd': 7}), ('zipf', {'a': 2}), ('betabinom', {'n': 5, 'a': 2, 'b': 3}),
                    ('nbinom', {'n': 3, 'p': 0.5}), ('geom', {'p': 0.3})]
        for dist, params in families:
            a = call('stats_dist', dict(dist=dist, fn='rvs', params=params, n=7, seed=9))
            b = call('stats_dist', dict(dist=dist, fn='rvs', params=params, n=23, seed=9))
            self.assertEqual((a['status'], b['status']), ('OK', 'OK'), dist)
            self.assertEqual(a['values'], b['values'][:7], dist)
            self.assertEqual(b['generator'], 'PCG64')

    def test_compound_samplers_match_their_laws(self):
        """The two-stream samplers for nct, skewnorm, betabinom and levy_stable keep the right distribution (moments and quantiles from closed forms)."""
        n = 40000
        with tempfile.TemporaryDirectory(dir='.') as d:
            def draw(dist, params, seed):
                out = str(pathlib.Path(d).resolve() / ('%s%d.safetensors' % (dist, seed)))
                call('stats_dist_to', dict(dist=dist, fn='rvs', params=params, n=n, seed=seed, out=out))
                return read_safetensors(out)['x']
            df, nc = 6.0, 1.5
            x = draw('nct', {'df': df, 'nc': nc}, 11)
            mean = nc * math.sqrt(df / 2) * math.gamma((df - 1) / 2) / math.gamma(df / 2)
            var = df * (1 + nc * nc) / (df - 2) - mean * mean
            self.assertLess(abs(float(np.mean(x)) - mean), 5 * math.sqrt(var / n))
            self.assertLess(abs(float(np.var(x)) - var) / var, 0.08)
            a = 3.0
            delta = a / math.sqrt(1 + a * a)
            x = draw('skewnorm', {'a': a, 'loc': 1.0, 'scale': 2.0}, 12)
            mean, var = 1.0 + 2.0 * delta * math.sqrt(2 / math.pi), 4.0 * (1 - 2 * delta * delta / math.pi)
            self.assertLess(abs(float(np.mean(x)) - mean), 5 * math.sqrt(var / n))
            self.assertLess(abs(float(np.var(x)) - var) / var, 0.03)
            x = draw('betabinom', {'n': 10, 'a': 2.0, 'b': 3.0}, 13)
            self.assertLess(abs(float(np.mean(x)) - 4.0), 5 * math.sqrt(10 * 2 * 3 * 15 / (25 * 6) / n))
            self.assertTrue(((x >= 0) & (x <= 10) & (x == np.floor(x))).all())
            x = draw('levy_stable', {'alpha': 2.0, 'beta': 0.0}, 14)  # N(0, 2)
            self.assertLess(abs(float(np.mean(x))), 5 * math.sqrt(2 / n))
            self.assertLess(abs(float(np.var(x)) - 2.0) / 2.0, 0.03)
            x = draw('levy_stable', {'alpha': 1.0, 'beta': 0.0}, 15)  # Cauchy: KS against the closed form
            self.assertLess(s8.ks_statistic(x.tolist(), lambda v: 0.5 + math.atan(v) / math.pi), 1.95 / math.sqrt(n))
            x = draw('levy_stable', {'alpha': 0.5, 'beta': 1.0}, 16)  # Levy(0, 1): cdf erfc(sqrt(1 / 2x))
            self.assertLess(s8.ks_statistic(x.tolist(), lambda v: math.erfc(math.sqrt(0.5 / v)) if v > 0 else 0.0), 1.95 / math.sqrt(n))
            x = draw('levy_stable', {'alpha': 1.0, 'beta': 0.5}, 17)
            q = values('levy_stable', 'ppf', [0.25, 0.5, 0.75], {'alpha': 1.0, 'beta': 0.5})
            for k, p in enumerate((0.25, 0.5, 0.75)):
                self.assertLess(abs(float(np.mean(x <= q[k])) - p), 0.01, (p, q[k]))

    def test_discrete_quantiles_at_the_probability_ends(self):
        """Fable m1 (correctness review): ppf(0) and isf(1) are the first support point, never one below it."""
        self.assertEqual(values('nbinom', 'ppf', [0.0, 1e-12], {'n': 3, 'p': 0.5}), [0.0, 0.0])
        self.assertEqual(values('binom', 'ppf', [0.0], {'n': 5, 'p': 0.3}), [0.0])
        self.assertEqual(values('betabinom', 'isf', [1.0], {'n': 5, 'a': 2, 'b': 3}), [0.0])
        self.assertEqual(values('zipf', 'ppf', [0.0], {'a': 2.0, 'loc': 10}), [11.0])
        self.assertEqual(values('geom', 'ppf', [0.0], {'p': 0.3}), [1.0])
        self.assertEqual(values('poisson', 'isf', [1.0, 0.0], {'lam': 2.0})[0], 0.0)
        self.assertEqual(values('binom', 'ppf', [1.0], {'n': 5, 'p': 0.3}), [5.0])
        self.assertEqual(values('geom', 'ppf', [1.0], {'p': 0.3}), ['Inf'])  # the unbounded end stays infinite

    def test_rank_rule_and_nan_count(self):
        """Argon m2 and m3: a non-vector x is BAD_PARAM, not INPUT_TOO_LARGE; the twin counts NaN inputs apart from NA."""
        rejected('stats_dist', dict(dist='norm', fn='cdf', params={}, x=[[0.0]]), 'BAD_PARAM')
        rejected('stats_dist', dict(dist='norm', fn='cdf', params={}, x=[0.0] * 1001), 'INPUT_TOO_LARGE')
        with tempfile.TemporaryDirectory(dir='.') as d:
            out = str(pathlib.Path(d).resolve() / 'nan.safetensors')
            r = call('stats_dist_to', dict(dist='norm', fn='cdf', params={}, x=[0.0, 'NaN', 'NA'], out=out))
            self.assertEqual((r['n_values'], r['n_missing'], r['n_nan']), (3, 1, 1))
            got = read_safetensors(out)['x']
            self.assertTrue(math.isnan(got[1]) and math.isnan(got[2]) and got[0] == 0.5)

    def test_far_tails_and_isf_are_not_derived_from_their_complements(self):
        """Kimi 4.3 and 4.4: logsf/logcdf deep in the tails and isf at 1e-300 come from the dedicated scipy functions."""
        mp.mp.dps = 50
        try:
            logsf = float(mp.log(mp.ncdf(-38)))
            isf = -float(mp.findroot(lambda z: mp.ncdf(z) - mp.mpf('1e-300'), -37))
        finally:
            mp.mp.dps = 30
        close(self, values('norm', 'logsf', [38.0]), [logsf], 1e-12)
        close(self, values('norm', 'logcdf', [-38.0]), [logsf], 1e-12)
        close(self, values('norm', 'isf', [1e-300]), [isf], 1e-12)
        close(self, values('norm', 'ppf', [1e-300]), [-isf], 1e-12)
        close(self, values('t', 'logsf', [1e6], {'df': 2.0}), [float(mp.log(mp.mpf(1) / 2 - mp.mpf(1e6) / (2 * mp.sqrt(2 + mp.mpf(1e6) ** 2))))], 1e-9)

    def test_pcg64_golden_streams(self):
        """Kimi 4.5: the generator is PCG64 in fact, not only in the echo (numpy's own stream for norm, a literal stream for genextreme)."""
        want = np.random.Generator(np.random.PCG64(42)).normal(0.0, 1.0, 3).tolist()
        self.assertEqual(values('norm', 'rvs', params={}, n=3, seed=42), want)
        self.assertEqual(values('gamma', 'rvs', params={'shape': 2.0, 'scale': 1.5}, n=3, seed=42), np.random.Generator(np.random.PCG64(42)).gamma(2.0, 1.5, 3).tolist())
        self.assertEqual(values('genextreme', 'rvs', params={'c': 0.2}, n=5, seed=42), [1.1919779497704852, 0.19043071644882123, 1.567598524048617, 0.9230340924162361, -0.9380950777027589])

    def test_family_count_and_df_exception_are_documented(self):
        """Kimi 3 and Fable m7: 31 families; Student-t df is the one parameter exempt from the 1e12 magnitude cap."""
        schema = json.loads(subprocess.run([BIN, '--schema'], capture_output=True, text=True, check=True).stdout)
        dist = [p for p in [o for o in schema['operations'] if o['name'] == 'stats_dist'][0]['params'] if p['name'] == 'dist'][0]
        self.assertEqual(len(dist['enum']), 31)
        for text in ((ROOT / 'README.md').read_text(), (ROOT / 'docs/s8b2.md').read_text(), (ROOT / 'docs/s8b.md').read_text()):
            self.assertNotIn('34 families', text)
            self.assertNotIn('34 `stats_dist` families', text)
        self.assertIn('except Student-t `df`', (ROOT / 'docs/s8b2.md').read_text())
        self.assertAlmostEqual(values('t', 'cdf', [0.0], {'df': 1e300})[0], 0.5, places=15)
        rejected('stats_dist', dict(dist='nct', fn='cdf', params={'df': 1e13, 'nc': 1.0}, x=[0.0]), 'BAD_PARAM')
        rejected('stats_dist', dict(dist='norm', fn='cdf', params={'loc': 1e13}, x=[0.0]), 'BAD_PARAM')


class DistMpmathFuzz(unittest.TestCase):
    """Every family at a random parameter set (fixed seed) against mpmath: pdf, logpdf, cdf, logcdf, sf, logsf, and the ppf(cdf) / isf(sf) round trips."""
    Phi = staticmethod(lambda z: mp.ncdf(z))
    phi = staticmethod(lambda z: mp.npdf(z))

    def families(self, rng):
        U, RI, Phi, phi = rng.uniform, rng.randint, self.Phi, self.phi
        reg = lambda a, b, x: mp.betainc(a, b, 0, x, regularized=True)
        out = {}
        out['norm'] = ({}, phi, Phi, (-mp.inf, mp.inf))
        v = U(1.5, 12)
        C = mp.gamma((v + 1) / 2) / (mp.sqrt(v * mp.pi) * mp.gamma(v / 2))
        out['t'] = ({'df': v}, lambda z, v=v, C=C: C * (1 + z * z / v) ** (-(v + 1) / 2), None, (-mp.inf, mp.inf))
        out['cauchy'] = ({}, lambda z: 1 / (mp.pi * (1 + z * z)), lambda z: mp.mpf(1) / 2 + mp.atan(z) / mp.pi, (-mp.inf, mp.inf))
        out['laplace'] = ({}, lambda z: mp.exp(-abs(z)) / 2, lambda z: (mp.exp(z) / 2 if z < 0 else 1 - mp.exp(-z) / 2), (-mp.inf, mp.inf))
        out['logistic'] = ({}, lambda z: mp.exp(-z) / (1 + mp.exp(-z)) ** 2, lambda z: 1 / (1 + mp.exp(-z)), (-mp.inf, mp.inf))
        out['gumbel_r'] = ({}, lambda z: mp.exp(-(z + mp.exp(-z))), lambda z: mp.exp(-mp.exp(-z)), (-mp.inf, mp.inf))
        out['rayleigh'] = ({}, lambda z: z * mp.exp(-z * z / 2), lambda z: 1 - mp.exp(-z * z / 2), (0, mp.inf))
        c = U(0.5, 4)
        out['weibull_min'] = ({'c': c}, lambda z, c=c: c * z ** (c - 1) * mp.exp(-z ** c), lambda z, c=c: 1 - mp.exp(-z ** c), (0, mp.inf))
        b = U(0.5, 5)
        out['pareto'] = ({'b': b}, lambda z, b=b: b / z ** (b + 1), lambda z, b=b: 1 - z ** (-b), (1, mp.inf))
        c = rng.choice([U(-0.8, -0.05), U(0.05, 0.8)])
        out['genextreme'] = ({'c': c}, lambda z, c=c: mp.exp(-(1 - c * z) ** (1 / c)) * (1 - c * z) ** (1 / c - 1), lambda z, c=c: mp.exp(-(1 - c * z) ** (1 / c)), (-mp.inf, 1 / c) if c > 0 else (1 / c, mp.inf))
        c = rng.choice([U(-0.8, -0.05), U(0.05, 1.5)])
        out['genpareto'] = ({'c': c}, lambda z, c=c: (1 + c * z) ** (-1 - 1 / c), lambda z, c=c: 1 - (1 + c * z) ** (-1 / c), (0, mp.inf) if c > 0 else (0, -1 / c))
        a = U(-4, 4)
        out['skewnorm'] = ({'a': a}, lambda z, a=a: 2 * phi(z) * Phi(a * z), None, (-mp.inf, mp.inf))
        a, b = U(-2, 2), U(0.3, 3)
        out['johnsonsu'] = ({'a': a, 'b': b}, lambda z, a=a, b=b: b / mp.sqrt(1 + z * z) * phi(a + b * mp.asinh(z)), lambda z, a=a, b=b: Phi(a + b * mp.asinh(z)), (-mp.inf, mp.inf))
        a = U(-2, 0.5)
        b = a + U(0.3, 3)
        Z = Phi(b) - Phi(a)
        out['truncnorm'] = ({'a': a, 'b': b}, lambda z, Z=Z: phi(z) / Z, lambda z, a=a, Z=Z: (Phi(z) - Phi(a)) / Z, (a, b))
        k = U(0.2, 8)
        Ck = 2 * mp.pi * mp.besseli(0, k)
        out['vonmises'] = ({'kappa': k}, lambda z, k=k, Ck=Ck: mp.exp(k * mp.cos(z)) / Ck, None, (-mp.pi, mp.pi))
        v, nc = U(2, 15), U(-3, 3)
        fS = lambda s, v=v: 2 * s * v * (s * s * v) ** (v / 2 - 1) * mp.exp(-s * s * v / 2) / (2 ** (v / 2) * mp.gamma(v / 2))
        out['nct'] = ({'df': v, 'nc': nc}, lambda x, nc=nc, fS=fS: mp.quad(lambda s: s * phi(x * s - nc) * fS(s), [0, 1, mp.inf]), lambda x, nc=nc, fS=fS: mp.quad(lambda s: Phi(x * s - nc) * fS(s), [0, 1, mp.inf]), (-mp.inf, mp.inf))
        d1, d2, lam = U(1.5, 10), U(2, 15), U(0.1, 6)
        fF = lambda x, a, b: mp.sqrt((a * x) ** a * b ** b / (a * x + b) ** (a + b)) / (x * mp.beta(a / 2, b / 2))
        FF = lambda x, a, b: reg(a / 2, b / 2, a * x / (a * x + b))
        out['ncf'] = ({'dfn': d1, 'dfd': d2, 'nc': lam},
                      lambda x, d1=d1, d2=d2, lam=lam: mp.nsum(lambda k: mp.exp(-lam / 2) * (lam / 2) ** k / mp.factorial(k) * (d1 / (d1 + 2 * k)) * fF(x * d1 / (d1 + 2 * k), d1 + 2 * k, d2), [0, mp.inf]),
                      lambda x, d1=d1, d2=d2, lam=lam: mp.nsum(lambda k: mp.exp(-lam / 2) * (lam / 2) ** k / mp.factorial(k) * FF(x * d1 / (d1 + 2 * k), d1 + 2 * k, d2), [0, mp.inf]), (0, mp.inf))
        a, b = U(1.5, 10), U(2, 15)
        out['f'] = ({'dfn': a, 'dfd': b}, lambda x, a=a, b=b: fF(x, a, b), lambda x, a=a, b=b: FF(x, a, b), (0, mp.inf))
        k = U(1.2, 12)
        out['chi2'] = ({'df': k}, lambda x, k=k: x ** (k / 2 - 1) * mp.exp(-x / 2) / (2 ** (k / 2) * mp.gamma(k / 2)), lambda x, k=k: mp.gammainc(k / 2, 0, x / 2, regularized=True), (0, mp.inf))
        k, th = U(0.5, 6), U(0.3, 3)
        out['gamma'] = ({'shape': k, 'scale': th}, lambda x, k=k, th=th: x ** (k - 1) * mp.exp(-x / th) / (mp.gamma(k) * th ** k), lambda x, k=k, th=th: mp.gammainc(k, 0, x / th, regularized=True), (0, mp.inf))
        a, b = U(0.5, 5), U(0.5, 5)
        out['beta'] = ({'a': a, 'b': b}, lambda x, a=a, b=b: x ** (a - 1) * (1 - x) ** (b - 1) / mp.beta(a, b), lambda x, a=a, b=b: reg(a, b, x), (0, 1))
        s = U(0.3, 3)
        out['expon'] = ({'scale': s}, lambda x, s=s: mp.exp(-x / s) / s, lambda x, s=s: 1 - mp.exp(-x / s), (0, mp.inf))
        lo = U(-3, 1)
        hi = lo + U(0.5, 4)
        out['uniform'] = ({'low': lo, 'high': hi}, lambda x, w=hi - lo: 1 / mp.mpf(w), lambda x, lo=lo, w=hi - lo: (x - lo) / mp.mpf(w), (lo, hi))
        mu, sg = U(-1, 1.5), U(0.2, 1.5)
        out['lognorm'] = ({'mu': mu, 'sigma': sg}, lambda x, mu=mu, sg=sg: mp.exp(-(mp.log(x) - mu) ** 2 / (2 * sg * sg)) / (x * sg * mp.sqrt(2 * mp.pi)), lambda x, mu=mu, sg=sg: Phi((mp.log(x) - mu) / sg), (0, mp.inf))
        out['levy_stable'] = ({'alpha': 2, 'beta': 0}, lambda z: phi(z / mp.sqrt(2)) / mp.sqrt(2), lambda z: Phi(z / mp.sqrt(2)), (-mp.inf, mp.inf))
        l = U(0.3, 20)
        out['poisson'] = ({'lam': l}, lambda k, l=l: mp.exp(-l) * mp.mpf(l) ** k / mp.factorial(k), lambda k, l=l: mp.gammainc(k + 1, mp.mpf(l), mp.inf, regularized=True), (0, 60))
        n, p = RI(1, 40), U(0.05, 0.95)
        out['binom'] = ({'n': n, 'p': p}, lambda k, n=n, p=p: mp.binomial(n, k) * mp.mpf(p) ** k * (1 - mp.mpf(p)) ** (n - k), None, (0, n))
        n, p = U(0.5, 8), U(0.1, 0.9)
        out['nbinom'] = ({'n': n, 'p': p}, lambda k, n=n, p=p: mp.gamma(k + n) / (mp.gamma(n) * mp.factorial(k)) * mp.mpf(p) ** n * (1 - mp.mpf(p)) ** k, lambda k, n=n, p=p: reg(n, k + 1, p), (0, 80))
        p = U(0.05, 0.95)
        out['geom'] = ({'p': p}, lambda k, p=p: (1 - mp.mpf(p)) ** (k - 1) * p, lambda k, p=p: 1 - (1 - mp.mpf(p)) ** k, (1, 60))
        a = U(1.3, 4)
        z = mp.zeta(a)
        out['zipf'] = ({'a': a}, lambda k, a=a, z=z: mp.mpf(k) ** (-a) / z, lambda k, a=a, z=z: 1 - mp.zeta(a, k + 1) / z, (1, 40))
        n, a, b = RI(1, 30), U(0.3, 5), U(0.3, 5)
        out['betabinom'] = ({'n': n, 'a': a, 'b': b}, lambda k, n=n, a=a, b=b: mp.binomial(n, k) * mp.beta(k + a, n - k + b) / mp.beta(a, b), None, (0, n))
        return out

    DISCRETE = {'poisson', 'binom', 'nbinom', 'geom', 'zipf', 'betabinom'}
    LOCSCALE = {'norm', 't', 'cauchy', 'laplace', 'logistic', 'gumbel_r', 'rayleigh', 'weibull_min', 'pareto', 'genextreme', 'genpareto', 'skewnorm', 'johnsonsu', 'truncnorm', 'vonmises', 'nct', 'ncf', 'levy_stable'}
    NUMERIC = {'nct', 'ncf', 'levy_stable', 'vonmises'}

    def test_every_family_against_mpmath(self):
        rng = random.Random(31)
        table = self.families(rng)
        self.assertEqual(len(table), 31)
        for fam in sorted(table):
            prm, f, Fc, (lo, hi) = table[fam]
            disc = fam in self.DISCRETE
            loc, scale = 0.0, 1.0
            if fam in self.LOCSCALE and rng.random() < 0.7:
                loc, scale = round(rng.uniform(-2, 2), 3), round(rng.uniform(0.3, 3), 3)
                prm = dict(prm, loc=loc, scale=scale)
            if disc:
                top = min(hi, 60)
                xs = sorted({lo, lo + 1, rng.randint(lo, top), rng.randint(lo, top), top})
                zs = xs
            else:
                L = lo if lo > -mp.inf else -4
                H = hi if hi < mp.inf else (8 if fam in ('pareto', 'lognorm') else 6)
                zs = sorted(z for z in (mp.mpf(rng.uniform(float(L), float(H))) for _ in range(4)) if lo < z < hi)
                xs = [float(loc + scale * z) for z in zs]
            refs = {}
            for z in zs:
                if disc:
                    pm = f(z)
                    c = Fc(z) if Fc else sum(f(j) for j in range(int(lo), int(z) + 1))
                else:
                    pm = f(z) / scale
                    c = Fc(z) if Fc else mp.quad(f, [lo if lo > -mp.inf else -mp.inf, z])
                refs.setdefault('pdf', []).append(pm)
                refs.setdefault('logpdf', []).append(mp.log(pm) if pm > 0 else -mp.inf)
                refs.setdefault('cdf', []).append(c)
                refs.setdefault('logcdf', []).append(mp.log(c) if c > 0 else -mp.inf)
                refs.setdefault('sf', []).append(1 - c)
                refs.setdefault('logsf', []).append(mp.log(1 - c) if c < 1 else -mp.inf)
            thr = 1e-5 if fam in self.NUMERIC else 1e-9
            got = {}
            for fn in ('pdf', 'logpdf', 'cdf', 'logcdf', 'sf', 'logsf'):
                got[fn] = values(fam, fn, xs, prm)
                for g, e, x in zip(got[fn], refs[fn], xs):
                    e = float(e)
                    g = {'Inf': INF, '-Inf': -INF}.get(g, g)
                    if fn in ('sf', 'logsf') and abs(1 - float(refs['cdf'][xs.index(x)])) < 1e-13:
                        continue  # the reference complement has no digits left at the top of the support
                    if fn.startswith('log') and abs(e) < 1:
                        self.assertLessEqual(abs(g - e), thr, (fam, fn, prm, x, g, e))
                    else:
                        self.assertLessEqual(abs(g - e), thr * abs(e) + 1e-300, (fam, fn, prm, x, g, e))
            for fn, inv in (('cdf', 'ppf'), ('sf', 'isf')):
                probs = [(p, x) for p, x in zip(got[fn], xs) if isinstance(p, float) and 1e-9 < p < 1 - 1e-9]
                if not probs:
                    continue
                back = values(fam, inv, [p for p, _ in probs], prm)
                for q, (p, x) in zip(back, probs):
                    if disc:
                        self.assertTrue(q == x or (inv == 'isf' and q == x + 1 and abs(values(fam, 'sf', [x], prm)[0] - p) < 1e-12), (fam, inv, prm, x, p, q))
                    else:
                        self.assertLessEqual(abs(q - x), (1e-4 if fam in self.NUMERIC else 1e-7) * max(1.0, abs(x)), (fam, inv, prm, x, p, q))


if __name__ == '__main__':
    unittest.main()
