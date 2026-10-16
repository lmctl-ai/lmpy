"""S8b-2 graph_paths / graph_flow (+ _to twins) against pure-Python hand algorithms (tests/oracle/s8b2.py): textbook graphs, seeded fuzz,
brute-force k shortest paths, Floyd-Warshall for all pairs, Edmonds-Karp with the min-cut duality check, successive shortest paths for min-cost
flow. Every pre-spawn rejection runs with a missing interpreter."""
import json, math, os, pathlib, random, subprocess, sys, tempfile, time, unittest
import numpy as np
BIN = str(pathlib.Path(sys.argv.pop(1)).resolve())
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests/oracle'))
from s1 import read_safetensors
import s8
import s8b2 as ref

NO_PYTHON = dict(os.environ, PYCLI_PYTHON='/no/such/python')
DESCRIPTOR_KEYS = {'ref', 'path', 'format', 'key', 'dtype', 'shape', 'nbytes'}


def call(op, params, rc=0, env=None):
    p = subprocess.run([BIN, '--input-json', json.dumps({'op': op, 'params': params})], capture_output=True, text=True, env=env)
    assert p.returncode == rc, (op, rc, p.returncode, p.stdout[:1500], p.stderr[:600])
    return json.loads(p.stdout)


def rejected(op, params, code=None, rc=2):
    r = call(op, params, rc, NO_PYTHON)
    if code is not None:
        assert r['error']['data']['tool_code'] == code, (code, r)
    return r


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir='.')
        self.dir = pathlib.Path(self.tmp.name).resolve()
        self.addCleanup(self.tmp.cleanup)
        self.count = 0

    def p(self, name):
        return str(self.dir / name)

    def edges(self, rows, header=('u', 'v', 'w'), name=None):
        """frame_import of an edge list through a CSV file."""
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


def paths(frame, **extra):
    return dict(frame=frame, from_column='u', to_column='v', **extra)


CLRS = [('s', 't', 10), ('s', 'y', 5), ('t', 'y', 2), ('t', 'x', 1), ('y', 't', 3), ('y', 'x', 9), ('y', 'z', 2), ('x', 'z', 4), ('z', 'x', 6), ('z', 's', 7)]


class Paths(Base):
    def test_textbook_graph_and_every_method(self):
        f = self.edges(CLRS)
        for method in ('dijkstra', 'bellman_ford'):
            r = call('graph_paths', paths(f, weight_column='w', source='s', method=method))
            self.assertEqual((r['status'], r['n_nodes'], r['n_reachable'], r['n_unreachable']), ('OK', 5, 5, 0))
            got = dict(zip(r['nodes'], r['distance']))
            self.assertEqual(got, {'s': 0, 't': 8, 'y': 5, 'x': 9, 'z': 7})  # CLRS figure 24.6
            self.assertEqual(r['nodes'], sorted(r['nodes']))
        b = call('graph_paths', paths(f, source='s', method='bfs'))
        self.assertEqual(dict(zip(b['nodes'], b['distance'])), {'s': 0, 't': 1, 'y': 1, 'x': 2, 'z': 2})
        t = call('graph_paths', paths(f, weight_column='w', source='s', target='x'))
        self.assertEqual((t['length'], t['hops'], t['path'][0], t['path'][-1]), (9, 3, 's', 'x'))
        self.assertEqual(t['path'], ['s', 'y', 't', 'x'])

    def test_fuzz_against_hand_algorithms(self):
        rng = random.Random(81)
        for case in range(30):
            directed = case % 2 == 0
            n = rng.randint(4, 12)
            nodes = list(range(100, 100 + n))
            edges = ref.random_graph(rng, n, rng.randint(n, 3 * n), directed, nodes=nodes, integer=case % 3 != 0, dag=(case % 5 == 0 and directed), negative=(case % 5 == 0 and directed))
            f = self.edges(edges)
            adj = ref.adjacency(edges, directed)
            source = rng.choice(nodes)
            if source not in adj:
                continue
            has_negative = any(w < 0 for _, _, w in edges)
            method = 'bellman_ford' if has_negative or case % 4 == 1 else 'dijkstra'
            r = call('graph_paths', paths(f, weight_column='w', source=source, method=method, directed=directed))
            want = ref.dijkstra(adj, source) if method == 'dijkstra' else ref.bellman_ford(adj, source)[0]
            self.assertEqual(r['status'], 'OK', (case, r))
            self.assertEqual(r['n_reachable'], len(want))
            self.assertEqual(r['n_unreachable'], len(adj) - len(want))
            for node, d in zip(r['nodes'], r['distance']):
                self.assertAlmostEqual(d, want[node], places=9, msg=(case, node))
            hops = ref.bfs_hops(adj, source)
            h = call('graph_paths', paths(f, source=source, method='bfs', directed=directed))
            self.assertEqual(dict(zip(h['nodes'], h['distance'])), hops, case)
            reachable = [v for v in want if v != source]
            if reachable:
                target = rng.choice(reachable)
                one = call('graph_paths', paths(f, weight_column='w', source=source, target=target, method=method, directed=directed))
                self.assertAlmostEqual(one['length'], want[target], places=9)
                total = 0.0
                for a, b in zip(one['path'], one['path'][1:]):
                    self.assertIn(b, adj[a], 'path uses a missing edge')
                    total += adj[a][b]
                self.assertAlmostEqual(total, one['length'], places=9)
                self.assertEqual(one['hops'], len(one['path']) - 1)

    def test_negative_weights_and_negative_cycles(self):
        f = self.edges([(1, 2, 4), (2, 3, -2), (1, 3, 5)])
        rejected('graph_paths', paths(f, weight_column='w', source=1, method='dijkstra'), 'BAD_PARAM')
        ok = call('graph_paths', paths(f, weight_column='w', source=1, method='bellman_ford'))
        self.assertEqual(dict(zip(ok['nodes'], ok['distance'])), {1: 0, 2: 4, 3: 2})
        cyc = self.edges([(1, 2, 1), (2, 3, -3), (3, 1, 1), (3, 4, 2)])
        r = call('graph_paths', paths(cyc, weight_column='w', source=1, method='bellman_ford'))
        self.assertEqual(r['status'], 'NEGATIVE_CYCLE')
        total = 0
        edge = {(a, b): w for a, b, w in [(1, 2, 1), (2, 3, -3), (3, 1, 1), (3, 4, 2)]}
        cycle = r['cycle']
        for a, b in zip(cycle, cycle[1:] + cycle[:1]):
            total += edge[(a, b)]
        self.assertLess(total, 0)
        self.assertNotIn('distance', r)
        away = call('graph_paths', paths(cyc, weight_column='w', source=4, method='bellman_ford'))  # the cycle is not reachable from 4
        self.assertEqual(away['status'], 'OK')
        self.assertEqual(away['n_reachable'], 1)
        undirected = call('graph_paths', paths(f, weight_column='w', source=1, method='bellman_ford', directed=False))
        self.assertEqual(undirected['status'], 'NEGATIVE_CYCLE')  # an undirected negative edge is a negative 2-cycle

    def test_unreachable_targets_and_duplicates_self_loops(self):
        f = self.edges([('a', 'b', 1), ('b', 'c', 1), ('d', 'e', 1), ('a', 'b', 0.5), ('c', 'c', 3)])
        r = call('graph_paths', paths(f, weight_column='w', source='a', target='e'))
        self.assertEqual((r['status'], r['n_reachable']), ('UNREACHABLE', 3))
        self.assertNotIn('path', r)
        t = call('graph_paths', paths(f, weight_column='w', source='a', target='c'))
        self.assertAlmostEqual(t['length'], 1.5, places=12)  # the lighter of two parallel edges
        full = call('graph_paths', paths(f, weight_column='w', source='a'))
        self.assertEqual((full['n_nodes'], full['n_reachable'], full['n_unreachable'], full['n_edges']), (5, 3, 2, 4))
        self.assertNotIn('d', full['nodes'])
        self.assertEqual(call('graph_paths', paths(f, source='a', target='a'))['length'], 0)
        und = call('graph_paths', paths(f, weight_column='w', source='c', target='a', directed=False))
        self.assertAlmostEqual(und['length'], 1.5, places=12)

    def test_node_ids_round_trip_exactly(self):
        weird = ['plain', 'with space', 'naïve—雪', 'a,b', '007', 'x"y']
        rows = [(weird[i], weird[i + 1], 1) for i in range(len(weird) - 1)]
        f = self.edges(rows)
        r = call('graph_paths', paths(f, weight_column='w', source=weird[0]))
        self.assertEqual(sorted(r['nodes']), sorted(weird))
        self.assertEqual(r['distance'][r['nodes'].index('x"y')], 5)
        big = [2 ** 53 + 1, -7, 0, 2 ** 40]
        g = self.edges([(big[0], big[1], 1), (big[1], big[2], 2), (big[2], big[3], 3)])
        q = call('graph_paths', paths(g, weight_column='w', source=big[0]))
        self.assertEqual(sorted(q['nodes']), sorted(big))
        self.assertTrue(all(isinstance(v, int) for v in q['nodes']))
        rejected('graph_paths', paths(g, source='a'), 'BAD_PARAM')  # a string node in an integer graph
        rejected('graph_paths', paths(f, source=1), 'BAD_PARAM')
        rejected('graph_paths', paths(f, source='nowhere'), 'UNKNOWN_NODE')
        rejected('graph_paths', paths(f, source=weird[0], target='nowhere'), 'UNKNOWN_NODE')

    def test_k_shortest_against_brute_force(self):
        rng = random.Random(82)
        for case in range(12):
            n = rng.randint(5, 8)
            edges = ref.random_graph(rng, n, rng.randint(n + 2, 2 * n + 2), True, weights=(1, 9), integer=False)
            adj = ref.adjacency(edges, True)
            nodes = sorted(adj)
            s, t = nodes[0], nodes[-1]
            every = ref.simple_paths(adj, s, t)
            f = self.edges(edges)
            r = call('graph_paths', paths(f, weight_column='w', source=s, target=t, k_shortest=4))
            if not every:
                self.assertEqual(r['status'], 'UNREACHABLE')
                continue
            self.assertEqual(r['k_found'], min(4, len(every)))
            for got, want in zip(r['paths'], every):
                self.assertAlmostEqual(got['length'], want[0], places=9)
                self.assertEqual(got['nodes'][0], s)
                self.assertEqual(got['nodes'][-1], t)
                self.assertEqual(len(set(got['nodes'])), len(got['nodes']))
            lengths = [p['length'] for p in r['paths']]
            self.assertEqual(lengths, sorted(lengths))
            if len({round(w[0], 9) for w in every[:5]}) == len(every[:5]):
                self.assertEqual([p['nodes'] for p in r['paths']], [w[1] for w in every[:r['k_found']]])
        rejected('graph_paths', paths(self.edges(CLRS), weight_column='w', source='s', k_shortest=3), 'BAD_PARAM')  # needs a target
        rejected('graph_paths', paths(self.edges(CLRS), weight_column='w', source='s', target='x', k_shortest=11), 'BAD_PARAM')
        rejected('graph_paths', paths(self.edges(CLRS), weight_column='w', source='s', target='x', k_shortest=0), 'BAD_PARAM')

    def test_column_and_parameter_validation_before_spawn(self):
        f = self.edges([(1, 2, 1.5), (2, 3, 2.0)])
        rejected('graph_paths', paths(f, weight_column='nope', source=1), 'UNKNOWN_COLUMN')
        rejected('graph_paths', dict(frame=f, from_column='nope', to_column='v', source=1), 'UNKNOWN_COLUMN')
        rejected('graph_paths', paths(f, source=1, method='bfs', weight_column='w'), 'BAD_PARAM')
        rejected('graph_paths', paths(f, source=1, all_pairs=True), 'BAD_PARAM')  # all pairs is a file result
        rejected('graph_paths', paths(f), 'BAD_PARAM')  # a source is needed
        rejected('graph_paths', paths(f, source=1, method='astar'))
        rejected('graph_paths', dict(from_column='u', to_column='v', source=1), 'BAD_PARAM')  # no edges at all
        rejected('graph_paths', paths(f, source=1, records=[{'u': 1, 'v': 2}]), 'BAD_PARAM')
        mixed = self.edges([(1, 'a', 1)])
        rejected('graph_paths', paths(mixed, source=1), 'BAD_DTYPE')
        words = self.edges([('x', 'y', 'heavy')])
        rejected('graph_paths', paths(words, weight_column='w', source='x'), 'BAD_DTYPE')
        flt = self.edges([(1.5, 2.5, 1)])
        rejected('graph_paths', paths(flt, source=1), 'BAD_DTYPE')  # float node ids are refused
        rejected('graph_paths', dict(frame=self.p('nope.npz'), from_column='u', to_column='v', source=1), None, rc=4)
        (self.dir / 'file.graphml').write_text('<graphml/>')
        rejected('graph_paths', dict(frame=self.p('file.graphml'), from_column='u', to_column='v', source=1), 'BAD_REF')  # graph file formats are never read
        bad = call('graph_paths', paths(self.edges([(1, 2, 1.0), (2, 3, '')]), weight_column='w', source=1), rc=2)
        self.assertEqual(bad['error']['data']['tool_code'], 'NA_NOT_ALLOWED')
        infinite = call('graph_paths', paths(self.edges([(1, 2, 'Inf')], name='inf.npz'), weight_column='w', source=1), rc=2)
        self.assertEqual(infinite['error']['data']['tool_code'], 'NON_FINITE')

    def test_inline_records(self):
        recs = [{'a': 'x', 'b': 'y', 'c': 2.0}, {'a': 'y', 'b': 'z', 'c': 3.0}, {'a': 'x', 'b': 'z', 'c': 10.0}]
        r = call('graph_paths', dict(records=recs, from_column='a', to_column='b', weight_column='c', source='x', target='z'))
        self.assertEqual((r['length'], r['path']), (5.0, ['x', 'y', 'z']))
        rejected('graph_paths', dict(records=recs, from_column='a', to_column='nope', source='x'), 'UNKNOWN_COLUMN')
        rejected('graph_paths', dict(records=[{'a': 1, 'b': 2}, {'a': 2}], from_column='a', to_column='b', source=1), 'BAD_DATA')
        rejected('graph_paths', dict(records=[{'a': 1, 'b': 2}] * 600, from_column='a', to_column='b', source=1), 'BAD_DATA')  # 1200 cells

    def test_caps_and_inline_budget_before_spawn(self):
        n = 460
        chain = self.edges([(i, i + 1, 1) for i in range(n)], name='chain.npz')
        r = rejected('graph_paths', paths(chain, weight_column='w', source=0), 'OUTPUT_TOO_LARGE', rc=1)  # 461 nodes listed inline
        target = call('graph_paths', paths(chain, weight_column='w', source=0, target=n))
        self.assertEqual(target['length'], n)
        (self.dir / 'big.csv').write_text('u,v\n' + ''.join('%d,%d\n' % (i, i + 1) for i in range(100001)))
        call('frame_import', dict(csv=self.p('big.csv'), out=self.p('big.npz')))
        rejected('graph_paths_to', dict(frame=self.p('big.npz'), from_column='u', to_column='v', source=0, out=self.p('o.npz')), 'INPUT_TOO_LARGE')
        ok = call('graph_paths_to', dict(frame=self.p('chain.npz'), from_column='u', to_column='v', source=0, out=self.p('chain-out.npz')))
        self.assertEqual(ok['status'], 'OK')

    def test_twin_table_for_single_source(self):
        rng = random.Random(83)
        nodes = ['n%02d' % i for i in range(30)]
        edges = ref.random_graph(rng, 30, 70, True, nodes=nodes)
        for cut in (nodes[7], nodes[11]):  # leave some nodes unreachable
            edges = [e for e in edges if e[1] != cut]
            edges.append((cut, nodes[0], 1))
        f = self.edges(edges)
        out = self.p('t.npz')
        r = call('graph_paths_to', paths(f, weight_column='w', source=nodes[0], out=out))
        self.assertEqual(r['status'], 'OK')
        self.assertEqual(set(r['artifacts']['table']), DESCRIPTOR_KEYS)
        t = s8.read_table(out)
        adj = ref.adjacency(edges, True)
        want = ref.dijkstra(adj, nodes[0])
        self.assertEqual(t['columns'], ['node', 'distance', 'hops', 'predecessor'])
        self.assertEqual(sorted(t['data']['node']), sorted(adj))
        hops = ref.bfs_hops(adj, nodes[0])
        for k, node in enumerate(t['data']['node']):
            d = t['data']['distance'][k]
            if node in want:
                self.assertAlmostEqual(d, want[node], places=9)
                pred = t['data']['predecessor'][k]
                if node == nodes[0]:
                    self.assertIsNone(pred)
                else:
                    self.assertIn(node, adj[pred])
                    self.assertAlmostEqual(want[pred] + adj[pred][node], want[node], places=9)  # the predecessor lies on a shortest path
                self.assertGreaterEqual(t['data']['hops'][k], hops[node])
            else:
                self.assertEqual((d, t['data']['hops'][k], t['data']['predecessor'][k]), (None, None, None))
        self.assertEqual(t['meta']['node']['dtype'], 'string')
        call('graph_paths_to', paths(f, weight_column='w', source=nodes[0], out=out), rc=5)

    def test_all_pairs_matrix_against_floyd_warshall(self):
        rng = random.Random(84)
        for directed in (True, False):
            for method, negative in (('dijkstra', False), ('bellman_ford', True), ('bfs', False)):
                if negative and not directed:
                    continue
                n = 14
                edges = ref.random_graph(rng, n, 30, directed, nodes=list(range(n)), dag=negative, negative=negative)
                f = self.edges(edges)
                adj = ref.adjacency(edges, directed)
                if method == 'bfs':
                    adj = ref.adjacency([(u, v, 1) for u, v, _ in edges], directed)
                nodes, want, _ = ref.floyd_warshall(adj)
                out = self.p('ap-%s-%s.safetensors' % (method, directed))
                r = call('graph_paths_to', paths(f, weight_column=None if method == 'bfs' else 'w', method=method, directed=directed, all_pairs=True, out=out) if method != 'bfs' else paths(f, method=method, directed=directed, all_pairs=True, out=out))
                self.assertEqual((r['status'], r['n_nodes']), ('OK', len(nodes)))
                self.assertEqual(set(r['artifacts']['distance']), DESCRIPTOR_KEYS)
                got = read_safetensors(out)
                self.assertEqual(got['nodes'].tolist(), nodes)
                self.assertEqual(got['distance'].shape, (len(nodes), len(nodes)))
                for i, u in enumerate(nodes):
                    for j, v in enumerate(nodes):
                        w = want[u][v]
                        if math.isinf(w):
                            self.assertTrue(math.isnan(got['distance'][i, j]), (method, u, v))
                        else:
                            self.assertAlmostEqual(got['distance'][i, j], w, places=9, msg=(method, directed, u, v))

    def test_all_pairs_negative_cycle_caps_and_string_ids(self):
        f = self.edges([(1, 2, 1), (2, 3, -3), (3, 1, 1)])
        r = call('graph_paths_to', paths(f, weight_column='w', method='bellman_ford', all_pairs=True, out=self.p('nc.safetensors')), rc=1)
        self.assertEqual(r['error']['data']['tool_code'], 'NEGATIVE_CYCLE')
        self.assertFalse(os.path.exists(self.p('nc.safetensors')))
        strings = self.edges([('a', 'b', 1)])
        rejected('graph_paths_to', paths(strings, weight_column='w', all_pairs=True, out=self.p('s.safetensors')), 'BAD_PARAM')  # an artifact holds numbers: integer node ids only
        rejected('graph_paths_to', paths(f, weight_column='w', all_pairs=True, source=1, out=self.p('s.safetensors')), 'BAD_PARAM')
        rejected('graph_paths_to', paths(f, weight_column='w', all_pairs=True, out=self.p('s.npz')), 'BAD_REF')
        rejected('graph_paths_to', paths(f, weight_column='w', all_pairs=True, method='dijkstra', out=self.p('s2.safetensors')), 'BAD_PARAM')  # negative weight
        big = self.edges([(i, i + 1, 1) for i in range(1400)], name='n1401.npz')  # 1401 nodes: 1,962,801 matrix entries > the 1.9M-number artifact budget
        rejected('graph_paths_to', paths(big, weight_column='w', all_pairs=True, out=self.p('big.safetensors')), 'OUTPUT_TOO_LARGE', rc=1)
        edge = self.edges([(i, i + 1, 1) for i in range(1376)], name='n1377.npz')  # 1377 nodes: 1,896,129 entries
        ok = call('graph_paths_to', paths(edge, weight_column='w', all_pairs=True, out=self.p('edge.safetensors')))
        self.assertEqual(ok['n_nodes'], 1377)
        d = read_safetensors(self.p('edge.safetensors'))['distance']
        self.assertEqual((d[0, 1376], d[1376, 0] != d[1376, 0], d[5, 9]), (1376.0, True, 4.0))

    def test_cooperative_budget(self):
        rng = random.Random(85)
        n = 20000
        edges = [(i, (i * 7 + 1) % n, rng.randint(1, 9)) for i in range(n)] + [(i, (i * 13 + 5) % n, rng.randint(1, 9)) for i in range(n)]
        f = self.edges(edges)
        started = time.monotonic()
        r = call('graph_paths', paths(f, weight_column='w', source=0, method='bellman_ford', time_limit=0.05))
        self.assertEqual(r['status'], 'TIME_LIMIT')
        self.assertLess(time.monotonic() - started, 20)
        self.assertNotIn('distance', r)

    def test_import_only_networkx_and_no_pandas(self):
        r = call('graph_paths', paths(self.edges(CLRS), source='s', method='bfs'))
        self.assertEqual((r['meta']['pandas'], r['meta']['sklearn']), ('unloaded', 'unloaded'))
        self.assertEqual(r['meta']['forbidden_modules_loaded'], [])


# --------------------------------------------------------------------------------------------------------------------- flow
CLRS_FLOW = [('s', 'v1', 16), ('s', 'v2', 13), ('v1', 'v3', 12), ('v2', 'v1', 4), ('v2', 'v4', 14), ('v3', 'v2', 9), ('v3', 't', 20), ('v4', 'v3', 7), ('v4', 't', 4)]


def flow_checks(test, result, edges, source, sink, value, tolerance=0.0):
    """Conservation, capacity and value of the reported flow edges, from the raw edge list (parallel edges summed)."""
    caps = {}
    for u, v, c in edges:
        caps[(u, v)] = caps.get((u, v), 0) + c
    net = {}
    for a, b, f in zip(result['flow_from'], result['flow_to'], result['flow']):
        test.assertLessEqual(f, caps[(a, b)] + tolerance, (a, b))
        test.assertGreaterEqual(f, -tolerance)
        net[a] = net.get(a, 0) - f
        net[b] = net.get(b, 0) + f
    for node, excess in net.items():
        if node not in (source, sink):
            test.assertLessEqual(abs(excess), tolerance + 1e-12, node)
    test.assertLessEqual(abs(-net.get(source, 0) - value), tolerance + 1e-12)


class Flow(Base):
    def flow(self, frame, **extra):
        return dict(frame=frame, from_column='u', to_column='v', capacity_column='w', **extra)

    def test_clrs_maximum_flow_and_min_cut_duality(self):
        f = self.edges(CLRS_FLOW)
        r = call('graph_flow', self.flow(f, kind='max_flow', source='s', sink='t'))
        self.assertEqual((r['status'], r['value'], r['integral']), ('OK', 23, True))
        flow_checks(self, r, CLRS_FLOW, 's', 't', 23)
        c = call('graph_flow', self.flow(f, kind='min_cut', source='s', sink='t'))
        self.assertEqual((c['status'], c['value']), ('OK', 23))
        caps = {(u, v): w for u, v, w in CLRS_FLOW}
        cut_total = sum(caps[(a, b)] for a, b in zip(c['cut_from'], c['cut_to']))
        self.assertEqual(cut_total, 23)  # max-flow = min-cut
        side = set(c['source_side'])
        self.assertIn('s', side)
        self.assertNotIn('t', side)
        crossing = {(u, v) for (u, v), w in caps.items() if u in side and v not in side and w > 0}
        self.assertEqual(crossing, set(zip(c['cut_from'], c['cut_to'])))

    def test_fuzz_max_flow_and_cuts_against_edmonds_karp(self):
        rng = random.Random(86)
        for case in range(25):
            n = rng.randint(4, 10)
            integer = case % 4 != 3
            edges = ref.random_graph(rng, n, rng.randint(n, 3 * n), True, weights=(1, 12), integer=integer)
            edges += [edges[0]] if case % 5 == 0 else []  # a parallel edge: capacities add
            f = self.edges(edges)
            caps = {}
            for u, v, c in edges:
                caps[(u, v)] = caps.get((u, v), 0) + c
            nodes = sorted({x for e in edges for x in e[:2]})
            s, t = nodes[0], nodes[-1]
            value, flow, side = ref.edmonds_karp(caps, s, t)
            r = call('graph_flow', self.flow(f, kind='max_flow', source=s, sink=t))
            self.assertAlmostEqual(r['value'], value, places=9, msg=case)
            self.assertEqual(r['integral'], integer)
            flow_checks(self, r, edges, s, t, r['value'], 0.0 if integer else 1e-9)
            c = call('graph_flow', self.flow(f, kind='min_cut', source=s, sink=t))
            self.assertAlmostEqual(c['value'], value, places=9)
            total = sum(caps[(a, b)] for a, b in zip(c['cut_from'], c['cut_to']))
            self.assertAlmostEqual(total, value, places=9)
            mine = set(c['source_side'])
            self.assertTrue(s in mine and t not in mine)
            self.assertTrue(all(a in mine and b not in mine for a, b in zip(c['cut_from'], c['cut_to'])))

    def test_min_cost_flow_against_successive_shortest_paths(self):
        rng = random.Random(87)
        done = 0
        for case in range(40):
            n = rng.randint(4, 8)
            nodes = list(range(n))
            base = ref.random_graph(rng, n, rng.randint(n + 2, 2 * n + 2), True, weights=(1, 9), nodes=nodes)
            edges = [(u, v, rng.randint(1, 6), rng.randint(-2, 9)) for u, v, _ in base]
            supplies = {nodes[0]: rng.randint(1, 6), nodes[-1]: -rng.randint(1, 6)}
            supplies[nodes[-1]] = -supplies[nodes[0]]
            expected = ref.min_cost_flow(edges, supplies)
            f = self.edges([(u, v, c, k) for u, v, c, k in edges], header=('u', 'v', 'w', 'k'))
            params = dict(frame=f, from_column='u', to_column='v', capacity_column='w', cost_column='k', kind='min_cost_flow', supplies=[[node, a] for node, a in supplies.items()])
            if expected is None:
                r = call('graph_flow', params)
                self.assertEqual(r['status'], 'INFEASIBLE', case)
                continue
            r = call('graph_flow', params)
            self.assertEqual(r['status'], 'OK', (case, r))
            self.assertEqual(r['cost'], expected[0], case)
            caps = {(u, v): c for u, v, c, _ in edges}
            net = {}
            for a, b, fl in zip(r['flow_from'], r['flow_to'], r['flow']):
                self.assertLessEqual(fl, caps[(a, b)])
                net[a] = net.get(a, 0) - fl
                net[b] = net.get(b, 0) + fl
            for node in nodes:
                self.assertEqual(-net.get(node, 0), supplies.get(node, 0), (case, node))
            done += 1
        self.assertGreater(done, 10)

    def test_min_cost_flow_domain_statuses_and_integer_rule(self):
        f = self.edges([(1, 2, 5, 1), (2, 3, 5, 1)], header=('u', 'v', 'w', 'k'))
        base = dict(frame=f, from_column='u', to_column='v', capacity_column='w', cost_column='k', kind='min_cost_flow')
        ok = call('graph_flow', dict(base, source=1, sink=3, flow_value=4))
        self.assertEqual((ok['status'], ok['cost']), ('OK', 8))
        self.assertEqual(call('graph_flow', dict(base, source=1, sink=3, flow_value=6))['status'], 'INFEASIBLE')
        unbalanced = call('graph_flow', dict(base, supplies=[[1, 4], [3, -3]]))
        self.assertEqual(unbalanced['status'], 'UNBALANCED')
        rejected('graph_flow', dict(base, source=1, sink=3), 'BAD_PARAM')  # a flow_value or supplies is needed
        rejected('graph_flow', dict(base, supplies=[[1, 4]], source=1, sink=3, flow_value=1), 'BAD_PARAM')
        rejected('graph_flow', dict(base, supplies=[[9, 4], [3, -4]]), 'UNKNOWN_NODE')
        rejected('graph_flow', dict(base, supplies=[[1, 1.5], [3, -1.5]]), 'BAD_PARAM')
        rejected('graph_flow', dict(base, supplies=[[1, 1], [1, -1]]), 'BAD_PARAM')
        half = self.edges([(1, 2, 2.5, 1), (2, 3, 5, 1)], header=('u', 'v', 'w', 'k'), name='half.npz')
        r = call('graph_flow', dict(base, frame=half, source=1, sink=3, flow_value=2), rc=2)
        self.assertEqual(r['error']['data']['tool_code'], 'NOT_INTEGER')
        dup = self.edges([(1, 2, 5, 1), (1, 2, 5, 3), (2, 3, 5, 1)], header=('u', 'v', 'w', 'k'), name='dup.npz')
        r = call('graph_flow', dict(base, frame=dup, source=1, sink=3, flow_value=2), rc=2)
        self.assertEqual(r['error']['data']['tool_code'], 'DUPLICATE_EDGE')
        nocost = {k: v for k, v in base.items() if k != 'cost_column'}
        rejected('graph_flow', dict(nocost, source=1, sink=3, flow_value=1), 'BAD_PARAM')

    def test_float_capacities_are_documented_with_a_tolerance(self):
        edges = [('a', 'b', 0.1), ('a', 'b', 0.2), ('b', 'c', 0.25), ('a', 'c', 0.125)]
        f = self.edges(edges)
        r = call('graph_flow', self.flow(f, kind='max_flow', source='a', sink='c'))
        self.assertFalse(r['integral'])
        self.assertAlmostEqual(r['value'], 0.375, places=12)  # min(0.3, 0.25) + 0.125
        self.assertEqual(r['tolerance'], 1e-9)
        flow_checks(self, r, edges, 'a', 'c', r['value'], 1e-9)

    def test_validation_before_spawn_and_inline_limits(self):
        f = self.edges(CLRS_FLOW)
        base = self.flow(f, kind='max_flow', source='s', sink='t')
        rejected('graph_flow', dict(base, sink='s'), 'BAD_PARAM')
        rejected('graph_flow', dict(base, sink='nope'), 'UNKNOWN_NODE')
        rejected('graph_flow', {k: v for k, v in base.items() if k != 'capacity_column'}, 'BAD_PARAM')
        rejected('graph_flow', {k: v for k, v in base.items() if k != 'sink'}, 'BAD_PARAM')
        rejected('graph_flow', dict(base, capacity_column='nope'), 'UNKNOWN_COLUMN')
        rejected('graph_flow', dict(base, cost_column='w'), 'BAD_PARAM')  # costs are for min_cost_flow
        rejected('graph_flow', dict(base, kind='max_flow', supplies=[[1, 1]]), 'BAD_PARAM')
        rejected('graph_flow', dict(base, directed=False))
        neg = self.edges([('a', 'b', -1)], name='neg.npz')
        r = call('graph_flow', self.flow(neg, kind='max_flow', source='a', sink='b'), rc=2)
        self.assertEqual(r['error']['data']['tool_code'], 'BAD_DATA')
        huge = self.edges([('a', 'b', 2e15)], name='huge.npz')
        r = call('graph_flow', self.flow(huge, kind='max_flow', source='a', sink='b'), rc=2)
        self.assertEqual(r['error']['data']['tool_code'], 'BAD_DATA')
        (self.dir / 'many.csv').write_text('u,v,w\n' + ''.join('%d,%d,1\n' % (i, i + 1) for i in range(20001)))
        call('frame_import', dict(csv=self.p('many.csv'), out=self.p('many.npz')))
        rejected('graph_flow', self.flow(self.p('many.npz'), kind='max_flow', source=0, sink=20001), 'INPUT_TOO_LARGE')
        wide = self.edges([(i, i + 1, 1) for i in range(400)], name='wide.npz')
        r = call('graph_flow', self.flow(wide, kind='max_flow', source=0, sink=400), rc=1)
        self.assertEqual(r['error']['data']['tool_code'], 'OUTPUT_TOO_LARGE')  # 400 flow edges do not fit inline: use graph_flow_to
        out = self.p('wide-flow.npz')
        t = call('graph_flow_to', self.flow(wide, kind='max_flow', source=0, sink=400, out=out))
        self.assertEqual((t['status'], t['value'], t['n_flow_edges']), ('OK', 1, 400))
        table = s8.read_table(out)
        self.assertEqual(table['columns'], ['from', 'to', 'flow'])
        self.assertEqual(sorted(table['data']['from']), list(range(400)))
        self.assertEqual(set(table['data']['flow']), {1})

    def test_min_cut_twin_and_zero_flow(self):
        f = self.edges([(1, 2, 3), (3, 4, 5)])
        r = call('graph_flow', self.flow(f, kind='max_flow', source=1, sink=4))
        self.assertEqual((r['status'], r['value'], r['flow']), ('OK', 0, []))  # disconnected: zero flow, no flow edges
        c = call('graph_flow', self.flow(f, kind='min_cut', source=1, sink=4))
        self.assertEqual((c['value'], c['cut_from'], c['source_side']), (0, [], [1, 2]))
        wide = self.edges([(0, 1 + i, 1) for i in range(500)] + [(1 + i, 501, 1) for i in range(500)], name='star.npz')
        out = self.p('cut.npz')
        t = call('graph_flow_to', self.flow(wide, kind='min_cut', source=0, sink=501, out=out))
        self.assertEqual((t['value'], t['n_cut_edges']), (500, 500))
        table = s8.read_table(out)
        self.assertEqual(table['columns'], ['from', 'to', 'capacity'])
        self.assertEqual(len(table['data']['from']), 500)

    def test_flow_budget_and_isolation(self):
        n = 3000
        rng = random.Random(88)
        edges = [(i, i + 1, rng.randint(1, 5)) for i in range(n)] + [(i, (i * 31 + 7) % (n + 1), rng.randint(1, 5)) for i in range(n)]
        f = self.edges(edges)
        r = call('graph_flow', self.flow(f, kind='max_flow', source=0, sink=n, time_limit=0.02), rc=0)
        self.assertEqual(r['status'], 'TIME_LIMIT')
        self.assertNotIn('flow', r)
        ok = call('graph_flow', self.flow(self.edges(CLRS_FLOW), kind='max_flow', source='s', sink='t'))
        self.assertEqual((ok['meta']['pandas'], ok['meta']['forbidden_modules_loaded']), ('unloaded', []))


class Schema(unittest.TestCase):
    def test_registration_access_and_documentation(self):
        schema = json.loads(subprocess.run([BIN, '--schema'], capture_output=True, text=True).stdout)
        ops = {o['name']: o for o in schema['operations']}
        for name in ('graph_paths', 'graph_flow'):
            self.assertEqual(ops[name]['access'], 'read')
            self.assertEqual(ops[name + '_to']['access'], 'write')
            for variant in (name, name + '_to'):
                for param in ops[variant]['params']:
                    self.assertTrue(param.get('description'), (variant, param['name']))
            self.assertIn('out', {p['name'] for p in ops[name + '_to']['params']})
            self.assertNotIn('out', {p['name'] for p in ops[name]['params']})
            self.assertIn('artifacts', ops[name + '_to']['description'])
        text = json.dumps(ops['graph_paths']) + json.dumps(ops['graph_flow'])
        for word in ('NEGATIVE_CYCLE', 'UNREACHABLE', 'INFEASIBLE', 'UNBALANCED'):
            self.assertIn(word, text)


if __name__ == '__main__':
    unittest.main()
