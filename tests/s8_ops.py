"""S8a: frame_to_array(_to), array_to_frame, frame_mutate, frame_merge, frame_rolling, ml_metrics and stats_dist rvs (+ _to twins).
Oracles are hand-built: a numpy-only table reader, pure-Python joins / rolling windows / metrics, closed-form moments and a hand-coded
Kolmogorov-Smirnov statistic. Every pre-spawn rejection runs with a missing interpreter."""
import json, math, os, pathlib, subprocess, sys, tempfile, time, unittest
import numpy as np
BIN = str(pathlib.Path(sys.argv.pop(1)).resolve())
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests/oracle'))
from s1 import read_safetensors, write_safetensors
import s8

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


def too_large(op, params):
    return rejected(op, params, 'OUTPUT_TOO_LARGE', rc=1)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir='.')
        self.dir = pathlib.Path(self.tmp.name).resolve()
        self.addCleanup(self.tmp.cleanup)

    def p(self, name):
        return str(self.dir / name)

    def table(self, columns, name='t.npz', **extra):
        call('frame_import', dict(columns=columns, out=self.p(name), **extra))
        return self.p(name)

    def csv_table(self, columns, name='t.npz'):
        """frame_import of a column map through a CSV file (inline tables hold at most 1000 cells)."""
        names = list(columns)
        rows = len(columns[names[0]])
        (self.dir / (name + '.csv')).write_text(','.join(names) + '\n' + ''.join(','.join(str(columns[c][i]).lower() if isinstance(columns[c][i], bool) else str(columns[c][i]) for c in names) + '\n' for i in range(rows)))
        call('frame_import', dict(csv=self.p(name + '.csv'), out=self.p(name)))
        return self.p(name)

    def read(self, name):
        return s8.read_table(self.p(name) if not os.path.isabs(name) else name)

    def npy(self, array, name=None):
        path = self.dir / (name or 'in%d.npy' % len(list(self.dir.glob('in*.npy'))))
        np.save(path, array)
        return str(path)


# --------------------------------------------------------------------------------------------------------------- bridge
class Bridge(Base):
    def test_to_array_orders_columns_maps_bool_and_reports_sources(self):
        t = self.table({'a': [1, 2, 3], 'b': [0.5, 1.5, 2.5], 'c': [True, False, True], 's': ['x', 'y', 'z']})
        r = call('frame_to_array', dict(frame=t, columns=['c', 'a', 'b']))
        self.assertEqual((r['status'], r['shape'], r['columns']), ('OK', [3, 3], ['c', 'a', 'b']))
        self.assertEqual(r['values'], [[1.0, 1.0, 0.5], [0.0, 2.0, 1.5], [1.0, 3.0, 2.5]])
        self.assertEqual(r['source_dtypes'], {'c': 'bool', 'a': 'int64', 'b': 'float64'})
        self.assertEqual((r['source_na_count'], r['source_nan_count'], r['missing_mask_lost'], r['n_rows_total']), (0, 0, False, 3))
        sub = call('frame_to_array', dict(frame=t, columns=['b'], offset=1, limit=1))
        self.assertEqual((sub['values'], sub['shape'], sub['offset']), ([[1.5]], [1, 1], 1))

    def test_missing_policy_is_explicit_and_loses_the_mask_only_when_asked(self):
        t = self.table({'x': [1.0, None, 3.0, 'NaN'], 'k': [1, 2, None, 4]}, dtypes={'x': 'float64', 'k': 'int64'})
        r = call('frame_to_array', dict(frame=t, columns=['x', 'k']), rc=2)
        self.assertEqual(r['error']['data']['tool_code'], 'NA_NOT_ALLOWED')
        r = call('frame_to_array', dict(frame=t, columns=['x', 'k'], missing='nan'))
        self.assertEqual((r['source_na_count'], r['source_nan_count'], r['missing_mask_lost']), (2, 1, True))
        v = r['values']
        self.assertEqual(v[0], [1.0, 1.0])
        self.assertEqual((v[1][0], v[1][1]), ('NaN', 2.0))
        self.assertEqual((v[2][0], v[2][1]), (3.0, 'NaN'))
        self.assertEqual((v[3][0], v[3][1]), ('NaN', 4.0))
        # a numeric NaN and a masked NA both become NaN: only the counts tell them apart
        clean = call('frame_to_array', dict(frame=t, columns=['x'], limit=1))
        self.assertFalse(clean['missing_mask_lost'])

    def test_integer_precision_boundary(self):
        exact = 2 ** 53
        t = self.table({'i': [exact, -exact, 1]})
        r = call('frame_to_array', dict(frame=t, columns=['i']))
        self.assertEqual([row[0] for row in r['values']], [float(exact), float(-exact), 1.0])
        t2 = self.table({'i': [exact + 1, 5]}, name='big.npz')
        bad = call('frame_to_array', dict(frame=t2, columns=['i']), rc=2)
        self.assertEqual(bad['error']['data']['tool_code'], 'INT64_NOT_EXACT')
        ok = call('frame_to_array', dict(frame=t2, columns=['i'], allow_lossy_int=True))
        self.assertEqual((ok['status'], ok['lossy_int_count']), ('OK', 1))
        self.assertEqual(ok['values'][0][0], float(exact + 1))

    def test_dtype_and_selection_validation_before_spawn(self):
        t = self.table({'a': [1, 2], 's': ['x', 'y'], 'd': ['2026-01-01', '2026-01-02'], 'c': ['u', 'v']},
                       dtypes={'d': 'datetime64[ns]', 'c': {'dtype': 'category', 'categories': ['u', 'v']}})
        for column in ('s', 'd', 'c'):
            rejected('frame_to_array', dict(frame=t, columns=['a', column]), 'BAD_DTYPE')
        rejected('frame_to_array', dict(frame=t, columns=['a', 'nope']), 'UNKNOWN_COLUMN')
        rejected('frame_to_array', dict(frame=t, columns=['a', 'a']), 'DUPLICATE_COLUMN')
        rejected('frame_to_array', dict(frame=t, columns=[]), 'BAD_DATA')
        rejected('frame_to_array', dict(frame=t, columns=['a'], offset=2), 'BAD_DATA')  # zero rows selected
        rejected('frame_to_array', dict(frame=t, columns=['a'], limit=0), 'BAD_PARAM')
        rejected('frame_to_array', dict(frame=t, columns=['a'], offset=-1), 'BAD_PARAM')
        rejected('frame_to_array', dict(frame=self.p('missing.npz'), columns=['a']), 'ARTIFACT_NOT_FOUND', rc=4)
        (self.dir / 'junk.npz').write_bytes(b'not a zip file at all' * 4)
        rejected('frame_to_array', dict(frame=self.p('junk.npz'), columns=['a']), 'BAD_DATA')

    def test_inline_budget_write_twin_descriptor_and_exclusive_out(self):
        n = 400
        t = self.csv_table({'a': list(range(n)), 'b': [i / 8 for i in range(n)], 'c': [i % 2 == 0 for i in range(n)]})
        too_large('frame_to_array', dict(frame=t, columns=['a', 'b', 'c']))  # 1200 values
        out = self.p('x.safetensors')
        r = call('frame_to_array_to', dict(frame=t, columns=['a', 'b', 'c'], out=out, key='design'))
        self.assertEqual((r['status'], r['shape'], r['outputs_in_artifact']), ('OK', [n, 3], True))
        self.assertNotIn('values', r)
        d = r['artifacts']['design']
        self.assertEqual(set(d), DESCRIPTOR_KEYS)
        self.assertEqual((d['format'], d['key'], d['dtype'], d['shape']), ('safetensors', 'design', 'F64', [n, 3]))
        got = read_safetensors(out)['design']
        np.testing.assert_array_equal(got[:, 0], np.arange(n, dtype=float))
        np.testing.assert_array_equal(got[:, 1], np.arange(n) / 8)
        np.testing.assert_array_equal(got[:, 2], (np.arange(n) % 2 == 0).astype(float))
        call('frame_to_array_to', dict(frame=t, columns=['a'], out=out), rc=5)
        default = call('frame_to_array_to', dict(frame=t, columns=['a'], out=self.p('y.safetensors')))
        self.assertIn('x', default['artifacts'])

    def test_value_cap_is_checked_before_spawn(self):
        (self.dir / 'big.csv').write_text('a,b,c\n' + ''.join('%d,%d,%d\n' % (i, i, i) for i in range(100000)))
        call('frame_import', dict(csv=self.p('big.csv'), out=self.p('big.npz')))
        rejected('frame_to_array_to', dict(frame=self.p('big.npz'), columns=['a', 'b', 'c'], out=self.p('o.safetensors')), 'INPUT_TOO_LARGE')
        ok = call('frame_to_array_to', dict(frame=self.p('big.npz'), columns=['a', 'b'], out=self.p('o.safetensors')))  # exactly 200000 values
        self.assertEqual(ok['shape'], [100000, 2])

    def test_readonly_and_dry_run(self):
        t = self.table({'a': [1, 2, 3]})
        out = self.p('o.safetensors')
        self.assertEqual(call('frame_to_array', dict(frame=t, columns=['a']), env=dict(os.environ, PYCLI_READONLY='1'))['status'], 'OK')
        call('frame_to_array_to', dict(frame=t, columns=['a'], out=out), rc=6, env=dict(os.environ, PYCLI_READONLY='1'))
        call('frame_to_array_to', dict(frame=t, columns=['a'], out=out), env=dict(os.environ, PYCLI_DRYRUN='1'))
        self.assertFalse(os.path.exists(out))

    def test_array_to_frame_dtypes_masks_and_nan_policy(self):
        floats = [[1.0, 2.0], ['NaN', 4.0], ['Inf', 6.0]]
        r = call('array_to_frame', dict(ref=[[1.0, 2.0], [3.0, 4.0]], columns=['p', 'q'], out=self.p('f.npz')))
        self.assertEqual((r['status'], r['shape'], r['column_meta']['p']['dtype']), ('OK', [2, 2], 'float64'))
        self.assertEqual(set(r['artifacts']['table']), DESCRIPTOR_KEYS)
        self.npy(np.array([[1, 10], [2, 20], [3, 30]], dtype=np.int64), 'ints.npy')
        r = call('array_to_frame', dict(ref=self.p('ints.npy'), columns=['i', 'j'], out=self.p('i.npz')))
        self.assertEqual([r['column_meta'][c]['dtype'] for c in ('i', 'j')], ['int64', 'int64'])
        self.npy(np.array([[True], [False]]), 'b.npy')
        r = call('array_to_frame', dict(ref=self.p('b.npy'), columns=['flag'], out=self.p('b.npz')))
        self.assertEqual(r['column_meta']['flag']['dtype'], 'bool')
        t = self.read('b.npz')
        self.assertEqual(t['data']['flag'], [True, False])
        call('array_to_frame', dict(ref=floats, columns=['u', 'v'], out=self.p('v.npz')))
        t = self.read('v.npz')
        self.assertTrue(math.isnan(t['data']['u'][1]) and t['data']['u'][2] == math.inf and t['data']['u'][1] is not None)
        r = call('array_to_frame', dict(ref=floats, columns=['u', 'v'], nan_policy='missing', out=self.p('m.npz')))
        t = self.read('m.npz')
        self.assertEqual(t['data']['u'][1], None)
        self.assertEqual((r['na_counts']['u'], r['nan_count']), (1, 0))
        o = call('array_to_frame', dict(ref=[[1.0, 0.0], [2.0, 1.0]], columns=['a', 'b'], dtypes={'a': 'int64', 'b': 'bool'}, out=self.p('o.npz')))
        t = self.read('o.npz')
        self.assertEqual((t['data']['a'], t['data']['b'], t['meta']['b']['dtype']), ([1, 2], [False, True], 'bool'))
        self.assertEqual(call('array_to_frame', dict(ref=[1.0, 2.0, 3.0], columns=['only'], out=self.p('one.npz')))['shape'], [3, 1])

    def test_array_to_frame_validation(self):
        ok = dict(ref=[[1.0, 2.0], [3.0, 4.0]], columns=['a', 'b'], out=self.p('x.npz'))
        rejected('array_to_frame', dict(ok, columns=['a']), 'BAD_DATA')
        rejected('array_to_frame', dict(ok, columns=['a', 'a']), 'DUPLICATE_COLUMN')
        rejected('array_to_frame', dict(ok, columns=['a', '']), 'BAD_DATA')
        rejected('array_to_frame', dict(ok, dtypes={'zz': 'int64'}), 'UNKNOWN_COLUMN')
        rejected('array_to_frame', dict(ok, dtypes={'a': 'string'}))
        rejected('array_to_frame', dict(ok, dtypes={'a': 'bool'}), 'BAD_DATA')  # values are not exactly 0/1
        rejected('array_to_frame', dict(ok, ref=[[1.5, 2.0]], dtypes={'a': 'int64'}), 'BAD_DATA')
        rejected('array_to_frame', dict(ok, ref=[['NaN', 2.0]], dtypes={'a': 'int64'}), 'BAD_DATA')
        rejected('array_to_frame', dict(ok, out=self.p('x.safetensors')), 'BAD_REF')
        rejected('array_to_frame', dict(ok, ref=self.p('nope.npy')), 'ARTIFACT_NOT_FOUND', rc=4)
        (self.dir / 'bad.npy').write_bytes(b'\x93NUMPY\x01\x00\xff\xff' + b'x' * 20)
        rejected('array_to_frame', dict(ok, ref=self.p('bad.npy')), 'BAD_NPY')
        rejected('array_to_frame', dict(ok, ref=[[[1.0]]]), None)
        big = np.zeros((200000, 2))
        rejected('array_to_frame', dict(ok, ref=self.npy(big), columns=['a', 'b']), 'INPUT_TOO_LARGE')
        rejected('array_to_frame', dict(ok, ref=self.npy(np.zeros((1, 300))), columns=['c%d' % i for i in range(300)]), 'INPUT_TOO_LARGE')
        call('array_to_frame', dict(ok))
        call('array_to_frame', dict(ok), rc=5)
        call('array_to_frame', dict(ok, out=self.p('r.npz')), rc=6, env=dict(os.environ, PYCLI_READONLY='1'))
        call('array_to_frame', dict(ok, out=self.p('d.npz')), env=dict(os.environ, PYCLI_DRYRUN='1'))
        self.assertFalse(os.path.exists(self.p('d.npz')))

    def test_round_trip_and_the_bridge_recipe(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=(50, 3))
        call('array_to_frame', dict(ref=self.npy(x), columns=['a', 'b', 'c'], out=self.p('t.npz')))
        call('frame_to_array_to', dict(frame=self.p('t.npz'), columns=['a', 'b', 'c'], out=self.p('t.safetensors')))
        np.testing.assert_array_equal(read_safetensors(self.p('t.safetensors'))['x'], x)
        # frame_import -> frame_to_array_to -> la_lstsq -> array_to_frame
        xs = list(range(1, 21))
        ys = [3.0 + 2.0 * v for v in xs]
        self.table({'one': [1.0] * 20, 'x': [float(v) for v in xs], 'y': ys}, name='d.npz')
        call('frame_to_array_to', dict(frame=self.p('d.npz'), columns=['one', 'x'], out=self.p('a.safetensors'), key='a'))
        call('frame_to_array_to', dict(frame=self.p('d.npz'), columns=['y'], out=self.p('b.safetensors'), key='b'))
        fit = call('la_lstsq', dict(a=self.p('a.safetensors') + '#a', b=self.p('b.safetensors') + '#b'))
        np.testing.assert_allclose(np.ravel(fit['x']), [3.0, 2.0], rtol=1e-12)
        call('array_to_frame', dict(ref=[[v] for v in np.ravel(fit['x'])], columns=['coef'], out=self.p('coef.npz')))
        self.assertAlmostEqual(self.read('coef.npz')['data']['coef'][1], 2.0, places=10)


# ------------------------------------------------------------------------------------------------------------- mutate
class Mutate(Base):
    def run_mutate(self, assign, name='m.npz', **extra):
        return call('frame_mutate', dict(frame=self.p('t.npz'), assign=assign, out=self.p(name), **extra))

    def test_expressions_match_hand_values_and_keep_other_columns(self):
        x = [1.0, 2.0, 3.0, 4.0]
        n = [1, 2, 3, 4]
        self.table({'x': x, 'n': n, 'name': ['a', 'b', 'c', 'd'], 'flag': [True, False, True, False]})
        r = self.run_mutate([{'name': 'z', 'expr': 'x * 2 + n'}, {'name': 'w', 'expr': 'z ^ 2 - sqrt(x)'}, {'name': 'big', 'expr': 'ifelse(x > 2, 1, 0) + flag'}])
        self.assertEqual((r['status'], r['shape']), ('OK', [4, 7]))
        t = self.read('m.npz')
        self.assertEqual(t['columns'], ['x', 'n', 'name', 'flag', 'z', 'w', 'big'])
        self.assertEqual(t['data']['name'], ['a', 'b', 'c', 'd'])
        z = [2 * a + b for a, b in zip(x, n)]
        self.assertTrue(s8.same(t['data']['z'], z))
        self.assertTrue(s8.same(t['data']['w'], [v ** 2 - math.sqrt(a) for v, a in zip(z, x)], 1e-15))
        self.assertTrue(s8.same(t['data']['big'], [0.0 + 1, 0.0, 1.0 + 1, 1.0 + 0]))
        self.assertEqual({t['meta'][c]['dtype'] for c in ('z', 'w', 'big')}, {'float64'})
        self.assertEqual(t['meta']['n']['dtype'], 'int64')  # untouched columns keep their dtype

    def test_masked_missing_values_propagate_and_nan_values_follow_the_expression_grammar(self):
        self.table({'x': [1.0, None, 3.0, 4.0], 'y': [10.0, 20.0, None, 40.0]}, dtypes={'x': 'float64', 'y': 'float64'})
        r = self.run_mutate([{'name': 's', 'expr': 'x + y'}, {'name': 'k', 'expr': 'x * 0 + 5'}])
        t = self.read('m.npz')
        self.assertEqual(t['data']['s'], [11.0, None, None, 44.0])  # NA in either operand -> NA
        self.assertEqual(t['data']['k'], [5.0, None, 5.0, 5.0])  # only the referenced columns matter
        self.assertEqual([a['n_masked'] for a in r['assigned']], [2, 1])
        # a derived column inherits the mask of the columns it references
        self.run_mutate([{'name': 'a1', 'expr': 'x + 1'}, {'name': 'a2', 'expr': 'a1 * 2'}], name='m2.npz')
        self.assertEqual(self.read('m2.npz')['data']['a2'], [4.0, None, 8.0, 10.0])

    def test_nonfinite_policy(self):
        self.table({'x': [1.0, 0.0, -1.0], 'y': [2.0, 0.0, 4.0]})
        bad = call('frame_mutate', dict(frame=self.p('t.npz'), assign=[{'name': 'q', 'expr': 'y / x'}], out=self.p('e.npz')), rc=2)
        self.assertEqual(bad['error']['data']['tool_code'], 'NON_FINITE')
        self.assertFalse(os.path.exists(self.p('e.npz')))
        r = self.run_mutate([{'name': 'q', 'expr': 'y / x'}], name='mask.npz', nonfinite='mask')
        t = self.read('mask.npz')
        self.assertEqual(t['data']['q'], [2.0, None, -4.0])
        self.assertEqual(r['assigned'][0]['n_nonfinite'], 1)
        self.run_mutate([{'name': 'q', 'expr': 'y / x'}], name='keep.npz', nonfinite='keep')
        q = self.read('keep.npz')['data']['q']
        self.assertTrue(math.isnan(q[1]) and q[0] == 2.0)
        self.run_mutate([{'name': 'q', 'expr': '1 / x'}], name='inf.npz', nonfinite='keep')

    def test_replace_requires_the_flag_and_assignments_are_sequential(self):
        self.table({'x': [1.0, 2.0]})
        rejected('frame_mutate', dict(frame=self.p('t.npz'), assign=[{'name': 'x', 'expr': 'x + 1'}], out=self.p('r.npz')), 'BAD_PARAM')
        r = self.run_mutate([{'name': 'x', 'expr': 'x + 1'}, {'name': 'x2', 'expr': 'x * 10'}], allow_replace=True)
        t = self.read('m.npz')
        self.assertEqual(t['columns'], ['x', 'x2'])
        self.assertEqual((t['data']['x'], t['data']['x2']), ([2.0, 3.0], [20.0, 30.0]))  # x2 sees the replaced x
        self.assertTrue(r['assigned'][0]['replaced'])
        rejected('frame_mutate', dict(frame=self.p('t.npz'), assign=[{'name': 'a', 'expr': '1'}, {'name': 'a', 'expr': '2'}], out=self.p('r2.npz')), 'DUPLICATE_COLUMN')

    def test_validation_before_spawn(self):
        self.table({'x': [1.0, 2.0], 's': ['a', 'b'], 'Bad Name': [1, 2]})
        base = dict(frame=self.p('t.npz'), out=self.p('v.npz'))
        for expr in ('x +', 'foo(x)', 'y + 1', 's + 1', 'sum(x)', 'x[0]', '__import__("os")', 'x ** 2 +' * 200):
            rejected('frame_mutate', dict(base, assign=[{'name': 'z', 'expr': expr}]))
        rejected('frame_mutate', dict(base, assign=[]), 'BAD_PARAM')
        rejected('frame_mutate', dict(base, assign=[{'name': 'z', 'expr': 'x'}] * 2), 'DUPLICATE_COLUMN')
        rejected('frame_mutate', dict(base, assign=[{'name': 'z%d' % i, 'expr': 'x'} for i in range(33)]), 'BAD_PARAM')
        rejected('frame_mutate', dict(base, assign=[{'name': '', 'expr': 'x'}]), 'BAD_DATA')
        rejected('frame_mutate', dict(base, assign=[{'name': 'z', 'expr': 'x', 'extra': 1}]), 'BAD_PARAM')
        rejected('frame_mutate', dict(base, assign=[{'name': 'z'}]), 'BAD_PARAM')
        rejected('frame_mutate', dict(base, assign=[{'name': 'z', 'expr': 'x'}], nonfinite='drop'))
        rejected('frame_mutate', dict(base, assign=[{'name': 'z', 'expr': 'x'}], out=self.p('v.safetensors')), 'BAD_REF')
        (self.dir / 'wide.csv').write_text(','.join('c%d' % i for i in range(250)) + '\n' + ','.join(['1'] * 250) + '\n')
        call('frame_import', dict(csv=self.p('wide.csv'), out=self.p('wide.npz')))
        rejected('frame_mutate', dict(frame=self.p('wide.npz'), assign=[{'name': 'z%d' % i, 'expr': 'c0'} for i in range(7)], out=self.p('w.npz')), 'TOO_MANY_COLUMNS')

    def test_int64_inputs_must_be_exact_and_cells_are_capped(self):
        self.table({'n': [2 ** 53 + 1, 1]})
        bad = call('frame_mutate', dict(frame=self.p('t.npz'), assign=[{'name': 'z', 'expr': 'n + 1'}], out=self.p('v.npz')), rc=2)
        self.assertEqual(bad['error']['data']['tool_code'], 'INT64_NOT_EXACT')
        (self.dir / 'rows.csv').write_text('a,b\n' + ''.join('%d,%d\n' % (i, i) for i in range(200000)))
        call('frame_import', dict(csv=self.p('rows.csv'), out=self.p('rows.npz')))  # 400000 cells
        rejected('frame_mutate', dict(frame=self.p('rows.npz'), assign=[{'name': 'c', 'expr': 'a + b'}], out=self.p('o.npz')), 'INPUT_TOO_LARGE')  # 600000 > 500000 cells

    def test_import_is_numpy_only(self):
        self.table({'x': [1.0, 2.0]})
        r = self.run_mutate([{'name': 'z', 'expr': 'x * 2'}])
        self.assertEqual(r['meta']['pandas'], 'unloaded')


# -------------------------------------------------------------------------------------------------------------- merge
LEFT = {'id': [1, 2, 2, 3, None, 5], 'k': ['a', 'b', 'b2', 'c', 'n', 'e'], 'v': [10.0, 20.0, 21.0, 30.0, 99.0, 50.0]}
RIGHT = {'id': [2, 2, 3, 4, None, 3], 'w': [200, 201, 300, 400, 999, 301], 'k': ['B', 'B2', 'C', 'D', 'N', 'C2']}


class Merge(Base):
    def setUp(self):
        super().setUp()
        self.count = 0
        self.table(LEFT, 'l.npz', dtypes={'id': 'int64', 'v': 'float64'})
        self.table(RIGHT, 'r.npz', dtypes={'id': 'int64'})

    def merge(self, how, **extra):
        self.count += 1
        self.last = 'o_%s_%d.npz' % (how, self.count)
        return call('frame_merge', dict(left=self.p('l.npz'), right=self.p('r.npz'), on=['id'], how=how, out=self.p(self.last), **extra))

    def expected(self, how):
        pairs = s8.hand_merge(LEFT, RIGHT, how, ['id'], ['id'])
        rows = {'id': [], 'k_x': [], 'v': [], 'w': [], 'k_y': []}
        for i, j in pairs:
            rows['id'].append(LEFT['id'][i] if i is not None else RIGHT['id'][j])
            rows['k_x'].append(LEFT['k'][i] if i is not None else None)
            rows['v'].append(LEFT['v'][i] if i is not None else None)
            rows['w'].append(RIGHT['w'][j] if j is not None else None)
            rows['k_y'].append(RIGHT['k'][j] if j is not None else None)
        return rows, pairs

    def test_all_join_types_match_the_hand_join_including_duplicates_order_and_na_keys(self):
        for how in ('inner', 'left', 'right', 'outer'):
            r = self.merge(how)
            rows, pairs = self.expected(how)
            t = self.read(self.last)
            self.assertEqual(t['columns'], ['id', 'k_x', 'v', 'w', 'k_y'], how)
            for column, want in rows.items():
                self.assertTrue(s8.same(t['data'][column], want), (how, column, t['data'][column], want))
            self.assertEqual(r['shape'], [len(pairs), 5])
            self.assertEqual(r['n_pairs'], sum(1 for i, j in pairs if i is not None and j is not None))
            outer = s8.hand_merge(LEFT, RIGHT, 'outer', ['id'], ['id'])  # unmatched input rows are counted for every join type, also the ones that drop them
            self.assertEqual((r['n_left_only'], r['n_right_only']), (sum(1 for i, j in outer if j is None), sum(1 for i, j in outer if i is None)))
        self.assertEqual(self.merge('inner')['n_pairs'], 2 * 2 + 1 * 2)  # id 2: 2x2, id 3: 1x2; NA keys never match

    def test_dtypes_survive_unmatched_rows_as_masks(self):
        self.merge('left')
        t = self.read(self.last)
        self.assertEqual(t['meta']['w']['dtype'], 'int64')  # no float promotion: the unmatched rows are masked
        self.assertEqual(t['data']['w'][0], None)

    def test_validate_modes(self):
        rejected_codes = {}
        for mode in ('one_to_one', 'one_to_many', 'many_to_one'):
            r = call('frame_merge', dict(left=self.p('l.npz'), right=self.p('r.npz'), on=['id'], how='inner', validate=mode, out=self.p('v_%s.npz' % mode)), rc=2)
            rejected_codes[mode] = r['error']['data']['tool_code']
        self.assertEqual(set(rejected_codes.values()), {'MERGE_VALIDATION'})
        self.table({'id': [1, 2, 3], 'a': [1, 2, 3]}, 'u1.npz')
        self.table({'id': [1, 2, 4], 'b': [4, 5, 6]}, 'u2.npz')
        self.table({'id': [1, 1, 2], 'b': [4, 5, 6]}, 'd2.npz')
        ok = call('frame_merge', dict(left=self.p('u1.npz'), right=self.p('u2.npz'), on=['id'], validate='one_to_one', out=self.p('x1.npz')))
        self.assertEqual(ok['shape'], [2, 3])
        call('frame_merge', dict(left=self.p('u1.npz'), right=self.p('d2.npz'), on=['id'], validate='one_to_many', out=self.p('x2.npz')))
        call('frame_merge', dict(left=self.p('u1.npz'), right=self.p('d2.npz'), on=['id'], validate='many_to_one', out=self.p('x3.npz')), rc=2)
        call('frame_merge', dict(left=self.p('d2.npz'), right=self.p('u1.npz'), on=['id'], validate='many_to_one', out=self.p('x4.npz')))
        call('frame_merge', dict(left=self.p('d2.npz'), right=self.p('u1.npz'), on=['id'], validate='one_to_many', out=self.p('x5.npz')), rc=2)

    def test_row_explosion_is_refused_before_anything_is_materialised(self):
        n = 3000
        (self.dir / 'a.csv').write_text('g,a\n' + ''.join('1,%d\n' % i for i in range(n)))
        call('frame_import', dict(csv=self.p('a.csv'), out=self.p('a.npz')))
        started = time.monotonic()
        r = call('frame_merge', dict(left=self.p('a.npz'), right=self.p('a.npz'), on=['g'], suffixes=['_l', '_r'], out=self.p('boom.npz')), rc=2)
        self.assertEqual(r['error']['data']['tool_code'], 'ROW_EXPLOSION')
        self.assertIn(str(n * n), r['error']['message'])  # 9,000,000 rows announced, never built
        self.assertLess(time.monotonic() - started, 20)
        self.assertFalse(os.path.exists(self.p('boom.npz')))
        small = call('frame_merge', dict(left=self.p('l.npz'), right=self.p('r.npz'), on=['id'], max_rows=5, out=self.p('s.npz')), rc=2)
        self.assertEqual(small['error']['data']['tool_code'], 'ROW_EXPLOSION')
        self.merge('inner', max_rows=6)

    def test_suffix_and_naming_rules(self):
        r = call('frame_merge', dict(left=self.p('l.npz'), right=self.p('r.npz'), on=['id'], suffixes=['_left', '_right'], out=self.p('s.npz')))
        self.assertEqual(self.read('s.npz')['columns'], ['id', 'k_left', 'v', 'w', 'k_right'])
        self.table({'a': [1, 2], 'v': [1.0, 2.0]}, 'p.npz')
        self.table({'b': [1, 2], 'v': [3.0, 4.0]}, 'q.npz')
        r = call('frame_merge', dict(left=self.p('p.npz'), right=self.p('q.npz'), left_on=['a'], right_on=['b'], out=self.p('lr.npz')))
        t = self.read('lr.npz')
        self.assertEqual(t['columns'], ['a', 'v_x', 'b', 'v_y'])  # different key names: both kept
        rejected('frame_merge', dict(left=self.p('l.npz'), right=self.p('r.npz'), on=['id'], suffixes=['_x', '_x'], out=self.p('z.npz')), 'BAD_PARAM')
        rejected('frame_merge', dict(left=self.p('l.npz'), right=self.p('r.npz'), on=['id'], suffixes=['', '_y'], out=self.p('z.npz')), 'BAD_PARAM')
        self.table({'id': [1], 'k': [1], 'k_x': [2]}, 'clash_l.npz')
        self.table({'id': [1], 'k': [3]}, 'clash_r.npz')
        rejected('frame_merge', dict(left=self.p('clash_l.npz'), right=self.p('clash_r.npz'), on=['id'], out=self.p('z.npz')), 'DUPLICATE_COLUMN')  # k -> k_x collides with the left k_x

    def test_key_validation(self):
        base = dict(left=self.p('l.npz'), right=self.p('r.npz'), out=self.p('z.npz'))
        rejected('frame_merge', dict(base), 'BAD_PARAM')  # keys required
        rejected('frame_merge', dict(base, on=['id'], left_on=['id'], right_on=['id']), 'BAD_PARAM')
        rejected('frame_merge', dict(base, left_on=['id'], right_on=['id', 'w']), 'BAD_PARAM')
        rejected('frame_merge', dict(base, on=['nope']), 'UNKNOWN_COLUMN')
        rejected('frame_merge', dict(base, on=['id', 'id']), 'DUPLICATE_COLUMN')
        self.table({'x': [1.5, 2.5]}, 'f.npz')
        rejected('frame_merge', dict(left=self.p('f.npz'), right=self.p('f.npz'), on=['x'], out=self.p('z.npz')), 'BAD_DTYPE')  # float keys are refused
        self.table({'id': ['1', '2']}, 'str.npz')
        mismatch = call('frame_merge', dict(left=self.p('l.npz'), right=self.p('str.npz'), on=['id'], out=self.p('z.npz')), rc=2)
        self.assertEqual(mismatch['error']['data']['tool_code'], 'KEY_DTYPE_MISMATCH')
        self.table({'name': ['a', 'b', 'c']}, 's1.npz', dtypes={'name': {'dtype': 'category', 'categories': ['a', 'b', 'c']}})
        self.table({'name': ['b', 'c', 'd'], 'z': [1, 2, 3]}, 's2.npz')
        cat = call('frame_merge', dict(left=self.p('s1.npz'), right=self.p('s2.npz'), on=['name'], how='left', out=self.p('cs.npz')))
        self.assertEqual(self.read('cs.npz')['data']['z'], [None, 1, 2])  # category and string keys compare by label

    def test_composite_keys_and_cap_on_cells(self):
        self.table({'a': [1, 1, 2], 'b': ['x', 'y', 'x'], 'l': [1, 2, 3]}, 'c1.npz')
        self.table({'a': [1, 2, 2], 'b': ['y', 'x', 'x'], 'r': [10, 20, 30]}, 'c2.npz')
        call('frame_merge', dict(left=self.p('c1.npz'), right=self.p('c2.npz'), on=['a', 'b'], how='inner', out=self.p('c.npz')))
        t = self.read('c.npz')
        self.assertEqual((t['data']['l'], t['data']['r']), ([2, 3, 3], [10, 20, 30]))

    def test_numpy_only(self):
        self.assertEqual(self.merge('inner')['meta']['pandas'], 'unloaded')

    def test_an_empty_join_is_a_status_and_writes_nothing(self):
        self.table({'id': [10, 11]}, 'nomatch.npz')
        out = self.p('empty.npz')
        r = call('frame_merge', dict(left=self.p('l.npz'), right=self.p('nomatch.npz'), on=['id'], out=out), rc=1)
        self.assertEqual(r['error']['data']['tool_code'], 'EMPTY_RESULT')
        self.assertFalse(os.path.exists(out))
        outer = call('frame_merge', dict(left=self.p('l.npz'), right=self.p('nomatch.npz'), on=['id'], how='outer', out=self.p('o.npz')))
        self.assertEqual(outer['shape'], [8, 3])


# ------------------------------------------------------------------------------------------------------------ rolling
class Rolling(Base):
    VALUES = [3.0, 1.0, None, 4.0, 1.5, 9.0, 2.0, 6.0]
    count = 0

    def roll(self, **extra):
        self.count += 1
        self.last = 'r%d.npz' % self.count
        params = dict(frame=self.p('t.npz'), columns=['v'], out=self.p(self.last))
        params.update(extra)
        return call('frame_rolling', params)

    def test_every_function_matches_the_hand_window(self):
        self.table({'v': self.VALUES}, dtypes={'v': 'float64'})
        fns = ['sum', 'mean', 'std', 'var', 'min', 'max', 'median', 'count']
        for window, min_periods, center in [(3, None, False), (3, 2, False), (4, 1, True), (5, 3, True), (1, None, False), (8, 1, False)]:
            extra = dict(window=window, functions=fns)
            if min_periods is not None:
                extra['min_periods'] = min_periods
            if center:
                extra['center'] = True
            r = self.roll(**extra)
            t = self.read(self.last)
            mp = window if min_periods is None else min_periods
            self.assertEqual(r['columns_added'], ['v__' + f for f in fns])
            for fn in fns:
                want = s8.hand_rolling(self.VALUES, fn, window, mp, center)
                self.assertTrue(s8.same(t['data']['v__' + fn], want, 1e-13), (window, mp, center, fn, t['data']['v__' + fn], want))
            self.assertEqual(t['meta']['v__count']['dtype'], 'int64')
            self.assertEqual(t['meta']['v__sum']['dtype'], 'float64')
            self.assertEqual(t['columns'][0], 'v')

    def test_integer_and_bool_sources_and_multiple_columns(self):
        self.table({'a': [1, 2, 3, 4, 5], 'b': [True, False, True, True, False], 's': list('abcde')})
        self.roll(columns=['a', 'b'], window=2, functions=['sum', 'max'])
        t = self.read(self.last)
        self.assertEqual(t['columns'], ['a', 'b', 's', 'a__sum', 'a__max', 'b__sum', 'b__max'])
        self.assertTrue(s8.same(t['data']['a__sum'], [None, 3.0, 5.0, 7.0, 9.0]))
        self.assertTrue(s8.same(t['data']['b__sum'], [None, 1.0, 1.0, 2.0, 1.0]))
        rejected('frame_rolling', dict(frame=self.p('t.npz'), columns=['s'], window=2, functions=['sum'], out=self.p('z.npz')), 'BAD_DTYPE')

    def test_time_window_needs_an_explicit_sorted_datetime_column(self):
        stamps = ['2026-01-01T00:00:00', '2026-01-01T00:00:01', '2026-01-01T00:00:03', '2026-01-01T00:00:03', '2026-01-01T00:00:10', '2026-01-01T00:00:11']
        vals = [1.0, 2.0, 4.0, 8.0, 16.0, None]
        self.table({'t': stamps, 'v': vals}, dtypes={'t': 'datetime64[ns]', 'v': 'float64'})
        r = self.roll(time_window='3s', on='t', functions=['sum', 'count', 'mean'])
        t = self.read(self.last)
        secs = [0, 1, 3, 3, 10, 11]
        for fn in ('sum', 'count', 'mean'):
            want = s8.hand_rolling(vals, fn, None, 1, False, times=secs, span=3)
            self.assertTrue(s8.same(t['data']['v__' + fn], want, 1e-13), (fn, t['data']['v__' + fn], want))
        self.table({'t': list(reversed(stamps)), 'v': vals}, 'unsorted.npz', dtypes={'t': 'datetime64[ns]', 'v': 'float64'})
        bad = call('frame_rolling', dict(frame=self.p('unsorted.npz'), columns=['v'], time_window='3s', on='t', functions=['sum'], out=self.p('u.npz')), rc=2)
        self.assertEqual(bad['error']['data']['tool_code'], 'NOT_SORTED')
        rejected('frame_rolling', dict(frame=self.p('t.npz'), columns=['v'], time_window='3s', functions=['sum'], out=self.p('z.npz')), 'BAD_PARAM')  # on is required
        rejected('frame_rolling', dict(frame=self.p('t.npz'), columns=['v'], time_window='3s', on='v', functions=['sum'], out=self.p('z.npz')), 'BAD_DTYPE')
        rejected('frame_rolling', dict(frame=self.p('t.npz'), columns=['v'], time_window='3s', on='t', center=True, functions=['sum'], out=self.p('z.npz')), 'BAD_PARAM')
        rejected('frame_rolling', dict(frame=self.p('t.npz'), columns=['v'], time_window='3 weeks', on='t', functions=['sum'], out=self.p('z.npz')), 'BAD_PARAM')
        rejected('frame_rolling', dict(frame=self.p('t.npz'), columns=['v'], time_window='3s', window=3, on='t', functions=['sum'], out=self.p('z.npz')), 'BAD_PARAM')

    def test_validation_and_work_cap_before_spawn(self):
        self.table({'v': self.VALUES}, dtypes={'v': 'float64'})
        base = dict(frame=self.p('t.npz'), columns=['v'], functions=['sum'], out=self.p('z.npz'))
        rejected('frame_rolling', dict(base), 'BAD_PARAM')  # a window is required
        rejected('frame_rolling', dict(base, window=0), 'BAD_PARAM')
        rejected('frame_rolling', dict(base, window=3, min_periods=4), 'BAD_PARAM')
        rejected('frame_rolling', dict(base, window=3, min_periods=0), 'BAD_PARAM')
        rejected('frame_rolling', dict(base, window=3, functions=[]), 'BAD_PARAM')
        rejected('frame_rolling', dict(base, window=3, functions=['sum', 'sum']), 'BAD_PARAM')
        rejected('frame_rolling', dict(base, window=3, functions=['kurtosis']))
        rejected('frame_rolling', dict(base, window=3, columns=['v', 'v']), 'DUPLICATE_COLUMN')
        rejected('frame_rolling', dict(base, window=3, columns=['nope']), 'UNKNOWN_COLUMN')
        rejected('frame_rolling', dict(base, window=100001), 'BAD_PARAM')
        (self.dir / 'rows.csv').write_text('v\n' + ''.join('%d\n' % i for i in range(150000)))
        call('frame_import', dict(csv=self.p('rows.csv'), out=self.p('rows.npz')))
        rejected('frame_rolling', dict(frame=self.p('rows.npz'), columns=['v'], window=5000, functions=['median'], out=self.p('z.npz')), 'INPUT_TOO_LARGE')  # 7.5e8 window-elements
        names = [f for f in ['sum', 'mean']]
        ok = call('frame_rolling', dict(frame=self.p('rows.npz'), columns=['v'], window=50, functions=names, out=self.p('big.npz')))
        self.assertEqual(ok['shape'], [150000, 3])

    def test_numpy_only(self):
        self.table({'v': self.VALUES}, dtypes={'v': 'float64'})
        self.assertEqual(self.roll(window=2, functions=['sum'])['meta']['pandas'], 'unloaded')


# ------------------------------------------------------------------------------------------------------------- metrics
class Metrics(Base):
    def metrics(self, **params):
        return call('ml_metrics', params)

    def test_classification_scalars_confusion_and_averages_match_hand_formulas(self):
        yt = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1]
        yp = [1, 0, 0, 1, 0, 1, 1, 0, 1, 0]
        r = self.metrics(y_true=yt, y_pred=yp, metrics=['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1', 'confusion_matrix'], average='binary', pos_label=1)
        self.assertEqual((r['status'], r['n']), ('OK', 10))
        acc = sum(a == b for a, b in zip(yt, yp)) / 10
        self.assertAlmostEqual(r['metrics']['accuracy'], acc, places=15)
        p, rc, f = s8.hand_prf(yt, yp, 'binary', pos=1)
        self.assertAlmostEqual(r['metrics']['precision'], p, places=15)
        self.assertAlmostEqual(r['metrics']['recall'], rc, places=15)
        self.assertAlmostEqual(r['metrics']['f1'], f, places=15)
        conf = s8.hand_confusion(yt, yp, [0, 1])
        self.assertEqual(r['confusion_matrix'], conf)
        self.assertEqual(r['labels'], [0, 1])
        bal = (conf[0][0] / sum(conf[0]) + conf[1][1] / sum(conf[1])) / 2
        self.assertAlmostEqual(r['metrics']['balanced_accuracy'], bal, places=15)
        yt3 = ['a', 'b', 'c', 'a', 'b', 'c', 'a', 'a', 'b', 'c', 'c']
        yp3 = ['a', 'b', 'a', 'a', 'c', 'c', 'b', 'a', 'b', 'c', 'a']
        self.table({'t': yt3, 'p': yp3}, 'c.npz')
        for average in ('macro', 'micro', 'weighted'):
            r = self.metrics(frame=self.p('c.npz'), y_true='t', y_pred='p', metrics=['precision', 'recall', 'f1'], average=average)
            want = s8.hand_prf(yt3, yp3, average)
            for name, w in zip(('precision', 'recall', 'f1'), want):
                self.assertAlmostEqual(r['metrics'][name], w, places=14, msg=(average, name))
        r = self.metrics(frame=self.p('c.npz'), y_true='t', y_pred='p', metrics=['confusion_matrix'])
        self.assertEqual((r['labels'], r['confusion_matrix']), (['a', 'b', 'c'], s8.hand_confusion(yt3, yp3, ['a', 'b', 'c'])))

    def test_scores_auc_logloss_brier(self):
        rng = np.random.default_rng(4)
        y = [int(v) for v in rng.integers(0, 2, 40)]
        score = [round(float(v), 2) for v in np.clip(0.5 * np.array(y) + rng.uniform(0, 0.6, 40), 0.01, 0.99)]  # ties included
        r = self.metrics(y_true=y, y_score=score, metrics=['roc_auc', 'log_loss', 'brier'], pos_label=1)
        self.assertAlmostEqual(r['metrics']['roc_auc'], s8.hand_auc(y, score, 1), places=14)
        ll = -sum(math.log(s if t == 1 else 1 - s) for t, s in zip(y, score)) / 40
        self.assertAlmostEqual(r['metrics']['log_loss'], ll, places=14)
        self.assertAlmostEqual(r['metrics']['brier'], sum((s - t) ** 2 for t, s in zip(y, score)) / 40, places=14)
        perfect = self.metrics(y_true=[0, 0, 1, 1], y_score=[0.1, 0.2, 0.8, 0.9], metrics=['roc_auc'], pos_label=1)
        self.assertEqual(perfect['metrics']['roc_auc'], 1.0)
        inverted = self.metrics(y_true=[1, 1, 0, 0], y_score=[0.1, 0.2, 0.8, 0.9], metrics=['roc_auc'], pos_label=1)
        self.assertEqual(inverted['metrics']['roc_auc'], 0.0)
        clipped = self.metrics(y_true=[1, 0], y_score=[0.0, 0.0], metrics=['log_loss'], pos_label=1)
        self.assertAlmostEqual(clipped['metrics']['log_loss'], (-math.log(1e-15) - math.log(1 - 1e-15)) / 2, places=12)
        self.assertEqual(clipped['log_loss_eps'], 1e-15)

    def test_regression_metrics(self):
        yt = [3.0, -0.5, 2.0, 7.0, 4.2]
        yp = [2.5, 0.0, 2.0, 8.0, 4.0]
        r = self.metrics(y_true=yt, y_pred=yp, metrics=['r2', 'mse', 'mae', 'mape'])
        n = len(yt)
        mean = sum(yt) / n
        self.assertAlmostEqual(r['metrics']['mse'], sum((a - b) ** 2 for a, b in zip(yt, yp)) / n, places=14)
        self.assertAlmostEqual(r['metrics']['mae'], sum(abs(a - b) for a, b in zip(yt, yp)) / n, places=14)
        self.assertAlmostEqual(r['metrics']['r2'], 1 - sum((a - b) ** 2 for a, b in zip(yt, yp)) / sum((a - mean) ** 2 for a in yt), places=14)
        self.assertAlmostEqual(r['metrics']['mape'], sum(abs(a - b) / abs(a) for a, b in zip(yt, yp)) / n, places=14)

    def test_undefined_cases_are_a_status_with_reasons_not_an_exception(self):
        r = self.metrics(y_true=[2.0, 2.0, 2.0], y_pred=[1.0, 2.0, 3.0], metrics=['r2', 'mse'])
        self.assertEqual(r['status'], 'UNDEFINED')
        self.assertEqual(set(r['undefined']), {'r2'})
        self.assertAlmostEqual(r['metrics']['mse'], 2 / 3, places=15)
        self.assertNotIn('r2', r['metrics'])
        r = self.metrics(y_true=[0.0, 1.0], y_pred=[1.0, 1.0], metrics=['mape'])
        self.assertEqual((r['status'], list(r['undefined'])), ('UNDEFINED', ['mape']))
        r = self.metrics(y_true=[1, 1, 1], y_score=[0.2, 0.5, 0.9], metrics=['roc_auc'], pos_label=1)
        self.assertEqual((r['status'], list(r['undefined'])), ('UNDEFINED', ['roc_auc']))
        # no predicted positives: precision is 0/0
        r = self.metrics(y_true=[1, 0, 1], y_pred=[0, 0, 0], metrics=['precision', 'recall', 'f1'], average='binary', pos_label=1)
        self.assertEqual((r['status'], 'precision' in r['undefined']), ('UNDEFINED', True))
        self.assertEqual(r['metrics']['recall'], 0.0)
        z0 = self.metrics(y_true=[1, 0, 1], y_pred=[0, 0, 0], metrics=['precision'], average='binary', pos_label=1, zero_division='0')
        self.assertEqual((z0['status'], z0['metrics']['precision']), ('OK', 0.0))
        z1 = self.metrics(y_true=[1, 0, 1], y_pred=[0, 0, 0], metrics=['precision'], average='binary', pos_label=1, zero_division='1')
        self.assertEqual(z1['metrics']['precision'], 1.0)
        multi = self.metrics(y_true=[0, 1, 2, 1], y_score=[0.1, 0.2, 0.3, 0.4], metrics=['roc_auc', 'brier'], pos_label=1)
        self.assertEqual(multi['status'], 'UNDEFINED')

    def test_validation_before_spawn(self):
        ok = dict(y_true=[1, 0, 1], y_pred=[1, 0, 0], metrics=['accuracy'])
        rejected('ml_metrics', dict(ok, metrics=[]), 'BAD_PARAM')
        rejected('ml_metrics', dict(ok, metrics=['accuracy', 'accuracy']), 'BAD_PARAM')
        rejected('ml_metrics', dict(ok, metrics=['kappa']))
        rejected('ml_metrics', dict(ok, y_pred=[1, 0]), 'BAD_DATA')
        rejected('ml_metrics', dict(ok, y_pred=None), 'BAD_PARAM')
        rejected('ml_metrics', dict(ok, metrics=['roc_auc']), 'BAD_PARAM')  # needs y_score
        rejected('ml_metrics', dict(ok, y_true=[1, "NaN", 0]), 'BAD_DATA')
        rejected('ml_metrics', dict(y_true=[1, 0], y_score=[0.5, 1.5], metrics=['log_loss'], pos_label=1), 'BAD_DATA')
        rejected('ml_metrics', dict(ok, metrics=['f1'], average='macro', pos_label=1), 'BAD_PARAM')
        rejected('ml_metrics', dict(ok, y_true='a', frame=self.p('none.npz')), 'ARTIFACT_NOT_FOUND', rc=4)

    def test_table_columns_and_masked_values(self):
        self.table({'t': [1, 0, 1, None], 'p': [1, 1, 0, 0], 's': [0.9, 0.2, 0.4, 0.1]}, 'm.npz', dtypes={'t': 'int64', 'p': 'int64'})
        r = call('ml_metrics', dict(frame=self.p('m.npz'), y_true='t', y_pred='p', metrics=['accuracy']), rc=2)
        self.assertEqual(r['error']['data']['tool_code'], 'NA_NOT_ALLOWED')
        rejected('ml_metrics', dict(frame=self.p('m.npz'), y_true='nope', y_pred='p', metrics=['accuracy']), 'UNKNOWN_COLUMN')
        t = self.table({'t': [1, 0, 1, 0], 'p': [1, 1, 0, 0], 's': [0.9, 0.2, 0.4, 0.1]}, 'm2.npz')
        r = call('ml_metrics', dict(frame=t, y_true='t', y_pred='p', y_score='s', metrics=['accuracy', 'roc_auc'], pos_label=1))
        self.assertEqual(r['metrics']['accuracy'], 0.5)
        self.assertEqual(r['metrics']['roc_auc'], s8.hand_auc([1, 0, 1, 0], [0.9, 0.2, 0.4, 0.1], 1))

    def test_pos_label_must_occur(self):
        r = call('ml_metrics', dict(y_true=[1, 0, 1], y_pred=[1, 0, 0], metrics=['f1'], average='binary', pos_label=7), rc=2)
        self.assertEqual(r['error']['data']['tool_code'], 'BAD_PARAM')
        t = self.table({'t': ['a', 'b', 'a'], 'p': ['a', 'a', 'b'], 's': [0.9, 0.2, 0.4]})
        r = call('ml_metrics', dict(frame=t, y_true='t', y_pred='p', y_score='s', metrics=['f1', 'roc_auc'], average='binary', pos_label='a'))
        self.assertEqual(r['status'], 'OK')
        self.assertAlmostEqual(r['metrics']['f1'], 2 * 1 / (2 * 1 + 1 + 1), places=15)
        self.assertEqual(r['metrics']['roc_auc'], s8.hand_auc(['a', 'b', 'a'], [0.9, 0.2, 0.4], 'a'))
        bad = call('ml_metrics', dict(frame=t, y_true='t', y_pred='p', metrics=['f1'], average='binary'), rc=2)
        self.assertEqual(bad['error']['data']['tool_code'], 'BAD_PARAM')  # string labels need an explicit pos_label

    def test_class_cap_and_import_cost(self):
        labels = list(range(40))
        rejected('ml_metrics', dict(y_true=labels, y_pred=labels, metrics=['confusion_matrix']), 'INPUT_TOO_LARGE')
        r = self.metrics(y_true=[1, 0], y_pred=[1, 1], metrics=['accuracy'])
        self.assertEqual((r['meta']['pandas'], r['meta']['sklearn']), ('unloaded', 'unloaded'))


# --------------------------------------------------------------------------------------------------------------- rvs
class Rvs(Base):
    def draw(self, dist, params, n, seed, **extra):
        return call('stats_dist', dict(dist=dist, fn='rvs', params=params, n=n, seed=seed, **extra))

    def test_seed_is_required_and_echoed_and_the_stream_is_reproducible(self):
        rejected('stats_dist', dict(dist='norm', fn='rvs', params={}, n=5), 'BAD_PARAM')
        a = self.draw('norm', {}, 20, 7)
        b = self.draw('norm', {}, 20, 7)
        c = self.draw('norm', {}, 20, 8)
        self.assertEqual((a['seed'], a['generator'], a['n'], a['status']), (7, 'PCG64', 20, 'OK'))
        self.assertEqual(a['values'], b['values'])
        self.assertNotEqual(a['values'], c['values'])
        self.assertEqual(len(set(a['values'])), 20)
        big = self.draw('norm', {}, 1000, 2 ** 63 - 1)
        self.assertEqual(big['seed'], 2 ** 63 - 1)
        # first draws are a prefix of a longer stream with the same seed
        longer = self.draw('norm', {}, 40, 7)
        self.assertEqual(longer['values'][:20], a['values'])

    def test_normal_moments_and_ks(self):
        out = self.p('n.safetensors')
        r = call('stats_dist_to', dict(dist='norm', fn='rvs', params={'loc': 3.0, 'scale': 2.0}, n=20000, seed=11, out=out))
        self.assertEqual((r['status'], r['seed'], r['generator']), ('OK', 11, 'PCG64'))
        self.assertEqual(set(r['artifacts']['x']), DESCRIPTOR_KEYS)
        x = read_safetensors(out)['x']
        self.assertEqual(x.shape, (20000,))
        n = len(x)
        mean = math.fsum(x) / n
        var = math.fsum((v - mean) ** 2 for v in x) / (n - 1)
        self.assertLess(abs(mean - 3.0), 5 * 2.0 / math.sqrt(n))
        self.assertLess(abs(var - 4.0), 5 * 4.0 * math.sqrt(2.0 / (n - 1)))
        d = s8.ks_statistic(x.tolist(), lambda v: s8.norm_cdf(v, 3.0, 2.0))
        self.assertLess(d, 1.95 / math.sqrt(n))  # alpha = 0.001
        # KS has power: the wrong scale is rejected
        self.assertGreater(s8.ks_statistic(x.tolist(), lambda v: s8.norm_cdf(v, 3.0, 2.2)), 1.95 / math.sqrt(n))

    def test_other_distributions_against_closed_form_moments(self):
        cases = [
            ('uniform', {'low': -1.0, 'high': 3.0}, 2.0 - 1.0, 16 / 12),
            ('expon', {'scale': 2.5}, 2.5, 6.25),
            ('gamma', {'shape': 3.0, 'scale': 2.0}, 6.0, 12.0),
            ('beta', {'a': 2.0, 'b': 5.0}, 2 / 7, (2 * 5) / (49 * 8)),
            ('lognorm', {'mu': 0.2, 'sigma': 0.5}, math.exp(0.2 + 0.125), (math.exp(0.25) - 1) * math.exp(0.4 + 0.25)),
            ('chi2', {'df': 4.0}, 4.0, 8.0),
            ('t', {'df': 8.0}, 0.0, 8 / 6),
            ('poisson', {'lam': 4.5}, 4.5, 4.5),
            ('binom', {'n': 20, 'p': 0.3}, 6.0, 4.2),
        ]
        for k, (dist, params, mean, var) in enumerate(cases):
            n = 100000
            r = call('stats_dist_to', dict(dist=dist, fn='rvs', params=params, n=n, seed=100 + k, out=self.p('%s.safetensors' % dist)))
            x = read_safetensors(self.p('%s.safetensors' % dist))['x']
            m = math.fsum(x) / n
            v = math.fsum((a - m) ** 2 for a in x) / (n - 1)
            kurt_allowance = 6.0 if dist in ('lognorm', 't') else 3.0
            self.assertLess(abs(m - mean), 5 * math.sqrt(var / n), (dist, m, mean))
            self.assertLess(abs(v - var), 5 * var * math.sqrt(kurt_allowance / n) + 0.02 * var * (dist in ('lognorm', 't')), (dist, v, var))
        u = read_safetensors(self.p('uniform.safetensors'))['x']
        self.assertTrue(((u >= -1.0) & (u < 3.0)).all())
        self.assertLess(s8.ks_statistic(u.tolist(), lambda v: (v + 1.0) / 4.0), 1.95 / math.sqrt(len(u)))
        e = read_safetensors(self.p('expon.safetensors'))['x']
        self.assertLess(s8.ks_statistic(e.tolist(), lambda v: 1 - math.exp(-v / 2.5)), 1.95 / math.sqrt(len(e)))
        ln = read_safetensors(self.p('lognorm.safetensors'))['x']
        self.assertLess(s8.ks_statistic(ln.tolist(), lambda v: s8.norm_cdf(math.log(v), 0.2, 0.5)), 1.95 / math.sqrt(len(ln)))
        counts = read_safetensors(self.p('binom.safetensors'))['x']
        self.assertTrue(((counts >= 0) & (counts <= 20) & (counts == np.floor(counts))).all())

    def test_caps_and_parameter_validation_before_spawn(self):
        base = dict(dist='norm', fn='rvs', params={}, n=10, seed=1)
        too_large('stats_dist', dict(base, n=1001))
        rejected('stats_dist_to', dict(base, n=1000001, out=self.p('z.safetensors')), 'INPUT_TOO_LARGE')
        rejected('stats_dist', dict(base, n=0), 'BAD_PARAM')
        rejected('stats_dist', dict(base, seed=-1), 'BAD_PARAM')
        rejected('stats_dist', dict(base, seed=2 ** 63))
        rejected('stats_dist', dict(base, seed=1.5))
        rejected('stats_dist', dict(base, x=[1.0]), 'BAD_PARAM')  # x is only for pdf/cdf/ppf/sf
        rejected('stats_dist', dict(dist='norm', fn='pdf', x=[0.0], params={}, n=3), 'BAD_PARAM')  # n and seed only for rvs
        rejected('stats_dist', dict(dist='norm', fn='pdf', x=[0.0], params={}, seed=3), 'BAD_PARAM')
        rejected('stats_dist', dict(dist='gamma', fn='pdf', x=[0.0], params={'shape': 1.0}), 'BAD_PARAM')  # new families sample only
        rejected('stats_dist', dict(base, params={'scale': 0.0}), 'BAD_PARAM')
        rejected('stats_dist', dict(base, params={'scale': -1.0}), 'BAD_PARAM')
        rejected('stats_dist', dict(base, params={'sigma': 1.0}), 'BAD_PARAM')
        rejected('stats_dist', dict(base, dist='uniform', params={'low': 2.0, 'high': 1.0}), 'BAD_PARAM')
        rejected('stats_dist', dict(base, dist='gamma', params={'shape': 1.0}), 'BAD_PARAM')
        rejected('stats_dist', dict(base, dist='gamma', params={'shape': 0.0, 'scale': 1.0}), 'BAD_PARAM')
        rejected('stats_dist', dict(base, dist='beta', params={'a': 1.0}), 'BAD_PARAM')
        rejected('stats_dist', dict(base, dist='poisson', params={'lam': -1.0}), 'BAD_PARAM')
        rejected('stats_dist', dict(base, dist='poisson', params={'lam': 1e12}), 'BAD_PARAM')
        rejected('stats_dist', dict(base, dist='binom', params={'n': 10, 'p': 1.5}), 'BAD_PARAM')
        rejected('stats_dist', dict(base, dist='binom', params={'n': 2.5, 'p': 0.5}), 'BAD_PARAM')
        rejected('stats_dist', dict(base, dist='t', params={'df': 0.0}), 'BAD_PARAM')
        rejected('stats_dist', dict(base, dist='cauchy'))
        rejected('stats_dist_to', dict(dist='norm', fn='pdf', x=[0.0], params={}, out=self.p('z.safetensors')))  # the write twin samples only (enum)
        call('stats_dist_to', dict(base, out=self.p('s.safetensors')))
        call('stats_dist_to', dict(base, out=self.p('s.safetensors')), rc=5)
        call('stats_dist_to', dict(base, out=self.p('r.safetensors')), rc=6, env=dict(os.environ, PYCLI_READONLY='1'))

    def test_existing_density_path_is_unchanged_and_sampling_is_numpy_only(self):
        r = call('stats_dist', dict(dist='norm', fn='cdf', x=[0.0, 1.96], params={}))
        self.assertAlmostEqual(r['values'][1], s8.norm_cdf(1.96), places=14)
        draw = self.draw('uniform', {'low': 0.0, 'high': 1.0}, 5, 3)
        self.assertEqual((draw['meta']['pandas'], draw['meta']['scipy']), ('unloaded', 'unloaded'))


class Schema(unittest.TestCase):
    def test_registration_documentation_and_access(self):
        schema = json.loads(subprocess.run([BIN, '--schema'], capture_output=True, text=True).stdout)
        ops = {o['name']: o for o in schema['operations']}
        reads = ['frame_to_array', 'ml_metrics', 'stats_dist']
        writes = ['frame_to_array_to', 'array_to_frame', 'frame_mutate', 'frame_merge', 'frame_rolling', 'stats_dist_to']
        for name in reads:
            self.assertEqual(ops[name]['access'], 'read', name)
        for name in writes:
            self.assertEqual(ops[name]['access'], 'write', name)
            self.assertIn('out', {p['name'] for p in ops[name]['params']}, name)
        for name in reads + writes:
            for param in ops[name]['params']:
                self.assertTrue(param.get('description'), (name, param['name']))
        self.assertIn('seed', {p['name']: p for p in ops['stats_dist']['params']})
        seed = next(p for p in ops['stats_dist']['params'] if p['name'] == 'seed')
        self.assertIn('required', seed['description'].lower())
        for name in writes:
            self.assertIn('artifacts', ops[name]['description'], name)


if __name__ == '__main__':
    unittest.main()
