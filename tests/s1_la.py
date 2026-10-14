"""S1 dense linear algebra: la_lstsq(_to), la_props, la_decomp. Oracles are independent:
exact rational arithmetic (fractions), NIST StRD certified values, closed forms, and
numpy/scipy called directly by the test (never the worker)."""
import hashlib, json, math, os, pathlib, random, subprocess, sys, tempfile, unittest
from fractions import Fraction as F
import numpy as np
import scipy.linalg as sla
BIN = str(pathlib.Path(sys.argv.pop(1)).resolve())
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / 'oracle'))
import s1
from s1 import read_safetensors

NO_PYTHON = dict(os.environ, PYCLI_PYTHON='/no/such/python')


def call(op, params=None, rc=0, env=None):
    proc = subprocess.run([BIN, '--input-json', json.dumps({'op': op, 'params': params or {}})], capture_output=True, text=True, env=env)
    assert proc.returncode == rc, (proc.returncode, proc.stdout[:2000], proc.stderr[:2000])
    return json.loads(proc.stdout)


def code(response):
    return response['error']['data']['tool_code']


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir='.')
        self.dir = pathlib.Path(self.tmp.name).resolve()
        self.addCleanup(self.tmp.cleanup)

    def out(self, name='o.safetensors'):
        return str(self.dir / name)


class Lstsq(Base):
    def test_exact_quadratic(self):
        x = np.arange(6.0)
        a = np.vander(x, 3, increasing=True)
        r = call('la_lstsq', {'a': a.tolist(), 'b': (2 + 3 * x - x * x).tolist()})
        self.assertEqual((r['status'], r['rank']), ('OK', 3))
        np.testing.assert_allclose(r['x'], [2, 3, -1], rtol=1e-12, atol=1e-12)
        self.assertLess(r['residual_norm'], 1e-12)
        self.assertEqual(len(r['residues']), 1)
        self.assertEqual(r['driver'], 'gelsd')

    def test_random_integer_problems_against_exact_rationals(self):
        rng = random.Random(99)
        for trial in range(25):
            m, n = rng.randint(3, 9), rng.randint(1, 3)
            a = [[rng.randint(-9, 9) for _ in range(n)] for _ in range(m)]
            b = [rng.randint(-9, 9) for _ in range(m)]
            if s1.rank_exact(a) < n:
                continue
            exact = [float(row[0]) for row in s1.lstsq_exact(a, b)]
            for driver in ('gelsd', 'gelsy', 'gelss'):
                r = call('la_lstsq', {'a': a, 'b': b, 'driver': driver})
                self.assertEqual((r['status'], r['rank']), ('OK', n), (trial, driver))
                np.testing.assert_allclose(r['x'], exact, rtol=1e-11, atol=1e-12, err_msg=f'{trial} {driver}')
                A, B = [[F(v) for v in row] for row in a], [F(v) for v in b]
                rss = sum((sum(A[i][j] * F(exact_x) for j, exact_x in enumerate(exact)) - B[i]) ** 2 for i in range(m))
                self.assertAlmostEqual(r['residues'][0], float(rss), delta=1e-9 * max(1, float(rss)))
                self.assertAlmostEqual(r['residual_norm'], math.sqrt(r['residues'][0]), delta=1e-12 * max(1, r['residual_norm']))

    def test_multiple_right_hand_sides_shape(self):
        a = [[1, 0], [0, 1], [1, 1]]
        b = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        r = call('la_lstsq', {'a': a, 'b': b})
        expected = np.linalg.lstsq(np.array(a, float), np.array(b, float), rcond=None)[0]
        np.testing.assert_allclose(r['x'], expected, rtol=1e-12)
        self.assertEqual(np.array(r['x']).shape, (2, 3))
        self.assertEqual(len(r['residues']), 3)

    def test_longley_certified_values_all_drivers(self):
        for driver in ('gelsd', 'gelsy', 'gelss'):
            r = call('la_lstsq', {'a': s1.LONGLEY_A, 'b': s1.LONGLEY_B, 'driver': driver})
            self.assertEqual((r['status'], r['rank']), ('OK', 7))
            lres = [s1.lre(c, q) for c, q in zip(r['x'], s1.LONGLEY_CERTIFIED)]
            self.assertGreaterEqual(min(lres), 9.0, (driver, lres))  # measured 10.8..13 on aarch64
            self.assertGreaterEqual(s1.lre(r['residues'][0], s1.LONGLEY_RESIDUAL_SS), 8.0)

    def test_wampler1_certified_all_ones(self):
        r = call('la_lstsq', {'a': s1.WAMPLER1_A, 'b': s1.WAMPLER1_B})
        self.assertEqual(r['rank'], 6)
        self.assertGreaterEqual(min(s1.lre(v, 1.0) for v in r['x']), 7.0)

    def test_ill_conditioned_hilbert_against_exact_rational_solution(self):
        m, n = 12, 5
        a = [[1.0 / (i + j + 1) for j in range(n)] for i in range(m)]
        b = [float(i + 1) for i in range(m)]
        exact = [row[0] for row in s1.lstsq_exact(a, b)]
        r = call('la_lstsq', {'a': a, 'b': b})
        self.assertGreaterEqual(min(s1.lre(c, q) for c, q in zip(r['x'], exact)), 10.0)
        self.assertAlmostEqual(r['singular_values'][0] / r['singular_values'][-1], np.linalg.cond(np.array(a)), delta=1e-6 * 66598)

    def test_matches_scipy_directly(self):
        rng = np.random.default_rng(21)
        a, b = rng.standard_normal((30, 4)), rng.standard_normal((30, 2))
        for driver in ('gelsd', 'gelsy', 'gelss'):
            x, residues, rank, sv = sla.lstsq(a, b, lapack_driver=driver)
            r = call('la_lstsq', {'a': a.tolist(), 'b': b.tolist(), 'driver': driver})
            np.testing.assert_allclose(r['x'], x, rtol=1e-12, atol=1e-13)
            np.testing.assert_allclose(r['residues'], residues, rtol=1e-10)
            self.assertEqual(r['rank'], rank)
            np.testing.assert_allclose(r['singular_values'], np.linalg.svd(a, compute_uv=False), rtol=1e-12)

    def test_rank_deficient_is_not_ok(self):
        r = call('la_lstsq', {'a': [[1, 1], [1, 1], [1, 1]], 'b': [1, 2, 3]})
        self.assertEqual((r['status'], r['rank']), ('RANK_DEFICIENT', 1))
        np.testing.assert_allclose(r['x'], [1, 1], rtol=1e-12)  # minimum-norm solution, closed form
        self.assertAlmostEqual(r['residues'][0], 2.0, delta=1e-12)  # (1-2)^2 + 0 + (3-2)^2
        self.assertLess(r['singular_values'][1], 1e-15)
        # rcond forces rank decisions: diag(1, 1e-9) is rank 2 by default, rank 1 with rcond=1e-6.
        a = [[1, 0], [0, 1e-9]]
        self.assertEqual(call('la_lstsq', {'a': a, 'b': [1, 1]})['status'], 'OK')
        r = call('la_lstsq', {'a': a, 'b': [1, 1], 'rcond': 1e-6})
        self.assertEqual((r['status'], r['rank']), ('RANK_DEFICIENT', 1))
        np.testing.assert_allclose(r['x'], [1, 0], atol=1e-12)

    def test_underdetermined_minimum_norm_is_ok(self):
        r = call('la_lstsq', {'a': [[1, 1]], 'b': [2]})
        self.assertEqual((r['status'], r['rank']), ('OK', 1))
        np.testing.assert_allclose(r['x'], [1, 1], rtol=1e-13)

    def test_validation_before_spawn(self):
        good = {'a': [[1, 0], [0, 1]], 'b': [1, 2]}
        for name, p in {'nan': {'a': [[1, 'NaN'], [0, 1]], 'b': [1, 2]}, 'inf_b': {'a': good['a'], 'b': [1, 'Inf']}, 'null': {'a': [[1, None], [0, 1]], 'b': [1, 2]},
                        'rows': {'a': good['a'], 'b': [1, 2, 3]}, 'rank1_a': {'a': [1, 2], 'b': [1, 2]}, 'rank3_b': {'a': good['a'], 'b': [[[1]], [[2]]]},
                        'neg_rcond': {**good, 'rcond': -1}, 'big_rcond': {**good, 'rcond': 1}, 'nan_rcond': {**good, 'rcond': 'NaN'}, 'driver': {**good, 'driver': 'gelsx'},
                        'ragged': {'a': [[1, 0], [0]], 'b': [1, 2]}, 'path': {'a': '/etc/passwd', 'b': [1]}, 'extra': {**good, 'x': 1}}.items():
            with self.subTest(case=name):
                r = call('la_lstsq', p, 2, NO_PYTHON)
                self.assertEqual(r['success'], False)

    def test_inline_output_cap_and_big_by_ref(self):
        rng = np.random.default_rng(4)
        a, b = rng.standard_normal((60, 20)), rng.standard_normal((60, 60))
        np.save(self.dir / 'a.npy', a)
        np.save(self.dir / 'b.npy', b)
        r = call('la_lstsq', {'a': str(self.dir / 'a.npy'), 'b': str(self.dir / 'b.npy')}, 1, NO_PYTHON)  # 20*60 = 1200 > 1000
        self.assertEqual(code(r), 'OUTPUT_TOO_LARGE')
        self.assertIn('la_lstsq_to', json.dumps(r))
        out = self.out('x.safetensors')
        r = call('la_lstsq_to', {'a': str(self.dir / 'a.npy'), 'b': str(self.dir / 'b.npy'), 'out': out})
        x = read_safetensors(out)['x']
        np.testing.assert_allclose(x, sla.lstsq(a, b)[0], rtol=1e-10, atol=1e-12)
        self.assertEqual((r['status'], r['x']['shape'], r['x']['dtype'], r['x']['ref'], r['rank']), ('OK', [20, 60], 'F64', out + '#x', 20))

    def test_to_policy_conflict_dryrun_readonly(self):
        p = {'a': [[1, 0], [0, 1], [1, 1]], 'b': [1, 2, 3], 'out': self.out()}
        call('la_lstsq_to', p, 0, dict(os.environ, PYCLI_DRYRUN='1'))
        self.assertFalse(pathlib.Path(p['out']).exists())
        call('la_lstsq_to', p, 6, dict(os.environ, PYCLI_READONLY='1'))
        self.assertFalse(pathlib.Path(p['out']).exists())
        r = call('la_lstsq_to', p)
        np.testing.assert_allclose(read_safetensors(p['out'])['x'], np.linalg.lstsq(np.array(p['a'], float), np.array(p['b'], float), rcond=None)[0], rtol=1e-12)
        before = hashlib.sha256(pathlib.Path(p['out']).read_bytes()).hexdigest()
        self.assertEqual(call('la_lstsq_to', p, 5)['error']['code'], 'CONFLICT')
        self.assertEqual(hashlib.sha256(pathlib.Path(p['out']).read_bytes()).hexdigest(), before)
        for bad in ('/tmp/pycli-escape.safetensors', str(self.dir / 'x.npy'), str(self.dir / 'nodir' / 'x.safetensors')):
            call('la_lstsq_to', {**p, 'out': bad}, 2, NO_PYTHON)

    def test_to_rank_deficient_still_writes_and_says_so(self):
        out = self.out()
        r = call('la_lstsq_to', {'a': [[1, 1], [1, 1], [1, 1]], 'b': [1, 2, 3], 'out': out})
        self.assertEqual(r['status'], 'RANK_DEFICIENT')
        np.testing.assert_allclose(read_safetensors(out)['x'], [1, 1], rtol=1e-12)


class Props(Base):
    def props(self, a, what, **kw):
        return call('la_props', {'a': a, 'what': what, **kw})

    def test_exact_rational_det_rank_trace_inverse(self):
        rng = random.Random(5)
        checked = 0
        while checked < 20:
            n = rng.randint(2, 6)
            a = [[rng.randint(-5, 5) for _ in range(n)] for _ in range(n)]
            d = s1.det_exact(a)
            r = self.props(a, ['det', 'slogdet', 'rank', 'trace', 'inverse'])
            self.assertEqual(r['rank'], s1.rank_exact(a))
            self.assertEqual(r['trace'], sum(a[i][i] for i in range(n)))
            if d != 0:
                checked += 1
                self.assertLessEqual(abs(r['det'] - float(d)), 1e-11 * max(1, abs(float(d))))
                self.assertEqual(r['slogdet']['sign'], 1 if d > 0 else -1)
                self.assertAlmostEqual(r['slogdet']['logabsdet'], math.log(abs(d)), delta=1e-12 * max(1, abs(math.log(abs(d)))))
                self.assertEqual(r['status'], 'OK', a)
                exact = s1.inverse_exact(a)
                np.testing.assert_allclose(r['inverse'], [[float(v) for v in row] for row in exact], rtol=1e-8, atol=1e-10)
                self.assertLess(r['inverse_residual'], 1e-9)
            else:
                self.assertEqual(r['status'], 'SINGULAR', a)
                self.assertNotIn('inverse', r)

    def test_hilbert3_closed_forms(self):
        h = [[1 / (i + j + 1) for j in range(3)] for i in range(3)]
        r = self.props(h, ['det', 'cond', 'inverse'])
        self.assertAlmostEqual(r['det'], 1 / 2160, delta=1e-15)
        np.testing.assert_allclose(r['inverse'], [[9, -36, 30], [-36, 192, -180], [30, -180, 180]], rtol=1e-11)
        self.assertAlmostEqual(r['cond'], 524.0567775860644, delta=1e-8)  # known 2-norm condition number of H3

    def test_norms_closed_form_and_scipy(self):
        a = [[1, -2, 3], [-4, 5, -6]]
        r = {o: self.props(a, ['norm'], norm_ord=o)['norm'] for o in ('fro', '1', 'inf', '2', 'nuc')}
        self.assertAlmostEqual(r['fro'], math.sqrt(91), delta=1e-14)
        self.assertEqual((r['1'], r['inf']), (9, 15))  # max column sum 3+6, max row sum 4+5+6
        s = sla.svdvals(np.array(a, float))
        self.assertAlmostEqual(r['2'], s[0], delta=1e-13)
        self.assertAlmostEqual(r['nuc'], s.sum(), delta=1e-13)
        self.assertAlmostEqual(self.props(a, ['norm'])['norm'], math.sqrt(91), delta=1e-14)  # default is Frobenius

    def test_cond_variants_and_diagonal_closed_form(self):
        d = [[1, 0], [0, 1e-8]]
        self.assertAlmostEqual(self.props(d, ['cond'])['cond'] / 1e8, 1.0, delta=1e-12)
        a = np.array([[2.0, 1], [1, 3]])
        for p, ref in (('2', np.linalg.cond(a, 2)), ('1', np.linalg.cond(a, 1)), ('inf', np.linalg.cond(a, np.inf)), ('fro', np.linalg.cond(a, 'fro'))):
            self.assertAlmostEqual(self.props(a.tolist(), ['cond'], cond_p=p)['cond'], ref, delta=1e-12)
        r = self.props([[1, 2], [2, 4]], ['cond'])
        self.assertEqual(r['cond'], 'Inf')
        r = self.props([[1, 2, 3], [4, 5, 6]], ['cond', 'rank', 'norm'])  # rectangular is fine for these
        self.assertEqual(r['rank'], 2)
        self.assertAlmostEqual(r['cond'], np.linalg.cond(np.array([[1, 2, 3], [4, 5, 6.]])), delta=1e-12)

    def test_singular_matrices_get_honest_statuses(self):
        r = self.props([[1, 2, 3], [4, 5, 6], [7, 8, 9]], ['det', 'rank', 'inverse', 'slogdet'])
        self.assertEqual(r['rank'], 2)  # exact rank is 2
        self.assertEqual(s1.rank_exact([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), 2)
        self.assertLess(abs(r['det']), 1e-12)  # exact det is 0; floating LU gives rounding-level noise
        self.assertEqual(r['status'], 'SINGULAR')
        self.assertNotIn('inverse', r)
        r = self.props([[1, 2], [2, 4]], ['inverse', 'det', 'slogdet'])
        self.assertEqual((r['status'], r['det']), ('SINGULAR', 0))
        self.assertEqual(r['slogdet'], {'sign': 0, 'logabsdet': '-Inf'})
        r = self.props([[1, 0], [0, 1e-20]], ['inverse', 'rank'])  # numerically singular to working precision: conservative
        self.assertEqual(r['status'], 'SINGULAR')

    def test_det_overflow_is_flagged_and_slogdet_is_exact(self):
        n = 40
        a = (np.eye(n) * 1e10).tolist()
        r = self.props(a, ['det', 'slogdet'])
        self.assertEqual(r['det'], 'Inf')
        self.assertTrue(any(w['category'] == 'NumericalWarning' for w in r['warnings']))
        self.assertAlmostEqual(r['slogdet']['logabsdet'], n * math.log(1e10), delta=1e-10)
        r = self.props((np.eye(n) * 1e-10).tolist(), ['det'])
        self.assertEqual(r['det'], 0)
        self.assertTrue(any(w['category'] == 'NumericalWarning' for w in r['warnings']))

    def test_defaults_and_requested_keys_only(self):
        r = call('la_props', {'a': [[2, 0], [0, 3]]})
        self.assertEqual(sorted(k for k in r if k in ('det', 'rank', 'cond', 'norm', 'trace', 'inverse', 'slogdet')), ['cond', 'det', 'norm', 'rank', 'trace'])
        self.assertEqual((r['det'], r['rank'], r['trace']), (6, 2, 5))
        r = self.props([[2, 0], [0, 3]], ['trace'])
        self.assertNotIn('det', r)
        self.assertEqual(r['status'], 'OK')

    def test_rank_tol_and_threshold(self):
        a = [[1, 0], [0, 1e-10]]
        self.assertEqual(self.props(a, ['rank'])['rank'], 2)
        self.assertEqual(self.props(a, ['rank'], rank_tol=1e-8)['rank'], 1)
        self.assertEqual(self.props(a, ['rank'], rank_tol=0)['rank'], 2)

    def test_validation_before_spawn(self):
        sq = [[1, 2], [3, 4]]
        for name, p in {'nan': {'a': [[1, 'NaN'], [0, 1]]}, 'rect_det': {'a': [[1, 2, 3], [4, 5, 6]], 'what': ['det']}, 'rect_inverse': {'a': [[1, 2, 3], [4, 5, 6]], 'what': ['inverse']},
                        'rect_slogdet': {'a': [[1, 2, 3], [4, 5, 6]], 'what': ['slogdet']}, 'rect_cond1': {'a': [[1, 2, 3], [4, 5, 6]], 'what': ['cond'], 'cond_p': '1'},
                        'unknown_what': {'a': sq, 'what': ['eigs']}, 'empty_what': {'a': sq, 'what': []}, 'bad_ord': {'a': sq, 'norm_ord': '3'}, 'bad_p': {'a': sq, 'cond_p': '7'},
                        'neg_tol': {'a': sq, 'rank_tol': -1}, 'vector': {'a': [1, 2]}, 'dup_what': {'a': sq, 'what': ['det', 'det']},
                        'inline_cap': {'a': np.eye(32).tolist(), 'what': ['rank']}}.items():
            with self.subTest(case=name):
                call('la_props', p, 2, NO_PYTHON)
        np.save(self.dir / 'big.npy', np.eye(40))
        r = call('la_props', {'a': str(self.dir / 'big.npy'), 'what': ['inverse']}, 1, NO_PYTHON)
        self.assertEqual(code(r), 'OUTPUT_TOO_LARGE')
        self.assertEqual(call('la_props', {'a': str(self.dir / 'big.npy'), 'what': ['det', 'rank']})['rank'], 40)

    def test_32x32_inverse_boundary(self):
        # 31x31 = 961 numbers inline; 32x32 = 1024 exceeds the inline input cap before the output cap.
        a = (np.eye(31) * 2 + np.eye(31, k=1)).tolist()
        r = self.props(a, ['inverse'])
        np.testing.assert_allclose(r['inverse'], np.linalg.inv(np.array(a)), rtol=1e-12)


def sorted_eig(w, v):
    order = np.lexsort((w.imag, w.real))
    return w[order], v[:, order]


class Decomp(Base):
    def decomp(self, a, kind, name='o.safetensors', **kw):
        out = self.out(name)
        r = call('la_decomp', {'a': a, 'kind': kind, 'out': out, **kw})
        return r, read_safetensors(out)

    def test_svd_matches_scipy_and_closed_form(self):
        rng = np.random.default_rng(1)
        for shape in ((5, 3), (3, 5), (4, 4)):
            a = rng.standard_normal(shape)
            for full in (False, True):
                r, t = self.decomp(a.tolist(), 'svd', f'{shape}{full}.safetensors', full=full)
                u, s, vt = sla.svd(a, full_matrices=full)
                self.assertEqual(sorted(t), ['s', 'u', 'vt'])
                for got, ref in ((t['u'], u), (t['s'], s), (t['vt'], vt)):
                    self.assertEqual(got.shape, ref.shape)
                    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-13)  # same LAPACK driver: signs agree
                self.assertTrue(np.all(np.diff(t['s']) <= 0) and t['s'][-1] >= 0)
                k = min(shape)
                np.testing.assert_allclose((t['u'][:, :k] * t['s']) @ t['vt'][:k], a, atol=1e-13)
                np.testing.assert_allclose(t['u'].T @ t['u'], np.eye(t['u'].shape[1]), atol=1e-13)
                self.assertLess(r['reconstruction_error'], 1e-14)
                self.assertEqual(r['status'], 'OK')
                np.testing.assert_allclose(r['values']['s'], s, rtol=1e-12)
                self.assertEqual({k: v['shape'] for k, v in r['tensors'].items()}, {k: list(t[k].shape) for k in t})
        r, t = self.decomp([[3, 0], [0, -2]], 'svd')
        np.testing.assert_allclose(t['s'], [3, 2], rtol=1e-15)  # closed form

    def test_qr_modes_and_pivoting(self):
        rng = np.random.default_rng(2)
        a = rng.standard_normal((6, 4))
        for full in (False, True):
            r, t = self.decomp(a.tolist(), 'qr', f'q{full}.safetensors', full=full)
            q, rr = sla.qr(a, mode='full' if full else 'economic')
            np.testing.assert_allclose(t['q'], q, rtol=1e-12, atol=1e-13)
            np.testing.assert_allclose(t['r'], rr, rtol=1e-12, atol=1e-13)
            np.testing.assert_allclose(t['q'].T @ t['q'], np.eye(t['q'].shape[1]), atol=1e-13)
            np.testing.assert_allclose(t['q'] @ t['r'], a, atol=1e-13)
            self.assertNotIn('p', t)
        r, t = self.decomp(a.tolist(), 'qr', 'p.safetensors', pivoting=True)
        q, rr, p = sla.qr(a, mode='economic', pivoting=True)
        self.assertEqual(t['p'].dtype, np.dtype('<i8'))
        self.assertEqual(t['p'].tolist(), p.tolist())
        np.testing.assert_allclose(t['q'] @ t['r'], a[:, t['p']], atol=1e-13)  # convention: A[:, p] = Q R
        self.assertTrue(np.all(np.diff(np.abs(np.diag(t['r']))) <= 1e-12))
        self.assertLess(r['reconstruction_error'], 1e-14)

    def test_eigh_ascending_and_closed_form(self):
        r, t = self.decomp([[2, 1], [1, 2]], 'eigh')
        np.testing.assert_allclose(t['w'], [1, 3], rtol=1e-14)
        s = 1 / math.sqrt(2)
        np.testing.assert_allclose(np.abs(t['v']), [[s, s], [s, s]], rtol=1e-14)  # sign is LAPACK's; magnitudes are closed form
        self.assertAlmostEqual(abs(t['v'][:, 0] @ np.array([1, -1]) * s), 1.0, delta=1e-14)
        rng = np.random.default_rng(3)
        b = rng.standard_normal((6, 6))
        a = b + b.T
        r, t = self.decomp(a.tolist(), 'eigh', 'e2.safetensors')
        w, v = sla.eigh(a)
        np.testing.assert_allclose(t['w'], w, rtol=1e-12, atol=1e-13)
        np.testing.assert_allclose(t['v'], v, rtol=1e-10, atol=1e-12)
        self.assertTrue(np.all(np.diff(t['w']) >= 0))
        np.testing.assert_allclose(a @ t['v'], t['v'] * t['w'], atol=1e-12)
        np.testing.assert_allclose(t['v'].T @ t['v'], np.eye(6), atol=1e-13)
        self.assertLess(r['reconstruction_error'], 1e-13)

    def test_eig_complex_split_sorting_and_closed_forms(self):
        r, t = self.decomp([[0, -1], [1, 0]], 'eig')  # rotation by 90 degrees: eigenvalues -i, +i (sorted by real, then imaginary)
        self.assertEqual(sorted(t), ['vr_im', 'vr_re', 'w_im', 'w_re'])
        np.testing.assert_allclose(t['w_re'], [0, 0], atol=1e-15)
        np.testing.assert_allclose(t['w_im'], [-1, 1], rtol=1e-14)
        v = t['vr_re'] + 1j * t['vr_im']
        np.testing.assert_allclose(np.array([[0, -1], [1, 0]]) @ v, v * (t['w_re'] + 1j * t['w_im']), atol=1e-14)
        # companion matrix of (x-1)(x-2)(x-3): real eigenvalues, imaginary parts exactly zero
        r, t = self.decomp([[6, -11, 6], [1, 0, 0], [0, 1, 0]], 'eig', 'c.safetensors')
        np.testing.assert_allclose(t['w_re'], [1, 2, 3], rtol=1e-12)
        self.assertTrue(np.all(t['w_im'] == 0))
        rng = np.random.default_rng(6)
        a = rng.standard_normal((8, 8))
        r, t = self.decomp(a.tolist(), 'eig', 'g.safetensors')
        w, v = sorted_eig(*sla.eig(a))
        np.testing.assert_allclose(t['w_re'], w.real, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(t['w_im'], w.imag, rtol=1e-10, atol=1e-12)
        got = t['vr_re'] + 1j * t['vr_im']
        np.testing.assert_allclose(np.abs(got), np.abs(v), rtol=1e-8, atol=1e-10)  # phase is LAPACK's: compare magnitudes
        wc = t['w_re'] + 1j * t['w_im']
        np.testing.assert_allclose(a @ got, got * wc, atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(got, axis=0), 1, rtol=1e-13)  # unit 2-norm columns
        self.assertTrue(all((x.real, x.imag) <= (y.real, y.imag) for x, y in zip(wc, wc[1:])))
        self.assertLess(r['reconstruction_error'], 1e-12)
        self.assertEqual(len(r['values']['w_re']), 8)

    def test_cholesky_lower_default_and_upper(self):
        r, t = self.decomp([[4, 2], [2, 5]], 'cholesky')
        self.assertEqual(sorted(t), ['l'])
        np.testing.assert_array_equal(t['l'], [[2, 0], [1, 2]])  # exact closed form
        r, t = self.decomp([[4, 2], [2, 5]], 'cholesky', 'u.safetensors', lower=False)
        self.assertEqual(sorted(t), ['u'])
        np.testing.assert_array_equal(t['u'], [[2, 1], [0, 2]])
        rng = np.random.default_rng(8)
        b = rng.standard_normal((7, 7))
        a = b @ b.T + 7 * np.eye(7)
        r, t = self.decomp(a.tolist(), 'cholesky', 'big.safetensors')
        np.testing.assert_allclose(t['l'], sla.cholesky(a, lower=True), rtol=1e-12, atol=1e-13)
        self.assertEqual(np.triu(t['l'], 1).any(), False)
        self.assertTrue(np.all(np.diag(t['l']) > 0))
        self.assertLess(r['reconstruction_error'], 1e-14)

    def test_lu_convention_p_l_u(self):
        r, t = self.decomp([[1, 2], [3, 4]], 'lu')
        self.assertEqual(sorted(t), ['l', 'p', 'u'])
        np.testing.assert_array_equal(t['p'], [[0, 1], [1, 0]])
        np.testing.assert_allclose(t['l'], [[1, 0], [1 / 3, 1]], rtol=1e-15)
        np.testing.assert_allclose(t['u'], [[3, 4], [0, 2 / 3]], rtol=1e-15)
        rng = np.random.default_rng(9)
        for shape in ((5, 5), (6, 4), (4, 6)):
            a = rng.standard_normal(shape)
            r, t = self.decomp(a.tolist(), 'lu', f'{shape}.safetensors')
            p, l, u = sla.lu(a)
            for got, ref in ((t['p'], p), (t['l'], l), (t['u'], u)):
                np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-14)
            np.testing.assert_allclose(t['p'] @ t['l'] @ t['u'], a, atol=1e-13)  # A = P L U
            self.assertTrue(np.all(np.diag(t['l']) == 1) and not np.triu(t['l'], 1).any())
            self.assertEqual(sorted(t['p'].sum(axis=0).tolist() + t['p'].sum(axis=1).tolist()), [1.0] * (2 * shape[0]))
            self.assertLess(r['reconstruction_error'], 1e-14)

    def test_not_positive_definite_and_symmetry_statuses(self):
        out = self.out()
        r = call('la_decomp', {'a': [[1, 2], [2, 1]], 'kind': 'cholesky', 'out': out}, 1)
        self.assertEqual(code(r), 'NOT_POSITIVE_DEFINITE')
        self.assertFalse(pathlib.Path(out).exists())
        r = call('la_decomp', {'a': [[0, 0], [0, 0]], 'kind': 'cholesky', 'out': out}, 1)
        self.assertEqual(code(r), 'NOT_POSITIVE_DEFINITE')
        for kind in ('cholesky', 'eigh'):
            r = call('la_decomp', {'a': [[1, 2], [3, 4]], 'kind': kind, 'out': out}, 2, NO_PYTHON)
            self.assertEqual(code(r), 'NOT_SYMMETRIC')

    def test_validation_before_spawn(self):
        sq = [[2, 1], [1, 2]]
        out = self.out()
        for name, p in {'nan': {'a': [[1, 'NaN'], [1, 2]], 'kind': 'svd'}, 'eig_rect': {'a': [[1, 2, 3], [4, 5, 6]], 'kind': 'eig'}, 'chol_rect': {'a': [[1, 2, 3], [4, 5, 6]], 'kind': 'cholesky'},
                        'kind': {'a': sq, 'kind': 'schur'}, 'vector': {'a': [1, 2], 'kind': 'svd'}, 'pivot_wrong_kind': {'a': sq, 'kind': 'svd', 'pivoting': True},
                        'full_wrong_kind': {'a': sq, 'kind': 'eig', 'full': True}, 'lower_wrong_kind': {'a': sq, 'kind': 'qr', 'lower': False},
                        'ext': {'a': sq, 'kind': 'svd', 'out': str(self.dir / 'x.npy')}, 'outside': {'a': sq, 'kind': 'svd', 'out': '/tmp/pycli-escape.safetensors'}}.items():
            with self.subTest(case=name):
                call('la_decomp', {'out': out, **p}, 2, NO_PYTHON)
        self.assertFalse(pathlib.Path(out).exists())

    def test_policy_conflict_dryrun_readonly(self):
        p = {'a': [[2, 1], [1, 2]], 'kind': 'eigh', 'out': self.out()}
        call('la_decomp', p, 0, dict(os.environ, PYCLI_DRYRUN='1'))
        self.assertFalse(pathlib.Path(p['out']).exists())
        call('la_decomp', p, 6, dict(os.environ, PYCLI_READONLY='1'))
        self.assertFalse(pathlib.Path(p['out']).exists())
        call('la_decomp', p)
        before = hashlib.sha256(pathlib.Path(p['out']).read_bytes()).hexdigest()
        self.assertEqual(call('la_decomp', p, 5)['error']['code'], 'CONFLICT')
        self.assertEqual(hashlib.sha256(pathlib.Path(p['out']).read_bytes()).hexdigest(), before)
        self.assertEqual([x.name for x in self.dir.iterdir()], ['o.safetensors'])

    def test_large_matrix_by_ref_goes_to_file_only(self):
        rng = np.random.default_rng(10)
        a = rng.standard_normal((120, 90))
        np.save(self.dir / 'a.npy', a)
        out = self.out('svd.safetensors')
        proc = subprocess.run([BIN, '--input-json', json.dumps({'op': 'la_decomp', 'params': {'a': str(self.dir / 'a.npy'), 'kind': 'svd', 'out': out}})], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertLess(len(proc.stdout), 20000)  # descriptors + 90 singular values, not the 120x90 factors
        t = read_safetensors(out)
        np.testing.assert_allclose((t['u'] * t['s']) @ t['vt'], a, atol=1e-12)
        r = json.loads(proc.stdout)
        self.assertEqual(r['tensors']['u']['ref'], out + '#u')
        self.assertNotIn('u', r.get('values', {}))
        # 447x447 is the largest square matrix by reference (200000 numbers cap) and must stay inside the wall-clock limit.
        np.save(self.dir / 'sq.npy', rng.standard_normal((447, 447)))
        r = call('la_decomp', {'a': str(self.dir / 'sq.npy'), 'kind': 'eig', 'out': self.out('eig.safetensors')})
        self.assertLess(r['reconstruction_error'], 1e-10)
        np.save(self.dir / 'toolarge.npy', np.ones((448, 448)))
        call('la_decomp', {'a': str(self.dir / 'toolarge.npy'), 'kind': 'svd', 'out': self.out('t.safetensors')}, 2, NO_PYTHON)

    def test_rank_deficient_inputs_still_decompose(self):
        r, t = self.decomp([[1, 2], [2, 4]], 'svd')
        self.assertLess(t['s'][1], 1e-15)
        r, t = self.decomp([[1, 2], [2, 4]], 'qr', 'q.safetensors', pivoting=True)
        self.assertLess(r['reconstruction_error'], 1e-15)
        r, t = self.decomp([[1, 2], [2, 4]], 'lu', 'l.safetensors')
        self.assertLess(r['reconstruction_error'], 1e-15)


class WorkerStatuses(unittest.TestCase):
    """LAPACK non-convergence cannot be provoked on finite data with a small input, so the
    worker's status mapping is exercised with the library call forced to fail."""

    def test_no_convergence_maps_to_status(self):
        import importlib.util
        root = pathlib.Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location('fixed_worker_s1', root / 'py/worker.py')
        w = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(w)
        import numpy as np
        tmp = tempfile.mkdtemp(dir='.')
        try:
            np.save(tmp + '/a.npy', np.array([[2.0, 1.0], [1.0, 2.0]]))
            base = {'a_file': tmp + '/a.npy', 'output_prefix': tmp + '/out_'}
            real_svd, real_eigh, real_eig, real_lstsq = sla.svd, sla.eigh, sla.eig, sla.lstsq

            def boom(*args, **kwargs):
                raise np.linalg.LinAlgError('SVD did not converge')
            sla.svd = sla.eigh = sla.eig = sla.lstsq = boom
            try:
                for kind in ('svd', 'eigh', 'eig'):
                    self.assertEqual(w.decomp({**base, 'kind': kind, 'full': False, 'pivoting': False, 'lower': True})['status'], 'NO_CONVERGENCE', kind)
                self.assertEqual(w.lstsq({'a_file': base['a_file'], 'b_file': base['a_file'], 'driver': 'gelsd', 'rcond': None})['status'], 'NO_CONVERGENCE')
            finally:
                sla.svd, sla.eigh, sla.eig, sla.lstsq = real_svd, real_eigh, real_eig, real_lstsq
        finally:
            import shutil
            shutil.rmtree(tmp)


if __name__ == '__main__':
    unittest.main()
