"""S8b: la_matrix_fn, la_sparse, special_fn (+ _to twins) and opt_root against independent oracles: closed forms (expm of a rotation generator, sqrtm of a
constructed SPD matrix, adjugate inverses), analytic eigenvalues of the tridiagonal Poisson matrix, a hand Thomas solver, exact rational linear systems and
mpmath at 50 digits (a test-image-only package). Every pre-spawn rejection runs with a missing interpreter."""
import json, math, os, pathlib, subprocess, sys, tempfile, unittest
from fractions import Fraction
import numpy as np
BIN = str(pathlib.Path(sys.argv.pop(1)).resolve())
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests/oracle'))
from s1 import read_safetensors
try:
    import mpmath
except ImportError:  # pragma: no cover - the test image carries it (docker/requirements-test.lock)
    mpmath = None

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


def close(test, got, want, rtol=1e-9, atol=0.0, msg=None):
    np.testing.assert_allclose(np.asarray(got, float), np.asarray(want, float), rtol=rtol, atol=atol, err_msg=msg or '')


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir='.')
        self.dir = pathlib.Path(self.tmp.name).resolve()
        self.addCleanup(self.tmp.cleanup)

    def p(self, name):
        return str(self.dir / name)

    def npy(self, array, name=None):
        path = self.dir / (name or 'in%d.npy' % len(list(self.dir.glob('in*.npy'))))
        np.save(path, np.asarray(array))
        return str(path)


def householder(n, seed):
    """A fixed orthogonal matrix from a Householder reflection product (numpy basics only)."""
    rng = np.random.default_rng(seed)
    q = np.eye(n)
    for _ in range(2):
        v = rng.normal(size=n)
        v /= np.linalg.norm(v)
        q = q @ (np.eye(n) - 2 * np.outer(v, v))
    return q


# -------------------------------------------------------------------------------------------------------------- matrices
class MatrixFn(Base):
    def fn(self, name, a, **extra):
        return call('la_matrix_fn', dict(a=a, fn=name, **extra))

    def test_expm_of_a_rotation_generator_and_simple_closed_forms(self):
        for t in (0.0, 0.7, 3.0, -2.2):
            r = self.fn('expm', [[0.0, -t], [t, 0.0]])
            self.assertEqual((r['status'], r['complex']), ('OK', False))
            close(self, r['y'], [[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]], 1e-14, 1e-15, 't=%g' % t)
        close(self, self.fn('expm', [[1.0, 0.0], [0.0, -2.0]])['y'], [[math.e, 0], [0, math.exp(-2)]], 1e-15, 1e-300)
        close(self, self.fn('expm', [[0.0, 1.0], [0.0, 0.0]])['y'], [[1.0, 1.0], [0.0, 1.0]], 1e-15, 1e-300)  # nilpotent: I + N
        close(self, self.fn('expm', [[2.0]])['y'], [[math.exp(2)]], 1e-15)

    def test_logm_is_the_generator_and_goes_complex_for_negative_eigenvalues(self):
        t = 0.7
        rot = [[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]]
        r = self.fn('logm', rot)
        close(self, r['y'], [[0, -t], [t, 0]], 1e-13, 1e-14)
        self.assertLess(r['residual'], 1e-13)
        close(self, self.fn('logm', [[math.e, 0], [0, math.e ** 2]])['y'], [[1, 0], [0, 2]], 1e-14, 1e-15)
        neg = self.fn('logm', [[-1.0, 0.0], [0.0, -1.0]])
        self.assertTrue(neg['complex'])
        close(self, neg['y_re'], np.zeros((2, 2)), atol=1e-14)
        close(self, neg['y_im'], math.pi * np.eye(2), 1e-14)

    def test_sqrtm_of_a_constructed_spd_matrix_and_a_negative_diagonal(self):
        q = householder(3, 5)
        a = q @ np.diag([4.0, 9.0, 16.0]) @ q.T
        a = (a + a.T) / 2
        r = self.fn('sqrtm', a.tolist())
        close(self, r['y'], q @ np.diag([2.0, 3.0, 4.0]) @ q.T, 1e-12, 1e-13)
        self.assertLess(r['residual'], 1e-13)
        c = self.fn('sqrtm', [[-4.0, 0.0], [0.0, 9.0]])
        self.assertTrue(c['complex'])
        close(self, c['y_re'], [[0, 0], [0, 3]], atol=1e-14)
        close(self, c['y_im'], [[2, 0], [0, 0]], 1e-14, 1e-15)

    def test_inverse_by_adjugate_condition_number_and_singularity(self):
        a = [[4.0, 7.0], [2.0, 6.0]]
        det = 4 * 6 - 7 * 2
        r = self.fn('inv', a)
        close(self, r['y'], [[6 / det, -7 / det], [-2 / det, 4 / det]], 1e-14)
        self.assertAlmostEqual(r['condition_number'], np.linalg.cond(np.array(a)), delta=1e-9)
        exact = Fraction
        m = [[2, 1, 0], [1, 3, 1], [0, 1, 4]]  # exact rational inverse by cofactors
        d = 2 * (3 * 4 - 1) - 1 * (1 * 4 - 0)
        cof = [[(3 * 4 - 1), -(1 * 4 - 0), 1], [-(1 * 4 - 0), 8, -2], [1, -2, 5]]
        want = [[float(exact(cof[i][j], d)) for j in range(3)] for i in range(3)]
        close(self, self.fn('inv', [[float(v) for v in row] for row in m])['y'], want, 1e-14)
        s = self.fn('inv', [[1.0, 2.0], [2.0, 4.0]])
        self.assertEqual(s['status'], 'SINGULAR')
        self.assertNotIn('y', s)
        tough = self.fn('inv', [[1.0, 1.0], [1.0, 1.0 + 1e-10]])  # invertible, condition number 4e10: accurate to ~1e-5
        self.assertEqual(tough['status'], 'OK')
        close(self, tough['y'], [[1e10 + 1, -1e10], [-1e10, 1e10]], 1e-5)

    def test_pinv_rank_one_cutoff_and_penrose(self):
        u, v = np.array([1.0, 2.0, 2.0]), np.array([3.0, 4.0])
        a = np.outer(u, v)
        r = self.fn('pinv', a.tolist())
        close(self, r['y'], np.outer(v, u) / ((u @ u) * (v @ v)), 1e-13)
        self.assertEqual((r['rank'], r['shape']), (1, [3, 2]))
        d = [[1.0, 0.0], [0.0, 1e-6]]
        cut = self.fn('pinv', d, rtol=1e-3)
        close(self, cut['y'], [[1, 0], [0, 0]], atol=1e-14)
        self.assertEqual(cut['rank'], 1)
        keep = self.fn('pinv', d, rtol=1e-8)
        close(self, keep['y'], [[1, 0], [0, 1e6]], 1e-12)
        self.assertEqual(keep['rank'], 2)

    def test_failures_are_statuses(self):
        r = self.fn('expm', [[800.0, 0.0], [0.0, 0.0]])
        self.assertEqual(r['status'], 'NUMERICAL_FAILURE')
        self.assertNotIn('y', r)
        s = self.fn('logm', [[0.0, 0.0], [0.0, 1.0]])
        self.assertEqual(s['status'], 'SINGULAR')

    def test_validation_before_spawn_and_inline_budget(self):
        sq = [[1.0, 2.0], [3.0, 4.0]]
        rejected('la_matrix_fn', dict(a=[[1.0, 2.0, 3.0]], fn='inv'), 'BAD_SHAPE')
        rejected('la_matrix_fn', dict(a=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], fn='expm'), 'BAD_SHAPE')
        rejected('la_matrix_fn', dict(a=sq, fn='inv', rtol=0.1), 'BAD_PARAM')
        rejected('la_matrix_fn', dict(a=sq, fn='pinv', rtol=1.0), 'BAD_PARAM')
        rejected('la_matrix_fn', dict(a=[[1.0, "NaN"], [3.0, 4.0]], fn='inv'), 'BAD_DATA')
        rejected('la_matrix_fn', dict(a=[1.0, 2.0], fn='inv'), 'BAD_SHAPE')
        rejected('la_matrix_fn', dict(a=sq, fn='funm'))
        too_large('la_matrix_fn', dict(a=self.npy(np.eye(40)), fn='expm'))  # 1600 values
        too_large('la_matrix_fn', dict(a=self.npy(np.eye(23)), fn='sqrtm'))  # may be complex: 2 * 529 numbers

    def test_write_twin_expm_against_the_eigendecomposition_closed_form(self):
        n = 40
        q = householder(n, 6)
        d = np.linspace(-1.5, 1.0, n)
        a = q @ np.diag(d) @ q.T
        a = (a + a.T) / 2
        out = self.p('e.safetensors')
        r = call('la_matrix_fn_to', dict(a=self.npy(a), fn='expm', out=out))
        self.assertEqual(r['status'], 'OK')
        self.assertEqual(set(r['artifacts']['y']), DESCRIPTOR_KEYS)
        got = read_safetensors(out)['y']
        w, v = np.linalg.eigh(a)  # closed form: V exp(W) V^T
        close(self, got, (v * np.exp(w)) @ v.T, 1e-12, 1e-13)
        c = call('la_matrix_fn_to', dict(a=self.npy(-np.eye(30)), fn='sqrtm', out=self.p('c.safetensors')))
        self.assertTrue(c['complex'])
        self.assertEqual(sorted(c['artifacts']), ['y_im', 'y_re'])
        im = read_safetensors(self.p('c.safetensors'))['y_im']
        close(self, im, np.eye(30), 1e-14)
        call('la_matrix_fn_to', dict(a=self.npy(np.eye(40)), fn='expm', out=out), rc=5)
        singular = call('la_matrix_fn_to', dict(a=self.npy(np.zeros((30, 30))), fn='inv', out=self.p('s.safetensors')), rc=1)
        self.assertEqual(singular['error']['data']['tool_code'], 'SINGULAR')
        self.assertFalse(os.path.exists(self.p('s.safetensors')))

    def test_matrix_functions_import_only_scipy_linalg(self):
        r = self.fn('inv', [[2.0]])
        self.assertEqual((r['meta']['pandas'], r['meta']['sklearn']), ('unloaded', 'unloaded'))
        self.assertNotIn('signal', r['meta']['scipy_subpackages'])


# -------------------------------------------------------------------------------------------------------------- special
@unittest.skipIf(mpmath is None, 'mpmath is not installed')
class Special(Base):
    @classmethod
    def setUpClass(cls):
        mpmath.mp.dps = 50

    def run_fn(self, fn, **args):
        r = call('special_fn', dict(fn=fn, args=args))
        self.assertEqual(r['status'], 'OK', (fn, args, r))
        return r

    def check(self, fn, points, reference, rtol=1e-12, atol=0.0, key='y'):
        names = list(points[0])
        args = {n: [pt[n] for pt in points] for n in names}
        r = self.run_fn(fn, **args)
        want = [float(reference(*[mpmath.mpf(pt[n]) if not isinstance(pt[n], int) else pt[n] for n in names])) for pt in points]
        got = np.array(r[key])
        for pt, g, w in zip(points, got, want):
            err = abs(g - w)
            self.assertLessEqual(err, atol + rtol * abs(w), (fn, pt, g, w, err))

    def test_error_function_family(self):
        xs = [{'x': v} for v in (-3.0, -0.5, 0.0, 0.25, 1.0, 2.5, 6.0)]
        self.check('erf', xs, mpmath.erf)
        self.check('erfc', xs, mpmath.erfc)
        self.check('erfcx', xs, lambda x: mpmath.exp(x * x) * mpmath.erfc(x))
        ys = [{'y': v} for v in (-0.99, -0.3, 0.0, 0.5, 0.9, 0.999)]
        self.check('erfinv', ys, mpmath.erfinv, 1e-11)
        self.check('erfcinv', [{'y': v} for v in (0.01, 0.5, 1.0, 1.5, 1.99)], lambda y: mpmath.erfinv(1 - y), 1e-11)
        self.check('log_ndtr', xs, lambda x: mpmath.log(mpmath.erfc(-x / mpmath.sqrt(2)) / 2))
        self.check('expit', xs, lambda x: 1 / (1 + mpmath.exp(-x)))
        self.check('logit', [{'p': v} for v in (0.01, 0.3, 0.5, 0.9, 0.999)], lambda p: mpmath.log(p / (1 - p)))

    def test_gamma_and_beta_incomplete_functions(self):
        pts = [{'a': a, 'x': x} for a, x in ((0.5, 0.3), (1.0, 2.0), (2.5, 1.0), (5.0, 7.5), (12.0, 10.0), (0.1, 5.0))]
        self.check('gammainc', pts, lambda a, x: mpmath.gammainc(a, 0, x, regularized=True))
        self.check('gammaincc', pts, lambda a, x: mpmath.gammainc(a, x, mpmath.inf, regularized=True))
        bpts = [{'a': a, 'b': b, 'x': x} for a, b, x in ((0.5, 0.5, 0.3), (2.0, 3.0, 0.4), (5.0, 1.5, 0.9), (10.0, 10.0, 0.5), (0.7, 4.0, 0.05))]
        self.check('betainc', bpts, lambda a, b, x: mpmath.betainc(a, b, 0, x, regularized=True))
        # inverses: the returned x must map back to y through the mpmath forward function
        ys = [0.05, 0.3, 0.5, 0.9]
        for a in (0.5, 2.5, 8.0):
            r = self.run_fn('gammaincinv', a=a, y=ys)['y']
            for x, y in zip(r, ys):
                self.assertAlmostEqual(float(mpmath.gammainc(a, 0, x, regularized=True)), y, delta=1e-11)
            r = self.run_fn('gammainccinv', a=a, y=ys)['y']
            for x, y in zip(r, ys):
                self.assertAlmostEqual(float(mpmath.gammainc(a, x, mpmath.inf, regularized=True)), y, delta=1e-11)
        for a, b in ((2.0, 3.0), (0.5, 0.5), (6.0, 1.5)):
            r = self.run_fn('betaincinv', a=a, b=b, y=ys)['y']
            for x, y in zip(r, ys):
                self.assertAlmostEqual(float(mpmath.betainc(a, b, 0, x, regularized=True)), y, delta=1e-11)
        r = self.run_fn('ndtri', p=ys)['y']
        for x, y in zip(r, ys):
            self.assertAlmostEqual(float(mpmath.erfc(-mpmath.mpf(x) / mpmath.sqrt(2)) / 2), y, delta=1e-12)

    def test_zeta_polygamma_lambertw_ellipt(self):
        self.check('zeta', [{'x': 2.0}, {'x': 3.5}, {'x': 10.0}, {'x': 1.1}], lambda x: mpmath.zeta(x))
        self.check('zeta', [{'x': 2.0, 'q': 0.5}, {'x': 3.0, 'q': 2.0}, {'x': 1.5, 'q': 10.0}], lambda x, q: mpmath.zeta(x, q))
        pg = [{'n': n, 'x': x} for n, x in ((0, 0.5), (1, 2.0), (2, 3.5), (3, 0.7), (5, 10.0))]
        self.check('polygamma', pg, lambda n, x: mpmath.polygamma(n, x))
        zs = [{'z': v} for v in (-0.3, 0.0, 0.5, 1.0, 10.0, 1000.0)]
        self.check('lambertw', zs, lambda z: mpmath.lambertw(z).real)
        self.check('lambertw', [{'z': -0.2, 'k': -1}], lambda z, k: mpmath.lambertw(z, -1).real)
        ms = [{'m': v} for v in (-3.0, 0.0, 0.3, 0.9, 0.999)]
        self.check('ellipk', ms, mpmath.ellipk)
        self.check('ellipe', ms, mpmath.ellipe)

    def test_hypergeometric_bessel_airy_struve_owens(self):
        self.check('hyp2f1', [{'a': a, 'b': b, 'c': c, 'z': z} for a, b, c, z in ((1.0, 2.0, 3.0, 0.5), (0.5, 0.5, 1.5, -0.3), (2.5, 1.5, 4.0, 0.9), (1.0, 1.0, 2.0, -5.0))], mpmath.hyp2f1, 1e-11)
        vz = [{'v': v, 'z': z} for v, z in ((0.0, 1.0), (1.0, 2.5), (2.5, 10.0), (0.5, 0.3), (3.0, 7.0))]
        self.check('jv', vz, mpmath.besselj, 1e-11, 1e-14)
        self.check('yv', vz, mpmath.bessely, 1e-11, 1e-14)
        self.check('iv', vz, mpmath.besseli)
        self.check('kv', vz, mpmath.besselk)
        self.check('spherical_jn', [{'n': n, 'z': z} for n, z in ((0, 1.0), (1, 2.5), (3, 5.0), (5, 12.0))], lambda n, z: mpmath.sqrt(mpmath.pi / (2 * z)) * mpmath.besselj(n + mpmath.mpf(1) / 2, z), 1e-11, 1e-14)
        xs = [{'x': v} for v in (-5.0, -1.0, 0.0, 0.7, 3.0)]
        for key, ref in (('ai', mpmath.airyai), ('aip', lambda x: mpmath.airyai(x, derivative=1)), ('bi', mpmath.airybi), ('bip', lambda x: mpmath.airybi(x, derivative=1))):
            self.check('airy', xs, ref, 1e-11, 1e-14, key=key)
        self.check('struve', [{'v': v, 'x': x} for v, x in ((0.0, 1.0), (1.0, 3.0), (2.5, 6.0), (0.5, 0.2))], mpmath.struveh, 1e-10, 1e-14)

        def owens(h, a):
            return mpmath.quad(lambda x: mpmath.exp(-h * h * (1 + x * x) / 2) / (1 + x * x), [0, a]) / (2 * mpmath.pi)
        self.check('owens_t', [{'h': h, 'a': a} for h, a in ((0.5, 1.0), (1.5, 0.3), (0.0, 2.0), (2.0, 5.0))], owens, 1e-10, 1e-15)

    def test_boxcox_xlogy_and_logsumexp(self):
        self.check('boxcox', [{'x': x, 'lmbda': l} for x, l in ((2.0, 0.5), (3.0, -1.0), (7.0, 0.0), (0.4, 2.0))], lambda x, l: mpmath.log(x) if l == 0 else (x ** l - 1) / l)
        self.check('inv_boxcox', [{'y': y, 'lmbda': l} for y, l in ((0.3, 0.5), (1.2, -0.5), (2.0, 0.0), (-0.2, 2.0))], lambda y, l: mpmath.exp(y) if l == 0 else (1 + l * y) ** (1 / l))
        self.check('xlogy', [{'x': x, 'y': y} for x, y in ((2.0, 3.0), (0.0, 5.0), (-1.5, 0.25))], lambda x, y: x * mpmath.log(y) if x != 0 else mpmath.mpf(0))
        self.check('xlog1py', [{'x': x, 'y': y} for x, y in ((2.0, 3.0), (0.0, 5.0), (-1.5, -0.25), (3.0, 1e-10))], lambda x, y: x * mpmath.log1p(y) if x != 0 else mpmath.mpf(0))
        a = [1000.0, 1001.0, 999.5, -3.0]
        r = self.run_fn('logsumexp', a=a)
        want = float(mpmath.log(sum(mpmath.exp(mpmath.mpf(v)) for v in a)))
        self.assertAlmostEqual(r['y'][0], want, delta=1e-12 * abs(want))

    def test_broadcasting_scalars_vectors_and_errors(self):
        r = self.run_fn('gammainc', a=2.0, x=[0.5, 1.0, 3.0])
        self.assertEqual(r['n'], 3)
        close(self, r['y'], [1 - math.exp(-x) * (1 + x) for x in (0.5, 1.0, 3.0)], 1e-14)  # P(2, x) in closed form
        pair = self.run_fn('betainc', a=[1.0, 2.0], b=[1.0, 1.0], x=0.5)
        close(self, pair['y'], [0.5, 0.25], 1e-15)
        scalar = self.run_fn('erf', x=0.3)
        self.assertEqual((scalar['n'], len(scalar['y'])), (1, 1))
        rejected('special_fn', dict(fn='gammainc', args={'a': [1.0, 2.0], 'x': [1.0, 2.0, 3.0]}), 'BAD_DATA')
        rejected('special_fn', dict(fn='gammainc', args={'a': 1.0}), 'BAD_PARAM')
        rejected('special_fn', dict(fn='erf', args={'x': 1.0, 'y': 2.0}), 'BAD_PARAM')
        rejected('special_fn', dict(fn='erf', args={'x': [1.0, "NaN"]}), 'BAD_DATA')
        rejected('special_fn', dict(fn='polygamma', args={'n': 1.5, 'x': 1.0}), 'BAD_DATA')
        rejected('special_fn', dict(fn='polygamma', args={'n': -1, 'x': 1.0}), 'BAD_DATA')
        rejected('special_fn', dict(fn='logsumexp', args={'a': 1.0}), 'BAD_PARAM')
        rejected('special_fn', dict(fn='bogus', args={'x': 1.0}))
        rejected('special_fn', dict(fn='erf', args=[1.0]))
        too_large('special_fn', dict(fn='erf', args={'x': self.npy(np.linspace(0, 1, 901))}))
        too_large('special_fn', dict(fn='airy', args={'x': self.npy(np.linspace(0, 1, 226))}))

    def test_domain_errors_and_poles_are_statuses_not_values(self):
        for fn, args, status, index in (('erfinv', {'y': [0.2, 2.0]}, 'DOMAIN_ERROR', 1), ('ellipk', {'m': [0.5, 2.0]}, 'DOMAIN_ERROR', 1), ('lambertw', {'z': [1.0, -1.0]}, 'DOMAIN_ERROR', 1),
                                        ('logit', {'p': [0.5, 0.0]}, 'NON_FINITE', 1), ('zeta', {'x': [2.0, 1.0]}, 'NON_FINITE', 1), ('erfinv', {'y': [1.0]}, 'NON_FINITE', 0), ('boxcox', {'x': [-1.0], 'lmbda': 0.5}, 'DOMAIN_ERROR', 0)):
            r = call('special_fn', dict(fn=fn, args=args))
            self.assertEqual(r['status'], status, (fn, r))
            self.assertNotIn('y', r)
            self.assertEqual(r['n_bad'], 1)
            self.assertEqual(r['first_index'], index, (fn, r))
        r = call('special_fn_to', dict(fn='logit', args={'p': [0.5, 0.0]}, out=self.p('bad.safetensors')), rc=1)
        self.assertEqual(r['error']['data']['tool_code'], 'NON_FINITE')
        self.assertFalse(os.path.exists(self.p('bad.safetensors')))

    def test_write_twin_and_airy_artifacts(self):
        xs = np.linspace(0.1, 8.0, 3000)
        out = self.p('j.safetensors')
        r = call('special_fn_to', dict(fn='jv', args={'v': 1.5, 'z': self.npy(xs)}, out=out))
        self.assertEqual((r['status'], r['n']), ('OK', 3000))
        self.assertEqual(set(r['artifacts']['y']), DESCRIPTOR_KEYS)
        y = read_safetensors(out)['y']
        for k in (0, 1500, 2999):
            self.assertAlmostEqual(y[k], float(mpmath.besselj(1.5, mpmath.mpf(float(xs[k])))), delta=1e-11)
        a = call('special_fn_to', dict(fn='airy', args={'x': [0.0, 1.0]}, out=self.p('a.safetensors')))
        self.assertEqual(sorted(a['artifacts']), ['ai', 'aip', 'bi', 'bip'])
        self.assertAlmostEqual(read_safetensors(self.p('a.safetensors'))['ai'][0], float(mpmath.airyai(0)), delta=1e-15)
        call('special_fn_to', dict(fn='jv', args={'v': 1.5, 'z': [1.0]}, out=out), rc=5)

    def test_only_scipy_special_is_imported(self):
        r = self.run_fn('erf', x=1.0)
        self.assertEqual((r['meta']['pandas'], r['meta']['sklearn']), ('unloaded', 'unloaded'))


# --------------------------------------------------------------------------------------------------------------- sparse
def poisson_coo(n):
    rows, cols, vals = [], [], []
    for i in range(n):
        rows.append(i); cols.append(i); vals.append(2.0)
        if i + 1 < n:
            rows += [i, i + 1]; cols += [i + 1, i]; vals += [-1.0, -1.0]
    return rows, cols, vals


def thomas(n, d):
    """Hand tridiagonal solve of (2, -1) x = d."""
    c, x = [0.0] * n, [0.0] * n
    c[0] = -1.0 / 2.0
    d = list(d)
    d[0] = d[0] / 2.0
    for i in range(1, n):
        m = 2.0 + c[i - 1]
        c[i] = -1.0 / m
        d[i] = (d[i] + d[i - 1]) / m
    x[-1] = d[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d[i] - c[i] * x[i + 1]
    return x


def poisson_eigenvalues(n):
    return [2.0 - 2.0 * math.cos(j * math.pi / (n + 1)) for j in range(1, n + 1)]


class Sparse(Base):
    def sparse(self, task, n=200, **extra):
        r, c, v = poisson_coo(n)
        if len(v) > 1000:  # inline arrays hold at most 1000 numbers
            r, c, v = self.npy(r), self.npy(c), self.npy(v)
            if 'b' in extra:
                extra['b'] = self.npy(extra['b'])
        params = dict(task=task, shape=[n, n], row=r, col=c, val=v)
        params.update(extra)
        return call('la_sparse', params)

    def test_solvers_match_the_hand_thomas_solution(self):
        n = 200
        b = [math.sin(0.1 * i) + 0.3 for i in range(n)]
        want = thomas(n, b)
        for task, extra in (('spsolve', {}), ('cg', {'rtol': 1e-12}), ('gmres', {'rtol': 1e-12, 'restart': 200, 'maxiter': 20})):
            r = self.sparse(task, n, b=b, **extra)
            self.assertEqual(r['status'], 'OK', (task, r))
            close(self, r['x'], want, 1e-8, 1e-10, task)
            self.assertLess(r['relative_residual'], 1e-9 if task == 'spsolve' else 1e-11)
        # known solution: b = A x0 with x0 = sin(i pi / (n+1))
        x0 = [math.sin((i + 1) * math.pi / (n + 1)) for i in range(n)]
        rhs = [2 * x0[i] - (x0[i - 1] if i else 0) - (x0[i + 1] if i + 1 < n else 0) for i in range(n)]
        r = self.sparse('cg', n, b=rhs, rtol=1e-12)
        close(self, r['x'], x0, 1e-8, 1e-10)
        self.assertGreater(r['iterations'], 10)

    def test_duplicate_entries_are_summed_and_frames_work(self):
        r = call('la_sparse', dict(task='spsolve', shape=[2, 2], row=[0, 0, 1, 1], col=[0, 0, 1, 1], val=[1.0, 1.0, 3.0, 1.0], b=[4.0, 8.0]))
        close(self, r['x'], [2.0, 2.0], 1e-15)  # diag(1+1, 3+1)
        rr, cc, vv = poisson_coo(30)
        (self.dir / 'coo.csv').write_text('i,j,v\n' + ''.join('%d,%d,%r\n' % t for t in zip(rr, cc, vv)))
        call('frame_import', dict(csv=self.p('coo.csv'), out=self.p('coo.npz')))
        b = [1.0] * 30
        framed = call('la_sparse', dict(task='spsolve', shape=[30, 30], frame=self.p('coo.npz'), row='i', col='j', val='v', b=b))
        close(self, framed['x'], thomas(30, b), 1e-10)
        rejected('la_sparse', dict(task='spsolve', shape=[30, 30], frame=self.p('coo.npz'), row='nope', col='j', val='v', b=b), 'UNKNOWN_COLUMN')
        rejected('la_sparse', dict(task='spsolve', shape=[30, 30], frame=self.p('coo.npz'), row='v', col='j', val='v', b=b), 'BAD_DTYPE')

    def test_eigsh_matches_the_analytic_poisson_spectrum(self):
        n = 200
        exact = poisson_eigenvalues(n)
        for which, want in (('SA', exact[:4]), ('LA', exact[-4:]), ('LM', exact[-4:])):
            r = self.sparse('eigsh', n, k=4, which=which, seed=1)
            self.assertEqual(r['status'], 'OK', r)
            close(self, r['w'], sorted(want), 1e-9, 1e-12, which)
            self.assertLess(r['max_residual'], 1e-8)
        near = self.sparse('eigsh', n, k=3, which='LM', sigma=0.5, seed=2)
        want = sorted(sorted(exact, key=lambda v: abs(v - 0.5))[:3])
        close(self, near['w'], want, 1e-9, 1e-12)
        vec = self.sparse('eigsh', n, k=2, which='SA', seed=3, vectors=True)
        for j in range(2):
            u = np.array([math.sin((i + 1) * (j + 1) * math.pi / (n + 1)) for i in range(n)])
            u /= np.linalg.norm(u)
            self.assertAlmostEqual(abs(float(np.dot(np.array(vec['v'])[:, j], u))), 1.0, places=8)

    def test_seed_makes_arpack_reproducible_and_is_required(self):
        a = self.sparse('eigsh', 120, k=3, which='SA', seed=7, vectors=True)
        b = self.sparse('eigsh', 120, k=3, which='SA', seed=7, vectors=True)
        self.assertEqual((a['w'], a['v']), (b['w'], b['v']))
        c = self.sparse('eigsh', 120, k=3, which='SA', seed=8)
        close(self, c['w'], a['w'], 1e-9)
        rejected('la_sparse', dict(task='eigsh', shape=[3, 3], row=[0, 1, 2], col=[0, 1, 2], val=[1.0, 2.0, 3.0], k=1), 'BAD_PARAM')
        rejected('la_sparse', dict(task='eigsh', shape=[3, 3], row=[0, 1, 2], col=[0, 1, 2], val=[1.0, 2.0, 3.0], k=1, seed=-1), 'BAD_PARAM')

    def test_eigs_triangular_and_complex_pairs_and_svds(self):
        n = 60
        rows = list(range(n)) + list(range(n - 1))
        cols = list(range(n)) + list(range(1, n))
        vals = [float(i + 1) for i in range(n)] + [1.0] * (n - 1)
        r = call('la_sparse', dict(task='eigs', shape=[n, n], row=rows, col=cols, val=vals, k=3, which='LR', seed=1))
        self.assertEqual(r['status'], 'OK', r)
        close(self, r['w'], [58.0, 59.0, 60.0], 1e-8)  # triangular matrix: the eigenvalues are the diagonal
        self.assertFalse(r['complex'])
        # blocks [[0, -r], [r, 0]] have eigenvalues +-i r
        radii = [1.0, 2.0, 3.0, 4.0, 5.0]
        row, col, val = [], [], []
        for b, rad in enumerate(radii):
            row += [2 * b, 2 * b + 1]; col += [2 * b + 1, 2 * b]; val += [-rad, rad]
        z = call('la_sparse', dict(task='eigs', shape=[10, 10], row=row, col=col, val=val, k=2, which='LI', seed=4))
        self.assertTrue(z['complex'])
        close(self, np.abs(z['w_im']), [5.0, 5.0], 1e-9)
        close(self, z['w_re'], [0.0, 0.0], atol=1e-9)
        d = [10.0, 7.0, 5.0, 3.0, 2.0, 1.0]
        sv = call('la_sparse', dict(task='svds', shape=[8, 6], row=list(range(6)), col=list(range(6)), val=d, k=3, which='LM', seed=5, vectors=True))
        close(self, sv['s'], [10.0, 7.0, 5.0], 1e-9)
        self.assertEqual((len(sv['u']), len(sv['u'][0]), len(sv['vt']), len(sv['vt'][0])), (8, 3, 3, 6))

    def test_non_convergence_and_singularity_are_statuses(self):
        n = 500
        r = self.sparse('eigsh', n, k=5, which='SA', seed=1, maxiter=2)
        self.assertEqual(r['status'], 'NO_CONVERGENCE', r)
        self.assertNotIn('w', r)
        self.assertIn('n_converged', r)
        c = self.sparse('cg', n, b=[1.0] * n, rtol=1e-14, maxiter=3)
        self.assertEqual(c['status'], 'NO_CONVERGENCE')
        self.assertNotIn('x', c)
        self.assertGreater(c['relative_residual'], 1e-3)
        g = self.sparse('gmres', n, b=[1.0] * n, rtol=1e-14, restart=3, maxiter=2)
        self.assertEqual(g['status'], 'NO_CONVERGENCE')
        s = call('la_sparse', dict(task='spsolve', shape=[3, 3], row=[0, 1], col=[0, 1], val=[1.0, 1.0], b=[1.0, 1.0, 1.0]))
        self.assertEqual(s['status'], 'SINGULAR')

    def test_validation_and_caps_before_spawn(self):
        base = dict(task='spsolve', shape=[3, 3], row=[0, 1, 2], col=[0, 1, 2], val=[1.0, 2.0, 3.0], b=[1.0, 1.0, 1.0])
        rejected('la_sparse', dict(base, shape=[3, 4]), 'BAD_SHAPE')
        rejected('la_sparse', dict(base, shape=[0, 0]), 'INPUT_TOO_LARGE')
        rejected('la_sparse', dict(base, shape=[100001, 100001]), 'INPUT_TOO_LARGE')
        rejected('la_sparse', dict(base, row=[0, 1, 3]), 'BAD_INDEX')
        rejected('la_sparse', dict(base, row=[0, -1, 2]), 'BAD_INDEX')
        rejected('la_sparse', dict(base, row=[0, 0.5, 2]), 'BAD_DATA')
        rejected('la_sparse', dict(base, val=[1.0, "NaN", 3.0]), 'BAD_DATA')
        rejected('la_sparse', dict(base, col=[0, 1]), 'BAD_DATA')
        rejected('la_sparse', {k: v for k, v in base.items() if k != 'b'}, 'BAD_PARAM')
        rejected('la_sparse', dict(base, b=[1.0, 1.0]), 'BAD_DATA')
        rejected('la_sparse', dict(base, k=2), 'BAD_PARAM')
        rejected('la_sparse', dict(base, seed=1), 'BAD_PARAM')
        rejected('la_sparse', dict(base, vectors=True), 'BAD_PARAM')
        rejected('la_sparse', dict(base, restart=5), 'BAD_PARAM')
        rejected('la_sparse', dict(task='cg', shape=[2, 2], row=[0, 0, 1], col=[0, 1, 1], val=[2.0, 1.0, 2.0], b=[1.0, 1.0]), 'NOT_SYMMETRIC')
        rejected('la_sparse', dict(task='eigsh', shape=[2, 2], row=[0, 0, 1], col=[0, 1, 1], val=[2.0, 1.0, 2.0], k=1, seed=1), 'NOT_SYMMETRIC')
        eig = dict(task='eigsh', shape=[4, 4], row=[0, 1, 2, 3], col=[0, 1, 2, 3], val=[1.0, 2.0, 3.0, 4.0], seed=1)
        rejected('la_sparse', dict(eig, k=4), 'BAD_PARAM')
        rejected('la_sparse', dict(eig, k=0), 'BAD_PARAM')
        rejected('la_sparse', dict(eig, k=2, which='LI'), 'BAD_PARAM')
        rejected('la_sparse', dict(eig, k=2, which='SA', sigma=0.5), 'BAD_PARAM')
        rejected('la_sparse', dict(eig, k=2, ncv=3), 'BAD_PARAM')
        rejected('la_sparse', dict(eig, task='eigs', k=3), 'BAD_PARAM')  # eigs needs k < n - 1
        rejected('la_sparse', dict(eig, k=2, rtol=1.5), 'BAD_PARAM')
        rejected('la_sparse', dict(base, rtol=0.0, task='cg'), 'BAD_PARAM')
        big = 20001
        rejected('la_sparse', dict(task='spsolve', shape=[big, big], row=self.npy(np.arange(big)), col=self.npy(np.arange(big)), val=self.npy(np.ones(big)), b=self.npy(np.ones(big))), 'INPUT_TOO_LARGE')
        too_large('la_sparse', dict(eig, k=3, vectors=True, shape=[2000, 2000], row=self.npy(np.arange(2000)), col=self.npy(np.arange(2000)), val=self.npy(np.arange(1, 2001.0))))

    def test_write_twin_solution_and_vectors(self):
        n = 2000
        r, c, v = poisson_coo(n)
        b = [math.cos(0.01 * i) for i in range(n)]
        out = self.p('x.safetensors')
        strict = call('la_sparse_to', dict(task='cg', shape=[n, n], row=self.npy(r), col=self.npy(c), val=self.npy(v), b=self.npy(b), rtol=1e-12, maxiter=20000, out=out), rc=1)
        self.assertEqual(strict['error']['data']['tool_code'], 'TOLERANCE_UNATTAINABLE')
        self.assertFalse(os.path.exists(out))
        # Actual binary64 residual is ~6e-11; request an attainable tolerance for the artifact oracle.
        res = call('la_sparse_to', dict(task='cg', shape=[n, n], row=self.npy(r), col=self.npy(c), val=self.npy(v), b=self.npy(b), rtol=1e-10, maxiter=20000, out=out))
        self.assertEqual(res['status'], 'OK', res)
        self.assertEqual(set(res['artifacts']['x']), DESCRIPTOR_KEYS)
        close(self, read_safetensors(out)['x'], thomas(n, b), 1e-7, 1e-9)
        e = call('la_sparse_to', dict(task='eigsh', shape=[n, n], row=self.npy(r), col=self.npy(c), val=self.npy(v), k=3, which='LM', sigma=0.0, seed=1, vectors=True, out=self.p('e.safetensors')))
        got = read_safetensors(self.p('e.safetensors'))
        close(self, got['w'], poisson_eigenvalues(n)[:3], 1e-8, 1e-12)
        self.assertEqual(got['v'].shape, (n, 3))
        no = call('la_sparse_to', dict(task='cg', shape=[n, n], row=self.npy(r), col=self.npy(c), val=self.npy(v), b=self.npy(b), rtol=1e-14, maxiter=2, out=self.p('no.safetensors')), rc=1)
        self.assertEqual(no['error']['data']['tool_code'], 'NO_CONVERGENCE')
        self.assertFalse(os.path.exists(self.p('no.safetensors')))

    def test_sparse_imports_only_scipy_sparse(self):
        r = self.sparse('spsolve', 20, b=[1.0] * 20)
        self.assertEqual((r['meta']['pandas'], r['meta']['sklearn']), ('unloaded', 'unloaded'))
        self.assertNotIn('signal', r['meta']['scipy_subpackages'])


# ------------------------------------------------------------------------------------------------------------- opt_root
class Root(Base):
    def root(self, equations, start, rc=0, **extra):
        return call('opt_root', dict(equations=equations, start=start, **extra), rc)

    def test_circle_line_and_rosenbrock_systems(self):
        for method in ('hybr', 'lm'):
            r = self.root(['x^2 + y^2 - 4', 'x - y'], {'x': 1.0, 'y': 1.0}, method=method)
            self.assertEqual(r['status'], 'OK', r)
            close(self, [r['x']['x'], r['x']['y']], [math.sqrt(2), math.sqrt(2)], 1e-8)
            self.assertLessEqual(r['residual_max'], 1e-8)
            r = self.root(['10 * (y - x^2)', '1 - x'], {'x': -1.2, 'y': 1.0}, method=method)
            self.assertEqual(r['status'], 'OK', (method, r))
            close(self, [r['x']['x'], r['x']['y']], [1.0, 1.0], 1e-7)
        t = self.root(['x^2 + y^2 - 4', 'x - y'], {'x': -1.0, 'y': -1.0})
        close(self, [t['x']['x'], t['x']['y']], [-math.sqrt(2), -math.sqrt(2)], 1e-8)  # the other root, chosen by the start

    def test_linear_system_against_exact_rational_arithmetic(self):
        a = [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]]
        b = [8, -11, -3]  # solution (2, 3, -1)
        eqs = ['%d*x + %d*y + %d*z - %d' % (a[i][0], a[i][1], a[i][2], b[i]) for i in range(3)]
        r = self.root(eqs, {'x': 0.0, 'y': 0.0, 'z': 0.0})
        self.assertEqual(r['status'], 'OK')
        close(self, [r['x']['x'], r['x']['y'], r['x']['z']], [2, 3, -1], 1e-9)
        m = [[Fraction(v) for v in row] + [Fraction(rhs)] for row, rhs in zip(a, b)]  # Gauss-Jordan over Q
        for c in range(3):
            piv = next(i for i in range(c, 3) if m[i][c] != 0)
            m[c], m[piv] = m[piv], m[c]
            m[c] = [v / m[c][c] for v in m[c]]
            for i in range(3):
                if i != c:
                    m[i] = [vi - m[i][c] * vc for vi, vc in zip(m[i], m[c])]
        self.assertEqual([m[i][3] for i in range(3)], [2, 3, -1])

    def test_bracketed_scalar_root_matches_the_known_value_and_num_root(self):
        known = 2.0945514815423265  # x^3 - 2x - 5 = 0
        r = self.root(['x^3 - 2*x - 5'], {'x': 2.0}, bracket=[2.0, 3.0], xtol=1e-14)
        self.assertEqual((r['status'], r['method']), ('OK', 'brentq'))
        self.assertAlmostEqual(r['x']['x'], known, delta=1e-12)
        n = call('num_root', dict(f='x^3 - 2*x - 5', lower=2.0, upper=3.0, xtol=1e-14))
        self.assertAlmostEqual(r['x']['x'], n['root'], delta=1e-12)
        end = self.root(['x - 3'], {'x': 0.0}, bracket=[0.0, 3.0])
        self.assertEqual((end['status'], end['x']['x']), ('OK', 3.0))
        bad = self.root(['x^2 + 1'], {'x': 0.0}, bracket=[-1.0, 1.0], rc=2)
        self.assertEqual(bad['error']['data']['tool_code'], 'BAD_INTERVAL')

    def test_success_is_verified_never_trusted(self):
        for method in ('hybr', 'lm'):
            r = self.root(['x^2 + 1'], {'x': 1.0}, method=method)
            self.assertEqual(r['status'], 'NO_CONVERGENCE', (method, r))
            self.assertGreaterEqual(r['residual_max'], 1.0 - 1e-9)  # the best point is reported, honestly not a root
        r = self.root(['x^2 + y^2 + 1', 'x - y'], {'x': 1.0, 'y': 1.0})
        self.assertEqual(r['status'], 'NO_CONVERGENCE')
        loose = self.root(['x^2 - 2'], {'x': 1.0}, residual_tol=1e-30)
        self.assertEqual(loose['status'], 'NO_CONVERGENCE')  # a tolerance tighter than rounding can reach is reported, not hidden
        limited = self.root(['x^2 + y^2 - 4', 'x - y'], {'x': 1.0, 'y': 1.0}, max_evals=2)
        self.assertEqual(limited['status'], 'EVAL_LIMIT')
        self.assertIn('x', limited)

    def test_validation_and_budgets_before_spawn(self):
        ok = dict(equations=['x^2 - 4'], start={'x': 1.0})
        rejected('opt_root', dict(ok, equations=['x - 1', 'y']), 'UNDER_DETERMINED')
        rejected('opt_root', dict(ok, equations=[]), 'BAD_PARAM')
        rejected('opt_root', dict(ok, equations=['x +']))
        rejected('opt_root', dict(ok, equations=['foo(x)']))
        rejected('opt_root', dict(ok, equations=['z - 1']))
        rejected('opt_root', dict(ok, equations=['sum(x)']))
        rejected('opt_root', dict(ok, start={'x': [1.0, 2.0]}), 'BAD_PARAM')
        rejected('opt_root', dict(ok, start={'X': 1.0}))
        rejected('opt_root', dict(ok, start={'sin': 1.0}))
        rejected('opt_root', dict(ok, method='brentq'), 'BAD_PARAM')
        rejected('opt_root', dict(equations=['x - y', 'x + y'], start={'x': 0.0, 'y': 0.0}, bracket=[0.0, 1.0]), 'BAD_PARAM')
        rejected('opt_root', dict(ok, bracket=[1.0, 0.0]), 'BAD_PARAM')
        rejected('opt_root', dict(ok, xtol=0.0), 'BAD_PARAM')
        rejected('opt_root', dict(ok, residual_tol=-1.0), 'BAD_PARAM')
        rejected('opt_root', dict(equations=['x%d' % i for i in range(17)], start={'x%d' % i: 0.0 for i in range(17)}), 'BAD_PARAM')
        nf = self.root(['log(x)'], {'x': -1.0}, rc=2)
        self.assertEqual(nf['error']['data']['tool_code'], 'NONFINITE_RESIDUAL')
        payload = "__import__('os').system('touch OWNED')"
        rejected('opt_root', dict(ok, equations=[payload]))
        self.assertFalse((self.dir / 'OWNED').exists())

    def test_schema_documents_opt_root(self):
        schema = json.loads(subprocess.run([BIN, '--schema'], capture_output=True, text=True).stdout)
        ops = {o['name']: o for o in schema['operations']}
        self.assertEqual(ops['opt_root']['access'], 'read')
        for param in ops['opt_root']['params']:
            self.assertTrue(param.get('description'), param['name'])


# ---------------------------------------------------------------------------------------------------- sm_tsa_tools
def ar1(n, phi, seed, scale=1.0):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    e = rng.normal(size=n) * scale
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


def hand_acf(x, nlags):
    v = x - x.mean()
    return [float(v[:len(v) - k] @ v[k:] / (v @ v)) for k in range(nlags + 1)]


def hand_pacf(acf, nlags):
    """Durbin-Levinson on the autocorrelations (Yule-Walker, n denominator)."""
    out, phi = [1.0], np.zeros(nlags + 1)
    prev = np.zeros(nlags + 1)
    for k in range(1, nlags + 1):
        num = acf[k] - sum(prev[j] * acf[k - j] for j in range(1, k))
        den = 1.0 - sum(prev[j] * acf[j] for j in range(1, k))
        phi[k] = num / den
        for j in range(1, k):
            phi[j] = prev[j] - phi[k] * prev[k - j]
        out.append(float(phi[k]))
        prev = phi.copy()
    return out


MACKINNON = {  # published tau-statistic response-surface coefficients (MacKinnon 2010, table for one variable): cv = b0 + b1/n + b2/n^2 + b3/n^3
    'c': {'1%': (-3.43035, -6.5393, -16.786, -79.433), '5%': (-2.86154, -2.8903, -4.234, -40.040), '10%': (-2.56677, -1.5384, -2.809, 0.0)},
    'ct': {'1%': (-3.95877, -9.0531, -28.428, -134.155), '5%': (-3.41049, -4.3904, -9.036, -45.374), '10%': (-3.12705, -2.5856, -3.925, -22.380)}}


def hand_adf(y, regression):
    """Dickey-Fuller regression with no augmentation lags: dy_t on [1, (t,) y_{t-1}]; the statistic is the t ratio of y_{t-1}."""
    dy, lag = np.diff(y), y[:-1]
    cols = [np.ones(len(dy))] + ([np.arange(1, len(dy) + 1.0)] if regression == 'ct' else []) + [lag]
    X = np.column_stack(cols)
    beta = np.linalg.solve(X.T @ X, X.T @ dy)
    resid = dy - X @ beta
    sigma2 = resid @ resid / (len(dy) - X.shape[1])
    se = math.sqrt(sigma2 * np.linalg.inv(X.T @ X)[-1, -1])
    return float(beta[-1] / se), len(dy)


def hand_kpss(y, regression, lags):
    n = len(y)
    if regression == 'c':
        e = y - y.mean()
    else:
        X = np.column_stack([np.ones(n), np.arange(1.0, n + 1)])
        e = y - X @ np.linalg.solve(X.T @ X, X.T @ y)
    s = np.cumsum(e)
    var = e @ e / n
    for k in range(1, lags + 1):
        var += 2.0 / n * (1.0 - k / (lags + 1.0)) * float(e[k:] @ e[:-k])
    return float(s @ s / (n * n * var))


def chi2_sf_even(x, dof):
    """Survival function of chi-square with an even number of degrees of freedom (closed form)."""
    half, term, total = x / 2.0, 1.0, 1.0
    for j in range(1, dof // 2):
        term *= half / j
        total += term
    return math.exp(-half) * total


class Tsa(Base):
    def tools(self, test, series, **extra):
        return call('sm_tsa_tools', dict(columns={'y': [float(v) for v in series]}, series='y', test=test, **extra))

    def test_acf_and_pacf_reuse_the_existing_estimators(self):
        x = ar1(300, 0.6, 3)
        a = self.tools('acf', x, nlags=12)
        self.assertEqual((a['status'], a['test']), ('OK', 'acf'))
        close(self, a['values'], hand_acf(x, 12), 1e-12, 1e-14)
        p = self.tools('pacf', x, nlags=8)
        close(self, p['values'], hand_pacf(hand_acf(x, 8), 8), 1e-10, 1e-12)
        old = call('sm_tsa_acf', dict(columns={'y': list(map(float, x))}, series='y', kind='acf', nlags=12))
        self.assertEqual(a['values'], old['values'])  # the same code path, not a second implementation
        self.assertEqual(a['ci_lower'], old['ci_lower'])

    def test_adfuller_statistic_critical_values_and_decisions(self):
        for regression in ('c', 'ct'):
            y = np.cumsum(ar1(120, 0.0, 5)) * 0.5 + ar1(120, 0.5, 6)
            r = self.tools('adfuller', y, regression=regression, autolag='none', maxlag=0)
            self.assertEqual((r['status'], r['used_lag'], r['nobs'], r['regression']), ('OK', 0, 119, regression))
            stat, nobs = hand_adf(y, regression)
            self.assertAlmostEqual(r['statistic'], stat, delta=1e-9)
            for level, coefficients in MACKINNON[regression].items():
                cv = sum(c / nobs ** i for i, c in enumerate(coefficients))
                self.assertAlmostEqual(r['critical_values'][level], cv, delta=1e-9, msg=(regression, level))
        stationary = self.tools('adfuller', ar1(400, 0.3, 7), autolag='AIC')
        self.assertLess(stationary['p_value'], 0.01)
        self.assertLessEqual(stationary['used_lag'], stationary['maxlag'])
        self.assertIn('icbest', stationary)
        walk = self.tools('adfuller', np.cumsum(ar1(400, 0.0, 8)), autolag='AIC')
        self.assertGreater(walk['p_value'], 0.05)
        self.assertGreater(walk['statistic'], walk['critical_values']['5%'])

    def test_kpss_statistic_and_interpolated_p_value(self):
        table = {'c': ([0.347, 0.463, 0.574, 0.739], [0.10, 0.05, 0.025, 0.01]), 'ct': ([0.119, 0.146, 0.176, 0.216], [0.10, 0.05, 0.025, 0.01])}
        found_interior = False
        for regression in ('c', 'ct'):
            for seed in range(20, 60):
                y = np.cumsum(ar1(150, 0.0, seed, 0.12)) + ar1(150, 0.2, seed + 100)
                stat = hand_kpss(y, regression, 4)
                if table[regression][0][0] < stat < table[regression][0][-1]:
                    break
            r = self.tools('kpss', y, regression=regression, kpss_lags=4)
            self.assertEqual((r['status'], r['lags']), ('OK', 4))
            self.assertAlmostEqual(r['statistic'], stat, delta=1e-9)
            self.assertAlmostEqual(r['p_value'], float(np.interp(stat, *table[regression])), delta=1e-9)
            self.assertEqual(r['p_value_bounded'], 'none')
            found_interior = True
            for level, crit in zip(('10%', '5%', '2.5%', '1%'), table[regression][0]):
                self.assertAlmostEqual(r['critical_values'][level], crit, delta=1e-12)
        self.assertTrue(found_interior)
        stationary = self.tools('kpss', ar1(300, 0.2, 9), kpss_lags=4)
        self.assertEqual((stationary['p_value'], stationary['p_value_bounded']), (0.1, 'above'))
        walk = self.tools('kpss', np.cumsum(ar1(300, 0.0, 10)), kpss_lags=4)
        self.assertEqual((walk['p_value'], walk['p_value_bounded']), (0.01, 'below'))
        legacy = self.tools('kpss', ar1(300, 0.2, 9), kpss_lags='legacy')
        self.assertEqual(legacy['lags'], math.ceil(12 * (300 / 100) ** 0.25))

    def test_ljung_box_statistic_and_closed_form_p_values(self):
        x = ar1(200, 0.4, 12)
        acf = hand_acf(x, 10)
        n = len(x)
        q = lambda m: n * (n + 2) * sum(acf[k] ** 2 / (n - k) for k in range(1, m + 1))
        r = self.tools('ljung_box', x, lags=[2, 4, 10])
        self.assertEqual(r['lags'], [2, 4, 10])
        close(self, r['statistic'], [q(2), q(4), q(10)], 1e-10)
        close(self, r['p_value'], [chi2_sf_even(q(m), m) for m in (2, 4, 10)], 1e-10)
        adj = self.tools('ljung_box', x, lags=[4, 10], model_df=2)
        close(self, adj['p_value'], [chi2_sf_even(q(4), 2), chi2_sf_even(q(10), 8)], 1e-10)
        upto = self.tools('ljung_box', x, lags=4)
        self.assertEqual(upto['lags'], [1, 2, 3, 4])
        close(self, upto['statistic'], [q(m) for m in (1, 2, 3, 4)], 1e-10)
        white = self.tools('ljung_box', np.random.default_rng(1).normal(size=500), lags=[10])
        self.assertGreater(white['p_value'][0], 0.01)
        strong = self.tools('ljung_box', ar1(500, 0.8, 2), lags=[10])
        self.assertLess(strong['p_value'][0], 1e-10)

    def test_validation_statuses_and_patsy_isolation(self):
        base = dict(columns={'y': [float(v) for v in ar1(100, 0.3, 1)]}, series='y')
        rejected('sm_tsa_tools', dict(base, test='kpss', regression='ctt'), 'BAD_PARAM')
        rejected('sm_tsa_tools', dict(base, test='adfuller', regression='x'))
        rejected('sm_tsa_tools', dict(base, test='adfuller', maxlag=-1), 'BAD_PARAM')
        rejected('sm_tsa_tools', dict(base, test='adfuller', autolag='BIC2'))
        rejected('sm_tsa_tools', dict(base, test='acf', regression='c'), 'BAD_PARAM')
        rejected('sm_tsa_tools', dict(base, test='acf', lags=[1]), 'BAD_PARAM')
        rejected('sm_tsa_tools', dict(base, test='ljung_box', nlags=5), 'BAD_PARAM')
        rejected('sm_tsa_tools', dict(base, test='ljung_box', lags=[]), 'BAD_PARAM')
        rejected('sm_tsa_tools', dict(base, test='ljung_box', lags=101), 'BAD_PARAM')
        rejected('sm_tsa_tools', dict(base, test='ljung_box', lags=[3], model_df=3), 'BAD_PARAM')
        rejected('sm_tsa_tools', dict(base, test='kpss', kpss_lags='many'), 'BAD_PARAM')
        rejected('sm_tsa_tools', dict(base, test='adfuller', kpss_lags=3), 'BAD_PARAM')
        rejected('sm_tsa_tools', dict(base, test='bogus'))
        call('sm_tsa_tools', dict(columns={'y': [1.0, 'NaN', 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]}, series='y', test='adfuller'), rc=2)
        short = call('sm_tsa_tools', dict(columns={'y': [1.0, 2.0, 1.5, 2.5]}, series='y', test='adfuller'), rc=2)
        self.assertIn(short['error']['data']['tool_code'], ('TOO_SHORT', 'BAD_DATA'))
        const = call('sm_tsa_tools', dict(columns={'y': [2.0] * 50}, series='y', test='kpss'), rc=2)
        self.assertEqual(const['error']['data']['tool_code'], 'CONSTANT_SERIES')
        r = self.tools('adfuller', ar1(100, 0.3, 1))
        self.assertEqual(r['meta']['forbidden_modules_loaded'], [])


class Schema(unittest.TestCase):
    def test_registration_access_and_documentation(self):
        schema = json.loads(subprocess.run([BIN, '--schema'], capture_output=True, text=True).stdout)
        ops = {o['name']: o for o in schema['operations']}
        for name in ('la_matrix_fn', 'la_sparse', 'special_fn'):
            self.assertEqual(ops[name]['access'], 'read', name)
            self.assertEqual(ops[name + '_to']['access'], 'write', name)
            for variant in (name, name + '_to'):
                for param in ops[variant]['params']:
                    self.assertTrue(param.get('description'), (variant, param['name']))
            self.assertIn('out', {p['name'] for p in ops[name + '_to']['params']})
            self.assertNotIn('out', {p['name'] for p in ops[name]['params']})
            self.assertIn('artifacts', ops[name + '_to']['description'])
        fns = next(p for p in ops['special_fn']['params'] if p['name'] == 'fn')['enum']
        self.assertEqual(len(fns), 34)
        seed = next(p for p in ops['la_sparse']['params'] if p['name'] == 'seed')
        self.assertIn('REQUIRED', seed['description'])


if __name__ == '__main__':
    unittest.main()
