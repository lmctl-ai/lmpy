"""S8b-2 stats_dist long tail: hand formulas and mpmath (test image only) for pdf/cdf/sf/ppf/moments of the new families, identities between the
log/inverse functions, seeded moment and KS checks for rvs, pre-spawn validation of every per-family table, and the _to twin."""
import json, math, os, pathlib, subprocess, sys, tempfile, unittest
import numpy as np
import mpmath as mp
BIN = str(pathlib.Path(sys.argv.pop(1)).resolve())
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests/oracle'))
from s1 import read_safetensors
import s8

mp.mp.dps = 40
NO_PYTHON = dict(os.environ, PYCLI_PYTHON='/no/such/python')


def call(op, params, rc=0, env=None):
    p = subprocess.run([BIN, '--input-json', json.dumps({'op': op, 'params': params})], capture_output=True, text=True, env=env)
    assert p.returncode == rc, (op, rc, p.returncode, p.stdout[:1500], p.stderr[:600])
    return json.loads(p.stdout)


def rejected(params, code=None, rc=2, op='stats_dist'):
    r = call(op, params, rc, NO_PYTHON)
    if code is not None:
        assert r['error']['data']['tool_code'] == code, (code, r)
    return r


def values(dist, fn, x=None, params=None, **extra):
    q = dict(dist=dist, fn=fn, params=params or {}, **extra)
    if x is not None:
        q['x'] = x
    return call('stats_dist', q)['values']


def close(testcase, got, want, rel, what='', atol=0.0):
    for g, w, in zip(got, want):
        w = float(w)
        testcase.assertLessEqual(abs(g - w), rel * abs(w) + atol + 1e-300, (what, g, w))


def Phi(x):
    return 0.5 * math.erfc(-x / math.sqrt(2))


def phi(x):
    return math.exp(-x * x / 2) / math.sqrt(2 * math.pi)


# ----------------------------------------------------------------------------------------------------- independent references
def gev_cdf(x, c):
    if c == 0:
        return math.exp(-math.exp(-x))
    t = 1 - c * x
    return math.exp(-t ** (1 / c)) if t > 0 else (0.0 if c > 0 else 1.0)


def gev_pdf(x, c):
    if c == 0:
        return math.exp(-x - math.exp(-x))
    t = 1 - c * x
    return t ** (1 / c - 1) * math.exp(-t ** (1 / c))


def gpd_cdf(x, c):
    return 1 - math.exp(-x) if c == 0 else 1 - (1 + c * x) ** (-1 / c)


def gpd_pdf(x, c):
    return math.exp(-x) if c == 0 else (1 + c * x) ** (-1 / c - 1)


def skewnorm_pdf(x, a):
    return 2 * phi(x) * Phi(a * x)


def skewnorm_cdf(x, a):
    return float(mp.quad(lambda t: 2 * mp.npdf(t) * mp.ncdf(a * t), [-mp.inf, 0, x]))


def johnson_cdf(x, a, b):
    return Phi(a + b * math.asinh(x))


def johnson_pdf(x, a, b):
    return b / math.sqrt(1 + x * x) * phi(a + b * math.asinh(x))


def vonmises_pdf(x, k):
    return float(mp.exp(k * mp.cos(x)) / (2 * mp.pi * mp.besseli(0, k)))


def vonmises_cdf(x, k):
    return float(mp.quad(lambda t: mp.exp(k * mp.cos(t)), [-mp.pi, x]) / (2 * mp.pi * mp.besseli(0, k)))


def truncnorm_pdf(x, a, b):
    return phi(x) / (Phi(b) - Phi(a)) if a <= x <= b else 0.0


def truncnorm_cdf(x, a, b):
    return min(1.0, max(0.0, (Phi(x) - Phi(a)) / (Phi(b) - Phi(a))))


def nct_cdf(x, df, nc):
    """T = (Z + nc) / sqrt(V / df): P(T <= x) = E[Phi(x sqrt(V / df) - nc)] over V ~ chi2(df)."""
    chi2 = lambda v: mp.exp((df / 2 - 1) * mp.log(v) - v / 2 - (df / 2) * mp.log(2) - mp.loggamma(df / 2))
    return float(mp.quad(lambda v: mp.ncdf(x * mp.sqrt(v / df) - nc) * chi2(v), [0, df, 4 * df + 40, mp.inf]))


def nct_pdf(x, df, nc):
    chi2 = lambda v: mp.exp((df / 2 - 1) * mp.log(v) - v / 2 - (df / 2) * mp.log(2) - mp.loggamma(df / 2))
    return float(mp.quad(lambda v: mp.sqrt(v / df) * mp.npdf(x * mp.sqrt(v / df) - nc) * chi2(v), [0, df, 4 * df + 40, mp.inf]))


def ncf_cdf(x, dfn, dfd, nc):
    """Poisson mixture of central F laws: sum_j Pois(j; nc/2) I_z(dfn/2 + j, dfd/2), z = dfn x / (dfn x + dfd)."""
    z = dfn * x / (dfn * x + dfd)
    return float(mp.nsum(lambda j: mp.exp(-nc / 2) * (nc / 2) ** j / mp.factorial(j) * mp.betainc(dfn / 2 + j, dfd / 2, 0, z, regularized=True), [0, mp.inf]))


def zipf_pmf(k, a):
    return float(mp.mpf(k) ** (-a) / mp.zeta(a))


def betabinom_pmf(k, n, a, b):
    return float(mp.binomial(n, k) * mp.beta(k + a, n - k + b) / mp.beta(a, b))


def levy_cdf(x, c):
    return math.erfc(math.sqrt(c / (2 * x))) if x > 0 else 0.0


def levy_pdf(x, c):
    return math.sqrt(c / (2 * math.pi)) * x ** -1.5 * math.exp(-c / (2 * x)) if x > 0 else 0.0


class Functions(unittest.TestCase):
    def test_genextreme(self):
        for c in (-0.4, 0.0, 0.3):
            xs = [-1.5, -0.2, 0.0, 0.7, 1.2]
            xs = [x for x in xs if c == 0 or 1 - c * x > 0]
            close(self, values('genextreme', 'pdf', xs, {'c': c}), [gev_pdf(x, c) for x in xs], 1e-12, ('pdf', c))
            close(self, values('genextreme', 'cdf', xs, {'c': c}), [gev_cdf(x, c) for x in xs], 1e-12, ('cdf', c))
        loc = values('genextreme', 'cdf', [3.0], {'c': 0.2, 'loc': 1.0, 'scale': 2.0})
        close(self, loc, [gev_cdf(1.0, 0.2)], 1e-12)  # (3 - 1) / 2 = 1

    def test_genpareto(self):
        for c in (-0.3, 0.0, 0.5):
            xs = [0.0, 0.4, 1.0, 2.5]
            if c < 0:
                xs = [x for x in xs if 1 + c * x > 0]
            close(self, values('genpareto', 'pdf', xs, {'c': c}), [gpd_pdf(x, c) for x in xs], 1e-12)
            close(self, values('genpareto', 'cdf', xs, {'c': c}), [gpd_cdf(x, c) for x in xs], 1e-12)
            close(self, values('genpareto', 'sf', xs, {'c': c}), [1 - gpd_cdf(x, c) for x in xs], 1e-9)

    def test_skewnorm(self):
        xs = [-2.0, -0.5, 0.0, 0.8, 2.5]
        for a in (-3.0, 0.0, 4.0):
            close(self, values('skewnorm', 'pdf', xs, {'a': a}), [skewnorm_pdf(x, a) for x in xs], 1e-12)
            close(self, values('skewnorm', 'cdf', xs, {'a': a}), [skewnorm_cdf(x, a) for x in xs], 1e-10, atol=1e-14)

    def test_johnsonsu(self):
        xs = [-3.0, -0.4, 0.0, 1.1, 5.0]
        close(self, values('johnsonsu', 'pdf', xs, {'a': 0.7, 'b': 1.9}), [johnson_pdf(x, 0.7, 1.9) for x in xs], 1e-12)
        close(self, values('johnsonsu', 'cdf', xs, {'a': 0.7, 'b': 1.9}), [johnson_cdf(x, 0.7, 1.9) for x in xs], 1e-12)
        p = [0.05, 0.5, 0.95]
        close(self, values('johnsonsu', 'ppf', p, {'a': 0.7, 'b': 1.9}), [math.sinh((float(mp.sqrt(2) * mp.erfinv(2 * q - 1)) - 0.7) / 1.9) for q in p], 1e-9)

    def test_vonmises(self):
        xs = [-3.0, -1.0, 0.0, 0.5, 2.9]
        for k in (0.5, 4.0, 40.0):
            close(self, values('vonmises', 'pdf', xs, {'kappa': k}), [vonmises_pdf(x, k) for x in xs], 1e-11)
            close(self, values('vonmises', 'cdf', xs, {'kappa': k}), [vonmises_cdf(x, k) for x in xs], 1e-9, atol=1e-12)

    def test_truncnorm(self):
        xs = [-0.9, 0.0, 0.7, 1.4]
        close(self, values('truncnorm', 'pdf', xs, {'a': -1.0, 'b': 1.5}), [truncnorm_pdf(x, -1.0, 1.5) for x in xs], 1e-12)
        close(self, values('truncnorm', 'cdf', xs, {'a': -1.0, 'b': 1.5}), [truncnorm_cdf(x, -1.0, 1.5) for x in xs], 1e-12)
        self.assertEqual(values('truncnorm', 'pdf', [-2.0, 2.0], {'a': -1.0, 'b': 1.5}), [0.0, 0.0])
        # loc/scale apply AFTER the standardised bounds: support is loc + scale * [a, b]
        self.assertEqual(values('truncnorm', 'cdf', [10.0 - 2.0 * 1.0 - 0.1, 10.0 + 2.0 * 1.5 + 0.1], {'a': -1.0, 'b': 1.5, 'loc': 10.0, 'scale': 2.0}), [0.0, 1.0])

    def test_nct_and_ncf(self):
        for x in (-1.0, 0.5, 2.5):
            close(self, values('nct', 'cdf', [x], {'df': 7.0, 'nc': 1.3}), [nct_cdf(x, 7, 1.3)], 1e-8, ('nct cdf', x))
            close(self, values('nct', 'pdf', [x], {'df': 7.0, 'nc': 1.3}), [nct_pdf(x, 7, 1.3)], 1e-7, ('nct pdf', x))
        for x in (0.3, 1.0, 3.0):
            close(self, values('ncf', 'cdf', [x], {'dfn': 4.0, 'dfd': 9.0, 'nc': 2.5}), [ncf_cdf(x, 4, 9, 2.5)], 1e-8, ('ncf', x))

    def test_zipf_and_betabinom_mass(self):
        close(self, values('zipf', 'pdf', [1, 2, 5, 10], {'a': 2.5}), [zipf_pmf(k, 2.5) for k in (1, 2, 5, 10)], 1e-12)
        cdf = values('zipf', 'cdf', [1, 3, 7], {'a': 2.5})
        close(self, cdf, [sum(zipf_pmf(k, 2.5) for k in range(1, m + 1)) for m in (1, 3, 7)], 1e-11)
        self.assertEqual(values('zipf', 'pdf', [0, 1.5, -2], {'a': 2.5}), [0.0, 0.0, 0.0])  # off the support (and non-integers) carry no mass
        close(self, values('betabinom', 'pdf', [0, 3, 10], {'n': 10, 'a': 2.0, 'b': 3.5}), [betabinom_pmf(k, 10, 2, 3.5) for k in (0, 3, 10)], 1e-12)
        close(self, values('betabinom', 'cdf', [3], {'n': 10, 'a': 2.0, 'b': 3.5}), [sum(betabinom_pmf(k, 10, 2, 3.5) for k in range(4))], 1e-12)
        # loc shifts the integer support
        self.assertEqual(values('zipf', 'pdf', [11], {'a': 2.5, 'loc': 10}), values('zipf', 'pdf', [1], {'a': 2.5}))

    def test_levy_stable_special_cases_and_warning(self):
        xs = [-2.0, 0.0, 0.7, 3.0]
        r = call('stats_dist', dict(dist='levy_stable', fn='cdf', x=xs, params={'alpha': 1.0, 'beta': 0.0}))
        close(self, r['values'], [0.5 + math.atan(x) / math.pi for x in xs], 1e-6, 'cauchy')
        self.assertTrue(any('levy_stable' in w['message'] for w in r['warnings']), r)
        close(self, values('levy_stable', 'pdf', xs, {'alpha': 2.0, 'beta': 0.0}), [math.exp(-x * x / 4) / math.sqrt(4 * math.pi) for x in xs], 1e-6, 'normal(0, sqrt 2)')
        ys = [0.3, 1.0, 4.0]
        close(self, values('levy_stable', 'cdf', ys, {'alpha': 0.5, 'beta': 1.0}), [levy_cdf(y, 1.0) for y in ys], 1e-5, 'levy')
        close(self, values('levy_stable', 'pdf', ys, {'alpha': 0.5, 'beta': 1.0}), [levy_pdf(y, 1.0) for y in ys], 1e-5, 'levy pdf')

    def test_common_families_against_closed_forms(self):
        xs = [-1.0, 0.3, 2.0]
        close(self, values('cauchy', 'cdf', xs, {'loc': 0.5, 'scale': 2.0}), [0.5 + math.atan((x - 0.5) / 2) / math.pi for x in xs], 1e-12)
        close(self, values('laplace', 'pdf', xs, {'loc': 0.5, 'scale': 2.0}), [math.exp(-abs(x - 0.5) / 2) / 4 for x in xs], 1e-12)
        close(self, values('logistic', 'cdf', xs, {}), [1 / (1 + math.exp(-x)) for x in xs], 1e-12)
        close(self, values('gumbel_r', 'cdf', xs, {}), [math.exp(-math.exp(-x)) for x in xs], 1e-12)
        ys = [0.5, 1.0, 3.0]
        close(self, values('weibull_min', 'cdf', ys, {'c': 1.7, 'scale': 2.0}), [1 - math.exp(-(y / 2) ** 1.7) for y in ys], 1e-12)
        close(self, values('rayleigh', 'pdf', ys, {'scale': 1.5}), [y / 2.25 * math.exp(-y * y / 4.5) for y in ys], 1e-12)
        zs = [1.0, 2.0, 5.0]
        close(self, values('pareto', 'cdf', zs, {'b': 2.5}), [1 - z ** -2.5 for z in zs], 1e-12)
        close(self, values('f', 'cdf', ys, {'dfn': 2.0, 'dfd': 2.0}), [y / (1 + y) for y in ys], 1e-12)  # F(2, 2): cdf = x / (1 + x)
        self.assertEqual(values('geom', 'pdf', [1, 2, 3], {'p': 0.25}), [0.25, 0.1875, 0.140625])
        close(self, values('nbinom', 'pdf', [0, 3], {'n': 2.0, 'p': 0.4}), [0.4 ** 2, 4 * 0.4 ** 2 * 0.6 ** 3], 1e-12)
        close(self, values('gamma', 'pdf', ys, {'shape': 2.0, 'scale': 1.5}), [y / 2.25 * math.exp(-y / 1.5) for y in ys], 1e-12)
        close(self, values('poisson', 'pdf', [0, 2, 5], {'lam': 2.0}), [math.exp(-2) * 2 ** k / math.factorial(k) for k in (0, 2, 5)], 1e-12)
        close(self, values('uniform', 'cdf', [0.0, 1.0, 5.0], {'low': -1.0, 'high': 3.0}), [0.25, 0.5, 1.0], 1e-15)
        close(self, values('lognorm', 'cdf', ys, {'mu': 0.2, 'sigma': 0.7}), [Phi((math.log(y) - 0.2) / 0.7) for y in ys], 1e-12)

    def test_inverse_and_log_identities(self):
        cases = [('genextreme', {'c': 0.2}), ('genpareto', {'c': 0.3}), ('skewnorm', {'a': 2.0}), ('johnsonsu', {'a': 0.5, 'b': 1.5}), ('vonmises', {'kappa': 3.0}),
                 ('truncnorm', {'a': -1.0, 'b': 2.0}), ('weibull_min', {'c': 2.0}), ('gamma', {'shape': 3.0, 'scale': 1.0}), ('t', {'df': 5.0}),
                 ('cauchy', {}), ('logistic', {}), ('nct', {'df': 6.0, 'nc': 0.8}), ('f', {'dfn': 5.0, 'dfd': 8.0})]
        for dist, params in cases:
            p = [0.01, 0.2, 0.5, 0.8, 0.99]
            x = values(dist, 'ppf', p, params)
            close(self, values(dist, 'cdf', x, params), p, 1e-9, (dist, 'cdf(ppf)'))
            close(self, [math.exp(v) for v in values(dist, 'logpdf', x, params)], values(dist, 'pdf', x, params), 1e-10, (dist, 'logpdf'))
            close(self, [math.exp(v) for v in values(dist, 'logcdf', x, params)], values(dist, 'cdf', x, params), 1e-10, (dist, 'logcdf'))
            close(self, [math.exp(v) for v in values(dist, 'logsf', x, params)], values(dist, 'sf', x, params), 1e-10, (dist, 'logsf'))
            close(self, values(dist, 'isf', p, params), values(dist, 'ppf', [1 - q for q in p], params), 1e-9, (dist, 'isf'), atol=1e-12)
            close(self, values(dist, 'sf', x, params), [1 - q for q in p], 1e-8, (dist, 'sf'))

    def test_discrete_ppf_returns_integers_and_mass_is_logmass(self):
        r = values('betabinom', 'ppf', [0.0, 0.1, 0.5, 0.9, 1.0], {'n': 12, 'a': 2.0, 'b': 3.0})
        self.assertTrue(all(float(v).is_integer() for v in r), r)
        k = [2, 4, 6]
        close(self, [math.exp(v) for v in values('zipf', 'logpdf', k, {'a': 3.0})], values('zipf', 'pdf', k, {'a': 3.0}), 1e-12)

    def test_missing_and_special_inputs(self):
        self.assertEqual(values('skewnorm', 'cdf', [0.0, 'NA', 'NaN', '-Inf', 'Inf'], {'a': 1.0}), [0.25, 'NA', 'NaN', 0.0, 1.0])  # skewnorm(a=1): cdf(0) = 1/2 - atan(1)/pi
        self.assertEqual(values('gumbel_r', 'ppf', [0.0, 1.0]), ['-Inf', 'Inf'])  # the unit-interval endpoints map to the support ends

    def test_moments(self):
        r = call('stats_dist', dict(dist='genpareto', fn='moments', params={'c': 0.2}))
        m = r['moments']
        self.assertAlmostEqual(m['mean'], 1 / (1 - 0.2), places=12)
        self.assertAlmostEqual(m['var'], 1 / ((1 - 0.2) ** 2 * (1 - 0.4)), places=12)
        self.assertEqual(r['moments_undefined'], [])
        r = call('stats_dist', dict(dist='genpareto', fn='moments', params={'c': 0.6}))  # finite mean only: the variance needs c < 1/2
        self.assertEqual((r['moments']['mean'], r['moments']['var'], r['moments']['skew'], r['moments']['kurtosis']), (r['moments']['mean'], 'NA', 'NA', 'NA'))
        self.assertAlmostEqual(r['moments']['mean'], 1 / 0.4, places=12)
        self.assertEqual(r['moments_undefined'], ['var', 'skew', 'kurtosis'])
        a = 3.0
        delta = a / math.sqrt(1 + a * a)
        m = call('stats_dist', dict(dist='skewnorm', fn='moments', params={'a': a, 'loc': 1.0, 'scale': 2.0}))['moments']
        self.assertAlmostEqual(m['mean'], 1.0 + 2.0 * delta * math.sqrt(2 / math.pi), places=12)
        self.assertAlmostEqual(m['var'], 4.0 * (1 - 2 * delta * delta / math.pi), places=12)
        m = call('stats_dist', dict(dist='johnsonsu', fn='moments', params={'a': 0.4, 'b': 1.6}))['moments']
        self.assertAlmostEqual(m['mean'], -math.exp(1 / (2 * 1.6 ** 2)) * math.sinh(0.4 / 1.6), places=12)
        m = call('stats_dist', dict(dist='zipf', fn='moments', params={'a': 4.5}))['moments']
        self.assertAlmostEqual(m['mean'], float(mp.zeta(3.5) / mp.zeta(4.5)), places=11)
        m = call('stats_dist', dict(dist='betabinom', fn='moments', params={'n': 10, 'a': 2.0, 'b': 3.0}))['moments']
        self.assertAlmostEqual(m['mean'], 10 * 2 / 5, places=12)
        self.assertAlmostEqual(m['var'], 10 * 2 * 3 * (5 + 10) / (25 * 6), places=12)
        m = call('stats_dist', dict(dist='nct', fn='moments', params={'df': 9.0, 'nc': 1.5}))['moments']
        self.assertAlmostEqual(m['mean'], 1.5 * math.sqrt(4.5) * math.exp(math.lgamma(4.0) - math.lgamma(4.5)), places=11)
        r = call('stats_dist', dict(dist='t', fn='moments', params={'df': 1.0}))
        self.assertEqual((r['moments']['mean'], r['moments']['var']), ('NA', 'NA'))
        self.assertEqual(r['moments_undefined'], ['mean', 'var', 'skew', 'kurtosis'])
        m = call('stats_dist', dict(dist='vonmises', fn='moments', params={'kappa': 2.0}))['moments']
        self.assertAlmostEqual(m['mean'], 0.0, places=12)  # symmetric about 0 (with loc != 0 scipy wraps into [-pi, pi]: see docs/s8b2.md)

    def test_original_ten_still_accept_their_parameter_names(self):
        for dist, params in [('norm', {}), ('t', {'df': 4.0}), ('uniform', {'low': 0.0, 'high': 2.0}), ('expon', {'scale': 2.0}), ('gamma', {'shape': 2.0, 'scale': 3.0}),
                             ('beta', {'a': 2.0, 'b': 3.0}), ('lognorm', {}), ('chi2', {'df': 3.0}), ('poisson', {'lam': 3.0}), ('binom', {'n': 5, 'p': 0.4})]:
            r = call('stats_dist', dict(dist=dist, fn='cdf', x=[0.5, 1.0, 2.0], params=params))
            self.assertEqual(r['status'], 'OK', r)
            self.assertEqual(len(r['values']), 3)


class Sampling(unittest.TestCase):
    def draw(self, dist, params, n=20000, seed=5, tmp=None):
        with tempfile.TemporaryDirectory(dir='.') as d:
            out = str(pathlib.Path(d).resolve() / 'x.safetensors')
            call('stats_dist_to', dict(dist=dist, fn='rvs', params=params, n=n, seed=seed, out=out))
            return read_safetensors(out)['x']

    def test_same_seed_same_stream_and_prefix_property(self):
        a = call('stats_dist', dict(dist='genextreme', fn='rvs', params={'c': 0.1}, n=50, seed=3))
        b = call('stats_dist', dict(dist='genextreme', fn='rvs', params={'c': 0.1}, n=20, seed=3))
        self.assertEqual(a['values'][:20], b['values'])
        self.assertEqual((a['seed'], a['generator']), (3, 'PCG64'))

    def test_ks_against_hand_cdfs(self):
        n = 20000
        bound = 1.95 / math.sqrt(n)
        checks = [('genextreme', {'c': 0.2}, lambda v: gev_cdf(v, 0.2)), ('genpareto', {'c': 0.3}, lambda v: gpd_cdf(v, 0.3)),
                  ('johnsonsu', {'a': 0.5, 'b': 1.5}, lambda v: johnson_cdf(v, 0.5, 1.5)), ('truncnorm', {'a': -1.0, 'b': 2.0}, lambda v: truncnorm_cdf(v, -1.0, 2.0)),
                  ('cauchy', {}, lambda v: 0.5 + math.atan(v) / math.pi), ('weibull_min', {'c': 1.5}, lambda v: 1 - math.exp(-v ** 1.5)),
                  ('levy_stable', {'alpha': 1.0, 'beta': 0.0}, lambda v: 0.5 + math.atan(v) / math.pi)]
        for k, (dist, params, cdf) in enumerate(checks):
            x = self.draw(dist, params, n, 40 + k)
            self.assertLess(s8.ks_statistic(x.tolist(), cdf), bound, dist)

    def test_moments_of_samples(self):
        n = 100000
        a = 3.0
        delta = a / math.sqrt(1 + a * a)
        cases = [('skewnorm', {'a': a}, delta * math.sqrt(2 / math.pi), 1 - 2 * delta * delta / math.pi),
                 ('betabinom', {'n': 10, 'a': 2.0, 'b': 3.0}, 4.0, 10 * 2 * 3 * 15 / (25 * 6)),
                 ('zipf', {'a': 6.0}, float(mp.zeta(5) / mp.zeta(6)), float(mp.zeta(4) / mp.zeta(6) - (mp.zeta(5) / mp.zeta(6)) ** 2)),
                 ('geom', {'p': 0.3}, 1 / 0.3, 0.7 / 0.09), ('nbinom', {'n': 3.0, 'p': 0.4}, 3 * 0.6 / 0.4, 3 * 0.6 / 0.16),
                 ]
        for k, (dist, params, mean, var) in enumerate(cases):
            x = self.draw(dist, params, n, 60 + k)
            sd = math.sqrt(var) if var else 1.0
            self.assertLess(abs(float(np.mean(x)) - mean), 5 * sd / math.sqrt(n), dist)
            if var:
                self.assertLess(abs(float(np.var(x, ddof=1)) - var), 0.05 * var, dist)
        x = self.draw('vonmises', {'kappa': 2.0, 'loc': 0.3}, n, 77)  # loc rotates the circle: compare the circular mean direction
        self.assertLess(abs(math.atan2(float(np.mean(np.sin(x))), float(np.mean(np.cos(x)))) - 0.3), 0.02)
        self.assertTrue(((x >= -math.pi - 1e-12) & (x <= math.pi + 1e-12)).all())
        x = self.draw('zipf', {'a': 6.0}, 1000, 3)
        self.assertTrue((x >= 1).all() and (x == np.floor(x)).all())


class Validation(unittest.TestCase):
    def test_per_family_tables(self):
        bad = [('genextreme', {}), ('genextreme', {'c': 0.1, 'shape': 2.0}), ('genextreme', {'c': 0.1, 'scale': 0.0}), ('skewnorm', {'a': 1.0, 'scale': -1.0}),
               ('levy_stable', {'alpha': 0.0}), ('levy_stable', {'alpha': 2.5}), ('levy_stable', {'alpha': 1.0, 'beta': 1.5}), ('johnsonsu', {'a': 1.0, 'b': 0.0}),
               ('nct', {'df': 3.0}), ('nct', {'df': 3.0, 'nc': 5000.0}), ('ncf', {'dfn': 3.0, 'dfd': 4.0, 'nc': -1.0}), ('ncf', {'dfn': 3.0, 'dfd': 4.0, 'nc': 2000.0}),
               ('vonmises', {'kappa': 0.0}), ('vonmises', {'kappa': 1e6}), ('betabinom', {'n': 5001, 'a': 1.0, 'b': 1.0}), ('truncnorm', {'a': 1.0, 'b': 1.0}), ('truncnorm', {'a': 2.0, 'b': 1.0}), ('zipf', {'a': 1.0}), ('zipf', {'a': 2.0, 'loc': 0.5}),
               ('betabinom', {'n': 0, 'a': 1.0, 'b': 1.0}), ('betabinom', {'n': 5.5, 'a': 1.0, 'b': 1.0}), ('betabinom', {'n': 5, 'a': 0.0, 'b': 1.0}),
               ('nbinom', {'n': 2.0, 'p': 0.0}), ('geom', {'p': 1.5}), ('weibull_min', {'c': 0.0}), ('pareto', {'b': -1.0}), ('f', {'dfn': 1.0}),
               ('cauchy', {'scale': 0.0}), ('gamma', {'shape': 1.0}), ('norm', {'loc': 1e13}), ('poisson', {'lam': -1.0})]
        for dist, params in bad:
            for fn, x in (('cdf', [0.5]), ('moments', None)):
                q = dict(dist=dist, fn=fn, params=params)
                if x:
                    q['x'] = x
                rejected(q, 'BAD_PARAM')
            q = dict(dist=dist, fn='rvs', params=params, n=3, seed=1)
            rejected(q, 'BAD_PARAM')

    def test_call_shape_rules(self):
        rejected(dict(dist='genextreme', fn='cdf', params={'c': 0.1}), 'BAD_PARAM')  # needs x
        rejected(dict(dist='genextreme', fn='moments', params={'c': 0.1}, x=[1.0]), 'BAD_PARAM')
        rejected(dict(dist='genextreme', fn='rvs', params={'c': 0.1}, n=3, seed=1, x=[1.0]), 'BAD_PARAM')
        rejected(dict(dist='genextreme', fn='pdf', params={'c': 0.1}, x=[1.0], n=3), 'BAD_PARAM')
        rejected(dict(dist='genextreme', fn='pdf', params={'c': 0.1}, x=[1.0], seed=3), 'BAD_PARAM')
        rejected(dict(dist='genextreme', fn='ppf', params={'c': 0.1}, x=[1.5]), 'BAD_PROBABILITY')
        rejected(dict(dist='genextreme', fn='isf', params={'c': 0.1}, x=[-0.1]), 'BAD_PROBABILITY')
        rejected(dict(dist='wald', fn='pdf', params={}, x=[1.0]))
        rejected(dict(dist='genextreme', fn='quantile', params={'c': 0.1}, x=[0.5]))
        rejected(dict(dist='genextreme', fn='cdf', params={'c': 0.1}, x=list(range(1001))), 'INPUT_TOO_LARGE')
        rejected(dict(dist='levy_stable', fn='cdf', params={'alpha': 1.5}, x=[0.1] * 201), 'INPUT_TOO_LARGE')  # numerical integration: 200 points at most
        rejected(dict(dist='vonmises', fn='cdf', params={'kappa': 2.0}, x=[0.1] * 201), 'INPUT_TOO_LARGE')
        rejected(dict(dist='zipf', fn='cdf', params={'a': 1.5}, x=[2e6]), 'INPUT_TOO_LARGE')  # cdf/sf sum the mass from loc + 1 up to x
        self.assertEqual(values('zipf', 'sf', [-2e6], {'a': 1.5}), [1.0])  # below the support nothing is summed
        rejected(dict(dist='genextreme', fn='cdf', params={'c': 0.1}, x=[[1.0]]), None)
        rejected(dict(dist='genextreme', fn='moments', params={'c': 0.1}, x=None, n=3), 'BAD_PARAM')


class Guards(unittest.TestCase):
    def test_zipf_quantile_beyond_the_summable_range_is_a_status(self):
        r = call('stats_dist', dict(dist='zipf', fn='ppf', params={'a': 1.2}, x=[0.5, 0.999]))
        self.assertEqual((r['status'], r['n_out_of_range'], r['values'][1]), ('OUT_OF_RANGE', 1, 'NA'))  # the in-range quantile is still answered
        self.assertGreaterEqual(r['values'][0], 1)
        r = call('stats_dist', dict(dist='zipf', fn='isf', params={'a': 1.2}, x=[0.5, 1e-3]))
        self.assertEqual((r['status'], r['values'][1]), ('OUT_OF_RANGE', 'NA'))
        r = call('stats_dist', dict(dist='zipf', fn='ppf', params={'a': 4.0}, x=[0.5, 0.999]))
        self.assertEqual((r['status'], r['values'][0]), ('OK', 1.0))
        self.assertTrue(r['values'][1] >= 2)

    def test_slow_cases_answer_within_the_budget(self):
        t = call('stats_dist', dict(dist='betabinom', fn='ppf', params={'n': 5000, 'a': 2.0, 'b': 3.0}, x=[0.01 * k for k in range(1, 100)], time_limit=3.0))
        self.assertIn(t['status'], ('OK', 'TIME_LIMIT'))
        if t['status'] == 'OK':
            self.assertTrue(all(float(v).is_integer() for v in t['values']))


class Twin(unittest.TestCase):
    def test_evaluation_artifact_beyond_inline_limit(self):
        with tempfile.TemporaryDirectory(dir='.') as d:
            d = pathlib.Path(d).resolve()
            x = np.linspace(-2.0, 2.0, 5001)
            np.save(d / 'x.npy', x)
            r = call('stats_dist', dict(dist='skewnorm', fn='pdf', params={'a': 2.0}, x=str(d / 'x.npy')), rc=2)
            self.assertEqual(r['error']['data']['tool_code'], 'INPUT_TOO_LARGE')
            out = str(d / 'pdf.safetensors')
            r = call('stats_dist_to', dict(dist='skewnorm', fn='pdf', params={'a': 2.0}, x=str(d / 'x.npy'), out=out))
            self.assertEqual((r['status'], r['n_values'], r['n_missing']), ('OK', 5001, 0))
            self.assertEqual(r['artifacts']['x']['shape'], [5001])
            got = read_safetensors(out)['x']
            close(self, got[::500].tolist(), [skewnorm_pdf(float(v), 2.0) for v in x[::500]], 1e-12)
            rejected(dict(dist='skewnorm', fn='pdf', params={'a': 2.0}, x=str(d / 'x.npy'), out=out), None, rc=5, op='stats_dist_to')
            rejected(dict(dist='skewnorm', fn='moments', params={'a': 2.0}, out=str(d / 'm.safetensors')), None, op='stats_dist_to')
            np.save(d / 'big.npy', np.zeros(200001))
            rejected(dict(dist='skewnorm', fn='pdf', params={'a': 2.0}, x=str(d / 'big.npy'), out=str(d / 'b.safetensors')), 'INPUT_TOO_LARGE', op='stats_dist_to')

    def test_missing_inputs_become_nan_in_the_artifact(self):
        with tempfile.TemporaryDirectory(dir='.') as d:
            out = str(pathlib.Path(d).resolve() / 'c.safetensors')
            r = call('stats_dist_to', dict(dist='cauchy', fn='cdf', params={}, x=[0.0, 'NA', 1.0], out=out))
            self.assertEqual((r['n_values'], r['n_missing']), (3, 1))
            got = read_safetensors(out)['x']
            self.assertEqual(got[0], 0.5)
            self.assertTrue(math.isnan(got[1]))
            self.assertAlmostEqual(got[2], 0.75, places=12)


if __name__ == '__main__':
    unittest.main()
