"""S7 interp_eval / sig_filter / sig_spectrum / sig_peaks / spatial_knn / spatial_hull / spatial_dist (+ _to twins).
Oracles are closed forms and brute force written here (numpy only, no scipy call computes an expected value): exact cubic
polynomials, hand piecewise-linear formulas, the digital Butterworth magnitude, Parseval, Euler's formula for Delaunay/Voronoi,
unit polytope volumes and hand-computed distance tables. Every pre-spawn rejection is run with a missing interpreter."""
import itertools, json, math, os, pathlib, subprocess, sys, tempfile, unittest
import numpy as np
BIN = str(pathlib.Path(sys.argv.pop(1)).resolve())
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests/oracle'))
from s1 import read_safetensors

NO_PYTHON = dict(os.environ, PYCLI_PYTHON='/no/such/python')


def call(op, params, rc=0, env=None):
    p = subprocess.run([BIN, '--input-json', json.dumps({'op': op, 'params': params})], capture_output=True, text=True, env=env)
    assert p.returncode == rc, (op, rc, p.returncode, p.stdout[:1500], p.stderr[:600])
    return json.loads(p.stdout)


def rejected(op, params, code=None, rc=2):
    """Refused by the C++ front end: no interpreter is available, so the worker cannot have produced the answer."""
    r = call(op, params, rc, NO_PYTHON)
    if code is not None:
        assert r['error']['data']['tool_code'] == code, (code, r)
    return r


def close(test, got, want, rtol=1e-9, atol=0.0, msg=None):
    np.testing.assert_allclose(np.asarray(got, float), np.asarray(want, float), rtol=rtol, atol=atol, err_msg=msg or '')


def too_large(op, params):
    """OUTPUT_TOO_LARGE is a pre-spawn CommandFailed (exit 1) when the output size is known from the request."""
    return rejected(op, params, 'OUTPUT_TOO_LARGE', rc=1)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir='.')
        self.dir = pathlib.Path(self.tmp.name).resolve()
        self.addCleanup(self.tmp.cleanup)

    def out(self, name='o.safetensors'):
        return str(self.dir / name)

    def npy(self, array):
        """Inline inputs hold at most 1000 numbers: larger inputs travel as .npy files inside the working directory."""
        path = self.dir / ('in%d.npy' % len(list(self.dir.glob('in*.npy'))))
        np.save(path, np.asarray(array, float))
        return str(path)


# f(x) = 2x^3 - 3x^2 + x - 5 and its derivatives / antiderivative (closed forms)
def poly(x):
    return 2 * x ** 3 - 3 * x ** 2 + x - 5


def poly_d(x, nu):
    return [poly(x), 6 * x ** 2 - 6 * x + 1, 12 * x - 6, 12 + 0 * x][nu]


def poly_int(a, b):
    anti = lambda x: x ** 4 / 2 - x ** 3 + x ** 2 / 2 - 5 * x
    return anti(b) - anti(a)


NODES = [0.0, 0.5, 1.5, 2.0, 3.5, 4.0, 6.0]


class Interp(Base):
    def run_interp(self, kind, x, y, x_new=None, **extra):
        params = dict(kind=kind, x=list(map(float, x)), y=list(map(float, y)))
        if x_new is not None:
            params['x_new'] = list(map(float, x_new))
        params.update(extra)
        return call('interp_eval', params)

    def test_not_a_knot_cubic_reproduces_a_cubic_with_derivatives_and_integral(self):
        grid = np.linspace(0.0, 6.0, 41)
        for nu in range(4):
            r = self.run_interp('cubic', NODES, poly(np.array(NODES)), grid, nu=nu)
            self.assertEqual((r['status'], r['kind'], r['nu']), ('OK', 'cubic', nu))
            close(self, r['values'], poly_d(grid, nu), 1e-10, 1e-9, 'nu=%d' % nu)
        r = self.run_interp('cubic', NODES, poly(np.array(NODES)), integrate=[0.7, 5.3])
        close(self, r['integral'], poly_int(0.7, 5.3), 1e-12)
        self.assertNotIn('values', r)

    def test_cubic_boundary_conditions(self):
        x = [0.0, 1.0, 2.5, 3.0, 4.5]
        y = [1.0, -2.0, 0.5, 3.0, 2.0]
        natural = self.run_interp('cubic', x, y, [0.0, 4.5], nu=2, bc_type='natural')
        close(self, natural['values'], [0.0, 0.0], atol=1e-12)
        clamped = self.run_interp('cubic', x, y, [0.0, 4.5], nu=1, bc_type='clamped')
        close(self, clamped['values'], [0.0, 0.0], atol=1e-12)
        periodic = self.run_interp('cubic', x, y[:-1] + [y[0]], [0.0, 4.5], nu=1, bc_type='periodic')
        close(self, periodic['values'][0], periodic['values'][1], 1e-12)
        rejected('interp_eval', dict(kind='cubic', x=x, y=y, x_new=[1.0], bc_type='periodic'), 'BAD_DATA')

    def test_linear_matches_the_hand_formula_derivative_and_trapezoid_integral(self):
        x = np.array([0.0, 1.0, 3.0, 4.0, 8.0])
        y = np.array([1.0, 3.0, -1.0, 2.0, 2.0])
        q = np.array([0.0, 0.25, 1.0, 2.0, 3.5, 6.0, 8.0])
        want = [y[i] + (v - x[i]) * (y[i + 1] - y[i]) / (x[i + 1] - x[i]) for v in q for i in [min(max(np.searchsorted(x, v, 'right') - 1, 0), 3)]]
        r = self.run_interp('linear', x, y, q)
        close(self, r['values'], want, 1e-14)
        mid = self.run_interp('linear', x, y, [0.5, 2.0, 3.5, 6.0], nu=1)
        close(self, mid['values'], [2.0, -2.0, 3.0, 0.0], 1e-14)
        r = self.run_interp('linear', x, y, integrate=[0.5, 7.0])
        grid = np.array([0.5, 1.0, 3.0, 4.0, 7.0])
        vals = np.interp(grid, x, y)
        close(self, r['integral'], float(np.sum(np.diff(grid) * (vals[1:] + vals[:-1]) / 2)), 1e-13)

    def test_pchip_is_monotone_hits_the_nodes_and_never_overshoots(self):
        x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        y = [0.0, 0.1, 0.1, 2.0, 2.0, 7.0, 7.5]
        grid = np.linspace(0.0, 6.0, 601)
        r = self.run_interp('pchip', x, y, grid)
        v = np.array(r['values'])
        self.assertGreaterEqual(float(np.min(np.diff(v))), -1e-14)
        self.assertGreaterEqual(float(v.min()), -1e-14)
        self.assertLessEqual(float(v.max()), 7.5 + 1e-14)
        close(self, self.run_interp('pchip', x, y, x)['values'], y, 1e-14)
        d = self.run_interp('pchip', x, y, grid, nu=1)
        self.assertGreaterEqual(float(np.min(d['values'])), -1e-12)

    def test_akima_and_makima_reproduce_lines_and_do_not_ring_on_a_step(self):
        x = np.arange(8.0)
        for kind in ('akima', 'makima'):
            line = self.run_interp(kind, x, 3.0 - 2.0 * x, np.linspace(0, 7, 29))
            close(self, line['values'], 3.0 - 2.0 * np.linspace(0, 7, 29), 1e-12, 1e-12)
            step = self.run_interp(kind, x, [0, 0, 0, 0, 1, 1, 1, 1], np.linspace(0, 7, 141))
            self.assertGreaterEqual(min(step['values']), -1e-12)
            self.assertLessEqual(max(step['values']), 1 + 1e-12)
        ring = self.run_interp('cubic', x, [0, 0, 0, 0, 1, 1, 1, 1], np.linspace(0, 7, 141))
        self.assertLess(min(ring['values']), -1e-3)  # the cubic spline does ring: this is why the other kinds exist

    def test_smoothing_spline_limits(self):
        x = np.linspace(0.0, 4.0, 15)
        line = 1.5 - 0.5 * x
        for lam in (None, 1e-3, 10.0):
            extra = {} if lam is None else {'lam': lam}
            r = self.run_interp('smoothing', x, line, [0.3, 2.2, 3.9], **extra)
            close(self, r['values'], 1.5 - 0.5 * np.array([0.3, 2.2, 3.9]), 1e-7, 1e-8)
            d = self.run_interp('smoothing', x, line, [0.3, 2.2], nu=1, **extra)
            close(self, d['values'], [-0.5, -0.5], 1e-6)
        rng = np.random.default_rng(4)
        noisy = 2.0 + 0.7 * x + 0.3 * rng.normal(size=x.size)
        heavy = self.run_interp('smoothing', x, noisy, x, lam=1e6)
        slope, intercept = np.polyfit(x, noisy, 1)
        close(self, heavy['values'], slope * x + intercept, 1e-6)
        # scipy's solver silently returns a wrong curve for a huge penalty (error 0.18 at lam=1e12 on this data): refused before spawn
        rejected('interp_eval', dict(kind='smoothing', x=list(x), y=list(noisy), x_new=[1.0], lam=1e12), 'BAD_PARAM')
        exact = self.run_interp('smoothing', x, noisy, x, lam=0.0)
        close(self, exact['values'], noisy, 1e-8, 1e-8)

    def test_rbf_reproduces_linear_functions_and_interpolates(self):
        rng = np.random.default_rng(5)
        pts = rng.uniform(-1, 1, (14, 2))
        f = lambda p: 2.0 + 3.0 * p[:, 0] - p[:, 1]
        q = rng.uniform(-0.8, 0.8, (9, 2))
        for kernel in ('thin_plate_spline', 'cubic', 'quintic', 'linear'):
            r = call('interp_eval', dict(kind='rbf', kernel=kernel, x=pts.tolist(), y=f(pts).tolist(), x_new=q.tolist()))
            self.assertEqual(r['status'], 'OK')
            if kernel in ('thin_plate_spline', 'cubic', 'quintic'):
                close(self, r['values'], f(q), 1e-8, 1e-8, kernel)
        g = call('interp_eval', dict(kind='rbf', kernel='gaussian', epsilon=1.5, x=pts.tolist(), y=f(pts).tolist(), x_new=pts.tolist()))
        close(self, g['values'], f(pts), 1e-7, 1e-7)
        s = call('interp_eval', dict(kind='rbf', kernel='gaussian', epsilon=1.5, smoothing=1.0, x=pts.tolist(), y=f(pts).tolist(), x_new=pts.tolist()))
        self.assertGreater(float(np.max(np.abs(np.array(s['values']) - f(pts)))), 1e-3)

    def test_rbf_singular_system_is_a_domain_status(self):
        pts = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 1.0]]
        r = call('interp_eval', dict(kind='rbf', kernel='gaussian', epsilon=1.0, x=pts, y=[0, 1, 2, 3, 4], x_new=[[0.5, 0.5]]))
        self.assertEqual(r['status'], 'SINGULAR')
        self.assertNotIn('values', r)

    def test_extrapolation_policy_is_explicit(self):
        y = poly(np.array(NODES))
        rejected('interp_eval', dict(kind='cubic', x=NODES, y=list(y), x_new=[6.5]), 'BAD_RANGE')
        rejected('interp_eval', dict(kind='cubic', x=NODES, y=list(y), integrate=[0.0, 7.0]), 'BAD_RANGE')
        r = self.run_interp('cubic', NODES, y, [-0.5, 3.0, 6.5], extrapolate='nan')
        self.assertEqual(r['values'][0], 'NaN')
        self.assertEqual(r['values'][2], 'NaN')
        self.assertEqual(r['n_extrapolated'], 2)
        r = self.run_interp('cubic', NODES, y, [-0.5, 6.5], extrapolate='extend')
        close(self, r['values'], poly(np.array([-0.5, 6.5])), 1e-9)
        self.assertEqual(r['n_extrapolated'], 2)

    def test_node_validation_happens_before_spawn(self):
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        base = dict(kind='pchip', x_new=[1.0])
        rejected('interp_eval', dict(base, x=[0, 1, 1, 2, 3], y=y), 'DUPLICATE_X')
        rejected('interp_eval', dict(base, x=[0, 2, 1, 3, 4], y=y), 'NOT_INCREASING')
        rejected('interp_eval', dict(base, x=[4, 3, 2, 1, 0], y=y), 'NOT_INCREASING')
        rejected('interp_eval', dict(base, x=[0, 1, 2, 3], y=y), 'BAD_DATA')
        rejected('interp_eval', dict(base, x=[0, 1, 2, 3, None], y=y), 'BAD_DATA')
        rejected('interp_eval', dict(base, x=[0, 1, 2, 3, 4], y=[1, 2, "NaN", 4, 5]), 'BAD_DATA')
        rejected('interp_eval', dict(kind='smoothing', x=[0, 1, 2, 3], y=[1, 2, 3, 4], x_new=[1.0]), 'BAD_DATA')
        rejected('interp_eval', dict(base, x=[0, 1, 2, 3, 4], y=y, nu=4), 'BAD_PARAM')
        rejected('interp_eval', dict(kind='linear', x=[0, 1, 2, 3, 4], y=y, x_new=[1.0], nu=2), 'BAD_PARAM')
        rejected('interp_eval', dict(kind='rbf', x=[[0, 0], [1, 0], [0, 1]], y=[1, 2, 3], x_new=[[0.2, 0.2]], nu=1), 'BAD_PARAM')
        rejected('interp_eval', dict(base, x=[0, 1, 2, 3, 4], y=y, bc_type='natural', lam=1.0), 'BAD_PARAM')
        rejected('interp_eval', dict(kind='cubic', x=[0, 1, 2, 3, 4], y=y), 'BAD_PARAM')  # neither x_new nor integrate
        rejected('interp_eval', dict(kind='cubic', x=[0, 1, 2, 3, 4], y=y, x_new=[1], integrate=[2, 1]), 'BAD_PARAM')
        rejected('interp_eval', dict(kind='bogus', x=[0, 1, 2], y=[1, 2, 3], x_new=[1.0]), None)

    def test_rbf_caps_before_spawn(self):
        rejected('interp_eval', dict(kind='rbf', x=self.npy([[float(i), 0.0] for i in range(5001)]), y=self.npy([0.0] * 5001), x_new=[[0.0, 0.0]]), 'INPUT_TOO_LARGE')
        rejected('interp_eval', dict(kind='rbf', x=[[float(i)] * 9 for i in range(12)], y=[0.0] * 12, x_new=[[0.0] * 9]), 'INPUT_TOO_LARGE')
        rejected('interp_eval', dict(kind='rbf', x=[[0, 0], [1, 0], [0, 1]], y=[1, 2, 3], x_new=[[0.2, 0.2, 0.2]]), 'BAD_DATA')
        rejected('interp_eval', dict(kind='rbf', kernel='gaussian', x=[[0, 0], [1, 0], [0, 1]], y=[1, 2, 3], x_new=[[0.2, 0.2]], epsilon=0), 'BAD_PARAM')

    def test_inline_budget_and_write_twin(self):
        x = np.linspace(0, 10, 50)
        grid = np.linspace(0, 10, 1500)
        too_large('interp_eval', dict(kind='pchip', x=list(x), y=list(np.sin(x)), x_new=self.npy(grid)))
        out = self.out()
        r = call('interp_eval_to', dict(kind='pchip', x=list(x), y=list(np.sin(x)), x_new=self.npy(grid), out=out))
        self.assertEqual(r['status'], 'OK')
        self.assertIn('values', r['tensors'])
        small = call('interp_eval', dict(kind='pchip', x=list(x), y=list(np.sin(x)), x_new=list(grid[:500])))
        close(self, read_safetensors(out)['values'][:500], small['values'], 1e-15)
        call('interp_eval_to', dict(kind='pchip', x=list(x), y=list(np.sin(x)), x_new=self.npy(grid), out=out), rc=5)  # exclusive

    def test_only_scipy_interpolate_is_imported(self):
        r = call('interp_eval', dict(kind='linear', x=[0, 1], y=[0, 1], x_new=[0.5]))
        loaded = r['meta']['scipy_subpackages']
        self.assertIn('interpolate', loaded)
        self.assertNotIn('signal', loaded)  # scipy.interpolate pulls in scipy.spatial, but never scipy.signal


class Signal(Base):
    def sos_design(self, **extra):
        return call('sig_filter', dict(extra))

    def test_butterworth_magnitude_matches_the_digital_analytic_response(self):
        fs, fc = 1000.0, 100.0
        freqs = [0.0, 5.0, 25.0, 60.0, 100.0, 140.0, 200.0, 300.0, 450.0]
        for n in (1, 2, 3, 4, 6, 8, 12):
            low = call('sig_filter', dict(family='butter', order=n, btype='lowpass', wn=fc, fs=fs, freqs=freqs))
            self.assertEqual((low['status'], low['stable']), ('OK', True))
            ratio = lambda f: math.tan(math.pi * f / fs) / math.tan(math.pi * fc / fs)
            want_low = [1 / math.sqrt(1 + ratio(f) ** (2 * n)) for f in freqs]
            close(self, low['magnitude'], want_low, 1e-8, 1e-12, 'lowpass n=%d' % n)
            high = call('sig_filter', dict(family='butter', order=n, btype='highpass', wn=fc, fs=fs, freqs=freqs[1:]))
            want_high = [1 / math.sqrt(1 + (1 / ratio(f)) ** (2 * n)) for f in freqs[1:]]
            close(self, high['magnitude'], want_high, 1e-8, 1e-12, 'highpass n=%d' % n)
            at_cut = call('sig_filter', dict(family='butter', order=n, btype='lowpass', wn=fc, fs=fs, freqs=[fc]))
            close(self, at_cut['magnitude'], [1 / math.sqrt(2)], 1e-12)
            self.assertEqual(low['n_sections'], (n + 1) // 2)
            self.assertEqual(len(low['sos']), (n + 1) // 2)
            self.assertTrue(all(len(row) == 6 for row in low['sos']))
        self.assertNotIn('b', low)
        self.assertNotIn('ba', low)

    def test_normalised_frequency_convention(self):
        # fs omitted: wn and freqs are fractions of Nyquist; same filter as fs = 2
        a = call('sig_filter', dict(family='butter', order=4, btype='lowpass', wn=0.25, freqs=[0.1, 0.25, 0.6]))
        b = call('sig_filter', dict(family='butter', order=4, btype='lowpass', wn=0.25, fs=2.0, freqs=[0.1, 0.25, 0.6]))
        close(self, a['magnitude'], b['magnitude'], 1e-14)
        close(self, a['sos'], b['sos'], 1e-14)

    def test_chebyshev_elliptic_and_bessel_responses(self):
        fs = 1000.0
        rp, rs = 1.0, 40.0
        dc_even = call('sig_filter', dict(family='cheby1', order=4, btype='lowpass', wn=100.0, fs=fs, rp=rp, freqs=[0.0, 100.0]))
        close(self, dc_even['magnitude'], [10 ** (-rp / 20)] * 2, 1e-9)  # even order: ripple floor at DC and at the edge
        dc_odd = call('sig_filter', dict(family='cheby1', order=5, btype='lowpass', wn=100.0, fs=fs, rp=rp, freqs=[0.0, 100.0]))
        close(self, dc_odd['magnitude'], [1.0, 10 ** (-rp / 20)], 1e-9)
        c2 = call('sig_filter', dict(family='cheby2', order=6, btype='lowpass', wn=150.0, fs=fs, rs=rs, freqs=[0.0, 150.0, 200.0, 400.0]))
        close(self, c2['magnitude'][0], 1.0, 1e-9)
        close(self, c2['magnitude'][1], 10 ** (-rs / 20), 1e-6)
        self.assertLessEqual(max(c2['magnitude'][2:]), 10 ** (-rs / 20) * (1 + 1e-6))
        el = call('sig_filter', dict(family='ellip', order=5, btype='lowpass', wn=120.0, fs=fs, rp=rp, rs=rs, freqs=[0.0, 60.0, 120.0, 250.0, 400.0]))
        self.assertGreaterEqual(min(el['magnitude'][:3]), 10 ** (-rp / 20) * (1 - 1e-9))
        self.assertLessEqual(max(el['magnitude'][3:]), 10 ** (-rs / 20) * (1 + 1e-6))
        be = call('sig_filter', dict(family='bessel', order=4, btype='lowpass', wn=80.0, fs=fs, freqs=[0.0, 400.0]))
        close(self, be['magnitude'][0], 1.0, 1e-9)
        self.assertLess(be['magnitude'][1], 0.05)
        bp = call('sig_filter', dict(family='butter', order=3, btype='bandpass', wn=[80.0, 120.0], fs=fs, freqs=[10.0, 98.0, 300.0]))
        self.assertEqual(bp['n_sections'], 3)
        self.assertGreater(bp['magnitude'][1], 0.95)
        self.assertLess(bp['magnitude'][0], 1e-3)
        bs = call('sig_filter', dict(family='butter', order=3, btype='bandstop', wn=[80.0, 120.0], fs=fs, freqs=[10.0, 98.0, 300.0]))
        self.assertLess(bs['magnitude'][1], 0.4)
        self.assertGreater(bs['magnitude'][0], 0.99)

    def tone(self, f, n=2000, fs=1000.0):
        t = np.arange(n) / fs
        return np.sin(2 * np.pi * f * t)

    def test_filtfilt_is_zero_phase_and_squares_the_magnitude_while_sosfilt_lags(self):
        fs = 1000.0
        x = self.tone(100.0)
        design = dict(family='butter', order=4, btype='lowpass', wn=100.0, fs=fs)
        gain = call('sig_filter', dict(design, freqs=[100.0]))['magnitude'][0]
        both = call('sig_filter', dict(design, x=list(x[:400]), mode='sosfiltfilt'))
        y = np.array(both['y'])
        mid = slice(100, 300)
        close(self, y[mid], gain ** 2 * x[mid], 1e-6, 1e-7)  # zero phase: y is exactly |H|^2 times x
        causal = np.array(call('sig_filter', dict(design, x=list(x[:400]), mode='sosfilt'))['y'])
        self.assertGreater(float(np.max(np.abs(causal[mid] - gain ** 2 * x[mid]))), 0.2)  # phase lag
        # a dc signal passes untouched in both modes
        dc = call('sig_filter', dict(design, x=[3.0] * 200, mode='sosfiltfilt'))
        close(self, dc['y'], [3.0] * 200, 1e-9)

    def test_unit_impulse_response_of_a_one_pole_filter(self):
        # first order butterworth lowpass is the bilinear one-pole: y[n] = k (x[n] + x[n-1]) + p y[n-1], k = t/(1+t), p = (1-t)/(1+t)
        t = math.tan(math.pi * 0.2 / 2)
        k, p = t / (1 + t), (1 - t) / (1 + t)
        x = np.zeros(30)
        x[0] = 1.0
        y = call('sig_filter', dict(family='butter', order=1, btype='lowpass', wn=0.2, x=list(x), mode='sosfilt'))['y']
        want = [k]
        want.append(k + p * want[0])
        for _ in range(28):
            want.append(p * want[-1])
        close(self, y, want, 1e-12, 1e-15)

    def test_stability_flag_is_consistent_with_the_poles(self):
        r = call('sig_filter', dict(family='butter', order=20, btype='bandpass', wn=[0.1, 0.1002], x=[0.0, 1.0] * 350, mode='sosfilt'))
        self.assertIn(r['status'], ('OK', 'UNSTABLE'))
        self.assertEqual(r['status'] == 'OK', r['max_pole_radius'] < 1.0)
        self.assertEqual(r['stable'], r['status'] == 'OK')
        if r['status'] == 'UNSTABLE':
            self.assertNotIn('y', r)
        normal = call('sig_filter', dict(family='butter', order=8, btype='lowpass', wn=0.3))
        self.assertTrue(normal['stable'])
        self.assertLess(normal['max_pole_radius'], 1.0)

    def test_marginal_designs_are_flagged_not_hidden(self):
        extreme = call('sig_filter', dict(family='butter', order=20, btype='lowpass', wn=1e-6))
        self.assertEqual((extreme['status'], extreme['stable'], extreme['marginal']), ('OK', True, True))
        self.assertGreater(extreme['max_pole_radius'], 1 - 1e-6)
        normal = call('sig_filter', dict(family='butter', order=4, btype='lowpass', wn=0.3))
        self.assertFalse(normal['marginal'])

    def test_filter_validation_before_spawn(self):
        base = dict(family='butter', order=4, btype='lowpass', wn=0.3)
        rejected('sig_filter', dict(base, order=21), 'BAD_PARAM')
        rejected('sig_filter', dict(base, order=0), 'BAD_PARAM')
        rejected('sig_filter', dict(base, wn=1.0), 'BAD_PARAM')
        rejected('sig_filter', dict(base, wn=0.0), 'BAD_PARAM')
        rejected('sig_filter', dict(base, wn=500.0, fs=1000.0), 'BAD_PARAM')
        rejected('sig_filter', dict(base, wn=[0.2, 0.4]), 'BAD_PARAM')
        rejected('sig_filter', dict(base, btype='bandpass', wn=0.3), 'BAD_PARAM')
        rejected('sig_filter', dict(base, btype='bandpass', wn=[0.4, 0.2]), 'BAD_PARAM')
        rejected('sig_filter', dict(base, family='cheby1'), 'BAD_PARAM')
        rejected('sig_filter', dict(base, family='cheby2'), 'BAD_PARAM')
        rejected('sig_filter', dict(base, family='ellip', rp=1.0), 'BAD_PARAM')
        rejected('sig_filter', dict(base, rp=1.0), 'BAD_PARAM')
        rejected('sig_filter', dict(base, family='cheby1', rp=0.0), 'BAD_PARAM')
        rejected('sig_filter', dict(base, fs=-1.0), 'BAD_PARAM')
        rejected('sig_filter', dict(base, mode='sosfilt'), 'BAD_PARAM')  # mode needs x
        rejected('sig_filter', dict(base, x=[1.0, 2.0, "NaN", 4.0], mode='sosfilt'), 'BAD_DATA')
        rejected('sig_filter', dict(base, x=[1.0] * 10, mode='sosfiltfilt'), 'BAD_DATA')  # shorter than the filtfilt padding
        rejected('sig_filter', dict(base, freqs=[0.1, 1.5]), 'BAD_PARAM')
        too_large('sig_filter', dict(base, x=[0.5] * 1000, mode='sosfilt'))
        rejected('sig_filter', dict(base, family='bogus'), None)
        rejected('sig_filter_to', dict(base, out=self.out()), 'BAD_PARAM')  # the write form needs a signal

    def test_filter_write_twin_matches_the_inline_result(self):
        x = list(self.tone(40.0, 600) + 0.1 * self.tone(300.0, 600))
        design = dict(family='butter', order=4, btype='lowpass', wn=100.0, fs=1000.0, mode='sosfiltfilt')
        inline = call('sig_filter', dict(design, x=x[:300]))
        out = self.out()
        r = call('sig_filter_to', dict(design, x=x, out=out))
        self.assertEqual(r['status'], 'OK')
        self.assertNotIn('y', r)
        got = read_safetensors(out)
        self.assertEqual(got['y'].shape, (600,))
        want = call('sig_filter_to', dict(design, x=x[:300], out=self.out('p.safetensors')))
        close(self, read_safetensors(self.out('p.safetensors'))['y'], inline['y'], 1e-14)

    def test_peaks_on_synthetic_gaussians(self):
        n = 1000
        i = np.arange(n, dtype=float)
        centres, heights = [100, 300, 500, 800], [1.0, 2.0, 3.0, 1.5]
        x = sum(h * np.exp(-0.5 * ((i - c) / 10.0) ** 2) for c, h in zip(centres, heights))
        r = call('sig_peaks', dict(x=list(x)))
        self.assertEqual((r['status'], r['n_peaks']), ('OK', 4))
        self.assertEqual(r['indices'], centres)
        close(self, r['heights'], heights, 1e-12)
        close(self, r['prominences'], heights, 1e-9)
        close(self, r['widths'], [2 * math.sqrt(2 * math.log(2)) * 10.0] * 4, 1e-2)  # FWHM of a gaussian
        self.assertEqual(call('sig_peaks', dict(x=list(x), height=1.8))['indices'], [300, 500])
        self.assertEqual(call('sig_peaks', dict(x=list(x), height=[1.2, 2.5]))['indices'], [300, 800])
        self.assertEqual(call('sig_peaks', dict(x=list(x), prominence=1.6))['indices'], [300, 500])
        self.assertEqual(call('sig_peaks', dict(x=list(x), distance=450))['indices'], [500])  # the tallest wins, all others are closer than 450

    def test_distance_threshold_plateau_and_edges(self):
        x = np.zeros(200)
        x[50], x[60] = 1.0, 2.0
        self.assertEqual(call('sig_peaks', dict(x=list(x)))['indices'], [50, 60])
        self.assertEqual(call('sig_peaks', dict(x=list(x), distance=20))['indices'], [60])  # the higher survives
        y = np.array([0, 1, 1, 1, 0, 3, 0, 0.0])
        self.assertEqual(call('sig_peaks', dict(x=list(y)))['indices'], [2, 5])  # plateau: middle sample
        z = np.array([0, 1, 0.9, 1.5, 0, 0.0])
        self.assertEqual(call('sig_peaks', dict(x=list(z), threshold=0.2))['indices'], [3])  # vs neighbours: 1.0 only exceeds by 0.1
        self.assertEqual(call('sig_peaks', dict(x=[3.0, 2.0, 1.0, 2.0, 3.0]))['indices'], [])  # endpoints are never peaks
        r = call('sig_peaks', dict(x=[0.0, 1.0, 0.0]))
        self.assertEqual((r['n_peaks'], r['indices'], r['heights']), (1, [1], [1.0]))

    def test_peaks_validation_and_write_twin(self):
        rejected('sig_peaks', dict(x=[0, 1, "Inf", 0]), 'BAD_DATA')
        rejected('sig_peaks', dict(x=[0, 1, 0], height=[2, 1]), 'BAD_PARAM')
        rejected('sig_peaks', dict(x=[0, 1, 0], distance=0.5), 'BAD_PARAM')
        rejected('sig_peaks', dict(x=[0, 1, 0], wlen=2), 'BAD_PARAM')
        rejected('sig_peaks', dict(x=[0, 1, 0], rel_height=0.0), 'BAD_PARAM')
        rejected('sig_peaks', dict(x=[0, 1, 0], prominence=-1.0), 'BAD_PARAM')
        rejected('sig_peaks', dict(x=[0, 1]), 'BAD_DATA')
        saw = [0.0, 1.0] * 400
        rejected_after = call('sig_peaks', dict(x=saw), rc=1)  # 399 peaks x 4 properties exceed 1000 numbers
        self.assertEqual(rejected_after['error']['data']['tool_code'], 'OUTPUT_TOO_LARGE')
        out = self.out()
        r = call('sig_peaks_to', dict(x=saw, out=out))
        self.assertEqual((r['status'], r['n_peaks']), ('OK', 399))
        got = read_safetensors(out)
        np.testing.assert_array_equal(got['indices'], np.arange(1, 799, 2))
        self.assertEqual(sorted(got), ['heights', 'indices', 'prominences', 'widths'])

    def test_tone_spectrum_parseval_and_amplitude(self):
        fs, n = 1000.0, 4096
        t = np.arange(n) / fs
        f0 = 125.0  # 128 bins of fs / 1024
        x = 2.0 * np.sin(2 * np.pi * f0 * t)
        out = self.out()
        w = call('sig_spectrum_to', dict(x=self.npy(x), fs=fs, method='welch', nperseg=1024, noverlap=512, scaling='spectrum', out=out))
        self.assertEqual(w['status'], 'OK')
        close(self, w['peak_freq'], f0, 1e-12)
        got = read_safetensors(out)
        close(self, got['pxx'][int(round(f0 / (fs / 1024)))], 2.0, 1e-9)  # A^2 / 2 for an on-bin tone
        self.assertEqual(int(np.argmax(got['pxx'])), 128)
        self.assertEqual((w['n_segments'], w['n_bins']), (1 + (n - 1024) // 512, 513))
        close(self, w['df'], fs / 1024, 1e-14)
        rng = np.random.default_rng(2)
        noise = rng.normal(size=1000)
        out2 = self.out('p.safetensors')
        pr = call('sig_spectrum_to', dict(x=list(noise), fs=fs, method='periodogram', window='boxcar', detrend='none', out=out2))
        close(self, pr['total_power'], float(np.mean(noise ** 2)), 1e-12)  # Parseval, one-sided density
        sp = call('sig_spectrum_to', dict(x=list(noise), fs=fs, method='periodogram', window='boxcar', detrend='none', scaling='spectrum', out=self.out('s.safetensors')))
        close(self, sp['total_power'], float(np.mean(noise ** 2)), 1e-12)

    def test_csd_and_coherence_identities(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=2048)
        xf = self.npy(x)
        common = dict(fs=1.0, nperseg=256, noverlap=128, window='hann')
        w = call('sig_spectrum', dict(common, x=xf, method='welch', nperseg=64, noverlap=32))
        c = call('sig_spectrum', dict(common, x=xf, y=xf, method='csd', nperseg=64, noverlap=32))
        close(self, c['pxy_re'], w['pxx'], 1e-12)
        close(self, c['pxy_im'], [0.0] * len(c['pxy_im']), atol=1e-14)
        coh = call('sig_spectrum', dict(common, x=xf, y=self.npy(2.5 * x), method='coherence', nperseg=64, noverlap=32))
        close(self, coh['cxy'], [1.0] * len(coh['cxy']), 1e-9)
        z = rng.normal(size=2048)
        other = call('sig_spectrum', dict(common, x=xf, y=self.npy(z), method='coherence', nperseg=64, noverlap=32))
        self.assertTrue(all(0.0 <= v <= 1.0 + 1e-12 for v in other['cxy']))
        self.assertLess(float(np.mean(other['cxy'])), 0.3)
        t = np.arange(4096) / 1000.0
        quad = call('sig_spectrum', dict(x=list(np.sin(2 * np.pi * 125 * t[:1000])), y=list(np.cos(2 * np.pi * 125 * t[:1000])), method='csd', fs=1000.0, nperseg=256, noverlap=0, window='boxcar', detrend='none'))
        k = int(np.argmax(np.hypot(quad['pxy_re'], quad['pxy_im'])))
        self.assertGreater(abs(quad['pxy_im'][k]) / math.hypot(quad['pxy_re'][k], quad['pxy_im'][k]), 0.999)  # 90 degrees apart: purely imaginary cross-spectrum

    def test_spectrum_validation_before_spawn(self):
        x = [float(v) for v in np.sin(np.arange(512))]
        base = dict(x=x, method='welch')
        rejected('sig_spectrum', dict(base, nperseg=600), 'BAD_PARAM')
        rejected('sig_spectrum', dict(base, nperseg=128, noverlap=128), 'BAD_PARAM')
        rejected('sig_spectrum', dict(base, nperseg=128, nfft=64), 'BAD_PARAM')
        rejected('sig_spectrum', dict(x=x, method='periodogram', nperseg=64), 'BAD_PARAM')
        rejected('sig_spectrum', dict(base, method='csd'), 'BAD_PARAM')  # needs y
        rejected('sig_spectrum', dict(base, y=x[:100], method='coherence'), 'BAD_DATA')
        rejected('sig_spectrum', dict(base, y=x), 'BAD_PARAM')  # y only for csd / coherence
        rejected('sig_spectrum', dict(base, fs=0.0), 'BAD_PARAM')
        rejected('sig_spectrum', dict(base, nfft=1 << 18), 'INPUT_TOO_LARGE')
        rejected('sig_spectrum', dict(base, nperseg=70000), 'BAD_PARAM')
        rejected('sig_spectrum', dict(base, window='kaiser'), None)
        rejected('sig_spectrum', dict(x=x + ["NaN"], method='welch'), 'BAD_DATA')
        long = self.npy(np.sin(np.arange(2000)))
        rejected('sig_spectrum', dict(x=long, method='welch', nperseg=1024), 'OUTPUT_TOO_LARGE', rc=1)  # 513 bins x 2 arrays
        rejected('sig_spectrum', dict(x=long, method='periodogram'), 'OUTPUT_TOO_LARGE', rc=1)
        rejected('sig_spectrum', dict(base, nperseg=64, average='mode'), None)

    def test_only_scipy_signal_is_imported(self):
        r = call('sig_peaks', dict(x=[0.0, 1.0, 0.0]))
        self.assertIn('signal', r['meta']['scipy_subpackages'])


def brute_knn(points, queries, k, p):
    if p == 'inf':
        d = np.max(np.abs(queries[:, None, :] - points[None, :, :]), axis=2)
    else:
        d = np.sum(np.abs(queries[:, None, :] - points[None, :, :]) ** p, axis=2) ** (1.0 / p)
    return d, np.sort(d, axis=1)[:, :k]


class Spatial(Base):
    def test_knn_matches_brute_force_for_several_norms(self):
        rng = np.random.default_rng(6)
        pts, qs = rng.normal(size=(150, 3)), rng.normal(size=(12, 3))
        for p in (1, 2, 3.5, 'inf'):
            r = call('spatial_knn', dict(points=pts.tolist(), queries=qs.tolist(), k=5, p=p))
            full, want = brute_knn(pts, qs, 5, p)
            self.assertEqual((r['status'], r['k']), ('OK', 5))
            close(self, r['distances'], want, 1e-12, 1e-14, 'p=%s' % p)
            idx = np.array(r['indices'])
            close(self, np.take_along_axis(full, idx, axis=1), want, 1e-12, 1e-14)  # each index really has that distance
            self.assertTrue((np.diff(np.array(r['distances']), axis=1) >= 0).all())

    def test_knn_ties_compare_distances_only(self):
        grid = np.array([[i, j] for i in range(5) for j in range(5)], float)
        r = call('spatial_knn', dict(points=grid.tolist(), queries=[[2.0, 2.0], [0.5, 0.5]], k=5))
        close(self, r['distances'], [[0, 1, 1, 1, 1], [math.sqrt(0.5)] * 4 + [math.sqrt(2.5)]], 1e-14)

    def test_knn_radius_and_exclude_self(self):
        rng = np.random.default_rng(7)
        pts, qs = rng.uniform(size=(120, 2)), rng.uniform(size=(6, 2))
        r = call('spatial_knn', dict(points=pts.tolist(), queries=qs.tolist(), radius=0.2))
        full, _ = brute_knn(pts, qs, 1, 2)
        counts = (full <= 0.2).sum(axis=1)
        self.assertEqual(r['counts'], counts.tolist())
        flat = np.array(r['indices'])
        self.assertEqual(len(flat), counts.sum())
        offset = 0
        for qi, c in enumerate(counts):
            got = flat[offset:offset + c]
            self.assertEqual(sorted(got.tolist()), sorted(np.flatnonzero(full[qi] <= 0.2).tolist()))
            close(self, np.array(r['distances'])[offset:offset + c], full[qi][got], 1e-12)
            offset += c
        s = call('spatial_knn', dict(points=pts.tolist(), k=3, exclude_self=True))
        full, _ = brute_knn(pts, pts, 3, 2)
        np.fill_diagonal(full, np.inf)
        near = np.sort(full, axis=1)[:, :3]
        out = call('spatial_knn_to', dict(points=pts.tolist(), k=3, exclude_self=True, out=self.out()))
        got = read_safetensors(self.out())
        close(self, got['distances'], near, 1e-12)
        self.assertFalse((got['indices'] == np.arange(120)[:, None]).any())
        self.assertEqual(got['indices'].dtype, np.dtype('<i8'))

    def test_radius_write_twin_layout(self):
        rng = np.random.default_rng(13)
        pts, qs = rng.uniform(size=(80, 2)), rng.uniform(size=(5, 2))
        out = self.out()
        r = call('spatial_knn_to', dict(points=pts.tolist(), queries=qs.tolist(), radius=0.3, out=out))
        got = read_safetensors(out)
        full, _ = brute_knn(pts, qs, 1, 2)
        np.testing.assert_array_equal(got['counts'], (full <= 0.3).sum(axis=1))
        self.assertEqual(len(got['indices']), int(got['counts'].sum()))
        self.assertEqual(r['n_pairs'], int(got['counts'].sum()))
        offset = 0
        for qi, c in enumerate(got['counts']):
            d = got['distances'][offset:offset + c]
            self.assertTrue((np.diff(d) >= 0).all())  # sorted by distance within a query
            close(self, d, full[qi][got['indices'][offset:offset + c]], 1e-12)
            offset += c

    def test_small_voronoi_inline(self):
        pts = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.4]]
        r = call('spatial_hull', dict(points=pts, kind='voronoi'))
        self.assertEqual(r['status'], 'OK')
        self.assertEqual(len(r['point_region']), 5)
        self.assertEqual(r['region_offsets'][-1], len(r['regions']))
        self.assertEqual(r['ridge_offsets'][-1], len(r['ridge_vertices']))
        self.assertGreaterEqual(r['n_unbounded_regions'], 4)  # the four corners of the square have open regions

    def test_knn_validation_and_caps_before_spawn(self):
        pts = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
        rejected('spatial_knn', dict(points=pts, queries=[[0.0, 0.0]], k=4), 'BAD_PARAM')
        rejected('spatial_knn', dict(points=pts, queries=[[0.0, 0.0]]), 'BAD_PARAM')
        rejected('spatial_knn', dict(points=pts, queries=[[0.0, 0.0]], k=1, radius=1.0), 'BAD_PARAM')
        rejected('spatial_knn', dict(points=pts, queries=[[0.0, 0.0]], radius=-1.0), 'BAD_PARAM')
        rejected('spatial_knn', dict(points=pts, queries=[[0.0, 0.0, 0.0]], k=1), 'BAD_DATA')
        rejected('spatial_knn', dict(points=pts, queries=[[0.0, "NaN"]], k=1), 'BAD_DATA')
        rejected('spatial_knn', dict(points=pts, queries=[[0.0, 0.0]], k=1, p=0.5), 'BAD_PARAM')
        rejected('spatial_knn', dict(points=pts, queries=[[0.0, 0.0]], k=1, p='two'), 'BAD_PARAM')
        rejected('spatial_knn', dict(points=pts, queries=[[0.0, 0.0]], k=1, exclude_self=True), 'BAD_PARAM')
        rejected('spatial_knn', dict(points=[[float(i)] * 33 for i in range(5)], queries=[[0.0] * 33], k=1), 'INPUT_TOO_LARGE')
        too_large('spatial_knn', dict(points=pts, queries=[[0.0, 0.0]] * 400, k=3))  # 400 x 3 x 2 numbers
        rejected('spatial_knn', dict(points=pts, queries=[[0.0, 0.0]], radius=1.0, max_pairs=0), 'BAD_PARAM')
        dense = call('spatial_knn', dict(points=[[0.0, 0.0]] * 10 + [[5.0, 5.0]], queries=[[0.0, 0.0]], radius=1.0, max_pairs=5), rc=1)
        self.assertEqual(dense['error']['data']['tool_code'], 'OUTPUT_TOO_LARGE')

    def test_convex_hull_of_unit_polytopes(self):
        square = [[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5], [0.2, 0.7]]
        r = call('spatial_hull', dict(points=square, kind='convex_hull'))
        self.assertEqual((r['status'], r['dim']), ('OK', 2))
        close(self, (r['volume'], r['area']), (1.0, 4.0), 1e-13)  # 2-D: area and perimeter
        self.assertEqual(sorted(r['vertices']), [0, 1, 2, 3])
        cube = [list(v) for v in itertools.product([0, 1], repeat=3)] + [[0.5, 0.5, 0.5], [0.3, 0.6, 0.2]]
        r = call('spatial_hull', dict(points=cube, kind='convex_hull'))
        close(self, (r['volume'], r['area']), (1.0, 6.0), 1e-13)
        self.assertEqual((r['n_vertices'], r['n_simplices']), (8, 12))
        tet = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
        r = call('spatial_hull', dict(points=tet, kind='convex_hull'))
        close(self, (r['volume'], r['area']), (1 / 6, 1.5 + math.sqrt(3) / 2), 1e-13)
        cross = [[s if a == b else 0 for b in range(3)] for a in range(3) for s in (1, -1)]
        r = call('spatial_hull', dict(points=cross, kind='convex_hull'))
        close(self, r['volume'], 4 / 3, 1e-13)
        hyper = [list(v) for v in itertools.product([0, 1], repeat=4)]
        r = call('spatial_hull', dict(points=hyper, kind='convex_hull'))
        close(self, r['volume'], 1.0, 1e-12)
        self.assertEqual(r['n_vertices'], 16)
        simplex5 = [[0] * 5] + [[1 if i == j else 0 for j in range(5)] for i in range(5)]
        r = call('spatial_hull', dict(points=simplex5, kind='convex_hull'))
        close(self, r['volume'], 1 / 120, 1e-12)

    def test_delaunay_counts_volume_and_euler(self):
        r = call('spatial_hull', dict(points=[[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]], kind='delaunay'))
        self.assertEqual((r['status'], r['n_simplices']), ('OK', 4))
        close(self, r['volume'], 1.0, 1e-13)
        grid = [[i, j] for i in range(3) for j in range(3)]
        r = call('spatial_hull', dict(points=grid, kind='delaunay'))
        self.assertEqual(r['n_simplices'], 8)
        rng = np.random.default_rng(8)
        pts = rng.uniform(size=(60, 2))
        d = call('spatial_hull_to', dict(points=pts.tolist(), kind='delaunay', out=self.out()))
        h = call('spatial_hull', dict(points=pts.tolist(), kind='convex_hull'))
        self.assertEqual(d['n_simplices'], 2 * 60 - 2 - h['n_vertices'])  # Euler for a triangulation of n points with h on the hull
        close(self, d['volume'], h['volume'], 1e-11)
        tri = read_safetensors(self.out())['simplices']
        self.assertEqual(tri.shape, (d['n_simplices'], 3))
        # empty-circumcircle property, brute force
        for t in tri[:20]:
            a, b, c = pts[t]
            m = np.array([[b[0] - a[0], b[1] - a[1]], [c[0] - a[0], c[1] - a[1]]])
            rhs = 0.5 * np.array([np.dot(b, b) - np.dot(a, a), np.dot(c, c) - np.dot(a, a)])
            centre = np.linalg.solve(m, rhs)
            radius = np.linalg.norm(a - centre)
            others = np.delete(np.arange(60), t)
            self.assertGreaterEqual(float(np.min(np.linalg.norm(pts[others] - centre, axis=1))), radius * (1 - 1e-9))
        cube = [list(v) for v in itertools.product([0, 1], repeat=3)]
        r = call('spatial_hull', dict(points=cube, kind='delaunay'))
        close(self, r['volume'], 1.0, 1e-12)

    def test_voronoi_vertices_are_the_delaunay_circumcentres(self):
        rng = np.random.default_rng(9)
        pts = rng.uniform(size=(30, 2))
        out = self.out()
        v = call('spatial_hull_to', dict(points=pts.tolist(), kind='voronoi', out=out))
        d = call('spatial_hull_to', dict(points=pts.tolist(), kind='delaunay', out=self.out('d.safetensors')))
        self.assertEqual((v['status'], v['n_vertices']), ('OK', d['n_simplices']))
        tri = read_safetensors(self.out('d.safetensors'))['simplices']
        centres = []
        for t in tri:
            a, b, c = pts[t]
            m = np.array([[b[0] - a[0], b[1] - a[1]], [c[0] - a[0], c[1] - a[1]]])
            rhs = 0.5 * np.array([np.dot(b, b) - np.dot(a, a), np.dot(c, c) - np.dot(a, a)])
            centres.append(np.linalg.solve(m, rhs))
        got = read_safetensors(out)
        mine = sorted(map(tuple, np.round(got['vertices'], 9).tolist()))
        want = sorted(map(tuple, np.round(np.array(centres), 9).tolist()))
        self.assertEqual(mine, want)
        edges = d['n_simplices'] + 30 - 1  # Euler: V - E + F = 1 for the triangulation
        self.assertEqual(v['n_ridges'], edges)
        self.assertEqual(got['ridge_points'].shape, (edges, 2))
        self.assertEqual(len(got['point_region']), 30)

    def test_degenerate_input_is_a_domain_status(self):
        line = [[float(i), 2.0 * i] for i in range(6)]
        for kind in ('convex_hull', 'delaunay', 'voronoi'):
            r = call('spatial_hull', dict(points=line, kind=kind))
            self.assertEqual(r['status'], 'DEGENERATE', kind)
            self.assertIn('message', r)
        plane = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0.5, 0.5, 0]]
        self.assertEqual(call('spatial_hull', dict(points=plane, kind='convex_hull'))['status'], 'DEGENERATE')
        self.assertEqual(call('spatial_hull', dict(points=[[0, 0, 0], [1, 1, 1], [2, 0, 1]], kind='convex_hull'))['status'], 'DEGENERATE')  # n < d + 1
        self.assertEqual(call('spatial_hull', dict(points=[[1.0, 1.0]] * 5, kind='delaunay'))['status'], 'DEGENERATE')
        out = self.out()
        r = call('spatial_hull_to', dict(points=line, kind='convex_hull', out=out), rc=1)
        self.assertEqual(r['error']['data']['tool_code'], 'DEGENERATE')
        self.assertFalse(os.path.exists(out))

    def test_qhull_options_are_never_caller_supplied_and_caps_are_prespawn(self):
        sq = [[0, 0], [1, 0], [1, 1], [0, 1]]
        rejected('spatial_hull', dict(points=sq, kind='convex_hull', qhull_options='Qt'))
        rejected('spatial_hull', dict(points=[[0.0] * 7, [1.0] * 7], kind='convex_hull'), 'INPUT_TOO_LARGE')
        rejected('spatial_hull', dict(points=[[1.0], [2.0]], kind='convex_hull'), 'BAD_DATA')
        rejected('spatial_hull', dict(points=sq + [[0, "Inf"]], kind='convex_hull'), 'BAD_DATA')
        big = [[float(i % 97), float(i // 97)] for i in range(20001)]
        rejected('spatial_hull', dict(points=self.npy(big), kind='delaunay'), 'INPUT_TOO_LARGE')
        rejected('spatial_hull', dict(points=self.npy([[float(i), float(i * i % 7), float(i % 5)] for i in range(5001)]), kind='delaunay'), 'INPUT_TOO_LARGE')
        rejected('spatial_hull', dict(points=self.npy([[float(i % 9)] * 4 for i in range(301)]), kind='delaunay'), 'INPUT_TOO_LARGE')
        rejected('spatial_hull', dict(points=self.npy([[0.0] * 6 for i in range(301)]), kind='convex_hull'), 'INPUT_TOO_LARGE')
        r = call('spatial_hull', dict(points=[[float(i % 9), float(i % 4), float(i % 5)] for i in range(60)] + [[10, 10, 10]], kind='convex_hull', ), rc=0)
        self.assertEqual(r['status'], 'OK')
        self.assertEqual(r['qhull_options'], 'Qt')

    def test_hull_inline_budget(self):
        rng = np.random.default_rng(10)
        pts = rng.uniform(size=(300, 3))
        r = call('spatial_hull', dict(points=pts.tolist(), kind='delaunay'), rc=1)
        self.assertEqual(r['error']['data']['tool_code'], 'OUTPUT_TOO_LARGE')
        s = call('spatial_hull_to', dict(points=pts.tolist(), kind='delaunay', out=self.out()))
        self.assertGreater(s['n_simplices'], 300)
        self.assertNotIn('simplices', s)

    def test_distance_tables_by_hand(self):
        a = [[0.0, 0.0], [3.0, 4.0]]
        b = [[0.0, 0.0], [1.0, 1.0], [6.0, 8.0]]
        want = {
            'euclidean': [[0, math.sqrt(2), 10], [5, math.sqrt(13), 5]],
            'sqeuclidean': [[0, 2, 100], [25, 13, 25]],
            'cityblock': [[0, 2, 14], [7, 5, 7]],
            'chebyshev': [[0, 1, 8], [4, 3, 4]],
        }
        for metric, table in want.items():
            r = call('spatial_dist', dict(a=a, b=b, metric=metric))
            self.assertEqual((r['status'], r['shape']), ('OK', [2, 3]))
            close(self, r['distances'], table, 1e-14, 1e-14, metric)
        cos = call('spatial_dist', dict(a=[[1, 0], [1, 1]], b=[[0, 1], [1, 1], [2, 0]], metric='cosine'))
        s = 1 - 1 / math.sqrt(2)
        close(self, cos['distances'], [[1, s, 0], [s, 0, s]], 1e-14, 1e-15)
        cor = call('spatial_dist', dict(a=[[1, 2, 3]], b=[[3, 2, 1], [2, 4, 6], [1, 3, 2]], metric='correlation'))
        close(self, cor['distances'], [[2.0, 0.0, 0.5]], 1e-13, 1e-14)
        mah = call('spatial_dist', dict(a=[[1.0, 1.0]], b=[[0.0, 0.0], [1.0, 1.0]], metric='mahalanobis', vi=[[2.0, 0.0], [0.0, 0.5]]))
        close(self, mah['distances'], [[math.sqrt(2.5), 0.0]], 1e-14, 1e-15)
        ham = call('spatial_dist', dict(a=[[1, 0, 1, 1]], b=[[1, 1, 0, 1]], metric='hamming'))
        close(self, ham['distances'], [[0.5]], 1e-15)
        jac = call('spatial_dist', dict(a=[[1, 0, 1, 1]], b=[[1, 1, 0, 1]], metric='jaccard'))
        close(self, jac['distances'], [[0.5]], 1e-15)
        # b omitted: the square within-set matrix, symmetric with a zero diagonal
        sq = call('spatial_dist', dict(a=[[0, 0], [3, 4], [6, 8]], metric='euclidean'))
        close(self, sq['distances'], [[0, 5, 10], [5, 0, 5], [10, 5, 0]], 1e-14, 1e-14)

    def test_undefined_distances_are_a_status_and_nothing_is_written(self):
        r = call('spatial_dist', dict(a=[[0.0, 0.0], [1.0, 1.0]], b=[[1.0, 0.0]], metric='cosine'))
        self.assertEqual((r['status'], r['n_undefined']), ('UNDEFINED', 1))
        self.assertNotIn('distances', r)
        out = self.out()
        w = call('spatial_dist_to', dict(a=[[0.0, 0.0], [1.0, 1.0]], b=[[1.0, 0.0]], metric='cosine', out=out), rc=1)
        self.assertEqual(w['error']['data']['tool_code'], 'UNDEFINED')
        self.assertFalse(os.path.exists(out))
        bad = call('spatial_dist', dict(a=[[1.0, 2.0], [1.0, 2.0]], b=[[1.0, 3.0], [2.0, 5.0]], metric='mahalanobis', vi=[[1.0, 0.0], [0.0, -1.0]]))
        self.assertEqual(bad['status'], 'UNDEFINED')

    def test_dist_validation_caps_and_twin(self):
        a = [[0.0, 0.0]] * 3
        rejected('spatial_dist', dict(a=a, metric='haversine'), None)
        rejected('spatial_dist', dict(a=a, b=[[0.0, 0.0, 0.0]], metric='euclidean'), 'BAD_DATA')
        rejected('spatial_dist', dict(a=a, metric='mahalanobis'), 'BAD_PARAM')
        rejected('spatial_dist', dict(a=a, metric='euclidean', vi=[[1.0, 0.0], [0.0, 1.0]]), 'BAD_PARAM')
        rejected('spatial_dist', dict(a=a, metric='mahalanobis', vi=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), 'BAD_DATA')
        rejected('spatial_dist', dict(a=a, metric='mahalanobis', vi=[[1.0, 2.0], [0.0, 1.0]]), 'NOT_SYMMETRIC')
        rejected('spatial_dist', dict(a=a, metric='euclidean', b=[["NaN", 0.0]]), 'BAD_DATA')
        too_large('spatial_dist', dict(a=[[0.0, 0.0]] * 40, b=[[0.0, 0.0]] * 40, metric='euclidean'))  # 1600 > 1000 inline
        rng = np.random.default_rng(11)
        pts = rng.normal(size=(1200, 2))
        np.save(self.dir / 'p.npy', pts)
        rejected('spatial_dist_to', dict(a=str(self.dir / 'p.npy'), b=str(self.dir / 'p.npy'), metric='euclidean', out=self.out('big.safetensors')), 'INPUT_TOO_LARGE')  # 1.44e6 > 1e6
        out = self.out()
        r = call('spatial_dist_to', dict(a=pts[:400].tolist(), b=pts[400:900].tolist(), metric='cityblock', out=out))
        self.assertEqual((r['status'], r['shape']), ('OK', [400, 500]))
        got = read_safetensors(out)['distances']
        close(self, got, np.abs(pts[:400, None, :] - pts[None, 400:900, :]).sum(axis=2), 1e-13)

    def test_only_scipy_spatial_is_imported(self):
        r = call('spatial_dist', dict(a=[[0.0, 0.0]], b=[[1.0, 1.0]], metric='euclidean'))
        self.assertEqual(r['meta']['scipy_subpackages'], ['spatial'])  # neither scipy.interpolate nor scipy.signal


class Budgets(Base):
    def test_write_twins_obey_readonly_and_dry_run(self):
        params = dict(a=[[0.0, 0.0], [3.0, 4.0]], metric='euclidean', out=self.out())
        call('spatial_dist_to', params, rc=6, env=dict(os.environ, PYCLI_READONLY='1'))
        call('spatial_dist_to', params, rc=0, env=dict(os.environ, PYCLI_DRYRUN='1'))
        self.assertFalse(os.path.exists(self.out()))
        call('spatial_dist_to', params)
        self.assertTrue(os.path.exists(self.out()))

    def test_time_budget_returns_a_status_not_a_kill(self):
        rng = np.random.default_rng(12)
        x = rng.uniform(size=(5000, 3))
        params = dict(kind='rbf', x=self.npy(x), y=self.npy(np.sin(x.sum(axis=1))), x_new=[[0.5, 0.5, 0.5]])
        r = call('interp_eval', params, env=dict(os.environ, PYCLI_TIMEOUT_S='12'))  # budget = 12 - 10 s, less than the ~6 s solve plus import
        self.assertEqual(r['status'], 'TIME_LIMIT')
        self.assertNotIn('values', r)


class WorkerCaps(Base):
    """The worker repeats the caps the front end enforces: a request that bypassed C++ is refused, never computed."""
    @classmethod
    def setUpClass(cls):
        import s3
        cls.w = s3.load_worker()
        import numpy
        numpy  # noqa: B018 - the assembled worker imports numpy lazily; make sure it is importable here
        import scipy.interpolate, scipy.signal, scipy.spatial, scipy.spatial.distance  # noqa: F401

    def files(self, **arrays):
        out = {}
        for name, value in arrays.items():
            path = self.dir / (name + '.npy')
            np.save(path, np.asarray(value, float))
            out[name + '_file'] = str(path)
        return out

    def refuse(self, function, **params):
        with self.assertRaises(self.w.BadData) as raised:
            function(dict(params, time_budget_s=30))
        self.assertEqual(raised.exception.code, 'INPUT_TOO_LARGE')

    def test_caps(self):
        w = self.w
        pts = np.zeros((5001, 2))
        self.refuse(w.interp_eval, kind='rbf', nu=0, extrapolate='extend', kernel='gaussian', epsilon=1.0, smoothing=0.0, **self.files(x=pts, y=np.zeros(5001)))
        self.refuse(w.sig_filter, family='butter', order=21, btype='lowpass', wn=0.3, **self.files())
        self.refuse(w.sig_spectrum, method='welch', fs=1.0, detrend='constant', scaling='density', average='mean', nperseg=70000, noverlap=0, **self.files(x=np.zeros(100)))
        self.refuse(w.spatial_knn, k=2000, exclude_self=False, p=2, **self.files(points=np.zeros((5, 33))))
        self.refuse(w.spatial_hull, kind='delaunay', qhull_options='Qbb Qc Qz Q12 Qt', **self.files(points=np.random.default_rng(0).uniform(size=(801, 3))))
        self.refuse(w.spatial_dist, metric='euclidean', **self.files(a=np.zeros((1001, 2)), b=np.zeros((1001, 2))))


class Schema(unittest.TestCase):
    def test_registration_access_and_documentation(self):
        schema = json.loads(subprocess.run([BIN, '--schema'], capture_output=True, text=True).stdout)
        ops = {o['name']: o for o in schema['operations']}
        names = ['interp_eval', 'sig_filter', 'sig_spectrum', 'sig_peaks', 'spatial_knn', 'spatial_hull', 'spatial_dist']
        for name in names:
            self.assertEqual(ops[name]['access'], 'read', name)
            self.assertEqual(ops[name + '_to']['access'], 'write', name)
            for variant in (name, name + '_to'):
                for param in ops[variant]['params']:
                    self.assertTrue(param.get('description'), (variant, param['name']))
            self.assertIn('out', {p['name'] for p in ops[name + '_to']['params']})
            self.assertNotIn('out', {p['name'] for p in ops[name]['params']})
        for name in ('spatial_hull', 'spatial_hull_to'):
            self.assertNotIn('qhull_options', {p['name'] for p in ops[name]['params']})
            self.assertIn('DEGENERATE', ops[name]['description'])


if __name__ == '__main__':
    unittest.main()
