"""S5 sm_ols / sm_glm / sm_arima / sm_tsa_acf (+ _to twins) against independent oracles: NIST Longley and Wampler,
exact rational WLS, closed-form GLM and ARIMA cases, closed-form robust covariances, R 4.3.3 golden values, and statsmodels'
own formula API (patsy lives in the test image only) as the design-matrix oracle."""
import hashlib, json, math, os, pathlib, subprocess, sys, tempfile, unittest
import numpy as np
from scipy import optimize, stats
BIN = str(pathlib.Path(sys.argv.pop(1)).resolve())
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests/oracle'))
import s1
import s5
from s1 import read_safetensors

NO_PYTHON = dict(os.environ, PYCLI_PYTHON='/no/such/python')
D, R = s5.DATA, s5.R
REG_TERMS = [{'const': True}, {'var': 'x1'}, {'var': 'x2'}, {'cat': 'g', 'ref': 'b'}, {'interact': ['x1', 'x2']}]
REG_NAMES = ['const', 'x1', 'x2', 'g[T.a]', 'g[T.c]', 'x1:x2']


def call(op, params, rc=0, env=None):
    p = subprocess.run([BIN, '--input-json', json.dumps({'op': op, 'params': params})], capture_output=True, text=True, env=env)
    assert p.returncode == rc, (rc, p.returncode, p.stdout[:1500], p.stderr[:600])
    return json.loads(p.stdout)


def rejected(op, params, code=None, rc=2):
    r = call(op, params, rc, NO_PYTHON)
    if code is not None:
        assert r['error']['data']['tool_code'] == code, (code, r)
    return r


def table(r):
    c = r['coefficients']
    return {k: np.array(v) if k != 'names' else v for k, v in c.items()}


def close(test, got, want, rtol, atol=0.0, msg=None):
    np.testing.assert_allclose(np.asarray(got, float), np.asarray(want, float), rtol=rtol, atol=atol, err_msg=msg or '')


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir='.')
        self.dir = pathlib.Path(self.tmp.name).resolve()
        self.addCleanup(self.tmp.cleanup)

    def out(self, name='o.safetensors'):
        return str(self.dir / name)


class Ols(Base):
    def test_nist_longley_certified(self):
        names = [f'x{i}' for i in range(1, 7)]
        columns = {n: [float(v) for v in col] for n, col in zip(names, s1.LONGLEY_X)}
        columns['y'] = s1.LONGLEY_B
        r = call('sm_ols', {'columns': columns, 'y': 'y', 'terms': [{'const': True}] + [{'var': n} for n in names]})
        t = table(r)
        self.assertEqual((r['status'], r['n_obs'], r['df_resid'], r['rank']), ('OK', 16, 9, 7))
        self.assertGreaterEqual(min(s5.lre(a, b) for a, b in zip(t['estimate'], s1.LONGLEY_CERTIFIED)), 8.0)
        self.assertGreaterEqual(min(s5.lre(a, b) for a, b in zip(t['std_err'], s5.LONGLEY_SE)), 6.0)
        self.assertGreaterEqual(s5.lre(r['sigma'], s5.LONGLEY_SIGMA), 8.0)
        self.assertGreaterEqual(s5.lre(r['r2'], s5.LONGLEY_R2), 9.0)
        self.assertGreaterEqual(s5.lre(r['ssr'], s1.LONGLEY_RESIDUAL_SS), 8.0)
        self.assertEqual(t['names'], ['const'] + names)

    def test_nist_wampler1_and_wampler2(self):
        x = [float(i) for i in range(21)]
        y1 = [float(sum(i ** p for p in range(6))) for i in range(21)]
        r = call('sm_ols', {'columns': {'x': x, 'y': y1}, 'y': 'y', 'terms': [{'const': True}, {'poly': ['x', 5]}]})
        t = table(r)
        self.assertEqual(t['names'], ['const', 'x', 'x^2', 'x^3', 'x^4', 'x^5'])
        self.assertGreaterEqual(min(s5.lre(v, 1.0) for v in t['estimate']), 6.0)
        self.assertGreater(r['r2'], 1 - 1e-12)
        # Wampler 2: y = 1 + .1x + .01x^2 + .001x^3 + 1e-4 x^4 + 1e-5 x^5 evaluated exactly, then rounded to doubles; the exact
        # rational least squares of those doubles is the reference, the certified (1, .1, ...) must agree to the data's rounding.
        from fractions import Fraction as Fr
        y2 = [float(sum(Fr(1, 10 ** k) * Fr(i) ** k for k in range(6))) for i in range(21)]
        r = call('sm_ols', {'columns': {'x': x, 'y': y2}, 'y': 'y', 'terms': [{'const': True}, {'poly': ['x', 5]}]})
        t = table(r)
        X = [[float(i) ** k for k in range(6)] for i in range(21)]
        exact, _, _, _ = s5.wls_exact(X, y2, [1] * 21)
        self.assertGreaterEqual(min(s5.lre(a, float(b)) for a, b in zip(t['estimate'], exact)), 5.0)
        self.assertGreaterEqual(min(s5.lre(a, 10.0 ** -k) for k, a in enumerate(t['estimate'])), 4.0)

    def test_exact_rational_ols_and_wls(self):
        import random
        rng = random.Random(41)
        for trial in range(12):
            n = rng.randint(7, 12)
            x = [rng.randint(-9, 9) for _ in range(n)]
            z = [rng.randint(-9, 9) for _ in range(n)]
            y = [rng.randint(-20, 20) for _ in range(n)]
            w = [rng.randint(1, 5) for _ in range(n)]
            X = [[1, x[i], z[i]] for i in range(n)]
            if s1.rank_exact(X) < 3:
                continue
            for weighted in (False, True):
                weights = w if weighted else [1] * n
                beta, inverse, rss, sigma2 = s5.wls_exact(X, y, weights)
                params = {'columns': {'x': x, 'z': z, 'y': y, 'w': w}, 'y': 'y', 'terms': [{'const': True}, {'var': 'x'}, {'var': 'z'}]}
                if weighted:
                    params['weights'] = 'w'
                r = call('sm_ols', params)
                t = table(r)
                close(self, t['estimate'], [float(b) for b in beta], 1e-11, 1e-12, f'{trial} {weighted}')
                se = [math.sqrt(float(sigma2 * inverse[i][i])) for i in range(3)]
                close(self, t['std_err'], se, 1e-11, 1e-12)
                self.assertAlmostEqual(r['ssr'], float(rss), delta=1e-9 * max(1.0, float(rss)))

    def test_r_golden_with_categorical_and_interaction(self):
        r = call('sm_ols', {'columns': s5.columns('y', 'x1', 'x2', 'g'), 'y': 'y', 'terms': REG_TERMS})
        t = table(r)
        self.assertEqual(t['names'], REG_NAMES)
        close(self, t['estimate'], R['lm.coef'], 1e-9)
        close(self, t['std_err'], R['lm.se'], 1e-9)
        close(self, t['statistic'], R['lm.t'], 1e-9)
        close(self, t['p_value'], R['lm.p'], 1e-7, 1e-300)
        ci = np.array(R['lm.ci']).reshape(-1, 2)
        close(self, t['ci_lower'], ci[:, 0], 1e-9)
        close(self, t['ci_upper'], ci[:, 1], 1e-9)
        for key, value in (('r2', 'lm.r2'), ('r2_adj', 'lm.adjr2'), ('sigma', 'lm.sigma'), ('f_statistic', 'lm.fstat'), ('llf', 'lm.loglik')):
            close(self, r[key], R[value][0], 1e-9, msg=key)
        # Documented parity difference: statsmodels (and lmstats) count p parameters in aic/bic, R's logLik counts p + 1 (the error variance).
        self.assertEqual(r['k_params'], 6)
        close(self, r['aic'] + 2, R['lm.aic'][0], 1e-12)
        close(self, r['bic'] + math.log(40), R['lm.bic'][0], 1e-12)
        self.assertEqual((r['df_resid'], r['df_model']), (R['lm.df'][0], 5))
        self.assertEqual(r['coding'], {'g': {'reference': 'b', 'levels': ['a', 'b', 'c']}})
        self.assertEqual(r['cov_type'], 'nonrobust')
        self.assertTrue(r['use_t'])

    def test_r_golden_wls_and_no_constant(self):
        r = call('sm_ols', {'columns': s5.columns('y', 'x1', 'x2', 'g', 'w'), 'y': 'y', 'weights': 'w', 'terms': [{'const': True}, {'var': 'x1'}, {'var': 'x2'}, {'cat': 'g', 'ref': 'b'}]})
        t = table(r)
        close(self, t['estimate'], R['wls.coef'], 1e-9)
        close(self, t['std_err'], R['wls.se'], 1e-9)
        close(self, r['r2'], R['wls.r2'][0], 1e-9)
        close(self, r['sigma'], R['wls.sigma'][0], 1e-9)
        r = call('sm_ols', {'columns': s5.columns('y', 'x1', 'x2'), 'y': 'y', 'terms': [{'const': False}, {'var': 'x1'}, {'var': 'x2'}]})
        t = table(r)
        close(self, t['estimate'], R['nocon.coef'], 1e-9)
        close(self, t['std_err'], R['nocon.se'], 1e-9)
        close(self, r['r2'], R['nocon.r2'][0], 1e-9)  # uncentered R^2 without an intercept, as R does
        self.assertEqual(t['names'], ['x1', 'x2'])

    def test_default_intercept_and_term_order(self):
        with_default = call('sm_ols', {'columns': s5.columns('y', 'x1'), 'y': 'y', 'terms': [{'var': 'x1'}]})
        self.assertEqual(table(with_default)['names'], ['const', 'x1'])  # an intercept is included unless {"const": false} says otherwise
        explicit = call('sm_ols', {'columns': s5.columns('y', 'x1'), 'y': 'y', 'terms': [{'var': 'x1'}, {'const': True}]})
        self.assertEqual(table(explicit)['names'], ['x1', 'const'])  # position follows the term list
        close(self, table(explicit)['estimate'], table(with_default)['estimate'][::-1], 1e-12)

    def test_design_matrix_matches_the_formula_api_oracle(self):
        import pandas as pd
        import statsmodels.formula.api as smf
        df = pd.DataFrame(s5.columns('y', 'x1', 'x2', 'g'))
        cases = {
            'x1 + x2 + C(g, Treatment("b")) + x1:x2': REG_TERMS[1:],
            'x1 + I(x1**2) + I(x1**3)': [{'poly': ['x1', 3]}],
            'x1 + C(g, Treatment("c")) + x1:C(g, Treatment("c"))': [{'var': 'x1'}, {'cat': 'g', 'ref': 'c'}, {'interact': ['x1', 'g']}],
            'C(g, Treatment("a")) + x1:x2': [{'cat': 'g', 'ref': 'a'}, {'interact': ['x1', 'x2']}],
        }
        for formula, terms in cases.items():
            with self.subTest(formula=formula):
                ref = smf.ols('y ~ ' + formula, data=df).fit()
                r = call('sm_ols', {'columns': s5.columns('y', 'x1', 'x2', 'g'), 'y': 'y', 'terms': terms})
                t = table(r)
                self.assertEqual(len(t['names']), len(ref.params))
                # same columns possibly in another order: compare sorted coefficient/SE multisets
                close(self, sorted(t['estimate']), sorted(ref.params.to_numpy()), 1e-9, 1e-11)
                close(self, sorted(t['std_err']), sorted(ref.bse.to_numpy()), 1e-8, 1e-11)
                close(self, r['r2'], ref.rsquared, 1e-10)
        # names: patsy's C(g, Treatment('b'))[T.a] <-> g[T.a]; interactions list the factors in the order given
        ref = smf.ols('y ~ x1 + C(g, Treatment("b")) + x1:C(g, Treatment("b"))', data=df).fit()
        r = call('sm_ols', {'columns': s5.columns('y', 'x1', 'g'), 'y': 'y', 'terms': [{'var': 'x1'}, {'cat': 'g', 'ref': 'b'}, {'interact': ['x1', 'g']}]})
        self.assertEqual(table(r)['names'], ['const', 'x1', 'g[T.a]', 'g[T.c]', 'x1:g[T.a]', 'x1:g[T.c]'])
        mapped = dict(zip(ref.params.index, ref.params.to_numpy()))
        self.assertAlmostEqual(table(r)['estimate'][2], mapped["C(g, Treatment(\"b\"))[T.a]"], delta=1e-9)
        self.assertAlmostEqual(table(r)['estimate'][4], mapped["x1:C(g, Treatment(\"b\"))[T.a]"], delta=1e-9)

    def test_robust_covariances_against_closed_forms(self):
        rng = np.random.default_rng(5)
        n = 25
        x, z = rng.standard_normal(n), rng.standard_normal(n)
        y = 1 + 2 * x - z + (0.3 + np.abs(x)) * rng.standard_normal(n)
        X = np.column_stack([np.ones(n), x, z])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        resid = y - X @ beta
        groups = [f'c{i % 6}' for i in range(n)]
        base = {'columns': {'x': x.tolist(), 'z': z.tolist(), 'y': y.tolist(), 'grp': groups}, 'y': 'y', 'terms': [{'const': True}, {'var': 'x'}, {'var': 'z'}]}
        for kind in ('HC0', 'HC1', 'HC2', 'HC3', 'cluster'):
            with self.subTest(cov_type=kind):
                params = dict(base, cov_type=kind)
                if kind == 'cluster':
                    params['cluster'] = 'grp'
                r = call('sm_ols', params)
                t = table(r)
                cov = s5.sandwich(X, resid, kind, groups)
                close(self, t['estimate'], beta, 1e-11)
                close(self, t['std_err'], np.sqrt(np.diag(cov)), 1e-9)
                self.assertFalse(r['use_t'])
                close(self, t['p_value'], 2 * stats.norm.sf(np.abs(beta / np.sqrt(np.diag(cov)))), 1e-7, 1e-300)
        import pandas as pd
        import statsmodels.api as sm
        for kind in ('HC0', 'HC3'):
            ref = sm.OLS(y, X).fit(cov_type=kind)  # statsmodels' own overall F (its path needs patsy, which only the test image has)
            mine = call('sm_ols', dict(base, cov_type=kind))
            close(self, mine['f_statistic'], float(ref.fvalue), 1e-9)
            close(self, mine['f_p_value'], float(ref.f_pvalue), 1e-8, 1e-300)
        r = call('sm_ols', dict(base, cov_type='HC1', use_t=True))
        self.assertTrue(r['use_t'])
        close(self, table(r)['p_value'], 2 * stats.t.sf(np.abs(beta / np.sqrt(np.diag(s5.sandwich(X, resid, 'HC1')))), n - 3), 1e-7, 1e-300)

    def test_alpha_and_inline_fitted_residuals(self):
        r = call('sm_ols', {'columns': s5.columns('y', 'x1'), 'y': 'y', 'terms': [{'var': 'x1'}], 'alpha': 0.1, 'include': ['fitted', 'residuals']})
        t = table(r)
        crit = stats.t.ppf(0.95, r['df_resid'])
        close(self, np.array(t['ci_upper']) - np.array(t['estimate']), crit * np.array(t['std_err']), 1e-11)
        X = np.column_stack([np.ones(40), D['x1']])
        beta = np.linalg.lstsq(X, np.array(D['y']), rcond=None)[0]
        close(self, r['fitted'], X @ beta, 1e-10)
        close(self, r['residuals'], np.array(D['y']) - X @ beta, 1e-9, 1e-12)
        r = call('sm_ols', {'columns': s5.columns('y', 'x1'), 'y': 'y', 'terms': [{'var': 'x1'}], 'include': ['fitted', 'residuals', 'fitted']}, 2, NO_PYTHON)

    def test_durbin_watson_condition_number_and_aic_by_hand(self):
        r = call('sm_ols', {'columns': s5.columns('y', 'x1', 'x2'), 'y': 'y', 'terms': [{'var': 'x1'}, {'var': 'x2'}], 'include': ['residuals']})
        e = np.array(r['residuals'])
        self.assertAlmostEqual(r['durbin_watson'], float(np.sum(np.diff(e) ** 2) / np.sum(e ** 2)), delta=1e-10)
        X = np.column_stack([np.ones(40), D['x1'], D['x2']])
        self.assertAlmostEqual(r['condition_number'], float(np.linalg.cond(X)), delta=1e-7 * np.linalg.cond(X))
        n, k = 40, 3
        llf = -n / 2 * (math.log(2 * math.pi) + math.log(float(e @ e) / n) + 1)
        self.assertAlmostEqual(r['llf'], llf, delta=1e-9)
        self.assertAlmostEqual(r['aic'], -2 * llf + 2 * k, delta=1e-8)
        self.assertAlmostEqual(r['bic'], -2 * llf + k * math.log(n), delta=1e-8)

    def test_missing_values_policy_is_explicit(self):
        cols = s5.columns('y', 'x1', 'x2')
        cols = {k: list(v) for k, v in cols.items()}
        cols['x1'][3] = None
        cols['y'][10] = 'NaN'
        base = {'columns': cols, 'y': 'y', 'terms': [{'var': 'x1'}, {'var': 'x2'}]}
        r = call('sm_ols', base, 2)
        self.assertEqual(r['error']['data']['tool_code'], 'MISSING_VALUES')
        self.assertIn('2 row', r['error']['message'])
        r = call('sm_ols', dict(base, na_action='drop'))
        self.assertEqual((r['n_obs'], r['n_dropped'], r['df_resid']), (38, 2, 35))
        keep = [i for i in range(40) if i not in (3, 10)]
        X = np.column_stack([np.ones(38), np.array(D['x1'])[keep], np.array(D['x2'])[keep]])
        close(self, table(r)['estimate'], np.linalg.lstsq(X, np.array(D['y'])[keep], rcond=None)[0], 1e-10)
        # infinities are never "missing": they are errors
        cols['x2'][5] = 'Inf'
        r = call('sm_ols', dict(base, columns=cols, na_action='drop'), 2)
        self.assertEqual(r['error']['data']['tool_code'], 'NONFINITE_DATA')

    def test_rank_deficient_design_is_a_status_not_a_crash(self):
        cols = {'x': D['x1'], 'x3': [2 * v for v in D['x1']], 'y': D['y']}
        r = call('sm_ols', {'columns': cols, 'y': 'y', 'terms': [{'var': 'x'}, {'var': 'x3'}]})
        self.assertEqual((r['status'], r['rank']), ('RANK_DEFICIENT', 2))
        t = r['coefficients']
        self.assertEqual(t['std_err'], ['NA'] * 3)
        self.assertEqual(t['p_value'], ['NA'] * 3)
        self.assertEqual(len(t['estimate']), 3)  # minimum-norm estimates are still returned, clearly flagged
        c = call('sm_ols', {'columns': {'x': D['x1'], 'k': [1.0] * 40, 'y': D['y']}, 'y': 'y', 'terms': [{'var': 'x'}, {'var': 'k'}]})
        self.assertEqual(c['status'], 'RANK_DEFICIENT')  # a constant column next to the intercept
        n1 = call('sm_ols', {'columns': {'x': [1.0, 2.0], 'y': [1.0, 2.0]}, 'y': 'y', 'terms': [{'var': 'x'}]})
        self.assertEqual(n1['status'], 'OK')
        self.assertEqual(n1['df_resid'], 0)
        self.assertEqual(n1['coefficients']['std_err'], ['NA', 'NA'])  # no residual degrees of freedom: nothing to estimate sigma with
        self.assertEqual(n1['no_dof'], True)

    def test_csv_and_table_sources(self):
        lines = ['y,x1,x2,g'] + [f"{D['y'][i]!r},{D['x1'][i]!r},{D['x2'][i]!r},{D['g'][i]}" for i in range(40)]
        csv = self.dir / 'd.csv'
        csv.write_text('\n'.join(lines) + '\n')
        a = call('sm_ols', {'csv': str(csv), 'y': 'y', 'terms': REG_TERMS})
        close(self, table(a)['estimate'], R['lm.coef'], 1e-9)
        npz = self.dir / 'd.npz'
        call('frame_import', {'csv': str(csv), 'out': str(npz)})
        b = call('sm_ols', {'frame': str(npz), 'y': 'y', 'terms': REG_TERMS})
        close(self, table(b)['estimate'], R['lm.coef'], 1e-9)
        categories = self.dir / 'c.npz'
        call('frame_import', {'csv': str(csv), 'dtypes': {'g': {'dtype': 'category', 'categories': ['c', 'b', 'a'], 'ordered': False}}, 'out': str(categories)})
        c = call('sm_ols', {'frame': str(categories), 'y': 'y', 'terms': REG_TERMS})
        self.assertEqual(c['coding']['g'], {'reference': 'b', 'levels': ['c', 'b', 'a']})  # declared category order is kept
        self.assertEqual(table(c)['names'], ['const', 'x1', 'x2', 'g[T.c]', 'g[T.a]', 'x1:x2'])
        close(self, sorted(table(c)['estimate']), sorted(R['lm.coef']), 1e-9)
        call('sm_ols', {'csv': str(self.dir / 'nothing.csv'), 'y': 'y', 'terms': [{'var': 'x1'}]}, 4)

    def test_categorical_reference_must_exist_and_be_explicit(self):
        r = call('sm_ols', {'columns': s5.columns('y', 'g'), 'y': 'y', 'terms': [{'cat': 'g', 'ref': 'zzz'}]}, 2)
        self.assertEqual(r['error']['data']['tool_code'], 'BAD_REFERENCE')
        self.assertIn('a', r['error']['message'])
        rejected('sm_ols', {'columns': s5.columns('y', 'g'), 'y': 'y', 'terms': [{'cat': 'g'}]}, 'BAD_TERMS')
        r = call('sm_ols', {'columns': {'y': D['y'], 'flag': [i % 2 == 0 for i in range(40)]}, 'y': 'y', 'terms': [{'cat': 'flag', 'ref': 'False'}]})
        self.assertEqual(table(r)['names'], ['const', 'flag[T.True]'])
        r = call('sm_ols', {'columns': {'y': D['y'], 'k': [i % 3 for i in range(40)]}, 'y': 'y', 'terms': [{'cat': 'k', 'ref': '1'}]})
        self.assertEqual(table(r)['names'], ['const', 'k[T.0]', 'k[T.2]'])
        r = call('sm_ols', {'columns': s5.columns('y', 'g'), 'y': 'y', 'terms': [{'var': 'g'}]}, 2)
        self.assertEqual(r['error']['data']['tool_code'], 'NOT_NUMERIC')
        r = call('sm_ols', {'columns': s5.columns('y', 'x1'), 'y': 'y', 'terms': [{'cat': 'nonesuch', 'ref': 'a'}]}, 2)
        self.assertEqual(r['error']['data']['tool_code'], 'UNKNOWN_COLUMN')

    def test_column_and_cell_caps_in_the_worker(self):
        many = {'y': [float(i % 7) for i in range(400)], 'g': [f'L{i % 300}' for i in range(400)]}
        r = call('sm_ols', {'csv': self.csv_of(many), 'y': 'y', 'terms': [{'cat': 'g', 'ref': 'L0'}]}, 2)
        self.assertEqual(r['error']['data']['tool_code'], 'TOO_MANY_COLUMNS')

    def csv_of(self, cols):
        names = list(cols)
        path = self.dir / f'{abs(hash(tuple(names)))}.csv'
        path.write_text('\n'.join([','.join(names)] + [','.join(str(cols[n][i]) for n in names) for i in range(len(cols[names[0]]))]) + '\n')
        return str(path)

    def test_to_twin_writes_fitted_residuals_and_coefficients(self):
        out = self.out()
        r = call('sm_ols_to', {'columns': s5.columns('y', 'x1', 'x2', 'g'), 'y': 'y', 'terms': REG_TERMS, 'out': out})
        t = read_safetensors(out)
        self.assertEqual(sorted(t), ['coef', 'fitted', 'residuals'])
        X = np.column_stack([np.ones(40), D['x1'], D['x2'], [float(v == 'a') for v in D['g']], [float(v == 'c') for v in D['g']], np.array(D['x1']) * np.array(D['x2'])])
        close(self, t['fitted'], X @ np.array(R['lm.coef']), 1e-9)
        close(self, t['residuals'], np.array(D['y']) - t['fitted'], 1e-12, 1e-12)
        close(self, t['coef'][:, 0], R['lm.coef'], 1e-9)
        self.assertEqual(t['coef'].shape, (6, 6))
        self.assertEqual(r['tensors']['fitted']['ref'], out + '#fitted')
        close(self, table(r)['estimate'], R['lm.coef'], 1e-9)
        before = hashlib.sha256(pathlib.Path(out).read_bytes()).hexdigest()
        call('sm_ols_to', {'columns': s5.columns('y', 'x1'), 'y': 'y', 'terms': [{'var': 'x1'}], 'out': out}, 5)
        self.assertEqual(hashlib.sha256(pathlib.Path(out).read_bytes()).hexdigest(), before)
        call('sm_ols_to', {'columns': s5.columns('y', 'x1'), 'y': 'y', 'terms': [{'var': 'x1'}], 'out': self.out('r.safetensors')}, 6, dict(os.environ, PYCLI_READONLY='1'))
        self.assertFalse(pathlib.Path(self.out('r.safetensors')).exists())
        call('sm_ols_to', {'columns': s5.columns('y', 'x1'), 'y': 'y', 'terms': [{'var': 'x1'}], 'out': self.out('d.safetensors')}, 0, dict(os.environ, PYCLI_DRYRUN='1'))
        self.assertFalse(pathlib.Path(self.out('d.safetensors')).exists())

    def test_wide_coefficient_table_goes_to_the_artifact(self):
        n, levels = 600, 130
        cols = {'y': [float((i * 7) % 11) + 0.1 * (i % levels) for i in range(n)], 'g': [f'L{i % levels:03d}' for i in range(n)]}
        csv = self.csv_of(cols)
        r = call('sm_ols', {'csv': csv, 'y': 'y', 'terms': [{'cat': 'g', 'ref': 'L000'}]}, 1)
        self.assertEqual(r['error']['data']['tool_code'], 'OUTPUT_TOO_LARGE')
        self.assertIn('sm_ols_to', r['error']['message'])
        out = self.out()
        r = call('sm_ols_to', {'csv': csv, 'y': 'y', 'terms': [{'cat': 'g', 'ref': 'L000'}], 'out': out})
        t = read_safetensors(out)
        self.assertEqual(t['coef'].shape, (levels, 6))
        self.assertTrue(r['coefficients_in_artifact'])
        self.assertEqual(len(r['coefficient_names']), levels)
        self.assertNotIn('coefficients', r)

    def test_patsy_is_never_loaded(self):
        for op, params in (('sm_ols', {'columns': s5.columns('y', 'x1'), 'y': 'y', 'terms': [{'var': 'x1'}]}),):
            r = call(op, params)
            self.assertEqual(r['meta']['forbidden_modules_loaded'], [])
            self.assertIn('statsmodels', r['meta'])


class OlsValidation(Base):
    def test_term_grammar_and_caps_are_enforced_before_spawn(self):
        cols = s5.columns('y', 'x1', 'x2', 'g')
        base = {'columns': cols, 'y': 'y'}
        bad = {
            'formula_string': ('y ~ x1 + x2', None), 'string_terms': (['x1', 'x2'], None), 'empty': ([], 'BAD_TERMS'),
            'unknown_key': ([{'variable': 'x1'}], 'BAD_TERMS'), 'two_keys': ([{'var': 'x1', 'cat': 'g'}], 'BAD_TERMS'), 'extra_key': ([{'var': 'x1', 'ref': 'a'}], 'BAD_TERMS'),
            'cat_no_ref': ([{'cat': 'g'}], 'BAD_TERMS'), 'cat_ref_number': ([{'cat': 'g', 'ref': 1}], 'BAD_TERMS'), 'cat_extra': ([{'cat': 'g', 'ref': 'a', 'x': 1}], 'BAD_TERMS'),
            'poly_degree_1': ([{'poly': ['x1', 1]}], 'BAD_TERMS'), 'poly_degree_7': ([{'poly': ['x1', 7]}], 'BAD_TERMS'), 'poly_float': ([{'poly': ['x1', 2.5]}], 'BAD_TERMS'),
            'poly_shape': ([{'poly': ['x1']}], 'BAD_TERMS'), 'poly_name': ([{'poly': [5, 2]}], 'BAD_NAME'), 'interact_one': ([{'interact': ['x1']}], 'BAD_TERMS'),
            'interact_four': ([{'interact': ['x1', 'x2', 'g', 'y']}], 'BAD_TERMS'), 'interact_dup': ([{'interact': ['x1', 'x1']}], 'BAD_TERMS'), 'interact_type': ([{'interact': 'x1:x2'}], 'BAD_TERMS'),
            'const_string': ([{'const': 'yes'}], 'BAD_TERMS'), 'const_twice': ([{'const': True}, {'const': False}], 'BAD_TERMS'), 'var_expression': ([{'var': 'x1*x2'}], 'BAD_NAME'),
            'var_empty': ([{'var': ''}], 'BAD_NAME'), 'var_control': ([{'var': 'x\n1'}], 'BAD_NAME'), 'var_nul': ([{'var': 'x\u0000'}], 'BAD_NAME'), 'var_long': ([{'var': 'x' * 300}], 'BAD_NAME'),
            'var_number': ([{'var': 5}], 'BAD_NAME'), 'nested': ([[{'var': 'x1'}]], None), 'too_many': ([{'var': f'v{i}'} for i in range(65)], 'BAD_TERMS'),
            'duplicate_var': ([{'var': 'x1'}, {'var': 'x1'}], 'BAD_TERMS'), 'duplicate_cat': ([{'cat': 'g', 'ref': 'a'}, {'cat': 'g', 'ref': 'b'}], 'BAD_TERMS'),
            'python_code': ([{'var': '__import__("os")'}], 'BAD_NAME'), 'format_string': ([{'var': '{x1}'}], 'BAD_NAME'),
        }
        for name, (terms, code) in bad.items():
            with self.subTest(case=name):
                r = call('sm_ols', dict(base, terms=terms), 2, NO_PYTHON)
                self.assertFalse(r['success'])
                if code:
                    self.assertEqual(r['error']['data']['tool_code'], code, r)

    def touch(self, name):
        path = self.dir / name
        path.write_text('a\n1\n')
        return str(path)

    def test_parameters_are_validated_before_spawn(self):
        cols = s5.columns('y', 'x1', 'g')
        base = {'columns': cols, 'y': 'y', 'terms': [{'var': 'x1'}]}
        bad = {
            'no_source': ({'y': 'y', 'terms': [{'var': 'x1'}]}, 'BAD_DATA'), 'two_sources': ({**base, 'csv': 'a.csv'}, 'BAD_DATA'), 'records': ({**base, 'records': [{'y': 1}]}, None),
            'y_missing': ({'columns': cols, 'terms': [{'var': 'x1'}]}, None), 'y_bad_name': ({**base, 'y': 'a b\nc'}, 'BAD_NAME'), 'cov_enum': ({**base, 'cov_type': 'HAC'}, None),
            'cluster_without_column': ({**base, 'cov_type': 'cluster'}, 'BAD_PARAM'), 'cluster_with_wrong_cov': ({**base, 'cluster': 'g'}, 'BAD_PARAM'), 'alpha_zero': ({**base, 'alpha': 0}, 'BAD_PARAM'),
            'alpha_one': ({**base, 'alpha': 1}, 'BAD_PARAM'), 'na_action_enum': ({**base, 'na_action': 'impute'}, None), 'include_enum': ({**base, 'include': ['design']}, None),
            'include_dup': ({**base, 'include': ['fitted', 'fitted']}, 'BAD_PARAM'), 'weights_bad_name': ({**base, 'weights': ''}, 'BAD_NAME'), 'inline_cap': ({**base, 'columns': {'y': [0.0] * 600, 'x1': [0.0] * 600}}, 'BAD_DATA'),
            'csv_outside': ({'csv': '/etc/passwd', 'y': 'y', 'terms': [{'var': 'x1'}]}, None), 'frame_ext': ({'frame': self.touch('a.csv'), 'y': 'y', 'terms': [{'var': 'x1'}]}, None),
            'out_on_read': ({**base, 'out': 'x.safetensors'}, None),
        }
        for name, (params, code) in bad.items():
            with self.subTest(case=name):
                r = call('sm_ols', params, 2, NO_PYTHON)
                if code:
                    self.assertEqual(r['error']['data']['tool_code'], code, r)
        for name, params in {'out_ext': {**base, 'out': 'x.npy'}, 'out_missing': dict(base), 'out_outside': {**base, 'out': '/tmp/pycli-s5.safetensors'}}.items():
            with self.subTest(case=name):
                call('sm_ols_to', params, 2, NO_PYTHON)


class Glm(Base):
    def fit(self, terms, y, family, link=None, **extra):
        params = {'columns': s5.columns(y, 'x1', 'x2', 'g', 'w'), 'y': y, 'terms': terms, 'family': family}
        if link:
            params['link'] = link
        params.update(extra)
        return call('sm_glm', params)

    def test_r_golden_poisson_binomial_gamma(self):
        r = self.fit([{'var': 'x1'}, {'var': 'x2'}, {'cat': 'g', 'ref': 'b'}], 'count', 'poisson')
        t = table(r)
        self.assertEqual((r['status'], r['family'], r['link'], r['converged']), ('OK', 'poisson', 'log', True))
        close(self, t['estimate'], R['poisson.coef'], 1e-6)
        close(self, t['std_err'], R['poisson.se'], 1e-6)
        close(self, t['statistic'], R['poisson.z'], 1e-6)
        close(self, r['deviance'], R['poisson.deviance'][0], 1e-8)
        close(self, r['null_deviance'], R['poisson.null'][0], 1e-8)
        close(self, r['llf'], R['poisson.loglik'][0], 1e-8)
        close(self, r['aic'], R['poisson.aic'][0], 1e-8)
        close(self, r['bic'], R['poisson.bic'][0], 1e-8)
        r = self.fit([{'var': 'x1'}, {'var': 'x2'}], 'binary', 'binomial')
        t = table(r)
        self.assertEqual(r['link'], 'logit')
        close(self, t['estimate'], R['binomial.coef'], 1e-6)
        close(self, t['std_err'], R['binomial.se'], 1e-6)
        close(self, r['deviance'], R['binomial.deviance'][0], 1e-8)
        close(self, r['null_deviance'], R['binomial.null'][0], 1e-8)
        close(self, r['aic'], R['binomial.aic'][0], 1e-8)
        close(self, r['bic'], R['binomial.bic'][0], 1e-8)
        r = self.fit([{'var': 'x1'}], 'gam', 'gamma', 'log')
        t = table(r)
        close(self, t['estimate'], R['gamma.coef'], 1e-6)
        close(self, t['std_err'], R['gamma.se'], 1e-6)
        close(self, r['deviance'], R['gamma.deviance'][0], 1e-8)
        close(self, r['null_deviance'], R['gamma.null'][0], 1e-8)
        close(self, r['scale'], R['gamma.dispersion'][0], 1e-6)  # Pearson-based dispersion, the same estimate R reports

    def test_closed_forms_intercept_only(self):
        y = np.array(D['count'], float)
        r = self.fit([{'const': True}], 'count', 'poisson', tol=1e-12)
        t = table(r)
        close(self, t['estimate'], [math.log(y.mean())], 1e-10)
        close(self, t['std_err'], [1 / math.sqrt(y.sum())], 1e-9)
        mu = y.mean()
        deviance = 2 * sum((v * math.log(v / mu) if v > 0 else 0.0) - (v - mu) for v in y)
        close(self, r['deviance'], deviance, 1e-9)
        llf = float(np.sum(y * math.log(mu) - mu - np.array([math.lgamma(v + 1) for v in y])))
        close(self, r['llf'], llf, 1e-9)
        close(self, r['aic'], -2 * llf + 2, 1e-9)
        close(self, r['bic'], -2 * llf + math.log(40), 1e-9)
        b = np.array(D['binary'], float)
        r = self.fit([{'const': True}], 'binary', 'binomial', tol=1e-12)
        pbar = b.mean()
        close(self, table(r)['estimate'], [math.log(pbar / (1 - pbar))], 1e-9)
        close(self, table(r)['std_err'], [math.sqrt(1 / (40 * pbar * (1 - pbar)))], 1e-9)
        g = np.array(D['gam'], float)
        r = self.fit([{'const': True}], 'gam', 'gamma', 'log', tol=1e-12)
        close(self, table(r)['estimate'], [math.log(g.mean())], 1e-9)
        scale = float(np.sum(((g - g.mean()) / g.mean()) ** 2) / 39)
        close(self, r['scale'], scale, 1e-8)
        close(self, table(r)['std_err'], [math.sqrt(scale / 40)], 1e-8)
        # gaussian + identity is OLS
        ols = call('sm_ols', {'columns': s5.columns('y', 'x1', 'x2'), 'y': 'y', 'terms': [{'var': 'x1'}, {'var': 'x2'}]})
        glm = self.fit([{'var': 'x1'}, {'var': 'x2'}], 'y', 'gaussian', tol=1e-12)
        close(self, table(glm)['estimate'], table(ols)['estimate'], 1e-9)
        close(self, table(glm)['std_err'], table(ols)['std_err'], 1e-9)
        close(self, glm['deviance'], ols['ssr'], 1e-9)

    def test_exposure_and_offset(self):
        exposure = [1.0 + (i % 5) for i in range(40)]
        cols = {'count': D['count'], 'e': exposure, 'le': [math.log(v) for v in exposure]}
        by_exposure = call('sm_glm', {'columns': cols, 'y': 'count', 'terms': [{'const': True}], 'family': 'poisson', 'exposure': 'e'})
        close(self, table(by_exposure)['estimate'], [math.log(sum(D['count']) / sum(exposure))], 1e-9)
        by_offset = call('sm_glm', {'columns': cols, 'y': 'count', 'terms': [{'const': True}], 'family': 'poisson', 'offset': 'le'})
        close(self, table(by_offset)['estimate'], table(by_exposure)['estimate'], 1e-10)
        rejected('sm_glm', {'columns': cols, 'y': 'count', 'terms': [{'const': True}], 'family': 'poisson', 'link': 'identity', 'exposure': 'e'}, 'BAD_PARAM')
        rejected('sm_glm', {'columns': cols, 'y': 'count', 'terms': [{'const': True}], 'family': 'poisson', 'exposure': 'e', 'offset': 'le'}, 'BAD_PARAM')

    def test_probit_and_cloglog_against_direct_maximum_likelihood(self):
        x1, x2, y = np.array(D['x1']), np.array(D['x2']), np.array(D['binary'], float)
        X = np.column_stack([np.ones(40), x1, x2])
        for link, cdf in (('probit', stats.norm.cdf), ('cloglog', lambda eta: 1 - np.exp(-np.exp(eta)))):
            def negll(beta):
                p = np.clip(cdf(X @ beta), 1e-300, 1 - 1e-16)
                return -np.sum(y * np.log(p) + (1 - y) * np.log1p(-p))
            best = optimize.minimize(negll, np.zeros(3), method='BFGS', options={'gtol': 1e-10})
            r = self.fit([{'var': 'x1'}, {'var': 'x2'}], 'binary', 'binomial', link)
            close(self, table(r)['estimate'], best.x, 1e-4, 1e-6, link)
            close(self, r['llf'], -best.fun, 1e-8)

    def test_robust_covariance_and_weights(self):
        r = self.fit([{'var': 'x1'}], 'gam', 'gamma', 'log', cov_type='HC0')
        self.assertEqual(r['cov_type'], 'HC0')
        plain = self.fit([{'var': 'x1'}], 'gam', 'gamma', 'log')
        close(self, table(r)['estimate'], table(plain)['estimate'], 1e-10)
        self.assertFalse(np.allclose(table(r)['std_err'], table(plain)['std_err'], rtol=1e-3))
        # var_weights: weights of 2 on every row equal a halved dispersion for gaussian; coefficients unchanged
        w2 = call('sm_glm', {'columns': {**s5.columns('y', 'x1'), 'w2': [2.0] * 40}, 'y': 'y', 'terms': [{'var': 'x1'}], 'family': 'gaussian', 'weights': 'w2'})
        w1 = call('sm_glm', {'columns': s5.columns('y', 'x1'), 'y': 'y', 'terms': [{'var': 'x1'}], 'family': 'gaussian'})
        close(self, table(w2)['estimate'], table(w1)['estimate'], 1e-10)

    def test_honest_statuses(self):
        r = self.fit([{'var': 'x1'}, {'var': 'x2'}], 'binary', 'binomial', maxiter=1)
        self.assertEqual(r['status'], 'NO_CONVERGENCE')
        self.assertFalse(r['converged'])
        self.assertEqual(r['n_iter'], 1)
        x = [float(i) - 10.5 for i in range(20)]
        sep = call('sm_glm', {'columns': {'x': x, 'y': [float(v > 0) for v in x]}, 'y': 'y', 'terms': [{'var': 'x'}], 'family': 'binomial'})
        self.assertIn(sep['status'], ('PERFECT_SEPARATION', 'NO_CONVERGENCE'))
        self.assertNotEqual(sep['status'], 'OK')
        dup = call('sm_glm', {'columns': {'x': D['x1'], 'x3': [2 * v for v in D['x1']], 'y': D['count']}, 'y': 'y', 'terms': [{'var': 'x'}, {'var': 'x3'}], 'family': 'poisson'})
        self.assertEqual(dup['status'], 'RANK_DEFICIENT')
        self.assertEqual(dup['coefficients']['std_err'], ['NA'] * 3)

    def test_response_domains_are_validated(self):
        base = {'columns': {'x': [1.0, 2.0, 3.0, 4.0, 5.0], 'y': [1.0, 2.0, 3.0, 4.0, 5.0]}, 'terms': [{'var': 'x'}], 'y': 'y'}
        for family, bad in (('poisson', [1.0, -2.0, 3.0, 4.0, 5.0]), ('poisson', [1.0, 2.5, 3.0, 4.0, 5.0]), ('gamma', [1.0, 0.0, 3.0, 4.0, 5.0]), ('binomial', [1.0, 2.0, 0.0, 1.0, 0.0])):
            with self.subTest(family=family, y=bad):
                p = dict(base, family=family)
                p['columns'] = {'x': base['columns']['x'], 'y': bad}
                r = call('sm_glm', p, 2)
                self.assertEqual(r['error']['data']['tool_code'], 'BAD_RESPONSE')
        ok = call('sm_glm', dict(base, family='binomial', columns={'x': [1.0, 2.0, 3.0, 4.0, 5.0], 'y': [0.1, 0.4, 0.5, 0.2, 0.9]}))
        self.assertIn(ok['status'], ('OK', 'NO_CONVERGENCE'))

    def test_family_link_matrix_is_closed(self):
        base = {'columns': s5.columns('count', 'x1'), 'y': 'count', 'terms': [{'var': 'x1'}]}
        for family, link in (('gaussian', 'logit'), ('poisson', 'logit'), ('binomial', 'log'), ('gamma', 'logit'), ('poisson', 'inverse_power'), ('gaussian', 'bogus')):
            with self.subTest(family=family, link=link):
                r = call('sm_glm', dict(base, family=family, link=link), 2, NO_PYTHON)
                self.assertFalse(r['success'])
        rejected('sm_glm', dict(base, family='tweedie'), None)
        rejected('sm_glm', dict(base), None)
        for key, value in (('maxiter', 0), ('maxiter', 101), ('tol', 0), ('tol', 0.5)):
            rejected('sm_glm', dict(base, family='poisson', **{key: value}), 'BAD_PARAM')

    def test_defaults_links(self):
        for family, link in (('gaussian', 'identity'), ('poisson', 'log'), ('binomial', 'logit'), ('gamma', 'inverse_power')):
            cols = {'gaussian': 'y', 'poisson': 'count', 'binomial': 'binary', 'gamma': 'gam'}[family]
            r = self.fit([{'var': 'x1'}], cols, family)
            self.assertEqual(r['link'], link)

    def test_to_twin(self):
        out = self.out()
        r = call('sm_glm_to', {'columns': s5.columns('count', 'x1', 'x2'), 'y': 'count', 'terms': [{'var': 'x1'}, {'var': 'x2'}], 'family': 'poisson', 'out': out})
        t = read_safetensors(out)
        self.assertEqual(sorted(t), ['coef', 'fitted', 'residuals'])
        X = np.column_stack([np.ones(40), D['x1'], D['x2']])
        close(self, t['fitted'], np.exp(X @ table(r)['estimate']), 1e-9)
        close(self, t['residuals'], np.array(D['count']) - t['fitted'], 1e-9, 1e-12)  # response residuals
        self.assertEqual(t['coef'].shape, (3, 6))
        bad = call('sm_glm_to', {'columns': s5.columns('binary', 'x1', 'x2'), 'y': 'binary', 'terms': [{'var': 'x1'}, {'var': 'x2'}], 'family': 'binomial', 'maxiter': 1, 'out': self.out('n.safetensors')}, 1)
        self.assertEqual(bad['error']['data']['tool_code'], 'NO_CONVERGENCE')
        self.assertFalse(pathlib.Path(self.out('n.safetensors')).exists())

    def test_patsy_is_never_loaded(self):
        r = self.fit([{'var': 'x1'}], 'count', 'poisson')
        self.assertEqual(r['meta']['forbidden_modules_loaded'], [])


class Arima(Base):
    def arima(self, series=None, **kw):
        params = {'columns': {'s': series if series is not None else D['series']}, 'series': 's', 'order': [1, 0, 1], 'trend': 'n'}
        params.update(kw)
        return call('sm_arima', params)

    def test_r_golden_arma11_and_forecasts(self):
        r = self.arima(forecast_steps=5)
        self.assertEqual((r['status'], r['converged'], r['cov_type']), ('OK', True, 'opg'))  # statsmodels' default covariance estimator
        t = table(r)
        self.assertEqual(t['names'], ['ar.L1', 'ma.L1', 'sigma2'])
        close(self, t['estimate'][:2], R['arma11.coef'], 2e-3, 2e-4)
        # R differentiates the likelihood numerically: statsmodels' equivalent estimator is cov_type 'approx' (the default opg differs by ~10 %)
        approx = self.arima(cov_type='approx')
        close(self, table(approx)['std_err'][:2], R['arma11.se'], 1e-2)
        self.assertFalse(np.allclose(t['std_err'][:2], R['arma11.se'], rtol=1e-2))
        close(self, r['sigma2'], R['arma11.sigma2'][0], 5e-3)
        close(self, r['llf'], R['arma11.loglik'][0], 1e-4)
        close(self, r['aic'], R['arma11.aic'][0], 1e-5)
        f = r['forecast']
        close(self, f['mean'], R['arma11.pred'], 5e-3, 5e-3)
        z = stats.norm.ppf(0.975)
        close(self, np.array(f['upper']) - np.array(f['mean']), z * np.array(R['arma11.pred.se']), 1e-2)
        close(self, np.array(f['upper']) - np.array(f['mean']), np.array(f['mean']) - np.array(f['lower']), 1e-12, 1e-12)

    def test_r_golden_ar2_with_mean_and_the_intercept_convention(self):
        r = self.arima(order=[2, 0, 0], trend='c')
        t = table(r)
        self.assertEqual(t['names'], ['intercept', 'ar.L1', 'ar.L2', 'sigma2'])
        close(self, t['estimate'][1:3], R['ar2.coef'][:2], 3e-3, 3e-4)
        # SARIMAX's trend "c" is the regression intercept; R's "intercept" is the process mean: mean = c / (1 - sum(phi))
        mean = t['estimate'][0] / (1 - t['estimate'][1] - t['estimate'][2])
        close(self, mean, R['ar2.coef'][2], 5e-2, 5e-3)
        close(self, r['llf'], R['ar2.loglik'][0], 1e-4)
        close(self, r['aic'], R['ar2.aic'][0], 1e-5)

    def test_closed_form_white_noise_with_mean(self):
        y = np.array(D['series'])
        r = self.arima(order=[0, 0, 0], trend='c')
        t = table(r)
        n = len(y)
        sigma2 = float(np.mean((y - y.mean()) ** 2))
        close(self, t['estimate'], [y.mean(), sigma2], 1e-5, 1e-8)
        llf = -n / 2 * (math.log(2 * math.pi * sigma2) + 1)
        close(self, r['llf'], llf, 1e-7)
        k = 2
        close(self, r['aic'], 2 * k - 2 * r['llf'], 1e-12)
        close(self, r['bic'], k * math.log(r['nobs_effective']) - 2 * r['llf'], 1e-12)
        self.assertEqual((r['nobs'], r['nobs_effective']), (n, n))

    def test_ar1_agrees_with_yule_walker(self):
        y = np.array(D['series'])
        r = call('sm_arima', {'columns': {'s': D['series']}, 'series': 's', 'order': [1, 0, 0], 'trend': 'n'})
        yd = y - y.mean()
        phi_yw = float(np.sum(yd[1:] * yd[:-1]) / np.sum(yd ** 2))
        self.assertAlmostEqual(table(r)['estimate'][0], phi_yw, delta=0.03)
        ols = float(np.sum(y[1:] * y[:-1]) / np.sum(y[:-1] ** 2))
        self.assertAlmostEqual(table(r)['estimate'][0], ols, delta=0.03)

    def test_differencing_seasonal_and_exog_shapes(self):
        r = self.arima(order=[1, 1, 1], forecast_steps=3)
        self.assertEqual(r['status'], 'OK')
        close(self, table(r)['estimate'][:2], R['arima111.coef'], 5e-3, 5e-4)
        close(self, r['llf'], R['arima111.loglik'][0], 1e-4)
        self.assertEqual(r['nobs_effective'], 119)  # d = 1 diffuse observation burned
        s = self.arima(order=[1, 0, 0], seasonal_order=[1, 0, 0, 4])
        self.assertEqual(table(s)['names'], ['ar.L1', 'ar.S.L4', 'sigma2'])
        x = [math.sin(i / 3.0) for i in range(120)]
        e = call('sm_arima', {'columns': {'s': D['series'], 'x': x}, 'series': 's', 'exog': ['x'], 'order': [1, 0, 0], 'trend': 'n', 'forecast_steps': 3, 'exog_future': {'x': [0.1, 0.2, 0.3]}})
        self.assertEqual(table(e)['names'], ['x', 'ar.L1', 'sigma2'])
        self.assertEqual(len(e['forecast']['mean']), 3)
        call('sm_arima', {'columns': {'s': D['series'], 'x': x}, 'series': 's', 'exog': ['x'], 'order': [1, 0, 0], 'trend': 'n', 'forecast_steps': 3}, 2, NO_PYTHON)

    def test_start_params_and_statuses(self):
        r = self.arima(order=[1, 0, 0], start_params=[0.5, 1.0])
        self.assertEqual(r['status'], 'OK')
        r = self.arima(order=[1, 0, 1], start_params=[0.5, 0.1, 1.0], maxiter=1)
        self.assertEqual(r['status'], 'NO_CONVERGENCE')
        self.assertFalse(r['converged'])
        bad = call('sm_arima', {'columns': {'s': D['series']}, 'series': 's', 'order': [1, 0, 0], 'start_params': [0.5]}, 2)
        self.assertEqual(bad['error']['data']['tool_code'], 'BAD_START_PARAMS')
        self.assertIn('2', bad['error']['message'])

    def test_missing_values_in_the_series_are_modelled_not_hidden(self):
        s = list(D['series'])
        s[10] = None
        s[50] = 'NaN'
        r = self.arima(series=s, order=[1, 0, 0])
        self.assertEqual((r['status'], r['nobs'], r['n_missing']), ('OK', 120, 2))

    def test_time_limit(self):
        n = 3000
        rng = np.random.default_rng(1)
        y = np.cumsum(rng.standard_normal(n)).tolist()
        r = call('sm_arima', {'csv': self.csv_series(y), 'series': 's', 'order': [2, 1, 2], 'seasonal_order': [1, 1, 1, 12], 'trend': 'n', 'maxiter': 1000}, 0, dict(os.environ, PYCLI_TIMEOUT_S='11'))
        self.assertEqual(r['status'], 'TIME_LIMIT')
        self.assertNotIn('coefficients', r)

    def csv_series(self, y):
        path = self.dir / 's.csv'
        path.write_text('s\n' + '\n'.join(repr(v) for v in y) + '\n')
        return str(path)

    def test_to_twin(self):
        out = self.out()
        r = call('sm_arima_to', {'columns': {'s': D['series']}, 'series': 's', 'order': [1, 0, 1], 'trend': 'n', 'forecast_steps': 20, 'out': out})
        t = read_safetensors(out)
        self.assertEqual(sorted(t), ['forecast', 'residuals'])
        self.assertEqual(t['forecast'].shape, (3, 20))  # mean, lower, upper
        self.assertEqual(t['residuals'].shape, (120,))
        self.assertNotIn('forecast', r)
        self.assertEqual(r['tensors']['forecast']['shape'], [3, 20])

    def test_forecast_cap_inline_and_validation(self):
        r = call('sm_arima', {'columns': {'s': D['series']}, 'series': 's', 'order': [1, 0, 0], 'forecast_steps': 101}, 2, NO_PYTHON)
        self.assertEqual(r['error']['data']['tool_code'], 'BAD_PARAM')
        base = {'columns': {'s': D['series']}, 'series': 's', 'order': [1, 0, 0]}
        bad = {'order_len': {'order': [1, 0]}, 'order_neg': {'order': [-1, 0, 0]}, 'p_cap': {'order': [6, 0, 0]}, 'd_cap': {'order': [1, 3, 0]}, 'order_float': {'order': [1.5, 0, 0]},
               'seasonal_len': {'seasonal_order': [1, 0, 0]}, 'season_1': {'seasonal_order': [1, 0, 0, 1]}, 'season_cap': {'seasonal_order': [1, 0, 0, 25]}, 'seasonal_P': {'seasonal_order': [3, 0, 0, 4]},
               'state_cap': {'order': [5, 2, 5], 'seasonal_order': [2, 1, 2, 24]}, 'trend': {'trend': 'ctt'}, 'method': {'method': 'newton'}, 'maxiter': {'maxiter': 0}, 'maxiter_big': {'maxiter': 1001},
               'alpha': {'alpha': 1.5}, 'start_nan': {'start_params': ['NaN', 1]}, 'start_long': {'start_params': [0.1] * 33}, 'exog_name': {'exog': ['a b\n']}, 'exog_many': {'exog': [f'x{i}' for i in range(9)]},
               'exog_future_missing_key': {'exog': ['x'], 'forecast_steps': 2, 'exog_future': {'z': [1, 2]}}, 'exog_future_len': {'exog': ['x'], 'forecast_steps': 2, 'exog_future': {'x': [1.0]}},
               'forecast_negative': {'forecast_steps': -1}, 'series_name': {'series': ''}}
        for name, extra in bad.items():
            with self.subTest(case=name):
                call('sm_arima', dict(base, **extra), 2, NO_PYTHON)
        ok = call('sm_arima_to', dict(base, forecast_steps=1000, out=self.out('big.safetensors')), 0, os.environ)
        self.assertEqual(read_safetensors(self.out('big.safetensors'))['forecast'].shape, (3, 1000))
        call('sm_arima_to', dict(base, forecast_steps=1001, out=self.out('big2.safetensors')), 2, NO_PYTHON)

    def test_patsy_is_never_loaded(self):
        r = self.arima(order=[1, 0, 0])
        self.assertEqual(r['meta']['forbidden_modules_loaded'], [])


class Acf(Base):
    def test_acf_and_pacf_against_r(self):
        a = call('sm_tsa_acf', {'columns': {'s': D['series']}, 'series': 's', 'kind': 'acf', 'nlags': 12})
        close(self, a['values'], R['acf'], 1e-10, 1e-14)
        self.assertEqual(a['lags'], list(range(13)))
        p = call('sm_tsa_acf', {'columns': {'s': D['series']}, 'series': 's', 'kind': 'pacf', 'nlags': 12})
        # R's pacf has lags 1..12; statsmodels includes lag 0 (=1)
        close(self, p['values'][1:], R['pacf'], 1e-9, 1e-12)
        self.assertEqual(p['values'][0], 1.0)

    def test_acf_by_hand_and_confidence_interval(self):
        y = np.array(D['series'])
        n = len(y)
        d = y - y.mean()
        c0 = float(np.sum(d * d) / n)
        by_hand = [float(np.sum(d[k:] * d[:n - k]) / n) / c0 for k in range(6)]
        a = call('sm_tsa_acf', {'columns': {'s': D['series']}, 'series': 's', 'kind': 'acf', 'nlags': 5, 'alpha': 0.05})
        close(self, a['values'], by_hand, 1e-12)
        # Bartlett: se_k = sqrt((1 + 2 sum_{j<k} r_j^2) / n); statsmodels returns intervals centred on the estimate for lags >= 1
        for k in range(1, 6):
            se = math.sqrt((1 + 2 * sum(by_hand[j] ** 2 for j in range(1, k))) / n)
            self.assertAlmostEqual(a['ci_upper'][k] - a['values'][k], 1.959963984540054 * se, delta=1e-9)
            self.assertAlmostEqual(a['values'][k] - a['ci_lower'][k], 1.959963984540054 * se, delta=1e-9)
        adjusted = call('sm_tsa_acf', {'columns': {'s': D['series']}, 'series': 's', 'kind': 'acf', 'nlags': 3, 'adjusted': True})
        by_hand_adj = [float(np.sum(d[k:] * d[:n - k]) / (n - k)) / float(np.sum(d * d) / n) for k in range(4)]
        close(self, adjusted['values'], by_hand_adj, 1e-12)

    def test_pacf_of_an_ar1_series_is_cut_off(self):
        rng = np.random.default_rng(3)
        x = [0.0]
        for _ in range(999):
            x.append(0.7 * x[-1] + rng.standard_normal())
        csv = self.dir / 'a.csv'
        csv.write_text('s\n' + '\n'.join(repr(v) for v in x) + '\n')
        p = call('sm_tsa_acf', {'csv': str(csv), 'series': 's', 'kind': 'pacf', 'nlags': 5})
        self.assertAlmostEqual(p['values'][1], 0.7, delta=0.08)
        self.assertTrue(all(abs(v) < 0.12 for v in p['values'][2:]))

    def test_degenerate_input_and_caps(self):
        r = call('sm_tsa_acf', {'columns': {'s': [1.0] * 30}, 'series': 's', 'kind': 'acf', 'nlags': 3}, 2)
        self.assertEqual(r['error']['data']['tool_code'], 'CONSTANT_SERIES')
        r = call('sm_tsa_acf', {'columns': {'s': [1.0, 2.0, 3.0, 5.0, 4.0, 7.0]}, 'series': 's', 'kind': 'acf', 'nlags': 6}, 2)
        self.assertEqual(r['error']['data']['tool_code'], 'BAD_PARAM')
        for extra in ({'nlags': 0}, {'nlags': 101}, {'kind': 'ccf'}, {'alpha': 0}, {'method': 'bogus'}):
            call('sm_tsa_acf', dict({'columns': {'s': D['series']}, 'series': 's', 'kind': 'acf', 'nlags': 5}, **extra), 2, NO_PYTHON)
        r = call('sm_tsa_acf', {'columns': {'s': [1.0, 'NaN'] + D['series'][:20]}, 'series': 's', 'kind': 'acf', 'nlags': 3}, 2)
        self.assertEqual(r['error']['data']['tool_code'], 'MISSING_VALUES')

    def test_patsy_is_never_loaded(self):
        r = call('sm_tsa_acf', {'columns': {'s': D['series']}, 'series': 's', 'kind': 'acf', 'nlags': 3})
        self.assertEqual(r['meta']['forbidden_modules_loaded'], [])


class Isolation(unittest.TestCase):
    """Fresh interpreter, the worker's own import phase and guard: patsy/formulaic never load and cannot be imported."""

    def test_no_sm_operation_loads_or_can_load_patsy(self):
        code = r'''
import sys, json
sys.path.insert(0, %r)
import s3, s5
w = s3.load_worker()
import numpy, scipy.linalg, scipy.stats, pandas
w.sm_import()
assert 'patsy' not in sys.modules and 'formulaic' not in sys.modules, 'import phase loaded patsy'
for module in list(sys.modules):
    if module.split('.')[0] in w.BLOCKED_MODULES:
        sys.modules.pop(module, None)
sys.meta_path.insert(0, w.BlockedImports())
sys.addaudithook(w.audit)
D = s5.DATA
common = {'alpha': 0.05, 'na_action': 'error', 'time_budget_s': 30}
ols = dict(common, columns=s5.columns('y', 'x1', 'x2', 'g'), y='y', terms=%r, cov_type='nonrobust')
glm = dict(common, columns=s5.columns('count', 'x1'), y='count', terms=[{'var': 'x1'}], family='poisson', link='log', maxiter=100, tol=1e-8, cov_type='nonrobust')
arima = dict(common, columns={'s': D['series']}, series='s', order=[1, 0, 0], trend='n', enforce_stationarity=True, enforce_invertibility=True, method='lbfgs', maxiter=100, forecast_steps=2)
acf = dict(common, columns={'s': D['series']}, series='s', kind='pacf', nlags=5, adjusted=False, method='ywm')
for name, params in (('sm_ols', ols), ('sm_glm', glm), ('sm_arima', arima), ('sm_tsa_acf', acf)):
    result = w.sm_operation(name)(params)
    assert result['status'] == 'OK', (name, result)
    leaked = [m for m in sys.modules if m.split('.')[0] in ('patsy', 'formulaic') or m == 'statsmodels.formula.api']
    assert not leaked, (name, leaked)
for blocked in ('patsy', 'formulaic', 'statsmodels.formula.api', 'patsy.util'):
    try:
        __import__(blocked)
    except ImportError:
        pass
    else:
        raise AssertionError(blocked + ' imported')
assert 'patsy' not in sys.modules
print('ok')
''' % (str(ROOT / 'tests/oracle'), REG_TERMS)
        proc = subprocess.run([sys.executable, '-P', '-s', '-B', '-c', code], capture_output=True, text=True, cwd=str(ROOT))
        self.assertEqual((proc.returncode, proc.stdout.strip()), (0, 'ok'), proc.stderr[-2000:])

    def test_worker_source_has_no_formula_or_patsy_import(self):
        import re
        source = ''.join((ROOT / 'py' / name).read_text() for name in ('sm.py', 'worker.py', 'numerics.py'))
        self.assertEqual(re.findall(r'^\s*(?:import|from)\s+(?:patsy|formulaic)\b', source, flags=re.M), [])
        self.assertNotIn('statsmodels.formula', ''.join(l for l in (ROOT / 'py/sm.py').read_text().splitlines(True) if not l.lstrip().startswith(('#', '"""'))).replace("'statsmodels.formula.api'", ''))
        self.assertNotRegex(source, r'from_formula|\bsmf\b|\bdmatrices?\b')


class Schema(unittest.TestCase):
    def test_ops_registered_read_write_split(self):
        schema = json.loads(subprocess.run([BIN, '--schema'], capture_output=True, text=True).stdout)
        ops = {o['name']: o for o in schema['operations']}
        for name in ('sm_ols', 'sm_glm', 'sm_arima', 'sm_tsa_acf'):
            self.assertEqual(ops[name]['access'], 'read', name)
        for name in ('sm_ols_to', 'sm_glm_to', 'sm_arima_to'):
            self.assertEqual(ops[name]['access'], 'write', name)
        for name in ('sm_ols', 'sm_glm', 'sm_arima'):
            self.assertIn('NO_CONVERGENCE' if name != 'sm_ols' else 'RANK_DEFICIENT', ops[name]['description'])
        self.assertNotIn('formula', json.dumps({k: v for k, v in ops.items() if k.startswith('sm_')}).replace('no formula', '').replace('formula strings', ''))


if __name__ == '__main__':
    unittest.main()
