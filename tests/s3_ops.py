"""S3 opt_minimize / opt_least_squares / num_quad / num_root against independent oracles:
textbook optima (Rosenbrock, Himmelblau, scipy-tutorial constrained problem), NIST Misra1a,
closed-form least squares, exact quadrature values and scipy called directly with test-side lambdas."""
import json, math, os, pathlib, subprocess, sys, tempfile, unittest
import numpy as np
from scipy import integrate, optimize
BIN = str(pathlib.Path(sys.argv.pop(1)).resolve())
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests/oracle'))
import s3
from s1 import read_safetensors, write_safetensors

NO_PYTHON = dict(os.environ, PYCLI_PYTHON='/no/such/python')


def call(op, params, rc=0, env=None):
    p = subprocess.run([BIN, '--input-json', json.dumps({'op': op, 'params': params})], capture_output=True, text=True, env=env)
    assert p.returncode == rc, (rc, p.returncode, p.stdout[:800], p.stderr[:800])
    return json.loads(p.stdout)


def rejected(op, params, code=None, rc=2):
    """Validation failures happen in C++ before any process starts: a nonexistent interpreter must not matter."""
    r = call(op, params, rc, NO_PYTHON)
    if code is not None:
        assert r['error']['data']['tool_code'] == code, (code, r)
    return r


def par_array(result, *names):
    return np.array([result['par'][n] for n in names])


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir='.')
        self.dir = pathlib.Path(self.tmp.name).resolve()
        self.addCleanup(self.tmp.cleanup)


class MinimizeOptima(Base):
    ROSEN = '100*(y-x^2)^2 + (1-x)^2'

    def test_rosenbrock_all_unconstrained_methods(self):
        for method, tol, solver_tol in (('BFGS', 1e-4, 1e-6), ('L-BFGS-B', 1e-4, 1e-10), ('Nelder-Mead', 1e-5, 1e-10), ('Powell', 1e-5, 1e-10), ('SLSQP', 1e-5, 1e-10), ('trust-constr', 1e-4, 1e-10)):
            with self.subTest(method=method):
                r = call('opt_minimize', {'objective': self.ROSEN, 'x0': {'x': -1.2, 'y': 1.0}, 'method': method, 'tol': solver_tol}, 0, os.environ)
                self.assertEqual(r['status'], 'OK', r)
                self.assertEqual(r['method'], method)
                np.testing.assert_allclose(par_array(r, 'x', 'y'), [1.0, 1.0], atol=tol)
                self.assertLess(r['fun'], 1e-6)
                self.assertGreater(r['nfev'], 5)

    def test_matches_scipy_called_directly_with_a_python_lambda(self):
        ref = optimize.minimize(lambda v: s3.rosenbrock(*v), [-1.2, 1.0], method='BFGS', tol=1e-6)
        r = call('opt_minimize', {'objective': self.ROSEN, 'x0': {'x': -1.2, 'y': 1.0}, 'method': 'BFGS', 'tol': 1e-6}, 0, os.environ)
        np.testing.assert_allclose(par_array(r, 'x', 'y'), ref.x, atol=1e-7)
        self.assertAlmostEqual(r['fun'], ref.fun, delta=1e-12)

    def test_himmelblau_reaches_all_four_minima(self):
        starts = [(2.0, 2.5), (-3.0, 2.5), (-3.0, -3.0), (3.0, -2.0)]
        found = []
        for x0 in starts:
            r = call('opt_minimize', {'objective': '(x^2+y-11)^2 + (x+y^2-7)^2', 'x0': {'x': x0[0], 'y': x0[1]}, 'method': 'BFGS', 'tol': 1e-7}, 0, os.environ)
            self.assertEqual(r['status'], 'OK', r)
            x, y = r['par']['x'], r['par']['y']
            self.assertLess(r['fun'], 1e-8)
            gx, gy = s3.himmelblau_grad(x, y)  # independent first-order optimality check
            self.assertLess(math.hypot(gx, gy), 1e-3)
            nearest = min(range(4), key=lambda k: math.hypot(x - s3.HIMMELBLAU_MINIMA[k][0], y - s3.HIMMELBLAU_MINIMA[k][1]))
            self.assertLess(math.hypot(x - s3.HIMMELBLAU_MINIMA[nearest][0], y - s3.HIMMELBLAU_MINIMA[nearest][1]), 1e-4)
            found.append(nearest)
        self.assertEqual(sorted(found), [0, 1, 2, 3])

    def test_equality_constrained_textbook_problem(self):
        for method in ('SLSQP', 'trust-constr'):
            with self.subTest(method=method):
                r = call('opt_minimize', {'objective': 'x^2+y^2', 'x0': {'x': 2.0, 'y': -1.0}, 'method': method, 'constraints': [{'type': 'eq', 'expr': 'x+y-1'}], 'tol': 1e-10}, 0, os.environ)
                self.assertEqual(r['status'], 'OK', r)
                np.testing.assert_allclose(par_array(r, 'x', 'y'), [0.5, 0.5], atol=1e-5)
                self.assertAlmostEqual(r['fun'], 0.5, delta=1e-8)
                self.assertLess(r['constraint_violation'], 1e-8)

    def test_scipy_tutorial_inequality_problem(self):
        # min (x-1)^2 + (y-2.5)^2 s.t. x-2y+2>=0, -x-2y+6>=0, -x+2y+2>=0, x,y>=0  ->  (1.4, 1.7), f = 0.8
        constraints = [{'type': 'ineq', 'expr': 'x-2*y+2'}, {'type': 'ineq', 'expr': '-x-2*y+6'}, {'type': 'ineq', 'expr': '-x+2*y+2'}]
        for method in ('SLSQP', 'trust-constr'):
            with self.subTest(method=method):
                r = call('opt_minimize', {'objective': '(x-1)^2+(y-2.5)^2', 'x0': {'x': 2.0, 'y': 0.0}, 'method': method, 'constraints': constraints,
                                          'bounds': {'x': [0, None], 'y': [0, 'Inf']}, 'tol': 1e-10}, 0, os.environ)
                self.assertEqual(r['status'], 'OK', r)
                np.testing.assert_allclose(par_array(r, 'x', 'y'), [1.4, 1.7], atol=1e-5)
                self.assertAlmostEqual(r['fun'], 0.8, delta=1e-8)
                self.assertLess(r['constraint_violation'], 1e-7)

    def test_box_bounds(self):
        for method in ('L-BFGS-B', 'SLSQP', 'Powell', 'Nelder-Mead', 'trust-constr'):
            with self.subTest(method=method):
                r = call('opt_minimize', {'objective': '(x-3)^2+(y+1)^2', 'x0': {'x': 1.0, 'y': 1.0}, 'method': method, 'bounds': {'x': [0, 2], 'y': [0, 5]}, 'tol': 1e-10}, 0, os.environ)
                self.assertEqual(r['status'], 'OK', r)
                np.testing.assert_allclose(par_array(r, 'x', 'y'), [2.0, 0.0], atol=1e-4)
                self.assertAlmostEqual(r['fun'], 2.0, delta=1e-3 if method == 'trust-constr' else 1e-6)
        # default method follows the problem: bounds -> L-BFGS-B, constraints -> SLSQP, otherwise BFGS
        self.assertEqual(call('opt_minimize', {'objective': '(x-3)^2', 'x0': {'x': 1.0}, 'bounds': {'x': [0, 2]}}, 0, os.environ)['method'], 'L-BFGS-B')
        self.assertEqual(call('opt_minimize', {'objective': '(x-3)^2', 'x0': {'x': 1.0}, 'constraints': [{'type': 'ineq', 'expr': '2-x'}]}, 0, os.environ)['method'], 'SLSQP')
        self.assertEqual(call('opt_minimize', {'objective': '(x-3)^2', 'x0': {'x': 1.0}}, 0, os.environ)['method'], 'BFGS')

    def test_vector_variables_with_literal_indexing(self):
        n = 4
        objective = ' + '.join(f'100*(x[{i + 1}]-x[{i}]^2)^2 + (1-x[{i}])^2' for i in range(n - 1))
        r = call('opt_minimize', {'objective': objective, 'x0': {'x': [-1.2, 1.0, -1.2, 1.0]}, 'method': 'L-BFGS-B', 'tol': 1e-12, 'maxiter': 2000}, 0, os.environ)
        self.assertEqual(r['status'], 'OK', r)
        self.assertEqual(len(r['par']['x']), n)
        np.testing.assert_allclose(r['par']['x'], np.ones(n), atol=1e-4)
        mixed = call('opt_minimize', {'objective': '(a[0]-1)^2 + (a[1]-2)^2 + (s-3)^2', 'x0': {'a': [0.0, 0.0], 's': 0.0}, 'method': 'BFGS'}, 0, os.environ)
        np.testing.assert_allclose(mixed['par']['a'] + [mixed['par']['s']], [1.0, 2.0, 3.0], atol=1e-5)
        self.assertIsInstance(mixed['par']['s'], float)

    def test_data_columns_with_reducers(self):
        t = np.linspace(0, 3, 25)
        y = 2.0 * np.exp(-0.8 * t)
        r = call('opt_minimize', {'objective': 'sum((y-a*exp(-b*t))^2)', 'x0': {'a': 1.0, 'b': 1.0}, 'data': {'t': t.tolist(), 'y': y.tolist()}, 'method': 'BFGS', 'tol': 1e-7}, 0, os.environ)
        self.assertEqual(r['status'], 'OK', r)
        np.testing.assert_allclose(par_array(r, 'a', 'b'), [2.0, 0.8], atol=1e-5)
        r = call('opt_minimize', {'objective': 'y-a', 'x0': {'a': 1.0}, 'data': {'y': [1.0, 2.0]}}, 2, os.environ)
        self.assertEqual(r['error']['data']['tool_code'], 'EXPR_NOT_SCALAR')

    def test_gradient_expressions_are_used_and_validated(self):
        plain = call('opt_minimize', {'objective': self.ROSEN, 'x0': {'x': -1.2, 'y': 1.0}, 'method': 'BFGS', 'tol': 1e-10}, 0, os.environ)
        grad = ['-400*x*(y-x^2) - 2*(1-x)', '200*(y-x^2)']  # variable order: x, y (sorted by name)
        exact = call('opt_minimize', {'objective': self.ROSEN, 'x0': {'x': -1.2, 'y': 1.0}, 'method': 'BFGS', 'tol': 1e-10, 'gradient': grad}, 0, os.environ)
        self.assertEqual(exact['status'], 'OK')
        np.testing.assert_allclose(par_array(exact, 'x', 'y'), [1, 1], atol=1e-7)
        self.assertLess(exact['nfev'], plain['nfev'])  # no finite-difference evaluations are charged to the objective
        self.assertEqual([w for w in exact['warnings'] if 'gradient' in w['message']], [])
        bad = call('opt_minimize', {'objective': self.ROSEN, 'x0': {'x': -1.2, 'y': 1.0}, 'method': 'BFGS', 'gradient': ['2*x', '2*y']}, 0, os.environ)
        self.assertTrue(any(w['category'] == 'NumericalWarning' and 'gradient' in w['message'] for w in bad['warnings']), bad['warnings'])
        rejected('opt_minimize', {'objective': self.ROSEN, 'x0': {'x': 1.0, 'y': 1.0}, 'gradient': ['2*x']}, 'GRADIENT_ARITY')
        rejected('opt_minimize', {'objective': self.ROSEN, 'x0': {'x': 1.0, 'y': 1.0}, 'gradient': ['2*x', '2*y', '1']}, 'GRADIENT_ARITY')
        rejected('opt_minimize', {'objective': self.ROSEN, 'x0': {'x': 1.0, 'y': 1.0}, 'gradient': ['2*x', 'z']}, 'EXPR_UNKNOWN_IDENT')
        rejected('opt_minimize', {'objective': self.ROSEN, 'x0': {'x': 1.0, 'y': 1.0}, 'gradient': ['2*x', 'sum(x)']}, 'EXPR_NOT_SCALAR')
        rejected('opt_minimize', {'objective': self.ROSEN, 'x0': {'x': 1.0, 'y': 1.0}, 'method': 'Nelder-Mead', 'gradient': ['2*x', '2*y']}, 'BAD_PARAM')
        # vector variable: one gradient expression per component
        r = call('opt_minimize', {'objective': '(v[0]-1)^2+(v[1]-2)^2', 'x0': {'v': [0.0, 0.0]}, 'method': 'BFGS', 'gradient': ['2*(v[0]-1)', '2*(v[1]-2)']}, 0, os.environ)
        np.testing.assert_allclose(r['par']['v'], [1.0, 2.0], atol=1e-8)
        rejected('opt_minimize', {'objective': '(v[0]-1)^2', 'x0': {'v': [0.0, 0.0]}, 'method': 'BFGS', 'gradient': ['2*(v[0]-1)']}, 'GRADIENT_ARITY')


class MinimizeStatuses(Base):
    ROSEN = MinimizeOptima.ROSEN

    def test_iteration_limit_is_not_ok(self):
        r = call('opt_minimize', {'objective': self.ROSEN, 'x0': {'x': -1.2, 'y': 1.0}, 'method': 'Nelder-Mead', 'maxiter': 5}, 0, os.environ)
        self.assertEqual(r['status'], 'NO_CONVERGENCE')
        self.assertTrue(r['message'])
        self.assertTrue(all(math.isfinite(v) for v in r['par'].values()))
        self.assertAlmostEqual(r['fun'], s3.rosenbrock(r['par']['x'], r['par']['y']), delta=1e-9)

    def test_eval_cap_returns_best_point_with_eval_limit(self):
        r = call('opt_minimize', {'objective': self.ROSEN, 'x0': {'x': -1.2, 'y': 1.0}, 'method': 'Nelder-Mead', 'max_evals': 30}, 0, os.environ)
        self.assertEqual(r['status'], 'EVAL_LIMIT')
        self.assertLessEqual(r['evals'], 31)
        self.assertLess(r['fun'], s3.rosenbrock(-1.2, 1.0))  # best-so-far, never worse than the start
        self.assertAlmostEqual(r['fun'], s3.rosenbrock(r['par']['x'], r['par']['y']), delta=1e-9)

    def test_wall_clock_budget_returns_time_limit(self):
        t = np.linspace(0, 1, 200000)
        np.save(self.dir / 't.npy', t)
        heavy = 'sum(' + '+'.join(['exp(t*a)*cos(t*a)'] * 14) + ')'
        env = dict(os.environ, PYCLI_TIMEOUT_S='11')  # worker budget = 1 s
        r = call('opt_minimize', {'objective': heavy, 'x0': {'a': 0.1}, 'data': {'t': str(self.dir / 't.npy')}, 'method': 'Nelder-Mead', 'tol': 1e-300, 'maxiter': 10000}, 0, env)
        self.assertEqual(r['status'], 'TIME_LIMIT')
        self.assertTrue(math.isfinite(r['fun']))

    def test_nonfinite_values_follow_the_policy(self):
        objective = 'ifelse(x>=1, (x-1)^2+x, 0/0)'
        r = call('opt_minimize', {'objective': objective, 'x0': {'x': 3.0}, 'method': 'Nelder-Mead', 'tol': 1e-12}, 0, os.environ)
        self.assertGreater(r['nonfinite_evals'], 0)
        self.assertAlmostEqual(r['par']['x'], 1.0, delta=1e-3)
        self.assertTrue(any(w['category'] == 'NumericalWarning' and 'non-finite' in w['message'] for w in r['warnings']))
        r = call('opt_minimize', {'objective': objective, 'x0': {'x': 3.0}, 'method': 'Nelder-Mead', 'nonfinite': 'error'}, 0, os.environ)
        self.assertEqual(r['status'], 'NON_FINITE')
        self.assertIn('x', r['par'])
        r = call('opt_minimize', {'objective': '1/(x-1)', 'x0': {'x': 1.0}}, 2, os.environ)
        self.assertEqual(r['error']['data']['tool_code'], 'NONFINITE_OBJECTIVE')
        r = call('opt_minimize', {'objective': 'log(x)', 'x0': {'x': -1.0}}, 2, os.environ)
        self.assertEqual(r['error']['data']['tool_code'], 'NONFINITE_OBJECTIVE')

    def test_infeasible_constraint_is_not_reported_ok(self):
        r = call('opt_minimize', {'objective': 'x^2', 'x0': {'x': 1.0}, 'method': 'SLSQP', 'constraints': [{'type': 'eq', 'expr': 'x-1'}, {'type': 'eq', 'expr': 'x+1'}]}, 0, os.environ)
        self.assertNotEqual(r['status'], 'OK')
        self.assertGreater(r['constraint_violation'], 0.5)

    def test_unbounded_objective_does_not_report_ok_with_garbage(self):
        r = call('opt_minimize', {'objective': '-x', 'x0': {'x': 0.0}, 'method': 'BFGS', 'maxiter': 50}, 0, os.environ)
        self.assertNotEqual(r['status'], 'OK')


class MinimizeValidation(Base):
    def test_param_validation_before_spawn(self):
        base = {'objective': '(x-1)^2', 'x0': {'x': 0.0}}
        cases = {
            'no_x0': ({'objective': '(x-1)^2'}, None), 'empty_x0': ({'objective': '1', 'x0': {}}, 'BAD_PARAM'),
            'bad_method': ({**base, 'method': 'COBYLA'}, None), 'eval_method': ({**base, 'method': 'eval'}, None),
            'nm_constraints': ({**base, 'method': 'Nelder-Mead', 'constraints': [{'type': 'eq', 'expr': 'x'}]}, 'BAD_PARAM'),
            'bfgs_bounds': ({**base, 'method': 'BFGS', 'bounds': {'x': [0, 1]}}, 'BAD_PARAM'),
            'lbfgs_constraints': ({**base, 'method': 'L-BFGS-B', 'constraints': [{'type': 'eq', 'expr': 'x'}]}, 'BAD_PARAM'),
            'constraint_type': ({**base, 'constraints': [{'type': 'le', 'expr': 'x'}]}, 'BAD_PARAM'),
            'constraint_extra_key': ({**base, 'constraints': [{'type': 'eq', 'expr': 'x', 'fun': 'x'}]}, 'BAD_PARAM'),
            'constraint_expr_bad': ({**base, 'constraints': [{'type': 'eq', 'expr': 'system(x)'}]}, 'EXPR_UNKNOWN_FUNCTION'),
            'too_many_constraints': ({**base, 'constraints': [{'type': 'ineq', 'expr': 'x'}] * 17}, 'BAD_PARAM'),
            'bounds_unknown_name': ({**base, 'bounds': {'y': [0, 1]}}, 'BAD_BOUNDS'), 'bounds_inverted': ({**base, 'bounds': {'x': [1, 0]}}, 'BAD_BOUNDS'),
            'bounds_len': ({**base, 'bounds': {'x': [0, 1, 2]}}, 'BAD_BOUNDS'), 'bounds_string': ({**base, 'bounds': {'x': [0, 'big']}}, 'BAD_BOUNDS'),
            'x0_outside': ({**base, 'bounds': {'x': [1, 2]}}, 'BAD_BOUNDS'), 'bounds_nan': ({**base, 'bounds': {'x': ['NaN', 1]}}, 'BAD_BOUNDS'),
            'x0_nan': ({'objective': 'x', 'x0': {'x': 'NaN'}}, 'BAD_PARAM'), 'x0_string': ({'objective': 'x', 'x0': {'x': 'a'}}, 'BAD_PARAM'),
            'x0_nested': ({'objective': 'x', 'x0': {'x': [[1]]}}, 'BAD_PARAM'), 'x0_empty_vec': ({'objective': 'x', 'x0': {'x': []}}, 'BAD_PARAM'),
            'x0_long_vec': ({'objective': 'x[0]', 'x0': {'x': [0.0] * 65}}, 'BAD_PARAM'),
            'too_many_vars': ({'objective': 'v0', 'x0': {f'v{i}': 0.0 for i in range(65)}}, 'BAD_PARAM'),
            'maxiter_cap': ({**base, 'maxiter': 10001}, 'BAD_PARAM'), 'maxiter_zero': ({**base, 'maxiter': 0}, 'BAD_PARAM'),
            'max_evals_cap': ({**base, 'max_evals': 2000001}, 'BAD_PARAM'), 'max_evals_zero': ({**base, 'max_evals': 0}, 'BAD_PARAM'),
            'tol_neg': ({**base, 'tol': -1}, 'BAD_PARAM'), 'tol_big': ({**base, 'tol': 2}, 'BAD_PARAM'), 'tol_zero': ({**base, 'tol': 0}, 'BAD_PARAM'),
            'nonfinite_enum': ({**base, 'nonfinite': 'zero'}, None), 'unknown_param': ({**base, 'seed': 1}, None),
            'data_collision': ({'objective': 'x', 'x0': {'x': 1.0}, 'data': {'x': [1.0]}}, 'EXPR_NAME_COLLISION'),
            'data_ragged': ({'objective': 'sum(a*b)', 'x0': {'z': 1.0}, 'data': {'a': [1.0, 2.0], 'b': [1.0]}}, 'BAD_DATA'),
            'data_name': ({'objective': 'x', 'x0': {'x': 1.0}, 'data': {'Y': [1.0]}}, 'BAD_NAME'),
            'data_too_many': ({'objective': 'x', 'x0': {'x': 1.0}, 'data': {f'c{i}': [1.0] for i in range(33)}}, 'BAD_PARAM'),
            'data_path': ({'objective': 'sum(d)', 'x0': {'x': 1.0}, 'data': {'d': '/etc/passwd'}}, None), 'data_inline_cap': ({'objective': 'sum(d)', 'x0': {'x': 1.0}, 'data': {'d': [1.0] * 1001}}, 'INPUT_TOO_LARGE'),
            'objective_long': ({'objective': 'x+' * 1100 + 'x', 'x0': {'x': 1.0}}, 'EXPR_TOO_LONG'),
            'vector_bare': ({'objective': 'v', 'x0': {'v': [1.0, 2.0]}}, 'EXPR_VECTOR_INDEX'), 'vector_oob': ({'objective': 'v[2]', 'x0': {'v': [1.0, 2.0]}}, 'EXPR_INDEX'),
            'scalar_indexed': ({'objective': 's[0]', 'x0': {'s': 1.0}}, 'EXPR_SYNTAX'),
        }
        for name, (params, code) in cases.items():
            with self.subTest(case=name):
                r = rejected('opt_minimize', params, code)
                self.assertFalse(r['success'])

    def test_unknown_order_is_sorted_by_name_not_by_json_key_order(self):
        # JSON objects are unordered: gradient expressions are matched to unknowns sorted by name, whatever the key order.
        x0_variants = [{'y': 1.0, 'x': -1.2}, {'x': -1.2, 'y': 1.0}]
        for x0 in x0_variants:
            r = call('opt_minimize', {'objective': MinimizeOptima.ROSEN, 'x0': x0, 'method': 'BFGS', 'tol': 1e-8, 'gradient': ['-400*x*(y-x^2) - 2*(1-x)', '200*(y-x^2)']}, 0, os.environ)
            self.assertEqual(r['status'], 'OK', (x0, r))
            np.testing.assert_allclose(par_array(r, 'x', 'y'), [1, 1], atol=1e-6)
            self.assertEqual([w for w in r['warnings'] if 'gradient' in w['message']], [])

    def test_x0_reported_in_sorted_name_order_with_vectors_flattened(self):
        r = call('opt_minimize', {'objective': '(b-1)^2+(a[0]-2)^2+(a[1]-3)^2+(c-4)^2', 'x0': {'c': 0.0, 'b': 0.0, 'a': [0.0, 0.0]}, 'method': 'BFGS'}, 0, os.environ)
        self.assertEqual(sorted(r['par']), ['a', 'b', 'c'])
        np.testing.assert_allclose(r['par']['a'] + [r['par']['b'], r['par']['c']], [2, 3, 1, 4], atol=1e-5)


class LeastSquares(Base):
    def test_misra1a_certified_values(self):
        for start in s3.MISRA1A_STARTS:
            for method in ('trf', 'lm', 'dogbox'):
                with self.subTest(start=start, method=method):
                    r = call('opt_least_squares', {'model': 'b1*(1-exp(-b2*x))', 'response': 'y', 'data': {'x': s3.MISRA1A_X, 'y': s3.MISRA1A_Y}, 'x0': start, 'method': method, 'tol': 1e-14}, 0, os.environ)
                    self.assertEqual(r['status'], 'OK', r)
                    for name in ('b1', 'b2'):
                        self.assertGreaterEqual(s3.lre(r['par'][name], s3.MISRA1A_CERTIFIED[name]), 6.0, (name, r['par']))
                        self.assertGreaterEqual(s3.lre(r['std_err'][name], s3.MISRA1A_SE[name]), 5.0, (name, r['std_err']))
                    self.assertGreaterEqual(s3.lre(r['rss'], s3.MISRA1A_RSS), 8.0)
                    self.assertAlmostEqual(r['cost'], 0.5 * s3.MISRA1A_RSS, delta=1e-9)
                    self.assertEqual((r['n_residuals'], r['n_parameters'], r['cov_status']), (14, 2, 'OK'))

    def test_exact_synthetic_exponential_decay_is_recovered(self):
        # Chwirut-style fixture with exactly known truth and no noise: parameters must be recovered to ~1e-10.
        t = np.linspace(0.5, 6.0, 40)
        y = 0.1 + 2.5 * np.exp(-0.7 * t)
        r = call('opt_least_squares', {'model': 'c+a*exp(-b*t)', 'response': 'y', 'data': {'t': t.tolist(), 'y': y.tolist()}, 'x0': {'a': 1.0, 'b': 0.2, 'c': 0.0}, 'tol': 1e-15}, 0, os.environ)
        self.assertEqual(r['status'], 'OK')
        np.testing.assert_allclose(par_array(r, 'a', 'b', 'c'), [2.5, 0.7, 0.1], rtol=1e-8)
        self.assertLess(r['cost'], 1e-18)

    def test_linear_fit_matches_closed_form_ols_with_standard_errors(self):
        rng = np.random.default_rng(5)
        x = np.arange(12.0)
        y = 2.0 + 3.0 * x + 0.1 * rng.standard_normal(12)
        X = np.column_stack([np.ones(12), x])
        beta = np.linalg.solve(X.T @ X, X.T @ y)
        s2 = float(np.sum((y - X @ beta) ** 2) / (12 - 2))
        se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
        r = call('opt_least_squares', {'model': 'a+b*x', 'response': 'y', 'data': {'x': x.tolist(), 'y': y.tolist()}, 'x0': {'a': 0.0, 'b': 0.0}, 'tol': 1e-15}, 0, os.environ)
        np.testing.assert_allclose(par_array(r, 'a', 'b'), beta, rtol=1e-7)
        np.testing.assert_allclose(par_array({'par': r['std_err']}, 'a', 'b'), se, rtol=1e-6)
        self.assertAlmostEqual(r['rss'], float(np.sum((y - X @ beta) ** 2)), delta=1e-10)

    def test_sigma_weights_and_absolute_sigma(self):
        x = np.arange(1.0, 9.0)
        y = np.array([2.1, 3.9, 6.2, 7.8, 10.3, 11.7, 14.2, 15.8])
        sigma = np.array([0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4])
        W = 1 / sigma ** 2
        X = np.column_stack([np.ones(8), x])
        cov = np.linalg.inv(X.T @ (X * W[:, None]))
        beta = cov @ (X.T @ (W * y))
        data = {'x': x.tolist(), 'y': y.tolist(), 's': sigma.tolist()}
        rel = call('opt_least_squares', {'model': 'a+b*x', 'response': 'y', 'sigma': 's', 'data': data, 'x0': {'a': 0.0, 'b': 0.0}, 'tol': 1e-15}, 0, os.environ)
        # The iterative solver stops where the weighted cost no longer decreases in binary64: the cost is known to about eps * chi2,
        # and along the weakest curvature direction of X'WX (smallest eigenvalue lambda_min = 87.55) that is a parameter error of
        # sqrt(eps * chi2 / lambda_min) = sqrt(2.2e-16 * 6.04 / 87.55) = 3.9e-9 absolute. The first amd64 CI run stopped 5.8e-10 away
        # on the intercept (rtol 1e-8 failed because the intercept is 0.057), aarch64 2.6e-11: both inside that floor. A solver that
        # ignores sigma lands 1e-2 away (scratch mutation), far outside it.
        chi2 = float(np.sum(((y - X @ beta) / sigma) ** 2))
        floor = math.sqrt(np.finfo(float).eps * chi2 / np.linalg.eigvalsh(X.T @ (X * W[:, None])).min())
        np.testing.assert_allclose(par_array(rel, 'a', 'b'), beta, rtol=0, atol=4 * floor)
        np.testing.assert_allclose([rel['std_err']['a'], rel['std_err']['b']], np.sqrt(np.diag(cov) * chi2 / 6), rtol=1e-6)
        absolute = call('opt_least_squares', {'model': 'a+b*x', 'response': 'y', 'sigma': 's', 'absolute_sigma': True, 'data': data, 'x0': {'a': 0.0, 'b': 0.0}, 'tol': 1e-15}, 0, os.environ)
        np.testing.assert_allclose([absolute['std_err']['a'], absolute['std_err']['b']], np.sqrt(np.diag(cov)), rtol=1e-6)
        scalar_sigma = call('opt_least_squares', {'model': 'a+b*x', 'response': 'y', 'sigma': 0.5, 'absolute_sigma': True, 'data': {'x': data['x'], 'y': data['y']}, 'x0': {'a': 0.0, 'b': 0.0}, 'tol': 1e-15}, 0, os.environ)
        cov_u = 0.25 * np.linalg.inv(X.T @ X)
        np.testing.assert_allclose([scalar_sigma['std_err']['a'], scalar_sigma['std_err']['b']], np.sqrt(np.diag(cov_u)), rtol=1e-6)

    def test_residual_expression_vector_solves_a_system(self):
        r = call('opt_least_squares', {'residuals': ['x^2+y^2-5', 'x-y-1', 'x*y-2'], 'x0': {'x': 3.0, 'y': 1.0}}, 0, os.environ)
        self.assertEqual(r['status'], 'OK')
        np.testing.assert_allclose(par_array(r, 'x', 'y'), [2.0, 1.0], atol=1e-8)
        self.assertEqual((r['n_residuals'], r['n_parameters']), (3, 2))
        # vector-valued residual expression over data columns, concatenated with a scalar prior
        t = np.arange(5.0)
        r = call('opt_least_squares', {'residuals': ['y-a*t', '10*(a-2)'], 'data': {'t': t.tolist(), 'y': (2 * t).tolist()}, 'x0': {'a': 0.0}}, 0, os.environ)
        self.assertEqual(r['n_residuals'], 6)
        self.assertAlmostEqual(r['par']['a'], 2.0, delta=1e-8)

    def test_bounds_and_methods(self):
        x = np.arange(1.0, 7.0)
        data = {'x': x.tolist(), 'y': (3 * x).tolist()}
        for method in ('trf', 'dogbox'):
            r = call('opt_least_squares', {'model': 'a*x', 'response': 'y', 'data': data, 'x0': {'a': 0.5}, 'bounds': {'a': [0, 2]}, 'method': method}, 0, os.environ)
            self.assertAlmostEqual(r['par']['a'], 2.0, delta=1e-8)
            self.assertEqual(r['status'], 'OK')
        r = call('opt_least_squares', {'model': 'a*x', 'response': 'y', 'data': data, 'x0': {'a': 0.5}, 'method': 'lm'}, 0, os.environ)
        self.assertAlmostEqual(r['par']['a'], 3.0, delta=1e-8)

    def test_robust_loss_matches_scipy_and_resists_an_outlier(self):
        x = np.arange(10.0)
        y = 2.0 * x + 1.0
        y[7] += 40.0
        data = {'x': x.tolist(), 'y': y.tolist()}
        ref = optimize.least_squares(lambda p: p[0] + p[1] * x - y, [0.0, 0.0], loss='soft_l1', f_scale=1.0, xtol=1e-14, ftol=1e-14, gtol=1e-14, x_scale='jac')
        r = call('opt_least_squares', {'model': 'a+b*x', 'response': 'y', 'data': data, 'x0': {'a': 0.0, 'b': 0.0}, 'loss': 'soft_l1', 'f_scale': 1.0, 'tol': 1e-14}, 0, os.environ)
        np.testing.assert_allclose(par_array(r, 'a', 'b'), ref.x, atol=1e-6)
        plain = call('opt_least_squares', {'model': 'a+b*x', 'response': 'y', 'data': data, 'x0': {'a': 0.0, 'b': 0.0}}, 0, os.environ)
        self.assertLess(abs(r['par']['b'] - 2.0), abs(plain['par']['b'] - 2.0))
        for loss in ('huber', 'cauchy', 'arctan'):
            self.assertEqual(call('opt_least_squares', {'model': 'a+b*x', 'response': 'y', 'data': data, 'x0': {'a': 0.0, 'b': 0.0}, 'loss': loss}, 0, os.environ)['loss'], loss)

    def test_degenerate_jacobian_and_no_degrees_of_freedom_are_reported(self):
        x = np.arange(6.0)
        data = {'x': x.tolist(), 'y': (1 + x).tolist()}
        r = call('opt_least_squares', {'model': 'a+b+0*x+x', 'response': 'y', 'data': data, 'x0': {'a': 0.0, 'b': 0.0}}, 0, os.environ)
        self.assertEqual(r['cov_status'], 'SINGULAR_JACOBIAN')
        self.assertEqual(r['std_err'], {'a': 'NA', 'b': 'NA'})
        r = call('opt_least_squares', {'model': 'a+b*x', 'response': 'y', 'data': {'x': [0.0, 1.0], 'y': [1.0, 3.0]}, 'x0': {'a': 0.0, 'b': 0.0}}, 0, os.environ)
        self.assertEqual(r['cov_status'], 'NO_DOF')
        self.assertEqual(r['std_err'], {'a': 'NA', 'b': 'NA'})
        np.testing.assert_allclose(par_array(r, 'a', 'b'), [1.0, 2.0], atol=1e-8)

    def test_budget_exhaustion_is_not_ok(self):
        r = call('opt_least_squares', {'model': 'b1*(1-exp(-b2*x))', 'response': 'y', 'data': {'x': s3.MISRA1A_X, 'y': s3.MISRA1A_Y}, 'x0': s3.MISRA1A_STARTS[0], 'max_nfev': 3}, 0, os.environ)
        self.assertEqual(r['status'], 'NO_CONVERGENCE')
        self.assertTrue(any('non-converged' in w['message'] for w in r['warnings']), r['warnings'])
        r = call('opt_least_squares', {'model': 'b1*(1-exp(-b2*x))', 'response': 'y', 'data': {'x': s3.MISRA1A_X, 'y': s3.MISRA1A_Y}, 'x0': s3.MISRA1A_STARTS[0], 'max_evals': 4}, 0, os.environ)
        self.assertEqual(r['status'], 'EVAL_LIMIT')
        self.assertTrue(all(math.isfinite(v) for v in r['par'].values()))

    def test_nonfinite_residuals(self):
        r = call('opt_least_squares', {'model': 'log(a)*x', 'response': 'y', 'data': {'x': [1.0, 2.0], 'y': [0.0, 0.0]}, 'x0': {'a': -1.0}}, 2, os.environ)
        self.assertEqual(r['error']['data']['tool_code'], 'NONFINITE_RESIDUAL')

    def test_large_data_through_refs(self):
        rng = np.random.default_rng(3)
        x = np.linspace(0, 1, 5000)
        y = 4.0 * x - 1.0 + 0.01 * rng.standard_normal(5000)
        np.save(self.dir / 'x.npy', x)
        write_safetensors(self.dir / 'd.safetensors', {'y': y})
        r = call('opt_least_squares', {'model': 'a*x+b', 'response': 'y', 'data': {'x': str(self.dir / 'x.npy'), 'y': str(self.dir / 'd.safetensors') + '#y'}, 'x0': {'a': 0.0, 'b': 0.0}}, 0, os.environ)
        beta = np.polyfit(x, y, 1)
        np.testing.assert_allclose(par_array(r, 'a', 'b'), beta, rtol=1e-7)
        self.assertEqual(r['n_residuals'], 5000)

    def test_validation_before_spawn(self):
        data = {'x': [0.0, 1.0, 2.0, 3.0], 'y': [1.0, 2.0, 3.0, 4.0]}
        base = {'model': 'a+b*x', 'response': 'y', 'data': data, 'x0': {'a': 0.0, 'b': 0.0}}
        cases = {
            'model_and_residuals': ({**base, 'residuals': ['x']}, 'BAD_PARAM'), 'neither': ({'x0': {'a': 0.0}, 'data': data}, 'BAD_PARAM'),
            'model_no_response': ({k: v for k, v in base.items() if k != 'response'}, 'BAD_PARAM'), 'response_unknown': ({**base, 'response': 'z'}, 'BAD_PARAM'),
            'response_in_residuals': ({'residuals': ['x'], 'response': 'y', 'data': data, 'x0': {'a': 0.0}}, 'BAD_PARAM'),
            'lm_bounds': ({**base, 'method': 'lm', 'bounds': {'a': [0, 1]}}, 'BAD_PARAM'), 'lm_loss': ({**base, 'method': 'lm', 'loss': 'huber'}, 'BAD_PARAM'),
            'lm_underdetermined': ({'model': 'a+b*x', 'response': 'y', 'data': {'x': [0.0], 'y': [1.0]}, 'x0': {'a': 0.0, 'b': 0.0}, 'method': 'lm'}, 'BAD_DATA'),
            'loss_enum': ({**base, 'loss': 'l1'}, None), 'method_enum': ({**base, 'method': 'exact'}, None), 'f_scale': ({**base, 'f_scale': 0}, 'BAD_PARAM'),
            'tol_low': ({**base, 'tol': 1e-17}, 'BAD_PARAM'), 'tol_high': ({**base, 'tol': 1}, 'BAD_PARAM'), 'max_nfev': ({**base, 'max_nfev': 100001}, 'BAD_PARAM'),
            'max_nfev_zero': ({**base, 'max_nfev': 0}, 'BAD_PARAM'), 'max_evals': ({**base, 'max_evals': 2000001}, 'BAD_PARAM'),
            'sigma_unknown': ({**base, 'sigma': 'zz'}, 'BAD_PARAM'), 'sigma_zero': ({**base, 'sigma': 0}, 'BAD_PARAM'), 'sigma_neg': ({**base, 'sigma': -1}, 'BAD_PARAM'),
            'sigma_col_zero': ({**base, 'sigma': 's', 'data': {**data, 's': [1.0, 0.0, 1.0, 1.0]}}, 'BAD_DATA'), 'sigma_type': ({**base, 'sigma': [1, 2]}, 'BAD_PARAM'),
            'ragged': ({**base, 'data': {'x': [0.0, 1.0], 'y': [1.0]}}, 'BAD_DATA'), 'nonfinite_data': ({**base, 'data': {'x': [0.0, 1.0], 'y': [1.0, 'NaN']}}, 'BAD_DATA'),
            'data_collision': ({**base, 'x0': {'x': 0.0, 'a': 0.0}}, 'EXPR_NAME_COLLISION'), 'model_reducer': ({**base, 'model': 'sum(x)'}, 'EXPR_NOT_SCALAR'),
            'model_hostile': ({**base, 'model': '__import__(x)'}, 'EXPR_SYNTAX'), 'residual_hostile': ({'residuals': ['x', 'exec(1)'], 'x0': {'x': 1.0}}, 'EXPR_UNKNOWN_FUNCTION'),
            'too_many_residuals': ({'residuals': ['x'] * 65, 'x0': {'x': 1.0}}, 'BAD_PARAM'), 'residuals_type': ({'residuals': [1], 'x0': {'x': 1.0}}, None),
            'empty_residuals': ({'residuals': [], 'x0': {'x': 1.0}}, 'BAD_PARAM'), 'bounds_bad': ({**base, 'bounds': {'a': [2, 1]}}, 'BAD_BOUNDS'),
            'x0_outside': ({**base, 'bounds': {'a': [1, 2]}}, 'BAD_BOUNDS'), 'no_x0': ({k: v for k, v in base.items() if k != 'x0'}, None),
        }
        for name, (params, code) in cases.items():
            with self.subTest(case=name):
                rejected('opt_least_squares', params, code)


class Quadrature(Base):
    def quad(self, f, a, b, **kw):
        return call('num_quad', {'f': f, 'a': a, 'b': b, **kw}, 0, os.environ)

    def test_exact_values(self):
        cases = [('exp(-x^2)', '-Inf', 'Inf', math.sqrt(math.pi), 1e-12), ('1/(1+x^2)', '-Inf', 'Inf', math.pi, 1e-12), ('1/(1+x^2)', 0, 1, math.pi / 4, 1e-14),
                 ('sin(x)', 0, math.pi, 2.0, 1e-14), ('log(x)', 0, 1, -1.0, 1e-12), ('1/sqrt(x)', 0, 1, 2.0, 1e-10), ('exp(-x)', 0, 'Inf', 1.0, 1e-12),
                 ('x^3 - 2*x', -1, 2, (2 ** 4 - 1) / 4 - (2 ** 2 - 1), 1e-13), ('exp(x)', 1, 0, -(math.e - 1), 1e-13), ('x', 2, 2, 0.0, 0.0),
                 ('exp(-x^2/2)/sqrt(2*pi)', '-Inf', 0, 0.5, 1e-12), ('abs(x - 1/3)', 0, 1, (1 / 3) ** 2 / 2 + (2 / 3) ** 2 / 2, 1e-8)]
        for f, a, b, expected, tol in cases:
            with self.subTest(f=f, a=a, b=b):
                r = self.quad(f, a, b)
                self.assertEqual(r['status'], 'OK', r)
                self.assertAlmostEqual(r['value'], expected, delta=max(tol, 1e-16))
                self.assertLessEqual(abs(r['value'] - expected), max(r['abserr'] * 10, tol))
                if a != b:
                    self.assertGreater(r['neval'], 0)

    def test_agrees_with_scipy_and_points_hint(self):
        ref = integrate.quad(lambda x: abs(x - 1 / 3), 0, 1, points=[1 / 3])
        r = self.quad('abs(x - 1/3)', 0, 1, points=[1 / 3])
        self.assertAlmostEqual(r['value'], ref[0], delta=1e-15)
        self.assertLess(r['abserr'], 1e-12)
        ref = integrate.quad(lambda x: math.exp(-x * x) * math.cos(3 * x), -math.inf, math.inf, epsabs=1e-12, epsrel=1e-12)
        r = self.quad('exp(-x^2)*cos(3*x)', '-Inf', 'Inf', epsabs=1e-12, epsrel=1e-12)
        self.assertAlmostEqual(r['value'], math.sqrt(math.pi) * math.exp(-9 / 4), delta=1e-11)
        self.assertAlmostEqual(r['value'], ref[0], delta=1e-12)

    def test_failures_are_reported_honestly(self):
        r = self.quad('sin(x)', 0, 'Inf')  # oscillatory and divergent: QUADPACK cannot certify it
        self.assertEqual(r['status'], 'NO_CONVERGENCE')
        self.assertTrue(r['message'])
        r = self.quad('1/x', 0, 1)
        self.assertNotEqual(r['status'], 'OK')
        r = self.quad('1/x', -1, 1)  # the pole sits on a Gauss-Kronrod node: f is infinite there
        self.assertEqual(r['status'], 'NON_FINITE')
        r = self.quad('sin(1000*x)', 0, 100, limit=5)
        self.assertEqual(r['status'], 'NO_CONVERGENCE')
        self.assertIn('message', r)
        r = self.quad('x^2', 0, 1, limit=2)  # one Gauss-Kronrod interval integrates a quadratic exactly (QUADPACK flags limit=1 itself)
        self.assertEqual(r['status'], 'OK')
        self.assertAlmostEqual(r['value'], 1 / 3, delta=1e-15)

    def test_eval_cap(self):
        r = self.quad('sin(1000*x)', 0, 100, limit=1000, max_evals=50)
        self.assertEqual(r['status'], 'EVAL_LIMIT')

    def test_validation_before_spawn(self):
        base = {'f': 'x', 'a': 0, 'b': 1}
        cases = {'a_nan': ({**base, 'a': 'NaN'}, 'BAD_PARAM'), 'b_string': ({**base, 'b': 'big'}, 'BAD_PARAM'), 'a_array': ({**base, 'a': [0]}, 'BAD_PARAM'), 'both_inf': ({**base, 'a': 'Inf', 'b': 'Inf'}, 'BAD_PARAM'),
                 'a_neg_inf_as_b': ({**base, 'b': '-Inf', 'a': '-Inf'}, 'BAD_PARAM'), 'limit_cap': ({**base, 'limit': 1001}, 'BAD_PARAM'), 'limit_zero': ({**base, 'limit': 0}, 'BAD_PARAM'),
                 'epsabs_neg': ({**base, 'epsabs': -1}, 'BAD_PARAM'), 'eps_both_zero': ({**base, 'epsabs': 0, 'epsrel': 0}, 'BAD_PARAM'), 'epsrel_big': ({**base, 'epsrel': 2}, 'BAD_PARAM'),
                 'points_infinite_range': ({**base, 'a': 0, 'b': 'Inf', 'points': [1]}, 'BAD_PARAM'), 'points_outside': ({**base, 'points': [5]}, 'BAD_PARAM'), 'points_many': ({**base, 'points': [0.5] * 65}, 'BAD_PARAM'),
                 'points_nan': ({**base, 'points': ['NaN']}, None), 'reducer': ({**base, 'f': 'sum(x)'}, 'EXPR_NOT_SCALAR'), 'other_var': ({**base, 'f': 'y'}, 'EXPR_UNKNOWN_IDENT'),
                 'hostile': ({**base, 'f': 'exp(x).__class__'}, 'EXPR_SYNTAX'), 'max_evals': ({**base, 'max_evals': 0}, 'BAD_PARAM'), 'no_f': ({'a': 0, 'b': 1}, None), 'unknown': ({**base, 'x0': 1}, None)}
        for name, (params, code) in cases.items():
            with self.subTest(case=name):
                rejected('num_quad', params, code)


class RootFinding(Base):
    def root(self, f, lower, upper, **kw):
        return call('num_root', {'f': f, 'lower': lower, 'upper': upper, **kw}, 0, os.environ)

    def test_known_roots_all_methods(self):
        for method in ('brentq', 'bisect', 'ridder', 'toms748'):
            for f, lo, hi, expected in (('x^2-2', 0, 2, math.sqrt(2)), ('cos(x)-x', 0, 1, 0.7390851332151607), ('exp(x)-10', 0, 5, math.log(10)), ('x^3-x-1', 1, 2, 1.324717957244746)):
                with self.subTest(method=method, f=f):
                    r = self.root(f, lo, hi, method=method, xtol=1e-14, maxiter=200)
                    self.assertEqual(r['status'], 'OK', r)
                    self.assertEqual(r['method'], method)
                    self.assertAlmostEqual(r['root'], expected, delta=1e-12)
                    self.assertLess(abs(r['f_root']), 1e-10)
                    self.assertGreater(r['function_calls'], 2)

    def test_matches_scipy_and_endpoint_roots(self):
        ref = optimize.brentq(lambda x: math.cos(x) - x, 0, 1)
        self.assertAlmostEqual(self.root('cos(x)-x', 0, 1)['root'], ref, delta=1e-12)
        r = self.root('x', 0, 1)
        self.assertEqual(r['root'], 0.0)
        r = self.root('x-1', 0, 1)
        self.assertAlmostEqual(r['root'], 1.0, delta=1e-12)

    def test_honest_failures(self):
        r = call('num_root', {'f': 'x^2+1', 'lower': -1, 'upper': 1}, 2, os.environ)
        self.assertEqual(r['error']['data']['tool_code'], 'BAD_INTERVAL')
        r = call('num_root', {'f': '1/x', 'lower': 0, 'upper': 1}, 2, os.environ)
        self.assertEqual(r['error']['data']['tool_code'], 'NONFINITE_BRACKET')
        r = self.root('x^2-2', 0, 2, method='bisect', maxiter=3)
        self.assertEqual(r['status'], 'NO_CONVERGENCE')
        r = self.root('x^2-2', 0, 2, max_evals=2)
        self.assertEqual(r['status'], 'EVAL_LIMIT')

    def test_validation_before_spawn(self):
        base = {'f': 'x-1', 'lower': 0, 'upper': 2}
        cases = {'order': ({**base, 'lower': 2, 'upper': 0}, 'BAD_PARAM'), 'equal': ({**base, 'lower': 1, 'upper': 1}, 'BAD_PARAM'), 'inf': ({**base, 'upper': 'Inf'}, None),
                 'maxiter': ({**base, 'maxiter': 1001}, 'BAD_PARAM'), 'maxiter0': ({**base, 'maxiter': 0}, 'BAD_PARAM'), 'xtol': ({**base, 'xtol': 0}, 'BAD_PARAM'), 'rtol': ({**base, 'rtol': 1e-16}, 'BAD_PARAM'),
                 'method': ({**base, 'method': 'newton'}, None), 'secant': ({**base, 'method': 'secant'}, None), 'reducer': ({**base, 'f': 'mean(x)'}, 'EXPR_NOT_SCALAR'), 'hostile': ({**base, 'f': 'system(x)'}, 'EXPR_UNKNOWN_FUNCTION')}
        for name, (params, code) in cases.items():
            with self.subTest(case=name):
                rejected('num_root', params, code)


class SchemaAndLimits(Base):
    def test_ops_are_read_access_and_documented(self):
        schema = json.loads(subprocess.run([BIN, '--schema'], capture_output=True, text=True).stdout)
        ops = {o['name']: o for o in schema['operations']}
        for name in ('opt_minimize', 'opt_least_squares', 'num_quad', 'num_root'):
            self.assertEqual(ops[name]['access'], 'read', name)
        self.assertIn('NO_CONVERGENCE', ops['opt_minimize']['description'])
        self.assertIn('EVAL_LIMIT', ops['opt_minimize']['description'])
        self.assertIn('std_err', ops['opt_least_squares']['description'])
        enum = {p['name']: p.get('enum') for p in ops['opt_minimize']['params']}
        self.assertEqual(enum['method'], ['Nelder-Mead', 'Powell', 'BFGS', 'L-BFGS-B', 'SLSQP', 'trust-constr'])

    def test_results_are_small_and_leave_no_files(self):
        # other test groups run in parallel and create their own temporary directories (tmpXXXXXXXX) in this working directory: only pycli's own files count
        listing = lambda: sorted(p.name for p in pathlib.Path('.').iterdir() if not p.name.startswith('tmp'))
        before = listing()
        r = call('opt_minimize', {'objective': '(a[0]-1)^2', 'x0': {'a': [0.0] * 64}, 'method': 'BFGS'}, 0, os.environ)
        self.assertEqual(len(r['par']['a']), 64)
        self.assertLess(len(json.dumps(r)), 8192)
        self.assertEqual(listing(), before)

    def test_readonly_policy_keeps_reads_allowed(self):
        r = call('num_quad', {'f': 'x', 'a': 0, 'b': 1}, 0, dict(os.environ, PYCLI_READONLY='1'))
        self.assertAlmostEqual(r['value'], 0.5, delta=1e-14)  # read operations stay allowed


class ReviewFixes(Base):
    """S3 review findings (Opus, Fable, Astra). Each test failed against 7fc6d7a."""
    HEAVY = 'sum(' + '+'.join(['exp(t*a[0])*cos(t*a[1])'] * 24) + ')'  # ~0.2 s per evaluation over 200000 rows

    def heavy_data(self):
        np.save(self.dir / 't.npy', np.linspace(0, 1, 200000))
        return {'t': str(self.dir / 't.npy')}

    def test_gradient_check_is_counted_and_budgeted(self):
        # The x0 gradient check used to evaluate the objective n+1 times outside the counter and the deadline.
        t0 = __import__('time').monotonic()
        r = call('opt_minimize', {'objective': self.HEAVY, 'x0': {'a': [0.1] * 64}, 'data': self.heavy_data(), 'method': 'BFGS',
                                  'gradient': ['0'] * 64, 'max_evals': 5}, 0, dict(os.environ, PYCLI_TIMEOUT_S='15'))
        self.assertEqual(r['status'], 'EVAL_LIMIT', r)
        self.assertLessEqual(r['evals'], 6)
        self.assertLess(__import__('time').monotonic() - t0, 14)
        r = call('opt_minimize', {'objective': self.HEAVY, 'x0': {'a': [0.1] * 64}, 'data': self.heavy_data(), 'method': 'BFGS',
                                  'gradient': ['0'] * 64}, 0, dict(os.environ, PYCLI_TIMEOUT_S='15'))
        self.assertEqual(r['status'], 'TIME_LIMIT', r)

    def test_budget_exhaustion_at_any_stage_is_a_domain_status(self):
        r = call('opt_minimize', {'objective': '(a-1)^2', 'x0': {'a': 0.0}, 'gradient': ['2*(a-1)'], 'max_evals': 1}, 0, os.environ)
        self.assertEqual(r['status'], 'EVAL_LIMIT')
        self.assertEqual(r['par'], {'a': 0.0})
        for budget in range(1, 80):
            r = call('opt_least_squares', {'model': 'a*exp(b*t)', 'response': 'y', 'data': {'t': [0.0, 0.5, 1.0, 1.5, 2.0], 'y': [1.0, 1.6, 2.7, 4.4, 7.4]},
                                           'x0': {'a': 1.0, 'b': 0.5}, 'max_evals': budget}, 0, os.environ)
            self.assertIn(r['status'], ('OK', 'EVAL_LIMIT'), (budget, r))
            self.assertLessEqual(r['evals'], budget)
            self.assertIn('std_err', r)

    def test_standard_errors_respect_the_deadline(self):
        # Post-solver evaluations (final residual, Jacobian steps) go through the same guard.
        r = call('opt_least_squares', {'model': 'a*exp(-b*t)', 'response': 'y', 'data': {'t': [0.0, 1.0, 2.0, 3.0], 'y': [2.0, 1.2, 0.7, 0.45]}, 'x0': {'a': 1.0, 'b': 0.1}, 'max_evals': 40}, 0, os.environ)
        self.assertLessEqual(r['evals'], 40)
        self.assertIn(r['status'], ('OK', 'EVAL_LIMIT'))

    def test_residual_count_times_unknowns_is_capped_before_spawn(self):
        np.save(self.dir / 'd.npy', np.linspace(0, 1, 200000))
        residuals = [f'd - a[{i % 16}]' for i in range(64)]
        r = rejected('opt_least_squares', {'residuals': residuals, 'x0': {'a': [1.0] * 16}, 'data': {'d': str(self.dir / 'd.npy')}}, 'INPUT_TOO_LARGE')
        self.assertIn('residual', r['error']['message'])
        # scalar residuals do not count as vectors; vector residuals with few unknowns are fine
        ok = call('opt_least_squares', {'residuals': ['d - a'], 'x0': {'a': 1.0}, 'data': {'d': str(self.dir / 'd.npy')}}, 0, os.environ)
        self.assertEqual(ok['n_residuals'], 200000)
        self.assertAlmostEqual(ok['par']['a'], 0.5, delta=1e-6)

    def test_standard_errors_for_near_zero_parameters_and_scaled_responses(self):
        for scale in (1000.0, 1e6, 1e-3):
            x = np.arange(1.0, 13.0)
            y = scale * x
            X = np.column_stack([np.ones(12), x])
            se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)))  # absolute_sigma, sigma = 1
            r = call('opt_least_squares', {'model': 'a+b*x', 'response': 'y', 'data': {'x': x.tolist(), 'y': y.tolist()}, 'x0': {'a': 0.0, 'b': 1.0},
                                           'absolute_sigma': True, 'tol': 1e-15}, 0, os.environ)
            with self.subTest(scale=scale):
                self.assertEqual(r['cov_status'], 'OK', r)
                np.testing.assert_allclose([r['std_err']['a'], r['std_err']['b']], se, rtol=1e-6)
        # a nonlinear parameter that is ~0 at the solution: reference from the analytic Jacobian
        t = np.linspace(0, 10, 25)
        y = 4.0 * np.exp(-1e-6 * t)
        J = np.column_stack([np.exp(-1e-6 * t), -4.0 * t * np.exp(-1e-6 * t)])
        cov = np.linalg.inv(J.T @ J)
        r = call('opt_least_squares', {'model': 'a*exp(-b*t)', 'response': 'y', 'data': {'t': t.tolist(), 'y': y.tolist()}, 'x0': {'a': 3.0, 'b': 0.05}, 'absolute_sigma': True, 'tol': 1e-15}, 0, os.environ)
        np.testing.assert_allclose([r['std_err']['a'], r['std_err']['b']], np.sqrt(np.diag(cov)), rtol=1e-5)

    def test_ill_conditioned_jacobian_is_flagged_not_silently_ok(self):
        x = np.linspace(1.0, 1.001, 20)
        y = 1 + 2 * x + 3 * x ** 2
        r = call('opt_least_squares', {'model': 'a+b*x+c*x^2', 'response': 'y', 'data': {'x': x.tolist(), 'y': y.tolist()}, 'x0': {'a': 0.0, 'b': 0.0, 'c': 0.0}}, 0, os.environ)
        self.assertIn('jacobian_condition', r)
        flagged = r['cov_status'] != 'OK' or any('ill-conditioned' in w['message'] for w in r['warnings'])
        self.assertTrue(flagged, r)
        good = call('opt_least_squares', {'model': 'a+b*x', 'response': 'y', 'data': {'x': [0.0, 1.0, 2.0, 3.0], 'y': [1.0, 3.1, 4.9, 7.2]}, 'x0': {'a': 0.0, 'b': 0.0}}, 0, os.environ)
        self.assertLess(good['jacobian_condition'], 100)
        self.assertEqual([w for w in good['warnings'] if 'ill-conditioned' in w['message']], [])

    def test_root_at_an_endpoint_and_tiny_brackets(self):
        for f, root in (('x*(x-1)', 0.0), ('x-1', 1.0)):
            r = call('num_root', {'f': f, 'lower': 0, 'upper': 1}, 0, os.environ)
            self.assertEqual((r['status'], r['root'], r['f_root'], r['iterations'], r['function_calls']), ('OK', root, 0.0, 0, 2), r)
        r = call('num_root', {'f': '1e-200*(x+3)', 'lower': 0, 'upper': 1}, 2, os.environ)
        self.assertEqual(r['error']['data']['tool_code'], 'BAD_INTERVAL')
        r = call('num_root', {'f': '1e-200*(x-0.5)', 'lower': 0, 'upper': 1}, 0, os.environ)  # opposite signs even though the product underflows
        self.assertEqual(r['status'], 'OK')
        self.assertAlmostEqual(r['root'], 0.5, delta=1e-9)
        self.assertGreaterEqual(r['iterations'], 0)

    def test_nonfinite_analytic_gradient_at_x0_is_a_validation_error(self):
        r = call('opt_minimize', {'objective': 'x^2', 'x0': {'x': 0}, 'method': 'BFGS', 'gradient': ['2*x + 0/0']}, 2, os.environ)
        self.assertEqual(r['error']['data']['tool_code'], 'NONFINITE_GRADIENT')

    def test_quad_singularity_on_the_centre_node_points_at_points(self):
        r = call('num_quad', {'f': '1/sqrt(abs(x))', 'a': -1, 'b': 1}, 0, os.environ)
        self.assertEqual(r['status'], 'NON_FINITE')
        self.assertIn('points', r['message'])
        r = call('num_quad', {'f': '1/sqrt(abs(x))', 'a': -1, 'b': 1, 'points': [0]}, 0, os.environ)
        self.assertEqual(r['status'], 'OK')
        self.assertAlmostEqual(r['value'], 4.0, delta=1e-6)

    def test_per_component_bounds_for_vector_unknowns(self):
        base = {'objective': '(v[0]-3)^2+(v[1]-7)^2', 'x0': {'v': [1.0, 1.0]}, 'method': 'L-BFGS-B'}
        r = call('opt_minimize', {**base, 'bounds': {'v': [[0, 2], [0, 5]]}}, 0, os.environ)
        np.testing.assert_allclose(r['par']['v'], [2.0, 5.0], atol=1e-5)
        r = call('opt_minimize', {**base, 'bounds': {'v': [0, 4]}}, 0, os.environ)  # a single pair still broadcasts
        np.testing.assert_allclose(r['par']['v'], [3.0, 4.0], atol=1e-5)
        r = call('opt_minimize', {**base, 'bounds': {'v': [[0, 2], [None, 'Inf']]}}, 0, os.environ)
        np.testing.assert_allclose(r['par']['v'], [2.0, 7.0], atol=1e-5)
        for bad in ([[0, 2]], [[0, 2], [0, 5], [0, 1]], [[0, 2], [5, 0]], [[0, 2], [0]], [[0, 2], 5], [[0, 2], [2, 5]]):
            with self.subTest(bad=bad):
                rejected('opt_minimize', {**base, 'bounds': {'v': bad}}, 'BAD_BOUNDS')

    def test_schema_documents_the_limits_users_hit(self):
        schema = json.loads(subprocess.run([BIN, '--schema'], capture_output=True, text=True).stdout)
        ops = {o['name']: o for o in schema['operations']}
        text = json.dumps(ops['opt_least_squares']) + json.dumps(ops['num_root']) + json.dumps(ops['opt_minimize'])
        for needle in ('1e-8', 'bracket width', 'infeasible', 'per-component'):
            self.assertIn(needle, text)


if __name__ == '__main__':
    unittest.main()
