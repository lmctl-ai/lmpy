"""pycli expression semantics against R (the language rcli evaluates in). tests/expr-semantics-golden.json holds R's answers for
every shared function and operator over a grid of edge values (0, negative, non-integer, huge, +-Inf, NaN); scripts/expr-semantics-diff.py
regenerates it from Rscript and proves it still matches R. Here pycli's real parser and builder must reproduce every cell."""
import json, math, os, pathlib, subprocess, sys, unittest
BIN = str(pathlib.Path(sys.argv.pop(1)).resolve())
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests/oracle'))
import semantics
import s3

SPEC = json.loads((ROOT / 'expr-spec.json').read_text())
GOLDEN = json.loads((ROOT / 'tests/expr-semantics-golden.json').read_text())
SPOT = json.loads((ROOT / 'tests/expr-semantics-spot.json').read_text())
W = s3.load_worker()


def check(case):
    return json.loads(subprocess.run([BIN, '--expr-check'], input=json.dumps(case), text=True, capture_output=True).stdout)


def call(op, params):
    proc = subprocess.run([BIN, '--input-json', json.dumps({'op': op, 'params': params})], capture_output=True, text=True)
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    return json.loads(proc.stdout)


class Semantics(unittest.TestCase):
    def test_golden_covers_the_whole_shared_table(self):
        self.assertEqual(sorted(GOLDEN['expressions']), sorted(semantics.semantic_expressions(SPEC)))
        self.assertEqual(GOLDEN['x'], [semantics.token(v) for v in semantics.SEMANTIC_X])
        self.assertEqual(GOLDEN['y'], [semantics.token(v) for v in semantics.SEMANTIC_Y])
        functions = {e.split('(')[0] for e in GOLDEN['expressions'] if e.endswith(')') and e.split('(')[0] in SPEC['functions']}
        reducers = {k for k, v in SPEC['functions'].items() if v.get('reducer')}
        self.assertEqual(functions - {'ifelse'}, set(SPEC['functions']) - reducers - {'ifelse'})

    def test_every_cell_matches_r(self):
        for expression, expected in GOLDEN['expressions'].items():
            got = s3.evaluate_semantics(W, SPEC, check, expression)
            ys = semantics.SEMANTIC_Y if semantics.uses_y(expression) else [None]
            pairs = [(x, y) for x in semantics.SEMANTIC_X for y in ys]
            self.assertEqual(len(got), len(expected), expression)
            wrong = [(x, y, want, have) for (x, y), want, have in zip(pairs, expected, got) if not semantics.semantic_cell_matches(expression, y, want, have)]
            self.assertEqual(wrong, [], expression)

    def test_documented_r_behaviours_by_name(self):
        def value(expression, **env):
            names = sorted(env)
            program = W.compile_expression(check({'expression': expression, 'names': names})['ast'], set(names), {}, SPEC)
            import numpy as np
            with np.errstate(all='ignore'):
                return float(W.evaluate(program, {k: np.float64(v) for k, v in env.items()}))
        nan = math.nan
        self.assertTrue(math.isnan(value('ifelse(x<1,1,2)', x=nan)))            # NA test -> NA, not FALSE
        self.assertTrue(math.isnan(value('ifelse(x<1 & y<1,1,2)', x=nan, y=0.5)))
        self.assertEqual(value('ifelse(x<1 & y<1,1,2)', x=nan, y=2.0), 2.0)       # NA & FALSE is FALSE
        self.assertEqual(value('ifelse(x<1 | y<1,1,2)', x=nan, y=0.5), 1.0)       # NA | TRUE is TRUE
        self.assertEqual(value('choose(5,2.5)'), 10.0)                            # k rounded half-even (2), R warns
        self.assertEqual(value('choose(5,3.5)'), 5.0)                             # ... to 4
        self.assertEqual(value('choose(-3,2)'), 6.0)
        self.assertAlmostEqual(value('lchoose(2000,1000)'), math.lgamma(2001) - 2 * math.lgamma(1001), delta=1e-9)
        self.assertTrue(math.isnan(value('beta(-0.5,2)')) and math.isnan(value('lbeta(-0.5,2)')))
        self.assertTrue(math.isnan(value('gamma(0)')) and math.isnan(value('digamma(0)')))

    def test_k_rounding_warns_once(self):
        import numpy as np, warnings
        program = W.compile_expression(check({'expression': 'choose(5,x)', 'names': ['x']})['ast'], {'x'}, {}, SPEC)
        W._WARNED.discard('choose_k')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            for _ in range(5):
                W.evaluate(program, {'x': np.float64(2.4)})
        self.assertEqual(len([c for c in caught if 'rounded' in str(c.message)]), 1)

    def test_spot_cases_agree_with_the_rcli_runtime_expectations(self):
        # The same cases are run through the real rcli binary by scripts/expr-semantics-diff.py --rcli.
        for case in SPOT['cases']:
            with self.subTest(f=case['f']):
                r = call('num_quad', {'f': case['f'], 'a': SPOT['lower'], 'b': SPOT['upper']})
                self.assertEqual(r['status'], case['status'], r)
                if case['status'] == 'OK':
                    self.assertAlmostEqual(r['value'], case['value'], delta=1e-12 * max(1.0, abs(case['value'])))

    def test_lchoose_does_not_overflow(self):
        r = call('num_quad', {'f': 'lchoose(2000,1000)', 'a': 0, 'b': 1})
        self.assertEqual(r['status'], 'OK')
        self.assertAlmostEqual(r['value'], 1382.26799353748, delta=1e-9)


if __name__ == '__main__':
    unittest.main()
