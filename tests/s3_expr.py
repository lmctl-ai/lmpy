"""S3 expression subsystem: C++ parser gate, closed Python builder, shared rcli corpus, hostile inputs.
Everything that must be rejected is rejected in C++ with a nonexistent interpreter (nothing spawns)."""
import hashlib, json, math, os, pathlib, re, subprocess, sys, unittest
import numpy as np
BIN = str(pathlib.Path(sys.argv.pop(1)).resolve())
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests/oracle'))
import s3
NO_PYTHON = dict(os.environ, PYCLI_PYTHON='/no/such/python')
SPEC_TEXT = (ROOT / 'expr-spec.json').read_text()
SPEC = json.loads(SPEC_TEXT)
CORPUS = json.loads((ROOT / 'tests/expr-corpus.json').read_text())
# SHA-256 of the rcli copy of the spec and corpus this slice was copied from (rcli commit f0055dd).
RCLI_SPEC_SHA256 = 'f0b596000cc99b796a9d5bb2e3a39f8fdc39d2794812bc189e9b1800418e01a4'
RCLI_CORPUS_SHA256 = 'd110faaacbf47285c879f0fe234f69f1727cc3d82fd78aae4f32c621e7d9ba5d'
W = s3.load_worker()


def check(case):
    p = subprocess.run([BIN, '--expr-check'], input=json.dumps(case, ensure_ascii=False), text=True, capture_output=True)
    assert p.returncode == 0, (case, p.stdout, p.stderr)
    return json.loads(p.stdout)


def call(op, params, rc=0, env=None):
    p = subprocess.run([BIN, '--input-json', json.dumps({'op': op, 'params': params}, ensure_ascii=False)], capture_output=True, text=True, env=env or NO_PYTHON)
    assert p.returncode == rc, (rc, p.returncode, p.stdout[:600], p.stderr[:600])
    return json.loads(p.stdout)


def evaluate(expression, names, env, dims=None, reducers=False):
    got = check({'expression': expression, 'names': sorted(names), 'allow_reducers': reducers, 'dims': dims or {}})
    assert got['accepted'], (expression, got)
    program = W.compile_expression(got['ast'], set(names), dims or {}, SPEC)
    with np.errstate(all='ignore'):
        return W.evaluate(program, env)


class SpecAndCorpus(unittest.TestCase):
    def test_spec_hash_matches_on_both_sides(self):
        out = subprocess.check_output([BIN, '--expr-spec-hash'], text=True).strip()
        self.assertEqual(out, RCLI_SPEC_SHA256)
        spec, digest = W.load_spec({'spec_file': str(ROOT / 'expr-spec.json')})
        self.assertEqual((digest, spec), (RCLI_SPEC_SHA256, SPEC))
        self.assertEqual(check({'expression': 'pi', 'names': []})['sha256'], RCLI_SPEC_SHA256)

    def test_every_spec_function_has_a_python_binding_and_matches_an_independent_reference(self):
        import numpy
        table = W.function_table(numpy)
        self.assertEqual(sorted(set(SPEC['functions']) - set(table)), [])
        reducers = {k for k, v in SPEC['functions'].items() if v.get('reducer')}
        for name, info in SPEC['functions'].items():
            if name == 'ifelse' or name in reducers:
                continue
            if info['arity'] == 1:
                for x in s3.POINTS_1.get(name, s3.DEFAULT_POINTS):
                    with self.subTest(fn=name, x=x):
                        got = evaluate(f'{name}(x)', ['x'], {'x': np.float64(x)})
                        want = s3.REFERENCE_1[name](x)
                        self.assertAlmostEqual(float(got), float(want), delta=1e-13 * max(1.0, abs(want)))
            else:
                pairs = [(2.0, 5.0), (3.0, 1.5), (7.0, 2.0)] if name in ('beta', 'lbeta') else [(5.0, 2.0), (7.0, 3.0), (6.0, 0.0)] if 'choose' in name else [(1.0, -2.0), (-3.5, 0.25), (2.0, 2.0)]
                for a, b in pairs:
                    with self.subTest(fn=name, a=a, b=b):
                        got = evaluate(f'{name}(a,b)', ['a', 'b'], {'a': np.float64(a), 'b': np.float64(b)})
                        want = s3.REFERENCE_2[name](a, b)
                        self.assertAlmostEqual(float(got), float(want), delta=1e-12 * max(1.0, abs(want)))
        # reducers and ifelse
        v = np.array([1.0, 4.0, -2.0, 7.5])
        for name, ref in (('sum', v.sum()), ('mean', v.mean()), ('prod', v.prod()), ('min', v.min()), ('max', v.max())):
            self.assertAlmostEqual(float(evaluate(f'{name}(x)', ['x'], {'x': v}, reducers=True)), float(ref), delta=1e-13)
        got = evaluate('ifelse(x>0 & x<5, x, -1)', ['x'], {'x': v})
        self.assertEqual(got.tolist(), [1.0, 4.0, -1.0, -1.0])

    def test_operators_precedence_and_unary_minus_match_python(self):
        cases = {'a+b*c': lambda a, b, c: a + b * c, 'a-b-c': lambda a, b, c: a - b - c, 'a/b/c': lambda a, b, c: a / b / c,
                 'a^b^c': lambda a, b, c: a ** (b ** c), '-a^2': lambda a, b, c: -(a ** 2), '--a + - -b': lambda a, b, c: a + b,
                 '(a+b)*c - a/(b+c)': lambda a, b, c: (a + b) * c - a / (b + c), '2^-a': lambda a, b, c: 2.0 ** -a,
                 'ifelse(a<b | b<c & c<a, a, b)': lambda a, b, c: a if (a < b or (b < c and c < a)) else b,
                 'ifelse(a==b, 1, ifelse(a!=c, 2, 3))': lambda a, b, c: 1.0 if a == b else (2.0 if a != c else 3.0),
                 'ifelse(a>=b & b<=c, a, c)': lambda a, b, c: a if (a >= b and b <= c) else c}
        for expression, ref in cases.items():
            for a, b, c in [(1.5, 2.0, 3.0), (2.0, 2.0, 2.0), (3.0, 1.0, 2.0)]:
                with self.subTest(expression=expression, a=a):
                    got = evaluate(expression, ['a', 'b', 'c'], {'a': np.float64(a), 'b': np.float64(b), 'c': np.float64(c)})
                    self.assertAlmostEqual(float(got), ref(a, b, c), delta=1e-13)

    def test_shared_rcli_corpus_counts_and_codes(self):
        self.assertEqual((len(CORPUS['accept']), len(CORPUS['reject'])), (13, 57))
        # The corpus file is byte-identical to the rcli original it was copied from.
        self.assertEqual(hashlib.sha256((ROOT / 'tests/expr-corpus.json').read_bytes()).hexdigest(), RCLI_CORPUS_SHA256)
        for case in CORPUS['accept']:
            got = check(case)
            self.assertTrue(got['accepted'], case)
            names = set(case.get('names', []))
            env = {n: np.float64(0.7) for n in names}
            program = W.compile_expression(got['ast'], names, {}, SPEC)
            if case['expression'] == 'sum(x)':
                env = {'x': np.array([1.0, 2.0])}
            with np.errstate(all='ignore'):
                value = W.evaluate(program, env)
            self.assertTrue(np.isfinite(value).all(), case)
        for case in CORPUS['reject']:
            got = check(case)
            self.assertFalse(got['accepted'], case)
            self.assertEqual(got['tool_code'], case['expected_tool_code'], case)

    def test_accepted_corpus_values_against_python_lambdas(self):
        lambdas = {'a*exp(-b*t)': lambda a, b, t: a * math.exp(-b * t), 'x^2': lambda x: x * x, 'ifelse(x>0,x,-x)': lambda x: abs(x),
                   'ifelse(x>0 & x<1,1,0)': lambda x: 1.0 if 0 < x < 1 else 0.0, 'pi+e': lambda: math.pi + math.e, 'pmin(x,2)': lambda x: min(x, 2.0),
                   '1e-12 + log1p(abs(x))': lambda x: 1e-12 + math.log1p(abs(x)), 'ifelse(x<1 | x>4 & x<0,1,0)': lambda x: 1.0 if (x < 1 or (x > 4 and x < 0)) else 0.0,
                   '--x + - -x': lambda x: 2 * x, '1e-310 + x': lambda x: 1e-310 + x, '--x': lambda x: x}
        for case in CORPUS['accept']:
            reference = lambdas.get(case['expression'])
            if reference is None:
                continue
            names = case.get('names', [])
            for value in (-2.5, 0.25, 0.9, 5.0):
                with self.subTest(expression=case['expression'], x=value):
                    env = {n: np.float64(value + i) for i, n in enumerate(names)}
                    got = float(evaluate(case['expression'], names, env))
                    self.assertAlmostEqual(got, reference(*[value + i for i in range(len(names))]), delta=1e-13)



class VectorIndexing(unittest.TestCase):
    """pycli extension: vector variables with integer-literal indexing; scalars may not be indexed."""

    def test_accepted_and_values(self):
        got = evaluate('100*(x[1]-x[0]^2)^2 + (1-x[0])^2', ['x'], {'x': np.array([-1.2, 1.0])}, dims={'x': 2})
        self.assertAlmostEqual(float(got), s3.rosenbrock(-1.2, 1.0), delta=1e-12)
        got = evaluate('x[ 0 ] + y[2] * z', ['x', 'y', 'z'], {'x': np.array([1.0]), 'y': np.array([1.0, 2.0, 3.0]), 'z': np.float64(2.0)}, dims={'x': 1, 'y': 3})
        self.assertEqual(float(got), 7.0)

    def test_rejections(self):
        for expression, code in [('x', 'EXPR_VECTOR_INDEX'), ('x+1', 'EXPR_VECTOR_INDEX'), ('x[2]', 'EXPR_INDEX'), ('x[99999999999]', 'EXPR_INDEX'), ('x[1.5]', 'EXPR_INDEX'),
                                 ('x[-1]', 'EXPR_INDEX'), ('x[0', 'EXPR_SYNTAX'), ('x[]', 'EXPR_INDEX'), ('x[0][0]', 'EXPR_SYNTAX'), ('x[y]', 'EXPR_INDEX'),
                                 ('x[0]()', 'EXPR_SYNTAX'), ('y[0]', 'EXPR_SYNTAX'), ('x[00]', 'EXPR_INDEX'), ('x[0x1]', 'EXPR_INDEX'), ('x[1e0]', 'EXPR_INDEX'), ('x[(0)]', 'EXPR_INDEX'), ('x [0]', 'EXPR_ACCEPTED')]:
            got = check({'expression': expression, 'names': ['x', 'y'], 'dims': {'x': 2}})
            if code == 'EXPR_ACCEPTED':
                self.assertTrue(got['accepted'], (expression, got))
            else:
                self.assertFalse(got['accepted'], (expression, got))
                self.assertEqual(got['tool_code'], code, expression)
        # Without a vector declaration '[' is plain syntax error, exactly like rcli.
        self.assertEqual(check({'expression': 'x[0]', 'names': ['x']})['tool_code'], 'EXPR_SYNTAX')

    def test_index_tokens_count_toward_the_512_token_cap(self):
        text = '+'.join(['x[0]'] * 100)  # 399 tokens, 199 nodes
        self.assertTrue(check({'expression': text, 'names': ['x'], 'dims': {'x': 1}})['accepted'])
        text = '+'.join(['x[0]'] * 200)  # 799 tokens
        self.assertEqual(check({'expression': text, 'names': ['x'], 'dims': {'x': 1}})['tool_code'], 'EXPR_TOO_LONG')


class HostileInputs(unittest.TestCase):
    ATTACKS = ["__import__('os').system('id')", 'x.__class__', 'x.__class__.__mro__', 'lambda: 1', 'lambda x: x', '[i for i in x]', "'a'", 'f"{x}"', '"{0}".format(x)',
               'x**2', '(x for x in y)', 'x if x else 1', 'x.real', 'x;1', 'x\\n1', "exec('1')", 'compile("1","","eval")', 'getattr(x,"real")', 'globals()', 'open("/etc/passwd")',
               '1_000', '0b1', '0o7', '1j', '1e', 'ех', 'x​', '１', 'ｅxp(x)', 'exṕ(x)', 'x−y', '\x00', 'exp(x)\x00', '‮', '', ' ', '(', ')', '()', '(x', 'x)',
               '((((((((((((((((((((((((((((((((((x))))))))))))))))))))))))))))))))))', 'x' + '+x' * 600, 'exp(' * 20 + 'x' + ')' * 20, '--' * 40 + 'x', 'x' * 2049,
               'ifelse(x>0)', 'ifelse(x>0,1)', 'ifelse(1,2,3)', 'exp(exp(exp(exp(exp(exp(exp(exp(exp(exp(exp(exp(exp(exp(exp(exp(exp(x)))))))))))))))))', 'sum(x)']

    def test_all_attacks_rejected_in_cpp_without_spawning(self):
        for text in self.ATTACKS:
            with self.subTest(text=text[:40]):
                got = check({'expression': text, 'names': ['x']})
                self.assertFalse(got['accepted'], text)
                self.assertTrue(got['tool_code'].startswith('EXPR_'), got)
                r = call('num_quad', {'f': text, 'a': 0, 'b': 1}, 2)
                self.assertTrue(r['error']['data']['tool_code'].startswith('EXPR_'), r)
                r = call('opt_minimize', {'objective': text, 'x0': {'x': 1.0}}, 2)
                self.assertTrue(r['error']['data']['tool_code'].startswith('EXPR_'), r)

    def test_shared_negative_corpus_through_ops_spawns_nothing(self):
        for case in CORPUS['reject']:
            names = [n for n in case.get('names', []) if re.fullmatch(r'[a-z][a-z0-9_]{0,31}', n) and n not in SPEC['functions'] and n not in ('pi', 'e')] or ['x']
            with self.subTest(expression=case['expression'][:40]):
                r = call('opt_minimize', {'objective': case['expression'], 'x0': {n: 1.0 for n in names}}, 2)
                self.assertTrue(r['error']['data']['tool_code'].startswith('EXPR_'), r)
                self.assertNotEqual(r['error']['code'], 'SERVICE_UNAVAILABLE')

    def test_unicode_in_identifiers_and_variable_names(self):
        for name in ('é', 'x́', 'ｘ', 'X', '1x', '_x', 'x-y', 'x y', 'x' * 33, 'exp', 'pi', 'e', 'ifelse', 'list', '', 'x.y', 'sum'):
            with self.subTest(name=name):
                r = call('opt_minimize', {'objective': '1', 'x0': {name: 1.0}}, 2)
                self.assertIn(r['error']['data']['tool_code'], ('BAD_NAME', 'EXPR_NAME_COLLISION'))

    def test_exponent_towers_are_bounded_work(self):
        # 2^2^2^2^2^2^2^2 = 2^65536: a legal expression whose value overflows. It must stop with an honest status.
        r = call('num_quad', {'f': '2^2^2^2^2^2^2^2 + x', 'a': 0, 'b': 1}, 0, os.environ)
        self.assertEqual(r['status'], 'NON_FINITE')
        r = call('opt_minimize', {'objective': '2^2^2^2^2^2^2^2 + x', 'x0': {'x': 1.0}}, 2, os.environ)
        self.assertEqual(r['error']['data']['tool_code'], 'NONFINITE_OBJECTIVE')
        self.assertIn(check({'expression': '2' + '^2' * 600, 'names': []})['tool_code'], ('EXPR_TOO_LONG', 'EXPR_TOO_DEEP'))


class ErrorMapping(unittest.TestCase):
    def fake(self, response, op='num_quad', params=None):
        import shutil, tempfile
        directory = tempfile.mkdtemp(dir='.')
        try:
            fake = pathlib.Path(directory) / 'python'
            fake.write_text('#!' + sys.executable + '\nimport json,sys\njson.dump(' + repr(response) + ',open(sys.argv[-1],"w"))\n')
            fake.chmod(0o755)
            proc = subprocess.run([BIN, '--input-json', json.dumps({'op': op, 'params': params or {'f': 'x', 'a': 0, 'b': 1}})], capture_output=True, text=True,
                                  env=dict(os.environ, PYCLI_PYTHON=str(fake)))
            return proc.returncode, json.loads(proc.stdout)
        finally:
            shutil.rmtree(directory)

    def test_worker_error_classes_map_to_protocol_codes(self):
        rc, r = self.fake({'ok': False, 'tool_code': 'EXPR_SPEC_MISMATCH', 'message': 'x', 'exception': 'ExprMismatch', 'error_class': 'internal'})
        self.assertEqual((rc, r['error']['code'], r['error']['data']['tool_code']), (1, 'INTERNAL_ERROR', 'EXPR_SPEC_MISMATCH'))
        rc, r = self.fake({'ok': False, 'tool_code': 'BAD_INTERVAL', 'message': 'x', 'exception': 'BadData', 'error_class': 'validation'})
        self.assertEqual((rc, r['error']['code'], r['error']['data']['tool_code']), (2, 'VALIDATION_ERROR', 'BAD_INTERVAL'))
        rc, r = self.fake({'ok': False, 'tool_code': 'PY_ERROR', 'message': 'x', 'exception': 'ValueError', 'error_class': 'failed'})
        self.assertEqual((rc, r['error']['code']), (1, 'COMMAND_FAILED'))

    def test_oversized_inline_result_is_refused(self):
        rc, r = self.fake({'ok': True, 'result': {'status': 'OK', 'values': [0.5] * 1001, 'warnings': [], 'warnings_truncated': False}})
        self.assertEqual((rc, r['error']['data']['tool_code']), (1, 'OUTPUT_TOO_LARGE'))


class BuilderIsClosed(unittest.TestCase):
    """The Python side re-validates ASTs: a hostile or corrupted AST cannot reach any callable outside the table."""
    NAMES = {'x', 'y'}

    def build(self, ast, dims=None):
        return W.compile_expression(ast, set(self.NAMES), dims or {}, SPEC)

    def test_hostile_asts_are_spec_mismatches(self):
        num = {'t': 'num', 'v': 1.0}
        var = {'t': 'var', 'n': 'x'}
        bad = [{'t': 'call', 'f': '__import__', 'a': [num]}, {'t': 'call', 'f': 'system', 'a': [num]}, {'t': 'call', 'f': 'eval', 'a': [num]},
               {'t': 'call', 'f': 'getattr', 'a': [num, num]}, {'t': 'call', 'f': 'ifelse', 'a': [num, num, num]}, {'t': 'call', 'f': 'exp', 'a': []},
               {'t': 'call', 'f': 'exp', 'a': [num, num]}, {'t': 'call', 'f': ['exp'], 'a': [num]}, {'t': 'call', 'f': 'exp'},
               {'t': 'op', 'o': '**', 'a': [num, num]}, {'t': 'op', 'o': '%', 'a': [num, num]}, {'t': 'op', 'o': '+', 'a': [num]}, {'t': 'op', 'o': '+', 'a': [num, num, num]},
               {'t': 'op', 'o': 'getattr', 'a': [num, num]}, {'t': 'var', 'n': 'z'}, {'t': 'var', 'n': '__class__'}, {'t': 'var', 'n': 5}, {'t': 'var'},
               {'t': 'const', 'n': 'tau'}, {'t': 'const', 'n': '__name__'}, {'t': 'num', 'v': 'nan'}, {'t': 'num', 'v': float('inf')}, {'t': 'num', 'v': True}, {'t': 'num', 'v': '1'}, {'t': 'num'},
               {'t': 'attr', 'o': var, 'n': 'real'}, {'t': 'lambda'}, {'t': 'neg', 'a': []}, {'t': 'neg', 'a': [num, num]}, {'t': 'idx', 'n': 'x', 'i': 0}, {'t': 'ifelse', 'c': num, 'a': num},
               {}, [], 'exp(1)', None, 5, {'t': None}, {'t': 'ifelse', 'c': num, 'a': num, 'b': {'t': 'call', 'f': 'system', 'a': [num]}}]
        for ast in bad:
            with self.subTest(ast=str(ast)[:60]):
                with self.assertRaises(W.ExprMismatch):
                    self.build(ast)

    def test_vector_asts(self):
        for ast in ({'t': 'idx', 'n': 'x', 'i': 2}, {'t': 'idx', 'n': 'x', 'i': -1}, {'t': 'idx', 'n': 'x', 'i': 1.0}, {'t': 'idx', 'n': 'x', 'i': True},
                    {'t': 'idx', 'n': 'y', 'i': 0}, {'t': 'idx', 'n': 'x', 'i': 10 ** 12}, {'t': 'var', 'n': 'x'}):
            with self.subTest(ast=ast):
                with self.assertRaises(W.ExprMismatch):
                    self.build(ast, dims={'x': 2})
        self.assertIsNotNone(self.build({'t': 'idx', 'n': 'x', 'i': 1}, dims={'x': 2}))

    def test_node_cap_and_no_recursion_on_huge_asts(self):
        node = {'t': 'num', 'v': 1.0}
        for _ in range(600):
            node = {'t': 'op', 'o': '+', 'a': [node, {'t': 'num', 'v': 1.0}]}
        with self.assertRaises(W.ExprMismatch):
            self.build(node)
        deep = {'t': 'num', 'v': 1.0}
        for _ in range(200000):  # would exhaust the interpreter stack in a recursive builder
            deep = {'t': 'neg', 'a': [deep]}
        with self.assertRaises(W.ExprMismatch):
            self.build(deep)
        node = {'t': 'num', 'v': 1.0}
        for _ in range(250):  # legal depth: iterative build and evaluation work without recursion
            node = {'t': 'op', 'o': '+', 'a': [node, {'t': 'num', 'v': 1.0}]}
        self.assertEqual(float(W.evaluate(self.build(node), {})), 251.0)

    def test_missing_binding_is_a_build_failure(self):
        import numpy
        spec = json.loads(SPEC_TEXT)
        spec['functions']['frobnicate'] = {'arity': 1, 'r': 'base::frobnicate'}
        with self.assertRaises(W.ExprMismatch):
            W.check_bindings(spec, W.function_table(numpy))

    def test_worker_source_has_no_dynamic_code_paths(self):
        source = ''.join((ROOT / 'py' / name).read_text() for name in ('frames.py', 'numerics.py', 'worker.py'))
        forbidden = r'(?<!re\.)\b(eval|exec|compile|literal_eval|setattr|delattr|globals|locals|vars|__import__|importlib|attrgetter|methodcaller|numexpr|pickle|query|sympy|marshal|codeop|runpy|shlex)\b'
        self.assertEqual(re.findall(forbidden, source), [])
        self.assertNotIn('getattr(', ''.join((ROOT / 'py' / 'numerics.py').read_text()))
        self.assertNotRegex(source, r'import\s+ast|from\s+ast')
        self.assertNotIn('pandas.eval', source)


if __name__ == '__main__':
    unittest.main()
