"""S1 array_import / array_inspect with independent oracles (numpy, math.hypot, a strict
hand-written safetensors reader). Import/inspect are pure C++: every call below runs with a
nonexistent interpreter to prove nothing is spawned."""
import hashlib, json, math, os, pathlib, random, resource, struct, subprocess, sys, tempfile, unittest
import numpy as np
BIN = str(pathlib.Path(sys.argv.pop(1)).resolve())
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / 'oracle'))
from s1 import read_safetensors, write_safetensors

ASAN = bool(os.environ.get('PYCLI_TEST_ASAN'))  # sanitized binaries reserve terabytes of address space: no RLIMIT_AS
NO_PYTHON = dict(os.environ, PYCLI_PYTHON='/no/such/python')


def call(op, params=None, rc=0, env=None, preexec=None):
    proc = subprocess.run([BIN, '--input-json', json.dumps({'op': op, 'params': params or {}})], capture_output=True,
                          text=True, env=env or NO_PYTHON, preexec_fn=preexec)
    assert proc.returncode == rc, (proc.returncode, proc.stdout, proc.stderr)
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

    def imported(self, params, key='x'):
        params = dict(params, out=self.out(f'{random.random()}.safetensors'))
        result = call('array_import', params)
        return result, read_safetensors(params['out'])[key]


class ImportInline(Base):
    def test_nested_json_bit_exact_and_descriptor(self):
        a = np.random.default_rng(7).standard_normal((4, 5)) * 10.0 ** np.random.default_rng(8).integers(-300, 300, (4, 5))
        result, got = self.imported({'format': 'json', 'data': a.tolist()})
        self.assertEqual(got.tobytes(), a.tobytes())
        self.assertEqual(got.dtype, np.dtype('<f8'))
        self.assertEqual((result['artifacts']['x']['shape'], result['artifacts']['x']['dtype'], result['artifacts']['x']['nbytes']), ([4, 5], 'F64', 160))
        self.assertTrue(result['artifacts']['x']['ref'].endswith('.safetensors#x'))
        self.assertEqual(result['na_count'], 0)
        self.assertEqual(result['finite_count'], 20)

    def test_1000_random_binary64_round_trip(self):
        rng = random.Random(2026)
        data = [0.0, -0.0, 5e-324, sys.float_info.max, sys.float_info.min, 0.1, 1e-310]
        while len(data) < 1000:
            v = struct.unpack('d', rng.getrandbits(64).to_bytes(8, 'little'))[0]
            if math.isfinite(v):
                data.append(v)
        _, got = self.imported({'format': 'json', 'data': data})
        self.assertEqual([struct.pack('<d', v) for v in data], [struct.pack('<d', v) for v in got])

    def test_envelope_special_strings_and_na(self):
        result, got = self.imported({'format': 'json', 'data': {'shape': [2, 3], 'data': [[1, 'NaN', 'Inf'], ['-Inf', None, 'NA']]}})
        self.assertEqual(got.shape, (2, 3))
        self.assertTrue(np.isnan(got[0, 1]) and got[0, 2] == np.inf and got[1, 0] == -np.inf and np.isnan(got[1, 1]) and np.isnan(got[1, 2]))
        self.assertEqual((result['na_count'], result['nan_count'], result['posinf_count'], result['neginf_count'], result['finite_count']), (2, 3, 1, 1, 1))
        r = call('array_import', {'format': 'json', 'data': [1, None], 'na': 'error', 'out': self.out('e.safetensors')}, 2)
        self.assertEqual(code(r), 'NA_NOT_ALLOWED')
        self.assertFalse(pathlib.Path(self.out('e.safetensors')).exists())
        # NaN spelled out is not "missing": it passes na=error.
        call('array_import', {'format': 'json', 'data': ['NaN', 1], 'na': 'error', 'out': self.out('n.safetensors')})

    def test_custom_key_and_key_validation(self):
        result, got = self.imported({'format': 'json', 'data': [1, 2], 'key': 'Design_1.a-b'}, key='Design_1.a-b')
        self.assertTrue(result['artifacts']['Design_1.a-b']['ref'].endswith('#Design_1.a-b'))
        for bad in ('', '1x', 'a b', 'a#b', 'x' * 65, '__metadata__', 'a/b'):
            with self.subTest(key=bad):
                call('array_import', {'format': 'json', 'data': [1], 'key': bad, 'out': self.out('k.safetensors')}, 2)

    def test_shape_rules(self):
        for bad in ([], [[]], [1, [2]], [[1, 2], [3]], [[[1]]], 5, 'x', {'shape': [3], 'data': [1, 2]}, {'shape': [4], 'data': [[1, 2], [3, 4]]}, {'shape': [4, 1], 'data': [[1, 2], [3, 4]]}, {'shape': [2, 2], 'data': [[1, 2, 3, 4]]}, {'shape': [0], 'data': []}, {'shape': [1.5], 'data': [1]}):
            with self.subTest(bad=bad):
                call('array_import', {'format': 'json', 'data': bad, 'out': self.out('s.safetensors')}, 2)
        call('array_import', {'format': 'json', 'data': ['x'], 'out': self.out('s.safetensors')}, 2)
        call('array_import', {'format': 'json', 'data': [True], 'out': self.out('s.safetensors')}, 2)

    def test_inline_cap(self):
        _, got = self.imported({'format': 'json', 'data': list(range(1000))})
        self.assertEqual(got.shape, (1000,))
        r = call('array_import', {'format': 'json', 'data': list(range(1001)), 'out': self.out('big.safetensors')}, 2)
        self.assertEqual(code(r), 'INPUT_TOO_LARGE')


class ImportCsv(Base):
    def test_inline_csv_matrix_header_crlf_and_na(self):
        text = 'a,b,c\r\n1,2.5,-3e2\r\nNA,NaN,Inf\r\n,-Inf,5e-324\r\n'
        result, got = self.imported({'format': 'csv', 'data': text, 'header': True})
        expected = np.array([[1, 2.5, -300], [np.nan, np.nan, np.inf], [np.nan, -np.inf, 5e-324]])
        np.testing.assert_array_equal(got, expected)  # NaN-aware, includes denormal exactness
        self.assertEqual(got[2, 2].tobytes(), np.float64(5e-324).tobytes())
        self.assertEqual((result['artifacts']['x']['shape'], result['na_count']), ([3, 3], 2))
        # empty field and NA are missing; the literal NaN is not.
        self.assertEqual(result['nan_count'], 3)

    def test_single_column_is_rank_one_single_row_is_rank_two(self):
        self.assertEqual(self.imported({'format': 'csv', 'data': '1\n2\n3'})[1].shape, (3,))
        self.assertEqual(self.imported({'format': 'csv', 'data': '1,2,3\n'})[1].shape, (1, 3))

    def test_plus_sign_and_precision(self):
        _, got = self.imported({'format': 'csv', 'data': '+1.5,0.1,0.30000000000000004,-0'})
        self.assertEqual(got.tobytes(), np.array([[1.5, 0.1, 0.30000000000000004, -0.0]]).tobytes())

    def test_rejected_cells(self):
        for text in ('1,2\n3', '1,2\n3,4,5', '0x10', '1e999', '1e-999', 'inf', 'nan', 'NAN', 'na', '"1"', "'1'", '1;2', '1 2', ' 1', '1,,x', '1..2', '--1', '1e', '.', '', '\n', '1\x00', 'é'):
            with self.subTest(text=text):
                r = call('array_import', {'format': 'csv', 'data': text, 'out': self.out('c.safetensors')}, 2)
                self.assertIn(code(r), ('BAD_DATA', 'BAD_CSV'))
        self.assertFalse(list(self.dir.glob('c.safetensors')))

    def test_header_only_rejected_and_inline_cap(self):
        call('array_import', {'format': 'csv', 'data': 'a,b\n', 'header': True, 'out': self.out('h.safetensors')}, 2)
        r = call('array_import', {'format': 'csv', 'data': '\n'.join(['1'] * 1001), 'out': self.out('b.safetensors')}, 2)
        self.assertEqual(code(r), 'INPUT_TOO_LARGE')
        self.assertEqual(self.imported({'format': 'csv', 'data': '\n'.join(['1'] * 1000)})[1].shape, (1000,))

    def test_csv_file_large_and_caps(self):
        values = np.random.default_rng(3).standard_normal((400, 500))
        path = self.dir / 'big.csv'
        np.savetxt(path, values, delimiter=',', fmt='%.17g')
        result, got = self.imported({'format': 'csv_file', 'data': str(path)})
        self.assertEqual(got.tobytes(), values.tobytes())
        self.assertEqual(result['artifacts']['x']['shape'], [400, 500])
        (self.dir / 'huge.csv').write_text('\n'.join(['1'] * 200001))
        r = call('array_import', {'format': 'csv_file', 'data': str(self.dir / 'huge.csv'), 'out': self.out('h.safetensors')}, 2)
        self.assertEqual(code(r), 'INPUT_TOO_LARGE')
        call('array_import', {'format': 'csv_file', 'data': '/etc/passwd', 'out': self.out('p.safetensors')}, 2)
        r = call('array_import', {'format': 'csv_file', 'data': str(self.dir / 'missing.csv'), 'out': self.out('p.safetensors')}, 4)
        self.assertEqual(r['error']['code'], 'NOT_FOUND')


class ImportBinary(Base):
    def test_npy_dtypes_convert_exactly(self):
        rng = np.random.default_rng(5)
        cases = {'f8': rng.standard_normal((3, 4)), 'f4': rng.standard_normal(7).astype('<f4'), 'i8': np.arange(-5, 6, dtype='<i8'),
                 'i4': np.array([[1, -2], [3, 4]], dtype='<i4'), 'b1': np.array([True, False, True]), 'u1': np.array([0, 7, 255], dtype='u1')}
        for name, array in cases.items():
            with self.subTest(dtype=name):
                np.save(self.dir / f'{name}.npy', array)
                result, got = self.imported({'format': 'npy', 'data': str(self.dir / f'{name}.npy')})
                self.assertEqual(got.tobytes(), array.astype('<f8').tobytes())
                self.assertEqual(result['source_dtype'], {'f8': 'F64', 'f4': 'F32', 'i8': 'I64', 'i4': 'I32', 'b1': 'BOOL', 'u1': 'U8'}[name])

    def test_npy_rejections_without_python(self):
        np.save(self.dir / 'object.npy', np.array(['bad'], dtype=object))
        np.save(self.dir / 'rank3.npy', np.ones((2, 2, 2)))
        np.save(self.dir / 'fortran.npy', np.asfortranarray(np.ones((3, 3))))
        np.save(self.dir / 'big_int.npy', np.array([2 ** 53 + 1], dtype='<i8'))
        np.save(self.dir / 'empty.npy', np.empty((0,)))
        np.save(self.dir / 'scalar.npy', np.float64(3))
        np.save(self.dir / 'complex.npy', np.ones(3, dtype=complex))
        np.save(self.dir / 'big.npy', np.ones(200001))
        np.save(self.dir / 'ok.npy', np.ones(3))
        raw = (self.dir / 'ok.npy').read_bytes()
        (self.dir / 'trunc.npy').write_bytes(raw[:-1])
        (self.dir / 'trailing.npy').write_bytes(raw + b'\0')
        (self.dir / 'hugehdr.npy').write_bytes(b'\x93NUMPY\x01\x00\xff\xff' + b' ' * 10)
        (self.dir / 'v3.npy').write_bytes(raw[:6] + b'\x03\x00' + raw[8:])
        (self.dir / 'shape_overflow.npy').write_bytes(self.header_file("{'descr': '<f8', 'fortran_order': False, 'shape': (99999999999999999999,), }"))
        (self.dir / 'shape_huge.npy').write_bytes(self.header_file("{'descr': '<f8', 'fortran_order': False, 'shape': (4294967296, 4294967296), }"))
        (self.dir / 'evil.npy').write_bytes(self.header_file("{'descr': '<f8', 'fortran_order': False, 'shape': (1,), 'x': __import__('os')}"))
        (self.dir / 'dir.npy').mkdir()
        for name in ('object', 'rank3', 'fortran', 'big_int', 'empty', 'scalar', 'complex', 'big', 'trunc', 'trailing', 'hugehdr', 'v3', 'shape_overflow', 'shape_huge', 'evil', 'dir', 'nothing'):
            with self.subTest(name=name):
                r = call('array_import', {'format': 'npy', 'data': str(self.dir / f'{name}.npy'), 'out': self.out(f'{name}.safetensors')}, 4 if name == 'nothing' else 2)
                self.assertIn(code(r), ('BAD_NPY', 'BAD_DATA', 'INPUT_TOO_LARGE', 'ARTIFACT_NOT_FOUND'))
                self.assertFalse(pathlib.Path(self.out(f'{name}.safetensors')).exists())

    def test_long_header_padding_does_not_blow_the_stack(self):
        text = "{'descr': '<f8', 'fortran_order': False, 'shape': (1,), }"
        header = text.encode() + b' ' * (65535 - len(text) - 1) + b'\n'
        (self.dir / 'pad.npy').write_bytes(b'\x93NUMPY\x01\x00' + struct.pack('<H', len(header)) + header + struct.pack('<d', 2.5))
        _, got = self.imported({'format': 'npy', 'data': str(self.dir / 'pad.npy')})
        self.assertEqual(got.tolist(), [2.5])
        bad = text.encode() + b' ' * 60000 + b'x\n'
        (self.dir / 'padbad.npy').write_bytes(b'\x93NUMPY\x01\x00' + struct.pack('<H', len(bad)) + bad + struct.pack('<d', 2.5))
        call('array_import', {'format': 'npy', 'data': str(self.dir / 'padbad.npy'), 'out': self.out('pb.safetensors')}, 2)

    @staticmethod
    def header_file(text):
        header = text.encode()
        header += b' ' * ((-(10 + len(header) + 1)) % 64) + b'\n'
        return b'\x93NUMPY\x01\x00' + struct.pack('<H', len(header)) + header + b'\0' * 24

    def test_path_confinement_and_extension(self):
        for path in ('/etc/passwd', '../outside.npy', str(self.dir / 'x.txt')):
            with self.subTest(path=path):
                call('array_import', {'format': 'npy', 'data': path, 'out': self.out('p.safetensors')}, 2)
        np.save(self.dir / 'a.npy', np.ones(2))
        os.symlink('/etc/passwd', self.dir / 'link.npy')
        call('array_import', {'format': 'npy', 'data': str(self.dir / 'link.npy'), 'out': self.out('p.safetensors')}, 2)
        call('array_import', {'format': 'npy', 'data': str(self.dir / 'a.npy') + '#x', 'out': self.out('p.safetensors')}, 2)

    def test_safetensors_key_dtypes_and_multiple_tensors(self):
        tensors = {'w': np.arange(6, dtype='<f8').reshape(2, 3) / 7, 'f': np.array([0.1, 0.2], dtype='<f4'), 'i': np.array([1, -2, 3], dtype='<i8'),
                   'j': np.array([[1], [2]], dtype='<i4'), 'u': np.array([0, 255], dtype='u1'), 'b': np.array([True, False])}
        write_safetensors(self.dir / 't.safetensors', tensors)
        for key, array in tensors.items():
            with self.subTest(key=key):
                result, got = self.imported({'format': 'safetensors', 'data': f'{self.dir}/t.safetensors#{key}', 'key': key}, key=key)
                self.assertEqual(got.tobytes(), array.astype('<f8').tobytes())
                self.assertEqual(result['artifacts'][key]['shape'], list(array.shape))
        self.assertEqual(result['source_dtype'], 'BOOL')

    def test_safetensors_rejections(self):
        write_safetensors(self.dir / 't.safetensors', {'a': np.ones(3), 'big': np.array([2 ** 60], dtype='<i8'), 'r3': np.ones((2, 2, 2)), 'z': np.empty((0,)), 'many': np.ones(200001)})
        for ref in ('t.safetensors', 't.safetensors#nokey', 't.safetensors#', '#a', 't.safetensors#big', 't.safetensors#r3', 't.safetensors#z', 't.safetensors#many', 't.safetensors#a#a', 'missing.safetensors#a'):
            with self.subTest(ref=ref):
                # protocol 6.1 (and torchcli): an absent file or tensor key is NOT_FOUND (4); every other defect is a validation error.
                rc = 4 if ref in ('t.safetensors#nokey', 'missing.safetensors#a') else 2
                r = call('array_import', {'format': 'safetensors', 'data': f'{self.dir}/{ref}', 'out': self.out('s.safetensors')}, rc)
                if rc == 4:
                    self.assertEqual(r['error']['code'], 'NOT_FOUND')
        self.assertFalse(pathlib.Path(self.out('s.safetensors')).exists())

    def test_header_validated_before_allocation(self):
        n = 20000000
        header = json.dumps({'x': {'dtype': 'F64', 'shape': [n], 'data_offsets': [0, n * 8]}}).encode()
        path = self.dir / 'oversize.safetensors'
        with path.open('wb') as f:
            f.write(len(header).to_bytes(8, 'little') + header)
            f.truncate(8 + len(header) + n * 8)

        def limited():
            if not ASAN:
                resource.setrlimit(resource.RLIMIT_AS, (64 * 1024 ** 2, 64 * 1024 ** 2))
        r = call('array_import', {'format': 'safetensors', 'data': f'{path}#x', 'out': self.out('o.safetensors')}, 2, preexec=limited)
        self.assertEqual(code(r), 'INPUT_TOO_LARGE')
        r = call('array_inspect', {'ref': f'{path}#x'}, 2, preexec=limited)
        self.assertEqual(code(r), 'INPUT_TOO_LARGE')


class ImportPolicy(Base):
    def test_conflict_dryrun_readonly_and_extension(self):
        target = self.dir / 'o.safetensors'
        params = {'format': 'json', 'data': [1, 2, 3], 'out': str(target)}
        call('array_import', params, 0, dict(NO_PYTHON, PYCLI_DRYRUN='1'))
        self.assertFalse(target.exists())
        call('array_import', params, 6, dict(NO_PYTHON, PYCLI_READONLY='1'))
        self.assertFalse(target.exists())
        call('array_import', params)
        before = hashlib.sha256(target.read_bytes()).hexdigest()
        r = call('array_import', dict(params, data=[9]), 5)
        self.assertEqual(r['error']['code'], 'CONFLICT')
        self.assertEqual(hashlib.sha256(target.read_bytes()).hexdigest(), before)
        os.symlink(target, self.dir / 'link.safetensors')
        call('array_import', dict(params, out=str(self.dir / 'link.safetensors')), 5)
        for bad in (str(self.dir / 'o.npy'), str(self.dir / 'missing' / 'o.safetensors'), '/tmp/pycli-escape.safetensors', '../escape.safetensors', ''):
            with self.subTest(out=bad):
                call('array_import', dict(params, out=bad), 2)
        self.assertFalse(pathlib.Path('/tmp/pycli-escape.safetensors').exists())
        self.assertEqual([p.name for p in self.dir.iterdir() if p.name.startswith('.') or '.tmp' in p.name], [])

    def test_artifact_is_independently_valid_and_not_world_writable(self):
        result, _ = self.imported({'format': 'json', 'data': [[1, 2], [3, 4]]})
        path = pathlib.Path(result['artifacts']['x']['path'])
        self.assertEqual(path.stat().st_mode & 0o022, 0)
        raw = path.read_bytes()
        n = struct.unpack('<Q', raw[:8])[0]
        self.assertEqual(n % 8, 0)
        self.assertEqual(json.loads(raw[8:8 + n])['x']['data_offsets'], [0, 32])
        self.assertEqual(len(raw), 8 + n + 32)


class Inspect(Base):
    def inspect(self, ref, **kw):
        return call('array_inspect', dict(ref=ref, **kw))

    def test_stats_match_numpy_and_hypot(self):
        a = np.random.default_rng(11).standard_normal((7, 13)) * 3
        r = self.inspect(a.tolist(), limit=20)
        self.assertEqual((r['shape'], r['dtype'], r['size']), ([7, 13], 'F64', 91))
        self.assertEqual((r['finite_count'], r['nan_count'], r['posinf_count'], r['neginf_count']), (91, 0, 0, 0))
        self.assertEqual((r['min'], r['max']), (a.min(), a.max()))
        ref = math.hypot(*a.ravel().tolist())
        self.assertLessEqual(abs(r['norm'] - ref), 2e-16 * ref)
        self.assertAlmostEqual(r['norm'], np.linalg.norm(a), delta=1e-14 * ref)
        self.assertEqual(r['preview'], a.ravel()[:20].tolist())
        self.assertTrue(r['preview_truncated'])

    def test_nonfinite_semantics(self):
        r = self.inspect([1, 'NaN', 'Inf', '-Inf', None, -4])
        self.assertEqual((r['finite_count'], r['nan_count'], r['posinf_count'], r['neginf_count']), (2, 2, 1, 1))
        self.assertEqual((r['min'], r['max']), ('-Inf', 'Inf'))  # NaN ignored, infinities kept (like nanmin/nanmax)
        self.assertEqual(r['norm'], 'NaN')
        self.assertEqual(r['preview'], [1, 'NaN', 'Inf', '-Inf', 'NaN', -4])
        r = self.inspect([1, 'Inf', 2])
        self.assertEqual(r['norm'], 'Inf')
        r = self.inspect(['NaN', 'NaN'])
        self.assertEqual((r['min'], r['max'], r['finite_count']), ('NA', 'NA', 0))

    def test_norm_neither_overflows_nor_underflows(self):
        r = self.inspect([1e200, 1e200])
        self.assertLessEqual(abs(r['norm'] / (math.sqrt(2) * 1e200) - 1), 4e-16)
        r = self.inspect([3e-200, 4e-200])
        self.assertLessEqual(abs(r['norm'] / 5e-200 - 1), 4e-16)
        self.assertEqual(self.inspect([0, -0.0])['norm'], 0)
        self.assertEqual(self.inspect([5e-324])['norm'], 5e-324)

    def test_preview_limit(self):
        r = self.inspect(list(range(1000)), limit=1000)
        self.assertEqual((len(r['preview']), r['preview_truncated']), (1000, False))
        r = self.inspect(list(range(1000)))
        self.assertEqual((len(r['preview']), r['preview_truncated']), (20, True))
        r = self.inspect([[1, 2], [3, 4]], limit=3)
        self.assertEqual(r['preview'], [1, 2, 3])
        for bad in (0, -1, 1001):
            call('array_inspect', {'ref': [1], 'limit': bad}, 2)
        np.save(self.dir / 'big.npy', np.arange(200000.0))
        r = self.inspect(str(self.dir / 'big.npy'), limit=1000)
        self.assertEqual((len(r['preview']), r['size'], r['max']), (1000, 200000, 199999))

    def test_files_report_stored_dtype(self):
        np.save(self.dir / 'f4.npy', np.array([[0.5, 1.5]], dtype='<f4'))
        r = self.inspect(str(self.dir / 'f4.npy'))
        self.assertEqual((r['dtype'], r['shape'], r['preview']), ('F32', [1, 2], [0.5, 1.5]))
        write_safetensors(self.dir / 't.safetensors', {'i': np.array([1, 2, 3], dtype='<i8'), 'f': np.ones((2, 2))})
        r = self.inspect(f'{self.dir}/t.safetensors#i')
        self.assertEqual((r['dtype'], r['shape'], r['max']), ('I64', [3], 3))
        self.assertAlmostEqual(r['norm'], math.sqrt(14), delta=1e-15)
        r = self.inspect(f'{self.dir}/t.safetensors')
        self.assertEqual(r['tensors'], {'f': {'dtype': 'F64', 'shape': [2, 2]}, 'i': {'dtype': 'I64', 'shape': [3]}})
        self.assertNotIn('preview', r)

    def test_listing_never_reads_payload_and_validates_header(self):
        n = 20000000
        header = json.dumps({'x': {'dtype': 'F64', 'shape': [n], 'data_offsets': [0, n * 8]}}).encode()
        path = self.dir / 'big.safetensors'
        with path.open('wb') as f:
            f.write(len(header).to_bytes(8, 'little') + header)
            f.truncate(8 + len(header) + n * 8)
        r = self.inspect(str(path))
        self.assertEqual(r['tensors'], {'x': {'dtype': 'F64', 'shape': [n]}})
        # Truncated payload: header lies about the data section.
        path.write_bytes(len(header).to_bytes(8, 'little') + header + b'\0' * 16)
        call('array_inspect', {'ref': str(path)}, 2)

    def test_bad_refs(self):
        for ref in ('/etc/passwd', '../x.npy', 'a.txt', '', 'x.npy#k', f'{self.dir}/none.npy', 5, {'a': 1}, [], [[1], [2, 3]]):
            with self.subTest(ref=ref):
                call('array_inspect', {'ref': ref}, 4 if ref == f'{self.dir}/none.npy' else 2)


class HeaderFuzz(Base):
    """Mutated safetensors/npy headers must end in a clean exit 0/2 (never crash or hang),
    and must not allocate memory proportional to a lying header."""

    def test_mutations_never_crash(self):
        rng = random.Random(1234)
        write_safetensors(self.dir / 'seed.safetensors', {'a': np.arange(6.0).reshape(2, 3), 'b': np.array([1, 2], dtype='<i8')}, metadata={'k': 'v'})
        seed = (self.dir / 'seed.safetensors').read_bytes()
        n = struct.unpack('<Q', seed[:8])[0]
        header = seed[8:8 + n]
        mutants = []
        for _ in range(250):
            data = bytearray(seed)
            for _ in range(rng.randint(1, 4)):
                if not data:
                    break
                kind = rng.randrange(5)
                if kind == 0:
                    data[rng.randrange(len(data))] = rng.randrange(256)
                elif kind == 1 and len(data) >= 8:
                    data[rng.randrange(8)] = rng.randrange(256)          # header length
                elif kind == 2:
                    del data[rng.randrange(len(data)):]                  # truncate
                elif kind == 3 and len(data) > 8:
                    i = rng.randrange(8, len(data))
                    data[i:i + 1] = rng.choice([b'9', b'-', b'[', b'{', b'"', b'e', b'0', b'1' * 30])
                else:
                    data += bytes(rng.randrange(256) for _ in range(rng.randrange(1, 40)))
            mutants.append(bytes(data))
        for text in (b'{"a":{"dtype":"F64","shape":[1],"data_offsets":[0,8],"dtype":"F32"}}', b'{"a":{"dtype":"F64","shape":[18446744073709551615],"data_offsets":[0,8]}}',
                     b'{"a":{"dtype":"F64","shape":[-1],"data_offsets":[0,8]}}', b'{"a":{"dtype":"F64","shape":[1.0],"data_offsets":[0,8]}}',
                     b'{"a":{"dtype":"F64","shape":[3037000500,3037000500],"data_offsets":[0,0]}}', b'{"a":{"dtype":"F64","shape":[1],"data_offsets":[8,0]}}',
                     b'{"a":{"dtype":"F64","shape":[1],"data_offsets":[0,18446744073709551615]}}', b'[' * 100000 + b']' * 100000, b'{"a":' * 5000, b'{}', b'null', b'{"__metadata__":{"a":1}}',
                     b'{"a":{"dtype":"F64","shape":[1],"data_offsets":[0,8]},"b":{"dtype":"F64","shape":[1],"data_offsets":[0,8]}}',
                     b'{"a":{"dtype":"F64","shape":[' + b'1,' * 33 + b'1],"data_offsets":[0,8]}}'):
            mutants.append(struct.pack('<Q', len(text)) + text + b'\0' * 64)
        mutants.append(struct.pack('<Q', 1 << 62) + b'x')
        mutants.append(struct.pack('<Q', 1 << 20) + b'x' * 10)
        mutants.append(b'')

        def limited():
            if not ASAN:
                resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 ** 2, 256 * 1024 ** 2))
        for i, data in enumerate(mutants):
            path = self.dir / f'm{i}.safetensors'
            path.write_bytes(data)
            for ref in (str(path), f'{path}#a', f'{path}#b'):
                proc = subprocess.run([BIN, '--input-json', json.dumps({'op': 'array_inspect', 'params': {'ref': ref}})], capture_output=True, text=True, timeout=20, preexec_fn=limited)
                self.assertIn(proc.returncode, (0, 2), (i, ref, proc.returncode, proc.stdout[:300], proc.stderr[:300]))
                json.loads(proc.stdout)
            path.unlink()


class ReviewFixes(Base):
    def test_mixed_dtype_file_only_fails_for_the_requested_tensor(self):
        # F16/BF16/I16/U32 entries are valid safetensors; only selecting one of them is unsupported.
        import numpy as np
        header = {'x': {'dtype': 'F64', 'shape': [2], 'data_offsets': [0, 16]}, 'h': {'dtype': 'F16', 'shape': [3], 'data_offsets': [16, 22]},
                  'b': {'dtype': 'BF16', 'shape': [1], 'data_offsets': [22, 24]}, 'i': {'dtype': 'I16', 'shape': [1], 'data_offsets': [24, 26]},
                  'u': {'dtype': 'U32', 'shape': [1], 'data_offsets': [26, 30]}, 'f8': {'dtype': 'F8_E4M3', 'shape': [2], 'data_offsets': [30, 32]}}
        text = json.dumps(header).encode()
        text += b' ' * ((-len(text)) % 8)
        payload = np.array([1.5, 2.5], dtype='<f8').tobytes() + b'\0' * 16
        (self.dir / 'mixed.safetensors').write_bytes(len(text).to_bytes(8, 'little') + text + payload)
        ref = f'{self.dir}/mixed.safetensors'
        r = call('array_inspect', {'ref': ref + '#x'})
        self.assertEqual((r['dtype'], r['preview']), ('F64', [1.5, 2.5]))
        listing = call('array_inspect', {'ref': ref})['tensors']
        self.assertEqual({k: v['dtype'] for k, v in listing.items()}, {'x': 'F64', 'h': 'F16', 'b': 'BF16', 'i': 'I16', 'u': 'U32', 'f8': 'F8_E4M3'})
        _, got = self.imported({'format': 'safetensors', 'data': ref + '#x'})
        self.assertEqual(got.tolist(), [1.5, 2.5])
        for key in ('h', 'b', 'i', 'u', 'f8'):
            r = call('array_inspect', {'ref': f'{ref}#{key}'}, 2)
            self.assertEqual(code(r), 'UNSUPPORTED_DTYPE')
        # A wrong byte count for a known dtype is still rejected for the whole file.
        header['h']['data_offsets'] = [16, 21]
        text = json.dumps(header).encode()
        text += b' ' * ((-len(text)) % 8)
        (self.dir / 'bad.safetensors').write_bytes(len(text).to_bytes(8, 'little') + text + payload)
        call('array_inspect', {'ref': f'{self.dir}/bad.safetensors'}, 2)

    def test_shape_envelope_accepts_flat_or_matching_nesting(self):
        _, got = self.imported({'format': 'json', 'data': {'shape': [2, 3], 'data': [1, 2, 3, 4, 5, 6]}})
        self.assertEqual(got.shape, (2, 3))
        _, got = self.imported({'format': 'json', 'data': {'shape': [2, 2], 'data': [[1, 2], [3, 4]]}})
        self.assertEqual(got.tolist(), [[1, 2], [3, 4]])


if __name__ == '__main__':
    unittest.main()
