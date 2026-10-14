"""Run each probe in a fresh interpreter because audit hooks cannot be removed."""
import importlib.util,json,pathlib,subprocess,sys,tempfile,unittest
ROOT=pathlib.Path(__file__).resolve().parents[1]
def load():
    spec=importlib.util.spec_from_file_location('fixed_worker',ROOT/'py/worker.py')
    w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w);return w
if len(sys.argv)>1 and sys.argv[1]=='probe':
    w=load();kind=sys.argv[2]
    if kind in ('ValueError','OverflowError'):
        with tempfile.TemporaryDirectory() as directory:
            req=pathlib.Path(directory)/'req.json';res=pathlib.Path(directory)/'resp.json'
            req.write_text(json.dumps({'op':'py_info','params':{}}))
            def fail(p):raise {'ValueError':ValueError,'OverflowError':OverflowError}[kind]('unexpected numerical failure')
            w.OPS['py_info']=fail;sys.argv=[str(ROOT/'py/worker.py'),str(req),str(res)];w.main()
            assert json.loads(res.read_text())['tool_code']=='PY_ERROR',res.read_text()
    elif kind=='ImportError':
        try:w.BlockedImports().find_spec('subprocess')
        except ImportError:pass
        else:raise AssertionError('guard did not raise ImportError')
    elif kind=='codec_keys':
        try:w.codec({1:2})
        except TypeError as exc: assert 'key' in str(exc)
        else:raise AssertionError('accepted non-string JSON key')
    elif kind=='codec_types':
        assert json.loads(w.codec(1+2j))=={'re':1.,'im':2.}
        try:w.codec(b'not supported')
        except TypeError as exc:assert 'bytes' in str(exc)
        else:raise AssertionError('accepted bytes')
    sys.exit(0)
class Review(unittest.TestCase):
    def probe(self,kind):
        p=subprocess.run([sys.executable,__file__,'probe',kind],capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stdout+p.stderr)
    def test_F10_value_mapping(self):self.probe('ValueError')
    def test_F10_overflow_mapping(self):self.probe('OverflowError')
    def test_O5_importerror(self):self.probe('ImportError')
    def test_F12_keys(self):self.probe('codec_keys')
    def test_F12_types(self):self.probe('codec_types')
if __name__=='__main__':unittest.main()
