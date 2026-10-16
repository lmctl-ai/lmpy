"""Regression cases from Opus/Fable, plus independent mathematical references."""
import json, math, os, pathlib, random, struct, subprocess, sys, tempfile, unittest
import numpy as np
BIN=str(pathlib.Path(sys.argv.pop(1)).resolve())
PROD=str(pathlib.Path(sys.argv.pop(1)).resolve()) if len(sys.argv)>1 and not sys.argv[1].startswith('-') else BIN

def call(op,p={},expected=0,env=None,cwd=None):
    proc=subprocess.run([BIN,'--input-json',json.dumps({'op':op,'params':p})],capture_output=True,text=True,env=env,cwd=cwd)
    assert proc.returncode==expected,(proc.returncode,proc.stdout,proc.stderr)
    return json.loads(proc.stdout)

class Review(unittest.TestCase):
    def test_F2_assumptions(self):
        for a,assume in [([[4,1],[0,3]],'pos'),([[1,2],[3,4]],'sym')]:
            with self.subTest(a=a):
                r=call('la_solve',{'a':a,'b':[1,1],'assume_a':assume},2)
                self.assertEqual(r['error']['data']['tool_code'],'NOT_SYMMETRIC')
        r=call('la_solve',{'a':[[1,2],[2,1]],'b':[1,1],'assume_a':'pos'})
        self.assertEqual(r['status'],'NOT_POSITIVE_DEFINITE')
        for assume in ('gen','sym','pos'):
            r=call('la_solve',{'a':[[4,1],[1,3]],'b':[[5,9],[4,7]],'assume_a':assume})
            np.testing.assert_allclose(r['x'],[[1,20/11],[1,19/11]],rtol=1e-14)
            self.assertLess(r['residual_norm'],1e-13)
    def test_F3_mixed_rank(self):
        for b in ([1,[2]],[[1],2],[[1,2],[3]],[[]]):
            with self.subTest(b=b): call('la_solve',{'a':[[1,0],[0,1]],'b':b},2)
        call('la_solve',{'a':[[1,0,0],[0,1,0]],'b':[1,2]},2)
    def test_F4_hash_isolation(self):
        with tempfile.TemporaryDirectory(dir='.') as directory:
            directory=pathlib.Path(directory).resolve()
            (directory/'numpy.py').write_text('raise RuntimeError("workspace module imported")')
            (directory/'sitecustomize.py').write_text('raise RuntimeError("workspace customization imported")')
            for _ in range(2):
                env=dict(os.environ,PYTHONPATH=str(directory),PYTHONHOME=str(directory),PYTHONUSERBASE=str(directory))
                r=call('py_info',env=env,cwd=directory)
                self.assertEqual(r['hash_randomization'],0)
                self.assertTrue(r['safe_path'])
                self.assertIsInstance(r['hash_probe'],str)
                self.assertFalse(r['user_site'])
                self.assertEqual(r['hash_probe'],call('py_info')['hash_probe'])
    def test_F6_binary64_boundary(self):
        rng=random.Random(2026)
        data=[0.,-0.,5e-324,sys.float_info.max,sys.float_info.min,0.1,1e-310]
        while len(data)<1000:
            v=struct.unpack('d',rng.getrandbits(64).to_bytes(8,'little'))[0]
            if math.isfinite(v): data.append(v)
        r=call('la_solve',{'a':[[1]],'b':[data]})['x'][0]
        self.assertEqual([struct.pack('d',v) for v in data],[struct.pack('d',v) for v in r])
    def test_F7_closed_forms(self):
        def values(dist,fn,x,params={}):return call('stats_dist',{'dist':dist,'fn':fn,'x':x,'params':params})['values']
        self.assertEqual(values('norm','cdf',[0]),[.5])
        self.assertEqual(values('norm','pdf',[0]),[1/math.sqrt(2*math.pi)])
        for x,y in zip([-3,-1,0,1,3],values('t','cdf',[-3,-1,0,1,3],{'df':1})):
            self.assertAlmostEqual(y,.5+math.atan(x)/math.pi,delta=5e-16)
        for p,y in zip([.25,.5,.75],values('t','ppf',[.25,.5,.75],{'df':1})):
            # SciPy 1.16.2's df=1 ppf is ~1.34e-11 off at .25/.75.
            self.assertAlmostEqual(y,math.tan(math.pi*(p-.5)),delta=2e-11)
        for got,want in zip(values('norm','ppf',[.025,.5,.975]),[-1.9599639845400545,0.,1.959963984540054]):
            self.assertAlmostEqual(got,want,delta=1e-15)  # scipy's ndtri (cephes C code, no BLAS) to within a few ulps; last-bit FMA differences across platforms are not a contract
        self.assertAlmostEqual(values('t','ppf',[.975],{'df':1e300})[0],1.959963984540054,delta=1e-14)
        self.assertEqual(values('norm','sf',[-2,0,2]),values('norm','cdf',[2,0,-2]))
        r=call('la_solve',{'a':[[1,0],[0,1e-20]],'b':[1,1]})
        self.assertAlmostEqual(r['rcond'],1e-20,delta=1e-34)
        self.assertEqual(r['warnings'][0]['category'],'LinAlgWarning')
        self.assertIn('Ill-conditioned',r['warnings'][0]['message'])
    def test_F8_probability(self):
        for x in ([1.5,-.5],['Inf'],['-Inf']):
            with self.subTest(x=x):
                r=call('stats_dist',{'dist':'norm','fn':'ppf','x':x},2)
                self.assertEqual(r['error']['data']['tool_code'],'BAD_PROBABILITY')
        r=call('stats_dist',{'dist':'t','fn':'ppf','x':[.5],'params':{'df':1e-300}})
        self.assertEqual(r['values'],['NaN'])
        self.assertTrue(any(w['category']=='NumericalWarning' for w in r['warnings']))
    def test_F9_no_artifact_no_success(self):
        with tempfile.TemporaryDirectory(dir='.') as directory:
            out=str(pathlib.Path(directory).resolve()/'singular.safetensors')
            r=call('la_solve_to',{'a':[[1,1],[1,1]],'b':[1,2],'out':out},1)
            self.assertEqual(r['error']['data']['tool_code'],'SINGULAR')
            self.assertFalse(pathlib.Path(out).exists())
    def test_F10_empty_message(self):
        with tempfile.TemporaryDirectory(dir='.') as directory:
            path=pathlib.Path(directory).resolve()/'empty.npy';np.save(path,np.empty((0,),dtype=float))
            r=call('la_solve',{'a':str(path),'b':[1]},2)
            self.assertIn('empty',r['error']['message'])
    def test_O6_header_before_payload(self):
        import resource
        with tempfile.TemporaryDirectory(dir='.') as directory:
            path=pathlib.Path(directory).resolve()/'oversize.safetensors'
            n=20000000;hdr=json.dumps({'x':{'dtype':'F64','shape':[n],'data_offsets':[0,n*8]}}).encode()
            with path.open('wb') as f:f.write(len(hdr).to_bytes(8,'little'));f.write(hdr);f.truncate(8+len(hdr)+n*8)
            def limited():resource.setrlimit(resource.RLIMIT_AS,(64*1024**2,64*1024**2))
            p=subprocess.run([BIN,'la_solve','--input-json',json.dumps({'a':str(path)+'#x','b':[1]})],capture_output=True,text=True,preexec_fn=limited)
            self.assertEqual(p.returncode,2,p.stdout)
            self.assertEqual(json.loads(p.stdout)['error']['data']['tool_code'],'INPUT_TOO_LARGE')
    def test_O7_prod_override(self):
        p=subprocess.run([PROD,'py_info'],env=dict(os.environ,PYCLI_PYTHON='/no/such/python'),capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stdout)

if __name__=='__main__': unittest.main()
