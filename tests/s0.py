import hashlib, json, os, pathlib, re, signal, subprocess, sys, tempfile, time
BIN = str(pathlib.Path(sys.argv[1]).absolute())
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests/oracle'))
def call(op, params={}, rc=0, env=None):
    p = subprocess.run([BIN, '--input-json',json.dumps({'op':op,'params':params})],capture_output=True,text=True,env=env)
    assert p.returncode == rc, (p.returncode,p.stdout,p.stderr)
    return json.loads(p.stdout)
def result(op,p={}): return call(op,p)
assert pathlib.Path(BIN).exists(), 'S0 binary has not been implemented'
import numpy as np
from la_solve import A,B,expected
from stats_dist import FIXTURES
np.testing.assert_allclose(result('la_solve',{'a':A,'b':B})['x'], expected(), rtol=1e-13, atol=1e-13)
for params, oracle in FIXTURES:
    assert result('stats_dist',params)['values'] == list(oracle)
assert result('stats_dist',{'dist':'norm','fn':'ppf','x':[0,1,'NaN','NA',None]})['values'] == ['-Inf','Inf','NaN','NA','NA']
assert result('stats_dist',{'dist':'norm','fn':'cdf','x':['-Inf','Inf','NaN','NA']})['values'] == [0.,1.,'NaN','NA']
info = result('py_info')
assert info['rng'] == 'PCG64' and info['blas'] and all(b['threads']==1 for b in info['blas']), info
assert info['worker_sha256']==hashlib.sha256((ROOT/'py/frames.py').read_bytes()+b'\n'+(ROOT/'py/numerics.py').read_bytes()+b'\n'+(ROOT/'py/s4.py').read_bytes()+b'\n'+(ROOT/'py/sm.py').read_bytes()+b'\n'+(ROOT/'py/ml.py').read_bytes()+b'\n'+(ROOT/'py/s7.py').read_bytes()+b'\n'+(ROOT/'py/worker.py').read_bytes()).hexdigest()
if info['arch']=='aarch64':
    assert info['coretype_requested']=='ARMV8' and all(b['core']=='armv8' for b in info['blas'])
worker = (ROOT/'py/frames.py').read_text() + (ROOT/'py/numerics.py').read_text() + (ROOT/'py/s4.py').read_text() + (ROOT/'py/sm.py').read_text() + (ROOT/'py/ml.py').read_text() + (ROOT/'py/s7.py').read_text() + (ROOT/'py/worker.py').read_text()
assert not re.search(r'\b(eval|exec|pickle|query)\b',worker)
env = dict(os.environ, PYCLI_PYTHON='/no/such/python', PATH='/nonexistent')
p = subprocess.run([BIN,'--schema'],capture_output=True,text=True,env=env)
assert p.returncode == 0 and json.loads(p.stdout)['tool']=='pycli'
call('py_info',rc=8,env=env)
with tempfile.TemporaryDirectory(dir='.') as tmp:
    tmp = pathlib.Path(tmp).resolve()
    env = dict(os.environ,TMPDIR=str(tmp),SECRET_SENTINEL='must-not-leak')
    capture = tmp/'capture.json'
    wrapper = tmp/'python'
    wrapper.write_text('#!'+sys.executable+'\nimport json,os,sys\njson.dump({"argv":sys.argv[1:],"env":dict(os.environ)},open('+repr(str(capture))+',"w"))\nos.execv('+repr(sys.executable)+', ['+repr(sys.executable)+']+sys.argv[1:])\n')
    wrapper.chmod(0o755)
    env['PYCLI_PYTHON']=str(wrapper)
    call('py_info',env=env)
    observed=json.loads(capture.read_text())
    assert observed['argv'][:5]==['-P','-s','-B','-X','faulthandler']
    assert observed['env']['OPENBLAS_NUM_THREADS']=='1' and 'SECRET_SENTINEL' not in observed['env']
    assert set(observed['env']) <= {'PATH','HOME','TMPDIR','LC_ALL','TZ','PYTHONHASHSEED','PYTHONNOUSERSITE','PYTHONDONTWRITEBYTECODE','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_MAX_THREADS','OPENBLAS_CORETYPE','MPLBACKEND','PYCLI_LIMIT_NUMBERS','LC_CTYPE'}
    assert not list(tmp.glob('pycli-*'))
    call('la_solve',{'a':[[1,1],[1,1]],'b':[1,2]},env=env)
    assert not list(tmp.glob('pycli-*'))
    wrapper.write_text('#!'+sys.executable+'\nimport sys,json\njson.dump({"ok":False,"tool_code":"PY_ERROR","message":"injected failure","exception":"RuntimeError"},open(sys.argv[-1],"w"))\n')
    call('py_info',rc=1,env=env)
    assert not list(tmp.glob('pycli-*'))
    wrapper.write_text('#!/bin/sh\nkill -SEGV $$\n')
    crash=call('py_info',rc=1,env=env)
    assert crash['error']['code']=='INTERNAL_ERROR' and crash['error']['data']['tool_code']=='TOOL_CRASH'
    assert not list(tmp.glob('pycli-*'))
    wrapper.write_text('#!/bin/sh\nsleep 10\n')
    env['PYCLI_TIMEOUT_S']='1'
    call('py_info',rc=7,env=env)
    assert not list(tmp.glob('pycli-*'))
    env.pop('PYCLI_TIMEOUT_S')
    proc=subprocess.Popen([BIN,'py_info'],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    deadline=time.monotonic()+5
    while not list(tmp.glob('pycli-*')) and time.monotonic()<deadline: time.sleep(.02)
    proc.send_signal(signal.SIGTERM); proc.communicate(timeout=5)
    assert not list(tmp.glob('pycli-*'))
    # Safe artifact round trip and exclusive install.
    env.pop('PYCLI_PYTHON')
    np.save(tmp/'a.npy',np.array(A)); np.save(tmp/'b.npy',np.array(B))
    out=tmp/'x.safetensors'
    r=result('la_solve_to',{'a':str(tmp/'a.npy'),'b':str(tmp/'b.npy'),'out':str(out)})
    assert out.exists() and r['artifacts']['x']['ref'].endswith('#x')
    np.testing.assert_allclose(result('la_solve',{'a':[[1,0,0],[0,1,0],[0,0,1]],'b':r['artifacts']['x']['ref']})['x'],expected(),rtol=1e-13,atol=1e-13)
    call('la_solve_to',{'a':A,'b':B,'out':str(out)},rc=5)
    np.save(tmp/'large.npy',np.ones((2,600)))
    call('la_solve',{'a':[[1,0],[0,1]],'b':str(tmp/'large.npy')},rc=1)
    large=result('la_solve_to',{'a':[[1,0],[0,1]],'b':str(tmp/'large.npy'),'out':str(tmp/'large.safetensors')})
    assert large['artifacts']['x']['shape']==[2,600] and large['artifacts']['x']['nbytes']==9600
    call('la_solve_to',{'a':A,'b':B,'out':str(tmp/'dry.safetensors')},env=dict(os.environ,PYCLI_DRYRUN='1'))
    assert not (tmp/'dry.safetensors').exists()
    (tmp/'bad.npy').write_bytes(b'\x93NUMPY\x01\x00\xff\xff')
    call('la_solve',{'a':str(tmp/'bad.npy'),'b':[1]},rc=2,env=dict(os.environ,PYCLI_PYTHON='/no/python'))
    call('la_solve_to',{'a':A,'b':B,'out':str(tmp/'dry.safetensors')},rc=6,env=dict(os.environ,PYCLI_READONLY='1'))
    np.save(tmp/'object.npy',np.array(['bad'],dtype=object))
    call('la_solve',{'a':str(tmp/'object.npy'),'b':[1]},rc=2)
    call('la_solve',{'a':'/etc/passwd','b':[1]},rc=2)

warn=result('la_solve',{'a':[[1,0],[0,1e-20]],'b':[1,1]})
assert warn['warnings'] and warn['warnings'][0]['category']=='LinAlgWarning'
assert result('la_solve',{'a':[[1,1],[1,1]],'b':[1,2]})['status']=='SINGULAR'
for bad in ({'dist':'t','fn':'cdf','x':[1]}, {'dist':'norm','fn':'pdf','x':[1],'params':{'df':3}}, {'dist':'norm','fn':'pdf','x':[1],'params':{'scale':0}}):
    call('stats_dist',bad,rc=2)

print('S0 numerical, protocol, isolation, crash, timeout, cleanup and artifact contracts passed')
