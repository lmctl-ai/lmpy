"""Compare the actual selected kernels and dense-solve overhead, offline."""
import json, os, subprocess
code='''import numpy as np, scipy.linalg as la, time, json, runpy
worker=runpy.run_path('/worker/worker.py')
rng=np.random.default_rng(2026); a=rng.normal(size=(400,400)); a=a@a.T+np.eye(400); b=np.ones(400)
la.solve(a,b)
values=[]
for _ in range(20):
 t=time.monotonic(); la.solve(a,b); values.append((time.monotonic()-t)*1000)
print(json.dumps({'blas':worker['blas_info'](),'solve400_ms':values}))
'''
results={}
for core in ('automatic','ARMV8'):
    args=['docker','run','--rm','--network=none','-e','OPENBLAS_NUM_THREADS=1','--mount','type=bind,src='+os.path.abspath('py')+',dst=/worker,readonly','--entrypoint','python3']
    if core!='automatic': args+=['-e','OPENBLAS_CORETYPE='+core]
    results[core]=json.loads(subprocess.check_output(args+['pycli:measure312','-I','-B','-c',code]))
print(json.dumps(results,indent=2))
