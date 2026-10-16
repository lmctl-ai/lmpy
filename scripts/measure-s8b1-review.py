"""Fresh hardened runtime probe: random-fill SuperLU at both pre-spawn work limits.
Mount this script read-only and run with the production shim's 2 CPU/4 GiB/128 pid/read-only/no-network constraints.
"""
import json, pathlib, resource, subprocess, tempfile, time
import numpy as np
with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
    d=pathlib.Path(tmp);rng=np.random.Generator(np.random.PCG64(19));n=2000
    # Exactly 10000 stored COO entries: random symmetric fill, diagonally dominant SPD.
    r=rng.integers(0,n,4000);c=rng.integers(0,n,4000);v=rng.uniform(-.1,.1,4000)
    rows=np.r_[np.arange(n),r,c];cols=np.r_[np.arange(n),c,r];vals=np.r_[np.full(n,10.),v,v]
    for key,value in [('r',rows),('c',cols),('v',vals),('b',np.ones(n))]:np.save(d/(key+'.npy'),value)
    base=dict(shape=[n,n],row='r.npy',col='c.npy',val='v.npy')
    results=[]
    for task,extra in [('spsolve',{'b':'b.npy'}),('eigsh',{'sigma':0.,'k':3,'which':'LM','seed':1,'vectors':True})]:
        p=dict(base,task=task,out=task+'.safetensors',**extra);start=time.monotonic()
        r=subprocess.run(['/usr/local/bin/pycli','--input-json',json.dumps({'op':'la_sparse_to','params':p})],cwd=d,text=True,capture_output=True)
        out=json.loads(r.stdout);peak=pathlib.Path('/sys/fs/cgroup/memory.peak')
        results.append(dict(task=task,stored_entries=10000,n_times_entries=20000000,wall_s=time.monotonic()-start,exit_code=r.returncode,result=out,cgroup_peak_bytes=int(peak.read_text()),cumulative_child_peak_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss))
        assert r.returncode==0 and out['status']=='OK',results[-1]
    print(json.dumps(results,indent=2))
