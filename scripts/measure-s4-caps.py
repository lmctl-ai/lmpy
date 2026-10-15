"""Inside the hardened runtime: exercise every matrix dimension cap and ODE output cap."""
import json,pathlib,subprocess,tempfile,time
import numpy as np
cases={}
with tempfile.TemporaryDirectory(prefix='pycli-s4-caps-') as directory:
 root=pathlib.Path(directory)
 for name,m,n in [('matrix_entries',500,400),('variables',390,512),('constraint_rows',4096,48)]:
  np.save(root/'c.npy',-np.ones(n));np.save(root/'a.npy',np.ones((m,n)));np.save(root/'b.npy',np.full(m,float(n)))
  p={'out':'lp.safetensors','c':'c.npy','a_ub':'a.npy','b_ub':'b.npy','bounds':[[0,1]]*n}
  started=time.monotonic();r=subprocess.run(['/usr/local/bin/pycli','opt_linprog_to','--input-json',json.dumps(p)],cwd=root,capture_output=True,text=True);elapsed=time.monotonic()-started
  assert r.returncode==0,(r.returncode,r.stdout,r.stderr);j=json.loads(r.stdout);assert j['status']=='OK' and abs(j['objective']+n)<1e-8,j;assert len(r.stdout.encode())<16384
  cases[name]={'rows':m,'variables':n,'entries':m*n,'seconds':elapsed,'response_bytes':len(r.stdout.encode()),'objective':j['objective']}
  pathlib.Path(j['artifact']['path']).unlink()
 n=32;points=(200000-128*(n+2)-n)//(n+1)-1
 p={'out':'ode.safetensors','rhs':['-0.1*y[%d]'%i for i in range(n)],'y0':[1]*n,'t_span':[0,10],'max_output_points':points,'method':'DOP853','rtol':1e-10,'atol':1e-12}
 started=time.monotonic();r=subprocess.run(['/usr/local/bin/pycli','ode_solve_to','--input-json',json.dumps(p)],cwd=root,capture_output=True,text=True);elapsed=time.monotonic()-started
 assert r.returncode==0,(r.returncode,r.stdout,r.stderr);j=json.loads(r.stdout);assert j['status']=='OK',j;assert all(abs(v-np.exp(-1))<1e-8 for v in j['final_state']);assert len(r.stdout.encode())<16384
 cases['ode_states_and_output']={'states':n,'points':points,'trajectory_numbers':points*(n+1),'seconds':elapsed,'response_bytes':len(r.stdout.encode()),'artifact_bytes':pathlib.Path(j['artifact']['path']).stat().st_size,'evals':j['evals']}
 peak=next((int(p.read_text()) for p in [pathlib.Path('/sys/fs/cgroup/memory/memory.max_usage_in_bytes'),pathlib.Path('/sys/fs/cgroup/memory.peak')] if p.exists()),None)
 print(json.dumps({'cases':cases,'cgroup_peak_bytes':peak},indent=2))
