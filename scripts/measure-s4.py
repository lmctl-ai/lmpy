"""Ten sequential production-shim calls per S4 workload; retain every sample."""
import json,os,pathlib,statistics,subprocess,tempfile,time
root=pathlib.Path(__file__).resolve().parents[1]
cases={
 'LP':('opt_linprog',{'c':[3,5],'a_ub':[[1,0],[0,2],[3,2]],'b_ub':[4,12,18],'maximize':True}),
 'MILP':('opt_linprog',{'c':[3,4,5,8],'a_ub':[[2,3,4,5]],'b_ub':[8],'integrality':['binary']*4,'maximize':True}),
 'ODE_decay_RK45':('ode_solve',{'rhs':['-y[0]'],'y0':[1],'t_span':[0,1]}),
 'ODE_Robertson_BDF':('ode_solve',{'rhs':['-0.04*y[0]+10000*y[1]*y[2]','0.04*y[0]-10000*y[1]*y[2]-30000000*y[1]^2','30000000*y[1]^2'],'y0':[1,0,0],'t_span':[0,40],'method':'BDF','rtol':1e-10,'atol':1e-13})}
report={'host_load_start':os.getloadavg(),'cases':{}}
with tempfile.TemporaryDirectory(prefix='pycli-s4-timing-',dir=pathlib.Path.home()) as directory:
 for name,(op,params) in cases.items():
  wall=[];imports=[];worker=[]
  for _ in range(10):
   started=time.monotonic();r=subprocess.run([str(root/'bin/pycli'),op,'--input-json',json.dumps(params)],cwd=directory,capture_output=True,text=True);wall.append((time.monotonic()-started)*1000)
   assert r.returncode==0,(name,r.stdout,r.stderr);j=json.loads(r.stdout);assert j['status']=='OK',j
   imports.append(j['meta']['import_ms']);worker.append(j['meta']['elapsed_ms'])
  report['cases'][name]={'p50_ms':statistics.median(wall),'p95_ms':max(wall),'samples_ms':wall,'import_p50_ms':statistics.median(imports),'worker_p50_ms':statistics.median(worker)}
  print(name,json.dumps(report['cases'][name]),flush=True)
report['host_load_end']=os.getloadavg();report['image']=json.loads(subprocess.run(['docker','image','inspect','pycli:runtime'],capture_output=True,text=True,check=True).stdout)[0]['Id']
(root/'docs/measurement-s4-shim.json').write_text(json.dumps(report,indent=2)+'\n')
