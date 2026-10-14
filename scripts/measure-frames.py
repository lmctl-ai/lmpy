"""Measure S2 per-call latency through the production shim; runtime image must exist."""
import json, pathlib, subprocess, tempfile, time, statistics, math, shutil
root=pathlib.Path(__file__).resolve().parents[1];shim=str(root/'bin/pycli')
with tempfile.TemporaryDirectory(prefix='pycli-s2-measure-',dir=pathlib.Path.home()) as tmp:
 cwd=pathlib.Path(tmp);results={}
 cols={'ts':['2026-03-0'+str(i)+'T12:00:00+00:00' for i in range(1,8)],'region':['E','W','E','W','E','W','E'],'sales':[10,20,30,40,50,60,70]}
 def call(op,p):
  r=subprocess.run([shim,'--input-json',json.dumps({'op':op,'params':p})],cwd=cwd,capture_output=True,text=True);assert r.returncode==0,(op,r.returncode,r.stdout,r.stderr);return json.loads(r.stdout)
 call('frame_import',{'columns':cols,'dtypes':{'ts':{'dtype':'datetime64[ns]','tz':'UTC'}},'out':'base.npz'})
 calls={'frame_import':{'columns':cols,'dtypes':{'ts':{'dtype':'datetime64[ns]','tz':'UTC'}},'out':'imp.npz'},'frame_inspect':{'frame':'base.npz','head':5,'describe':True},'frame_export':{'frame':'base.npz','format':'csv','out':'exp.csv'},'frame_select':{'frame':'base.npz','where':{'col':'sales','op':'ge','value':30},'out':'sel.npz'},'frame_groupby':{'frame':'base.npz','keys':['region'],'aggs':{'sales':['sum','mean']},'out':'grp.npz'},'frame_resample':{'frame':'base.npz','time':'ts','rule':'1W','aggs':{'sales':['sum']},'out':'res.npz'},'la_solve':{'a':[[1]],'b':[2]}}
 for op,p in calls.items():
  wall=[];imports=[];worker=[]
  for i in range(10):
   if 'out' in p:(cwd/p['out']).unlink(missing_ok=True)
   start=time.monotonic();r=call(op,p);wall.append((time.monotonic()-start)*1000);imports.append(r['meta']['import_ms']);worker.append(r['meta']['elapsed_ms'])
   if op=='la_solve':assert r['meta']['pandas']=='unloaded'
  results[op]={'p50_ms':statistics.median(wall),'p95_ms':sorted(wall)[math.floor(.95*len(wall))],'samples':wall,'import_p50_ms':statistics.median(imports),'worker_p50_ms':statistics.median(worker)}
  print(op,json.dumps({k:v for k,v in results[op].items() if k!='samples'}),flush=True)
 overhead=[]
 for i in range(10):
  start=time.monotonic();r=subprocess.run(['docker','run','--rm','--pull=never','--network=none','--entrypoint','/bin/true','pycli:runtime']);assert r.returncode==0;overhead.append((time.monotonic()-start)*1000)
 results['docker_overhead']={'p50_ms':statistics.median(overhead),'p95_ms':max(overhead),'samples':overhead}
 results['image']=json.loads(subprocess.run(['docker','image','inspect','pycli:runtime'],capture_output=True,text=True,check=True).stdout)[0]['Id']
 (root/'docs/measurement-s2-shim.json').write_text(json.dumps(results,indent=2)+'\n')
