"""Sequential fresh calls through the production shim. Run on idle host after builds/tests."""
import json,pathlib,statistics,subprocess,tempfile,time
root=pathlib.Path(__file__).resolve().parents[1]
def summary(v):
 v=sorted(v);return {'p50_ms':round(statistics.median(v),3),'p95_ms':round(v[min(len(v)-1,int(.95*len(v)))],3),'n':len(v)}
results={'image_inspect_bytes':int(subprocess.check_output(['docker','image','inspect','pycli:runtime','--format','{{.Size}}'])),'image_display_size':subprocess.check_output(['docker','image','ls','pycli:runtime','--format','{{.Size}}'],text=True).strip(),'method':'sequential fresh calls, no concurrent builds or tests; 10 calls per case'}
with tempfile.TemporaryDirectory(dir=root) as d:
 work=pathlib.Path(d);counter=0
 def call(op,p):
  start=time.monotonic();r=subprocess.run([str(root/'bin/pycli'),op,'--input-json',json.dumps(p)],cwd=work,capture_output=True,text=True);wall=(time.monotonic()-start)*1000
  if r.returncode:raise RuntimeError((op,r.returncode,r.stdout,r.stderr))
  return json.loads(r.stdout),wall
 columns={'x':[(i-20)/5 for i in range(40)],'y':[int(i>20) for i in range(40)]}
 base=dict(family='logistic',columns=columns,features=['x'],target='y',preprocess={'scale':True,'one_hot':[]},seed=42)
 call('ml_fit',dict(base,out='base.safetensors'))
 for name,op,p in [('py_info','py_info',{}),('ml_fit_logistic_40x1','ml_fit',base),('ml_predict_proba_40x1','ml_predict',dict(model='base.safetensors',columns=columns,kind='predict_proba')),('ml_cv_logistic_4fold_40x1','ml_cv',dict(base,cv='stratified',n_splits=4,shuffle=True,metrics=['accuracy','log_loss']))]:
  walls=[];imports=[];elapsed=[]
  for i in range(10):
   counter+=1;params=dict(p)
   if op=='ml_fit':params['out']=f'm{counter}.safetensors'
   if op=='ml_predict':params['out']=f'p{counter}.npz'
   j,w=call(op,params);assert op=='py_info' or j['status']=='OK',j;walls.append(w);imports.append(j['meta']['import_ms']);elapsed.append(j['meta']['elapsed_ms'])
  results[name]={'wall':summary(walls),'worker_import_ms':summary(imports),'worker_elapsed_ms':summary(elapsed)}
print(json.dumps(results,indent=2))
