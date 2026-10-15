"""S6 limits in the production hardened container; no network, subprocess is only the trusted measurement harness."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json,pathlib,subprocess,tempfile,time
import numpy as np
def peak():
 for p in ['/sys/fs/cgroup/memory.peak','/sys/fs/cgroup/memory/memory.max_usage_in_bytes']:
  if pathlib.Path(p).exists():return int(pathlib.Path(p).read_text())
def run(root,op,p):
 start=time.monotonic();r=subprocess.run(['/usr/local/bin/pycli',op,'--input-json',json.dumps(p)],cwd=root,capture_output=True,text=True);j=json.loads(r.stdout)
 return {'exit':r.returncode,'seconds':round(time.monotonic()-start,3),'status':j.get('status'),'error':j.get('error'),'response_bytes':len(r.stdout.encode()),'peak_bytes_so_far':peak()}
rng=np.random.default_rng(6);results={}
with tempfile.TemporaryDirectory(dir='/tmp') as d:
 root=pathlib.Path(d);x=rng.normal(size=(20000,9));y=x@np.arange(1,10)+rng.normal(size=20000)
 np.savetxt(root/'reg.csv',np.c_[x,y],delimiter=',',header=','.join(['x'+str(i) for i in range(9)]+['y']),comments='')
 features=['x'+str(i) for i in range(9)];base=dict(csv='reg.csv',features=features,target='y',preprocess={'scale':False,'one_hot':[]},time_limit=30)
 results['ridge_20000x9_200000_raw_cells']=run(root,'ml_fit',dict(base,family='ridge',out='ridge.safetensors'))
 for f in ['random_forest_regressor','hist_gradient_boosting_regressor']:
  # Prediction source created with the public frame importer, not worker internals.
  if not (root/'future.npz').exists():results['future_import']=run(root,'frame_import',dict(columns={name:x[:20,i].tolist() for i,name in enumerate(features)},out='future.npz'))
  params={'n_estimators':64,'max_depth':12} if f.startswith('random_') else {'max_iter':100,'max_depth':8}
  results[f+'_20000x9']=run(root,'ml_fit',dict(base,family=f,seed=6,params=params,predict_frame='future.npz',predict_out=f+'.npz'))
 results['ridge_cv_20fold_20000x9']=run(root,'ml_cv',dict(base,family='ridge',seed=6,cv='kfold',n_splits=20,shuffle=True,metrics=['mse','r2']))
 a=rng.normal(size=(3125,64));np.savetxt(root/'pca.csv',a,delimiter=',',header=','.join('x'+str(i) for i in range(64)),comments='')
 results['pca_3125x64_200000_cells']=run(root,'ml_fit',dict(csv='pca.csv',features=['x'+str(i) for i in range(64)],family='pca',params={'n_components':32},preprocess={'scale':False,'one_hot':[]},out='pca.safetensors',time_limit=30))
 np.savetxt(root/'km.csv',rng.normal(size=20000),header='x',comments='')
 results['kmeans_20000x1_k32_192M_budget']=run(root,'ml_fit',dict(csv='km.csv',features=['x'],family='kmeans',params={'n_clusters':32,'n_init':1,'max_iter':300},seed=6,preprocess={'scale':False,'one_hot':[]},out='km.safetensors',time_limit=30))
 np.savetxt(root/'knn.csv',np.c_[np.arange(2000),np.arange(2000)],delimiter=',',header='x,y',comments='')
 results['knn_fit_2000']=run(root,'ml_fit',dict(csv='knn.csv',features=['x'],target='y',family='knn_regressor',preprocess={'scale':False,'one_hot':[]},out='knn.safetensors'))
 np.savetxt(root/'query.csv',np.arange(2500),header='x',comments='')
 results['knn_predict_5M_pairs']=run(root,'ml_predict',dict(model='knn.safetensors',csv='query.csv',out='knn.npz',time_limit=30))
 results['peak_bytes']=peak()
print(json.dumps(results,indent=2))
assert all(v['exit']==0 and v['status']=='OK' for v in results.values() if isinstance(v,dict)),results
