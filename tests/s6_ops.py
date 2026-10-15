"""S6: independent numerical oracles, model round trips and hostile preflight."""
import hashlib,itertools,json,math,os,pathlib,struct,subprocess,sys,tempfile,unittest
import numpy as np
BIN=sys.argv.pop(1);ROOT=pathlib.Path(__file__).resolve().parents[1]
def bundle(path):
 b=pathlib.Path(path).read_bytes();n=struct.unpack('<Q',b[:8])[0];h=json.loads(b[8:8+n]);types={'F64':'<f8','I64':'<i8'}
 return h,{k:np.frombuffer(b[8+n+v['data_offsets'][0]:8+n+v['data_offsets'][1]],dtype=types[v['dtype']]).reshape(v['shape']).copy() for k,v in h.items() if k!='__metadata__'}
def tamper(path,dest,change):
 b=pathlib.Path(path).read_bytes();n=struct.unpack('<Q',b[:8])[0];h=json.loads(b[8:8+n]);change(h);text=json.dumps(h,separators=(',',':')).encode();text+=b' '*((-len(text))%8);pathlib.Path(dest).write_bytes(struct.pack('<Q',len(text))+text+b[8+n:])
class ML(unittest.TestCase):
 def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.cwd=pathlib.Path(self.tmp.name)
 def tearDown(self):self.tmp.cleanup()
 def call(self,op,rc=0,env=None,**p):
  r=subprocess.run([BIN,op,'--input-json',json.dumps(p)],cwd=self.cwd,env=env,capture_output=True,text=True,timeout=48)
  self.assertEqual(r.returncode,rc,(op,r.stdout,r.stderr));return json.loads(r.stdout)
 def fit(self,family,columns,features,target=None,**extra):
  p=dict(family=family,columns=columns,features=features,preprocess={'scale':False,'one_hot':[]},seed=123,out=family+'.safetensors')
  if target:p['target']=target
  p.update(extra);return self.call('ml_fit',**p)
 def predict(self,model,columns,kind='predict',**p):return self.call('ml_predict',model=model,columns=columns,kind=kind,out='pred.npz',**p)
 def output(self,path):return self.call('frame_inspect',frame=path,head=20)['head']['data']
 def test_linear_ridge_closed_form(self):
  x=np.array([[-2,1],[-1,-1],[0,2],[1,0],[2,-2],[3,1.]],float);y=np.array([1,-2,2,3,4,8.]);cols={'a':x[:,0].tolist(),'b':x[:,1].tolist(),'y':y.tolist()}
  for fam,alpha in [('linear',0),('ridge',2)]:
   r=self.fit(fam,cols,['a','b'],'y',params={'alpha':alpha} if fam=='ridge' else {});self.assertEqual(r['status'],'OK');_,t=bundle(self.cwd/(fam+'.safetensors'))
   xc=x-x.mean(0);yc=y-y.mean();want=np.linalg.solve(xc.T@xc+alpha*np.eye(2),xc.T@yc);np.testing.assert_allclose(t['coef_'],want,rtol=1e-10,atol=1e-12)
   np.testing.assert_allclose(t['intercept_'],[y.mean()-x.mean(0)@want],rtol=1e-10,atol=1e-12)
 def test_logistic_numpy_newton(self):
  x=np.array([-3,-2,-1,0,1,2,3,4.]);y=np.array([0,0,1,0,1,0,1,1]);C=2.;a=np.c_[np.ones(len(x)),x];b=np.zeros(2)
  for _ in range(50):
   q=1/(1+np.exp(-(a@b)));g=a.T@(q-y)+np.array([0,b[1]/C]);h=a.T@((q*(1-q))[:,None]*a)+np.diag([0,1/C]);step=np.linalg.solve(h,g);b-=step
   if np.linalg.norm(step)<1e-13:break
  self.fit('logistic',{'x':x.tolist(),'y':y.tolist()},['x'],'y',params={'C':C,'tol':1e-12,'max_iter':1000});_,t=bundle(self.cwd/'logistic.safetensors');np.testing.assert_allclose(np.r_[t['intercept_'].ravel(),t['coef_'].ravel()],b,rtol=1e-7,atol=1e-9)
 def test_nb_lda_hand(self):
  cols={'x':[-3,-2,-1,1,2,4],'z':[0,1,-1,0,2,1],'y':['a']*3+['b']*3}
  for f in ['gaussian_nb','lda']:
   self.fit(f,cols,['x','z'],'y');_,t=bundle(self.cwd/(f+'.safetensors'));x=np.c_[cols['x'],cols['z']];means=np.array([x[:3].mean(0),x[3:].mean(0)])
   np.testing.assert_allclose(t['theta_' if f=='gaussian_nb' else 'means_'],means,rtol=1e-12)
   if f=='gaussian_nb':
    variance=np.array([x[:3].var(0),x[3:].var(0)])+1e-9*x.var(0).max();np.testing.assert_allclose(t['var_'],variance,rtol=1e-12)
   if f=='lda':
    centered=np.r_[x[:3]-means[0],x[3:]-means[1]];cov=centered.T@centered/4;w=np.linalg.solve(cov,means[1]-means[0]);np.testing.assert_allclose(t['coef_'].ravel(),w,rtol=1e-10,atol=1e-12)
 def test_all_family_roundtrips(self):
  for f in ['linear','ridge','lasso','elastic_net','logistic','linear_svc','gaussian_nb','lda','knn_classifier','knn_regressor','kmeans','pca','standard_scaler']:
   with self.subTest(family=f):
    cls=f in ['logistic','linear_svc','gaussian_nb','lda','knn_classifier'];uns=f in ['kmeans','pca','standard_scaler'];cols={'x':[-4,-3,-2,-1,1,2,3,4],'z':[0,1,-1,2,1,0,2,-1]};cols.update({} if uns else {'y':['low']*4+['high']*4 if cls else [-3,-2,0,0,2,4,3,6]})
    params={'n_clusters':2} if f=='kmeans' else {'n_components':1} if f=='pca' else {'n_neighbors':3} if f.startswith('knn') else {}
    fit=self.fit(f,cols,['x','z'],None if uns else 'y',params=params);self.assertEqual(fit['status'],'OK',fit)
    self.predict(f+'.safetensors',cols,'transform' if f in ('pca','standard_scaler') else 'predict');loaded=self.output('pred.npz');self.assertEqual(loaded,fit['prediction_preview']);(self.cwd/'pred.npz').unlink()
 def test_pca_svd_kmeans_knn_scaler(self):
  x=np.array([[-2,0],[-1,1],[0,2],[1,-1],[2,3],[3,1.]])
  self.fit('pca',{'x':x[:,0].tolist(),'z':x[:,1].tolist()},['x','z'],params={'n_components':2});_,t=bundle(self.cwd/'pca.safetensors');u,s,v=np.linalg.svd(x-x.mean(0),full_matrices=False);np.testing.assert_allclose(t['explained_variance_'],s*s/(len(x)-1),rtol=1e-12)
  self.fit('standard_scaler',{'x':x[:,0].tolist(),'z':x[:,1].tolist()},['x','z']);_,t=bundle(self.cwd/'standard_scaler.safetensors');np.testing.assert_allclose(t['mean_'],x.mean(0),rtol=1e-12);np.testing.assert_allclose(t['var_'],x.var(0),rtol=1e-12)
  r=self.fit('kmeans',{'x':[-10,-9,-11,9,10,11]},['x'],params={'n_clusters':2});lab=r['prediction_preview']['prediction'];self.assertTrue(lab[0]==lab[1]==lab[2] and lab[3]==lab[4]==lab[5] and lab[0]!=lab[3])
  self.fit('knn_regressor',{'x':[0,2,4,8],'y':[1,3,9,15]},['x'],'y',params={'n_neighbors':2});self.predict('knn_regressor.safetensors',{'x':[1,5]});np.testing.assert_allclose(self.output('pred.npz')['prediction'],[2,6],rtol=1e-12)
 def test_preprocess_and_missing(self):
  cols={'x':[1,2,3,4,5,6],'g':['a','b']*3,'y':[1,3,3,5,5,7]}
  r=self.fit('linear',cols,['x','g'],'y',preprocess={'scale':True,'one_hot':['g']});self.predict('linear.safetensors',cols);self.assertEqual(self.output('pred.npz'),r['prediction_preview'])
  self.fit('linear',{'x':[1,None,3],'y':[1,2,3]},['x'],'y',rc=2,out='missing.safetensors')
 def test_cv_hand_indices(self):
  for cv,want,kwargs in [('kfold',[[0,1],[2,3],[4,5]],{}),('stratified',[[0,2],[1,3],[4,5]],{}),('group',[[4,5],[2,3],[0,1]],{'groups':'g'}),('timeseries',[[2,3],[4,5]],{})]:
   c={'x':[0,1,2,3,4,5],'y':[0,0,1,1,0,1],'g':['a','a','b','b','c','c']};classification=cv=='stratified'
   r=self.call('ml_cv',family='gaussian_nb' if classification else 'ridge',columns=c,features=['x'],target='y',preprocess={'scale':False,'one_hot':[]},seed=123,cv=cv,n_splits=len(want),shuffle=False,metrics=['accuracy' if classification else 'mse'],**kwargs);self.assertEqual(r['status'],'OK');self.assertEqual([f['test_indices'] for f in r['folds']],want)
 def test_prespawn_tampering_policies(self):
  env=dict(os.environ,PYCLI_PYTHON='/missing-python');base=dict(family='linear',columns={'x':[1,2,3],'y':[2,4,6]},features=['x'],target='y',preprocess={'scale':False,'one_hot':[]},out='a.safetensors')
  for extra in [dict(params={'n_jobs':2}),dict(params={'solver':'__import__'}),dict(features=['missing']),dict(params={'max_iter':10000000}),dict(features=['x']*65),dict(preprocess={'scale':'yes','one_hot':[]})]:self.call('ml_fit',rc=2,env=env,**dict(base,**extra))
  self.call('ml_fit',**base)
  for i,change in enumerate([lambda h:h.update(extra=dict(h['coef_'])),lambda h:h['coef_'].update(shape=[999]),lambda h:h['__metadata__'].update(sklearn_version='0.0.0')]):
   tamper(self.cwd/'a.safetensors',self.cwd/'bad.safetensors',change);self.call('ml_predict',rc=2,env=env,model='bad.safetensors',columns={'x':[1]},out='bad.npz');(self.cwd/'bad.safetensors').unlink()
  (self.cwd/'truncated.safetensors').write_bytes((self.cwd/'a.safetensors').read_bytes()[:-1]);self.call('ml_predict',rc=2,env=env,model='truncated.safetensors',columns={'x':[1]},out='bad.npz')
  self.call('ml_predict',rc=2,env=env,model='a.safetensors',columns={'z':[1]},out='bad.npz')
  self.call('ml_fit',rc=6,env=dict(os.environ,PYCLI_READONLY='1'),**dict(base,out='none.safetensors'));self.call('ml_fit',env=dict(os.environ,PYCLI_DRYRUN='1'),**dict(base,out='none.safetensors'));self.assertFalse((self.cwd/'none.safetensors').exists())
 def test_trees_and_nonconvergence(self):
  train={'x':[-3,-2,-1,1,2,3],'y':[0,0,0,1,1,1]};self.call('frame_import',columns={'x':[-2,2]},out='test.npz')
  for f in ['random_forest_classifier','random_forest_regressor','hist_gradient_boosting_classifier','hist_gradient_boosting_regressor']:
   self.call('ml_fit',family=f,columns=train,features=['x'],target='y',seed=123,preprocess={'scale':False,'one_hot':[]},predict_frame='test.npz',predict_out='tree.npz');self.assertTrue((self.cwd/'tree.npz').exists());(self.cwd/'tree.npz').unlink()
   self.call('ml_fit',rc=2,env=dict(os.environ,PYCLI_PYTHON='/missing-python'),family=f,columns=train,features=['x'],target='y',seed=123,preprocess={'scale':False,'one_hot':[]},out='no.safetensors',predict_frame='test.npz',predict_out='tree.npz')
  r=self.fit('logistic',{'x':list(range(20)),'y':[0]*10+[1]*10},['x'],'y',params={'max_iter':1},rc=1);self.assertEqual(r['error']['data']['tool_code'],'NO_CONVERGENCE')
 def test_wide_preview_budget(self):
  cols={"f"+str(i):list(range(15)) for i in range(64)}
  r=self.fit('standard_scaler',cols,list(cols));self.assertLess(len(json.dumps(r)),65536)
  p=self.predict('standard_scaler.safetensors',cols,'transform');self.assertLess(len(json.dumps(p)),65536)
 def test_csv_npz_caps_and_nan_preflight(self):
  base=dict(family='ridge',features=['x'],target='y',preprocess={'scale':False,'one_hot':[]},out='fit.safetensors')
  (self.cwd/'train.csv').write_text('x,y\n1,2\n2,4\n3,6\n')
  self.call('ml_fit',csv='train.csv',**base)
  self.call('frame_import',columns={'x':[1,2,3],'y':[2,4,6]},out='train.npz')
  self.call('ml_fit',frame='train.npz',**dict(base,out='f2.safetensors'))
  env=dict(os.environ,PYCLI_PYTHON='/missing-python')
  (self.cwd/'large.csv').write_text('x,y\n'+'1,2\n'*20001)
  self.call('ml_fit',rc=2,env=env,csv='large.csv',**dict(base,out='big.safetensors'))
  for value in [None,'NaN','Inf',1e13]:self.call('ml_fit',rc=2,env=env,columns={'x':[1,value,3],'y':[2,4,6]},**dict(base,out='bad.safetensors'))
 def test_proba_decision_transform_and_skew(self):
  cols={'x':[-3,-2,-1,1,2,3],'y':[0,0,0,1,1,1]};self.fit('logistic',cols,['x'],'y')
  for kind in ['predict_proba','decision_function']:
   r=self.predict('logistic.safetensors',cols,kind);self.assertTrue(all(t['num_threads']==1 for t in r['threads']));data=self.output('pred.npz');(self.cwd/'pred.npz').unlink()
   if kind=='predict_proba':
    np.testing.assert_allclose(np.array(list(data.values())).sum(0),1,atol=1e-15);_,t=bundle(self.cwd/'logistic.safetensors');want=1/(1+np.exp(-(np.array(cols['x'])*t['coef_'].item()+t['intercept_'].item())));np.testing.assert_allclose(data['probability_1'],want,rtol=1e-14)
  tamper(self.cwd/'logistic.safetensors',self.cwd/'old.safetensors',lambda h:h['__metadata__'].update(sklearn_version='0.0.0'))
  r=self.predict('old.safetensors',cols,allow_version_skew=True);self.assertTrue(r['version_skew_accepted'])
 def test_budgets_seeds_laziness(self):
  base=dict(columns={'x':[-3,-2,-1,1,2,3],'y':[0,0,0,1,1,1]},features=['x'],target='y',preprocess={'scale':False,'one_hot':[]},out='fit.safetensors')
  self.call('ml_fit',rc=2,env=dict(os.environ,PYCLI_PYTHON='/missing-python'),family='logistic',**base)
  r=self.call('ml_fit',rc=1,family='logistic',seed=123,time_limit=0.000001,**base);self.assertEqual(r['error']['data']['tool_code'],'TIME_LIMIT');self.assertFalse((self.cwd/'fit.safetensors').exists())
  for op,args in [('py_info',{}),('stats_dist',{'dist':'norm','fn':'cdf','x':[0]})]:self.assertEqual(self.call(op,**args)['meta']['sklearn'],'unloaded')
class Isolation(unittest.TestCase):
 def test_dependency_inventory(self):
  import importlib.metadata as md
  for package,version in [('scikit-learn','1.7.2'),('joblib','1.5.2'),('threadpoolctl','3.6.0')]:self.assertEqual(md.version(package),version)
 def test_installed_deserialization_guard(self):
  code="import sys;sys.path.insert(0,'/repo/tests/oracle');from s3 import load_worker;w=load_worker();w.ml_import();import pickle;sys.addaudithook(w.audit);pickle.loads(b'cos\\nsystem\\n.')"
  code=code.replace('/repo',str(ROOT))
  r=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
  self.assertNotEqual(r.returncode,0);self.assertIn('blocked serialization audit event',r.stderr)
 def test_guard_audit_event(self):
  sys.path.insert(0,str(ROOT/'tests/oracle'));from s3 import load_worker
  w=load_worker()
  with self.assertRaises(RuntimeError):w.audit('pi'+'ckle.find_class',('os','system'))
if __name__=='__main__':unittest.main()
