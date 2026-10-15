"""S6 review regressions; mathematical references do not call sklearn estimators."""
import json, os, pathlib, struct, subprocess, sys, unittest
import numpy as np
import s6_ops as base

class Review(unittest.TestCase):
 setUp=base.ML.setUp; tearDown=base.ML.tearDown; call=base.ML.call
 fit=base.ML.fit; predict=base.ML.predict; output=base.ML.output
 def test_knn_label_bitflip_before_spawn(self):
  for labels in [['a','b'],['a','b','c']]:
   self.fit('knn_classifier',{'x':list(range(len(labels))),'y':labels},['x'],'y',params={'n_neighbors':1},out='knn.safetensors')
   data=bytearray((self.cwd/'knn.safetensors').read_bytes());n=struct.unpack('<Q',data[:8])[0];h=json.loads(data[8:8+n]);offset=h['train_y_']['data_offsets'][0]
   # Binary: 0->1; multiclass: middle class 1->0, leaving a gap.
   data[8+n+offset+(8 if len(labels)==3 else 0)]^=1
   (self.cwd/'bad.safetensors').write_bytes(data)
   for kind in ['predict','predict_proba']:
    r=self.predict('bad.safetensors',{'x':[0]},kind,rc=2,env=dict(os.environ,PYCLI_PYTHON='/missing-python'))
    self.assertEqual(r['error']['data']['tool_code'],'BAD_MODEL')
   (self.cwd/'knn.safetensors').unlink()
 def test_prediction_partition_invariance(self):
  cols={'x':[0,1,2,3,4,5],'z':[0,1,-1,2,0,1]}
  for family,params,kind in [('knn_regressor',{'n_neighbors':3},'predict'),('knn_classifier',{'n_neighbors':3},'predict_proba'),('pca',{'n_components':2},'transform'),('kmeans',{'n_clusters':2,'n_init':1},'transform'),('lda',{'n_components':2},'transform')]:
   train=dict(cols);target=None
   if family.startswith('knn') or family=='lda':target='y';train['y']=[0,0,1,1,2,2] if family!='knn_regressor' else [0,2,4,6,8,10]
   self.fit(family,train,['x','z'],target,params=params,seed=19)
   self.predict(family+'.safetensors',cols,kind);full=self.output('pred.npz');(self.cwd/'pred.npz').unlink()
   parts={k:[] for k in full}
   for i in range(6):
    self.predict(family+'.safetensors',{k:[v[i]] for k,v in cols.items()},kind);one=self.output('pred.npz');(self.cwd/'pred.npz').unlink()
    for k in parts:parts[k]+=one[k]
   for k in full:np.testing.assert_allclose(parts[k],full[k],rtol=1e-13,atol=1e-13)
 def test_rank_deficient_model_is_preserved(self):
  r=self.fit('linear',{'a':[1,2,3,4,5,6],'b':[2,4,6,8,10,12],'y':[2,3,4,5,6,7]},['a','b'],'y')
  self.assertEqual(r['status'],'RANK_DEFICIENT');self.assertTrue((self.cwd/'linear.safetensors').exists());self.assertEqual(r['rank'],1)
  self.predict('linear.safetensors',{'a':[1],'b':[2]});np.testing.assert_allclose(self.output('pred.npz')['prediction'],[2])
 def test_degenerate_cluster_and_lda_status(self):
  r=self.fit('kmeans',{'x':[1]*6},['x'],params={'n_clusters':2},rc=1)
  self.assertEqual(r['error']['data']['tool_code'],'DEGENERATE_CLUSTERS');self.assertFalse((self.cwd/'kmeans.safetensors').exists())
  r=self.fit('lda',{'x':[1]*6,'y':[0,0,0,1,1,1]},['x'],'y',rc=1)
  self.assertEqual(r['error']['data']['tool_code'],'DEGENERATE_DATA');self.assertFalse((self.cwd/'lda.safetensors').exists())
 def test_logistic_is_deterministic(self):
  r=self.call('ml_fit',family='logistic',columns={'x':[-2,-1,1,2],'y':[0,0,1,1]},features=['x'],target='y',preprocess={'scale':False,'one_hot':[]},out='a.safetensors')
  self.assertEqual(r['status'],'OK');self.assertIsNone(r['generator']);self.assertIsNone(r['seed'])
 def cv(self,labels,metrics):
  return self.call('ml_cv',family='logistic',columns={'x':list(range(len(labels))),'y':labels},features=['x'],target='y',preprocess={'scale':False,'one_hot':[]},seed=123,cv='kfold',n_splits=2,shuffle=False,metrics=metrics)
 def test_fold_missing_classes(self):
  for labels,metrics,status in [(['p']*5+['q']*4+['r'],['log_loss'],'UNDEFINED_METRIC'),([0]*15+[1]*15,['accuracy'],'INSUFFICIENT_CLASSES')]:
   r=self.cv(labels,metrics);self.assertEqual(r['status'],status);self.assertEqual(r['completed_folds'],len(r['folds']));self.assertIn('fold_index',r);self.assertTrue(r['missing_classes']);self.assertIn('stratified',r['message'])
 def test_undefined_metric_diagnostic(self):
  r=self.cv([0,0,0,1,0,1],['roc_auc']);self.assertEqual(r['status'],'UNDEFINED_METRIC');self.assertEqual(r['metric'],'roc_auc');self.assertIn('fold_index',r)
 def test_csv_inline_artifact_class_identity(self):
  for labels in [[0,0,1,1],['00','00','01','01'],['+0','+0','+1','+1']]:
   cols={'x':[-2,-1,1,2],'y':labels};(self.cwd/'train.csv').write_text('x,y\n'+''.join(f'{x},{y}\n' for x,y in zip(cols['x'],labels)))
   self.call('frame_import',columns=cols,out='train.npz')
   for i,source in enumerate([{'columns':cols},{'csv':'train.csv'},{'frame':'train.npz'}]):
    r=self.call('ml_fit',family='gaussian_nb',features=['x'],target='y',preprocess={'scale':False,'one_hot':[]},out=f'{i}.safetensors',**source);self.assertEqual(r['classes'],sorted(set(labels)))
    self.predict(f'{i}.safetensors',{'x':[-2,2]});self.assertEqual(self.output('pred.npz')['prediction'],[labels[0],labels[-1]]);(self.cwd/'pred.npz').unlink();(self.cwd/f'{i}.safetensors').unlink()
   (self.cwd/'train.npz').unlink()
 def test_small_parameters_and_wide_budget(self):
  r=self.fit('ridge',{'x':[0,1,2],'y':[1,3,5]},['x'],'y',params={'alpha':1})
  self.assertIn('parameters',r);np.testing.assert_allclose(r['parameters']['coef_'],[4/3],atol=1e-14);self.assertFalse(r['parameters_omitted'])
  cols={'f'+str(i):list(range(15)) for i in range(64)};r=self.fit('standard_scaler',cols,list(cols));self.assertTrue(r['parameters_omitted']);self.assertLess(len(json.dumps(r).encode()),65536)
 def test_knn_ties(self):
  self.fit('knn_classifier',{'x':[-1,1],'y':['b','a']},['x'],'y',params={'n_neighbors':2})
  self.predict('knn_classifier.safetensors',{'x':[0]},'predict_proba');self.assertEqual(self.output('pred.npz'),{'probability_0':[.5],'probability_1':[.5]});(self.cwd/'pred.npz').unlink()
  self.predict('knn_classifier.safetensors',{'x':[0]});self.assertEqual(self.output('pred.npz')['prediction'],['a']);(self.cwd/'pred.npz').unlink()
  self.fit('knn_regressor',{'x':[-1,1,3],'y':[8,2,100]},['x'],'y',params={'n_neighbors':1})
  self.predict('knn_regressor.safetensors',{'x':[0]});self.assertEqual(self.output('pred.npz')['prediction'],[8])
 def test_lasso_elastic_kkt_and_pca_sign(self):
  x=np.array([[-2,1],[-1,-1],[0,2],[1,0],[2,-2],[3,1.]]);y=np.array([1,-2,2,3,4,8.]);xc=x-x.mean(0)
  for family,ratio in [('lasso',1.),('elastic_net',.4)]:
   params={'alpha':.2,'tol':1e-12,'max_iter':1000}
   if family=='elastic_net':params['l1_ratio']=ratio
   self.fit(family,{'x':x[:,0].tolist(),'z':x[:,1].tolist(),'y':y.tolist()},['x','z'],'y',params=params)
   _,t=base.bundle(self.cwd/(family+'.safetensors'));w=t['coef_'];res=x@w+t['intercept_'][0]-y;g=x.T@res/len(x)+.2*(1-ratio)*w
   self.assertLess(abs(res.mean()),1e-10)
   for a,b in zip(w,g):self.assertLess(abs(b+.2*ratio*np.sign(a)) if a else max(0,abs(b)-.2*ratio),1e-9)
  self.fit('pca',{'x':x[:,0].tolist(),'z':x[:,1].tolist()},['x','z'],params={'n_components':2});_,t=base.bundle(self.cwd/'pca.safetensors');_,s,v=np.linalg.svd(xc,full_matrices=False);v*=np.sign(v[np.arange(len(v)),np.abs(v).argmax(1)])[:,None];np.testing.assert_allclose(t['components_'],v,atol=1e-14)
 def test_schema_and_objective_contract(self):
  schema=json.loads(subprocess.run([base.BIN,'--schema'],capture_output=True,text=True,check=True).stdout)
  ops={o['name']:o for o in schema['operations']}
  params={p['name']:p for p in ops['ml_fit']['params']}
  self.assertIn('l1_ratio',params['params']['description']);self.assertIn('default',params['params']['description']);self.assertIn('one_hot',params['preprocess']['description'])
  for name in ['ml_fit','ml_predict','ml_cv']:
   for p in ops[name]['params']:self.assertNotIn('parameter for',p['description'])
  doc=(base.ROOT/'docs/s6.md').read_text()
  for phrase in ['1/(2n)','glmnet','population variance','kth','sorted label','max-absolute','C sum']:self.assertIn(phrase,doc)

class WorkerMapping(unittest.TestCase):
 def test_valueerror_is_numerical(self):
  sys.path.insert(0,str(base.ROOT/'tests/oracle'));from s3 import load_worker
  w=load_worker();w.ml_import()
  def fail(p):raise ValueError('numerical backend failed')
  w.ml_cv=fail
  r=w.ml_operation('ml_cv',{'time_budget_s':10,'time_limit':10});self.assertEqual(r['status'],'NUMERICAL_FAILURE')
 def test_probability_width_guard(self):
  sys.path.insert(0,str(base.ROOT/'tests/oracle'));from s3 import load_worker
  w=load_worker()
  class Broken:
   def predict_proba(self,x):return np.ones((len(x),1))
  with self.assertRaises(w.NumericalFailure):w.ml_result(Broken(),np.zeros((1,1)),'predict_proba','knn_classifier',['a','b'])
if __name__=='__main__':unittest.main()
