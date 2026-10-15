"""Review reproducers and independent hand/NumPy references; no scipy oracle."""
import json, math, os, pathlib, signal, subprocess, sys, tempfile, time, unittest
from unittest import mock
import numpy as np
import s7_ops as old
BIN=old.BIN;ROOT=old.ROOT

class Review(old.Base):
 def runop(self,op,rc=0,env=None,**p):return old.call(op,p,rc,env)
 def test_knn_write_cap_before_spawn(self):
  p=self.npy(np.arange(200000)[:,None]);t=time.monotonic()
  r=self.runop('spatial_knn_to',1,old.NO_PYTHON,points=p,k=1000,out=self.out());self.assertEqual(r['error']['data']['tool_code'],'OUTPUT_TOO_LARGE');self.assertLess(time.monotonic()-t,2)
 def test_radius_inline_count_cap_before_spawn(self):
  r=self.runop('spatial_knn',1,old.NO_PYTHON,points=self.npy(np.arange(1000)),radius=1);self.assertEqual(r['error']['data']['tool_code'],'OUTPUT_TOO_LARGE')
 def test_high_p_identity_distance_and_membership(self):
  for scale in [1e-100,1e-6,1,1e6,1e100]:
   with self.subTest(scale=scale):
    p=dict(points=[[0],[scale]],queries=[[2*scale]],p=64)
    r=self.runop('spatial_knn',**p,k=1);self.assertEqual(r['indices'],[[1]]);np.testing.assert_allclose(r['distances'],[[scale]],rtol=1e-14,atol=0)
    r=self.runop('spatial_knn',**p,radius=scale/2);self.assertEqual(r['counts'],[0]);self.assertEqual(r['indices'],[])
 def test_spectrum_padding_and_windows(self):
  x=[1,0,-1,0]*8
  for method,window in [('periodogram','boxcar'),('periodogram','hann'),('welch','hann'),('welch','hamming')]:
   with self.subTest(method=method,window=window):
    params=dict(method=method,x=x,window=window,detrend='none')
    if method=='welch':params.update(nperseg=8,noverlap=4)
    a=self.runop('sig_spectrum',**params,scaling='spectrum',nfft=32);b=self.runop('sig_spectrum',**params,scaling='spectrum',nfft=64)
    density=self.runop('sig_spectrum',**params,scaling='density',nfft=64)
    self.assertAlmostEqual(a['total_power'],b['total_power'],13);self.assertAlmostEqual(b['total_power'],density['total_power'],13)
    if window=='boxcar':self.assertAlmostEqual(b['total_power'],.5,13)
 def test_nan_policy_does_not_hide_failure(self):
  r=self.runop('interp_eval',kind='linear',x=[0,1e-310],y=[0,1],x_new=[0,5e-311,1e-310],extrapolate='nan');self.assertEqual(r['status'],'NUMERICAL_FAILURE')
 def test_intentional_nan_write(self):
  r=self.runop('interp_eval_to',kind='linear',x=[0,1],y=[0,1],x_new=[-1,.5,2],extrapolate='nan',out=self.out())
  values=old.read_safetensors(self.out())['values'];self.assertTrue(np.isnan(values[[0,2]]).all());self.assertEqual(values[1],.5)
 def test_integral_only_write_is_preflight_error(self):
  r=self.runop('interp_eval_to',2,old.NO_PYTHON,kind='linear',x=[0,1],y=[0,1],integrate=[0,1],out=self.out());self.assertEqual(r['error']['data']['tool_code'],'BAD_PARAM')
 def test_rbf_polynomial_preflight(self):
  for params in [dict(x=[[0,0],[1,1]],y=[0,1]),dict(x=[[0,0],[1,0],[0,1]],y=[0,1,1],kernel='quintic'),dict(x=[[0],[1],[2]],y=[0,1,2],kernel='quintic',degree=0)]:
   with self.subTest(params=params):
    r=self.runop('interp_eval',2,old.NO_PYTHON,kind='rbf',x_new=[[.5]*len(params['x'][0])],**params);self.assertEqual(r['error']['data']['tool_code'],'BAD_PARAM')
 def test_ellip_ripple_preflight(self):
  for rp,rs in [(20,10),(20,20)]:
   r=self.runop('sig_filter',2,old.NO_PYTHON,family='ellip',order=4,wn=.2,rp=rp,rs=rs);self.assertEqual(r['error']['data']['tool_code'],'BAD_PARAM')
 def test_coherence_needs_multiple_segments(self):
  x=[1,0,-1,0,1,0,-1,0,1,0];y=[0,2,1,-3,0,5,-2,1,4,-1]
  r=self.runop('sig_spectrum',method='coherence',x=x,y=y);self.assertGreaterEqual(r['n_segments'],2);self.assertFalse(np.allclose(r['cxy'],1))
  r=self.runop('sig_spectrum',2,old.NO_PYTHON,method='coherence',x=x,y=y,nperseg=10);self.assertEqual(r['error']['data']['tool_code'],'BAD_PARAM')
 def test_akima_method_specific_hand_values(self):
  expected={'akima':[-.375,1.1536,6.],'makima':[-.328125,1.1536,5.875],'pchip':[0,1,5]}
  for kind,want in expected.items():
   r=self.runop('interp_eval',kind=kind,x=[0,3,4,6,7,9],y=[0,0,1,1,5,5],x_new=[1.5,4.4,8]);np.testing.assert_allclose(r['values'],want,atol=1e-13)
 def test_bessel_phase_frequency(self):
  for order,want in [(2,1/math.sqrt(3)),(4,.4179214269322225)]:
   r=self.runop('sig_filter',family='bessel',order=order,fs=1000,wn=80,freqs=[80]);self.assertAlmostEqual(r['magnitude'][0],want,13)
   shifted=self.runop('sig_filter',family='bessel',order=order,fs=1000,wn=40,freqs=[80]);self.assertLess(shifted['magnitude'][0],want)
 def test_width_is_half_prominence(self):
  r=self.runop('sig_peaks',x=[0,0,4,2,2]);self.assertEqual(r['heights'],[4]);self.assertEqual(r['prominences'],[2]);self.assertEqual(r['widths'],[.75])
 def test_spectrum_field_contract_and_signed_csd(self):
  t=np.arange(256);x=np.sin(2*np.pi*32*t/256);y=np.cos(2*np.pi*32*t/256)
  for method in ['periodogram','welch','csd','coherence']:
   p=dict(method=method,x=x.tolist(),window='boxcar',detrend='none',fs=1000)
   if method!='periodogram':p.update(nperseg=128 if method=='coherence' else 256,noverlap=0)
   if method in ('csd','coherence'):p['y']=y.tolist()
   r=self.runop('sig_spectrum',**p);self.assertEqual('total_power' in r,method in ('periodogram','welch'))
   if method=='csd':self.assertAlmostEqual(r['pxy_re'][32],0,13);self.assertAlmostEqual(r['pxy_im'][32],.128,13)
 def test_voronoi_exact_connectivity_inline_and_artifact(self):
  pts=[[0,0],[2,0],[2,2],[0,2],[1,1]]
  expected=[{(1.,0.),(0.,1.),None},{(1.,0.),(2.,1.),None},{(2.,1.),(1.,2.),None},{(0.,1.),(1.,2.),None},{(1.,0.),(0.,1.),(2.,1.),(1.,2.)}]
  edges={(0,1):{None,(1.,0.)},(1,2):{None,(2.,1.)},(2,3):{None,(1.,2.)},(0,3):{None,(0.,1.)},(0,4):{(1.,0.),(0.,1.)},(1,4):{(1.,0.),(2.,1.)},(2,4):{(2.,1.),(1.,2.)},(3,4):{(0.,1.),(1.,2.)}}
  for write in [False,True]:
   r=self.runop('spatial_hull_to' if write else 'spatial_hull',points=pts,kind='voronoi',**({'out':self.out()} if write else {}))
   a=old.read_safetensors(self.out()) if write else r
   vertices=np.asarray(a['vertices']);region=np.asarray(a['regions'],dtype=int);offset=np.asarray(a['region_offsets'],dtype=int)
   cv=lambda ids:{None if int(i)==-1 else tuple(vertices[int(i)]) for i in ids}
   for i,rid in enumerate(a['point_region']):self.assertEqual(cv(region[offset[int(rid)]:offset[int(rid)+1]]),expected[i])
   offsets=np.asarray(a['ridge_offsets'],dtype=int);flat=np.asarray(a['ridge_vertices'],dtype=int)
   got={tuple(sorted(map(int,pair))):cv(flat[offsets[j]:offsets[j+1]]) for j,pair in enumerate(a['ridge_points'])}
   self.assertEqual(got,edges)
 def test_mahalanobis_roundoff(self):
  vi=np.array([[2,.25],[.25+5e-17,1.]]);a=np.array([[1,0],[0,1],[2,2.]])
  r=self.runop('spatial_dist',a=a.tolist(),b=[[1,1]],metric='mahalanobis',vi=vi.tolist());self.assertTrue(r['vi_symmetrised']);diff=a-[1,1];want=np.sqrt(np.einsum('ij,jk,ik->i',diff,(vi+vi.T)/2,diff));np.testing.assert_allclose(np.ravel(r['distances']),want,rtol=1e-14)
  self.runop('spatial_dist',2,old.NO_PYTHON,a=a.tolist(),metric='mahalanobis',vi=[[2,.25],[.3,1]])
 def test_undefined_names_rows_and_preserves_write_diagnostics(self):
  p=dict(a=[[0,0],[1,1]],metric='cosine')
  r=self.runop('spatial_dist',**p);self.assertEqual(r['undefined_a_rows'],[0]);self.assertEqual(r['undefined_b_rows'],[0])
  r=self.runop('spatial_dist_to',1,**p,out=self.out());d=r['error']['data'];self.assertEqual(d['n_undefined'],3);self.assertEqual(d['shape'],[2,2]);self.assertEqual(d['metric'],'cosine');self.assertFalse(pathlib.Path(self.out()).exists())
 def test_prespawn_error_codes(self):
  cases=[('sig_spectrum',dict(method='welch',x=[1,0,-1,0]*2,nperseg=70000),'INPUT_TOO_LARGE'),('sig_filter',dict(family='butter',order=2,wn=.2,freqs=[[.1,.2]]),'BAD_DATA'),('sig_filter',dict(family='butter',order=2,wn=.2,freqs=[.1]*201),'INPUT_TOO_LARGE')]
  for op,p,code in cases:
   with self.subTest(op=op,p=p):self.assertEqual(self.runop(op,2,old.NO_PYTHON,**p)['error']['data']['tool_code'],code)
 def test_time_limit_uniform_and_guarded(self):
  for op,p in [('sig_filter',dict(family='butter',order=2,wn=.2)),('sig_spectrum',dict(method='welch',x=[1,0,-1,0]*20)),('sig_peaks',dict(x=[0,1,0])),('spatial_knn',dict(points=[[0],[1]],k=1))]:
   r=self.runop(op,**p,time_limit=1e-9);self.assertEqual(r['status'],'TIME_LIMIT')
 def test_near_unit_poles_use_actual_quadratics(self):
  import decimal
  r=self.runop('sig_filter',family='butter',order=20,wn=1e-10)
  with decimal.localcontext() as ctx:
   ctx.prec=80;expected=decimal.Decimal(0)
   for row in r['sos']:
    a,b,c=[decimal.Decimal.from_float(float(v)) for v in row[3:]];disc=b*b-4*a*c
    radius=(c/a).sqrt() if disc<0 else max(abs((-b+disc.sqrt())/(2*a)),abs((-b-disc.sqrt())/(2*a)))
    expected=max(expected,radius)
  self.assertAlmostEqual(r['max_pole_radius'],float(expected),14)
 def test_schema_documentation_contract(self):
  schema=json.loads(subprocess.check_output([BIN,'--schema'],text=True));ops={o['name']:o for o in schema['operations']}
  for name in ['interp_eval','interp_eval_to']:
   p={p['name']:p for p in ops[name]['params']};self.assertIn('can overshoot',p['kind']['description']);self.assertNotIn('do not overshoot',p['kind']['description'])
  for name in ['sig_filter','sig_filter_to']:
   p={p['name']:p for p in ops[name]['params']};self.assertIn('phase-normalisation',p['wn']['description'])
  p={p['name']:p for p in ops['sig_peaks']['params']};self.assertIn('half prominence',p['rel_height']['description'])
  for name in ['opt_minimize','sm_ols','sig_peaks','spatial_knn']:self.assertIn('time_limit',{p['name'] for p in ops[name]['params']})
 def test_signal_cleanup_staging(self):
  marker=self.dir/'started';wrapper=self.dir/'wait-python';wrapper.write_text('#!/usr/local/bin/python3\nimport pathlib,sys,time\npathlib.Path('+repr(str(marker))+').write_text(sys.argv[-2])\ntime.sleep(30)\n');wrapper.chmod(0o755)
  p=subprocess.Popen([BIN,'--input-json',json.dumps({'op':'interp_eval_to','params':dict(kind='linear',x=[0,1],y=[0,1],x_new=[.5],out=self.out())})],env=dict(os.environ,PYCLI_PYTHON=str(wrapper)),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
  try:
   for _ in range(200):
    if marker.exists():break
    time.sleep(.01)
   self.assertTrue(marker.exists());staging=pathlib.Path(marker.read_text()).parent;self.assertTrue((staging/'y.npy').exists());p.send_signal(signal.SIGTERM);p.communicate(timeout=4);self.assertFalse(staging.exists())
  finally:
   if p.poll() is None:p.kill();p.communicate()

class WorkerWorkBounds(unittest.TestCase):
 def worker(self):
  sys.path.insert(0,str(ROOT/'tests/oracle'));from s3 import load_worker
  return load_worker()
 def test_peaks_refuse_before_expensive_prominences(self):
  w=self.worker();x=np.arange(200000,dtype=float)+2*(np.arange(200000)%2)
  with tempfile.TemporaryDirectory() as d:
   file=pathlib.Path(d)/'x.npy';np.save(file,x);p=dict(x_file=str(file),rel_height=.5,time_budget_s=10)
   with mock.patch('scipy.signal.peak_prominences',side_effect=AssertionError('unbounded prominence search')):
    with self.assertRaises(w.OutputTooLarge):w.sig_peaks(p)
   p['output_prefix']=str(pathlib.Path(d)/'out_')
   with mock.patch('scipy.signal.peak_prominences',side_effect=AssertionError('unbounded prominence search')):
    with self.assertRaises(w.BadData):w.sig_peaks(p)
 def test_radius_early_budget_exit(self):
  w=self.worker()
  class Tree:
   calls=0
   def __init__(self,points):self.n=len(points)
   def query_ball_point(self,q,r,**kw):raise AssertionError('counts every pair before cap')
   def query(self,q,k,**kw):
    self.calls+=1
    if self.calls>12:raise AssertionError('did not stop early')
    return np.zeros(k),np.arange(k)
  with tempfile.TemporaryDirectory() as d:
   file=pathlib.Path(d)/'x.npy';np.save(file,np.zeros((10000,1)))
   with mock.patch('scipy.spatial.cKDTree',Tree):
    with self.assertRaises(w.OutputTooLarge):w.spatial_knn(dict(points_file=str(file),p=2,radius=1,max_pairs=5,exclude_self=False))

class ArtifactContract(old.Base):
 def test_one_descriptor_schema_across_slices(self):
  expected={'ref','path','format','key','dtype','shape','nbytes'}
  cases=[('array_import',dict(format='json',data=[1,2]),'x'),('la_solve_to',dict(a=[[2]],b=[4]),'x'),('la_lstsq_to',dict(a=[[1],[2]],b=[2,4]),'x'),('la_decomp',dict(a=[[2]],kind='svd'),'u'),('opt_linprog_to',dict(c=[1],bounds=[[0,1]]),'x'),('ode_solve_to',dict(rhs=['-y[0]'],y0=[1],t_span=[0,1],t_eval=[0,1]),'y'),('sm_ols_to',dict(columns={'x':[0,1,2,3],'y':[1,3,5,7]},y='y',terms=[{'const':True},{'var':'x'}]),'fitted'),('frame_import',dict(columns={'x':[1,2]}),'table'),('ml_fit',dict(family='ridge',columns={'x':[0,1,2],'y':[1,3,5]},features=['x'],target='y',preprocess={'scale':False,'one_hot':[]}),'model'),('interp_eval_to',dict(kind='linear',x=[0,1],y=[0,1],x_new=[.5]),'values')]
  for i,(op,p,key) in enumerate(cases):
   with self.subTest(op=op):
    out=self.out(str(i)+('.npz' if op=='frame_import' else '.safetensors'));r=old.call(op,dict(p,out=out));self.assertTrue(r['outputs_in_artifact']);self.assertIn(key,r['artifacts']);self.assertFalse({'tensors','artifact','model'}&set(r))
    for descriptor in r['artifacts'].values():self.assertEqual(set(descriptor),expected);self.assertEqual(descriptor['path'],out);self.assertGreater(descriptor['nbytes'],0)
if __name__=='__main__':unittest.main()
