"""S4 independent hand optima, exhaustive integer search, analytic ODEs and primary ODEPACK data."""
import itertools,json,math,os,pathlib,subprocess,sys,tempfile,unittest,struct,re
BIN=sys.argv.pop(1);ROOT=pathlib.Path(__file__).resolve().parents[1]
class S4(unittest.TestCase):
 def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.cwd=pathlib.Path(self.tmp.name)
 def tearDown(self):self.tmp.cleanup()
 def call(self,op,rc=0,env=None,**p):
  r=subprocess.run([BIN,op,'--input-json',json.dumps(p)],cwd=self.cwd,env=env,capture_output=True,text=True,timeout=48)
  self.assertEqual(r.returncode,rc,(op,r.stdout,r.stderr));return json.loads(r.stdout)
 def ode(self,**p):return self.call('ode_solve',**dict({'rhs':['-y[0]'],'y0':[1],'t_span':[0,1]},**p))
 def test_production_diet_duals(self):
  # Wyndor: max 3x+5y subject to x<=4, 2y<=12, 3x+2y<=18; optimum (2,6), 36.
  p=dict(c=[3,5],a_ub=[[1,0],[0,2],[3,2]],b_ub=[4,12,18],maximize=True)
  r=self.call('opt_linprog',**p);self.assertEqual(r['status'],'OK');self.assertTrue(r['primal_feasible']);self.assertEqual(r['warnings'],[]);self.assertAlmostEqual(r['objective'],36);self.assertAlmostEqual(r['x'][0],2);self.assertAlmostEqual(r['x'][1],6)
  for i in range(3):
   q=dict(p,b_ub=list(p['b_ub']));q['b_ub'][i]+=1e-4;s=self.call('opt_linprog',**q);self.assertAlmostEqual((s['objective']-r['objective'])/1e-4,r['duals']['ub'][i],places=6)
  # min x+2y, x+y>=3, x+3y>=5: intersection (2,1), cost4.
  r=self.call('opt_linprog',c=[1,2],a_ub=[[-1,-1],[-1,-3]],b_ub=[-3,-5]);self.assertAlmostEqual(r['objective'],4)
 def test_eq_and_bound_dual_sensitivity(self):
  p={'c':[1,2],'a_eq':[[1,1]],'b_eq':[1]};r=self.call('opt_linprog',**p);s=self.call('opt_linprog',**dict(p,b_eq=[1.0001]));self.assertAlmostEqual((s['objective']-r['objective'])/.0001,r['duals']['eq'][0],places=7)
  for maximize,which,base,changed in [(False,'lower',[2,5],[2.0001,5]),(True,'upper',[2,5],[2,5.0001])]:
   r=self.call('opt_linprog',c=[1],bounds=[base],maximize=maximize);s=self.call('opt_linprog',c=[1],bounds=[changed],maximize=maximize);self.assertAlmostEqual((s['objective']-r['objective'])/.0001,r['duals'][which][0],places=7)
 def test_lp_statuses_degenerate_and_fixed_bounds(self):
  r=self.call('opt_linprog',c=[1,1],a_eq=[[1,1]],b_eq=[1]);self.assertEqual(r['status'],'OK');self.assertAlmostEqual(sum(r['x']),1);self.assertAlmostEqual(r['objective'],1)
  self.assertEqual(self.call('opt_linprog',c=[1],a_ub=[[1],[-1]],b_ub=[0,-1])['status'],'INFEASIBLE')
  self.assertEqual(self.call('opt_linprog',c=[-1])['status'],'UNBOUNDED')
  r=self.call('opt_linprog',c=[1],bounds=[[2,2]]);self.assertAlmostEqual(r['objective'],2)
  r=self.call('opt_linprog',c=[1],bounds=[[None,None]]);self.assertEqual(r['status'],'UNBOUNDED')
 def test_milp_knapsack_assignment_bruteforce(self):
  weights=[2,3,4,5];values=[3,4,5,8];poss=list(itertools.product((0,1),repeat=4));want=max(sum(a*b for a,b in zip(x,values)) for x in poss if sum(a*b for a,b in zip(x,weights))<=8)
  r=self.call('opt_linprog',c=values,a_ub=[weights],b_ub=[8],integrality=['binary']*4,maximize=True);self.assertEqual(r['status'],'OK');self.assertAlmostEqual(r['objective'],want);self.assertTrue(all(abs(v-round(v))<1e-9 for v in r['x']));self.assertIn('mip_gap',r);self.assertNotIn('duals',r)
  costs=[[9,2,7],[6,4,3],[5,8,1]];rows=[[int(k//3==i) for k in range(9)] for i in range(3)]+[[int(k%3==j) for k in range(9)] for j in range(3)]
  want=min(sum(costs[i][j] for i,j in enumerate(perm)) for perm in itertools.permutations(range(3)))
  r=self.call('opt_linprog',c=sum(costs,[]),a_eq=rows,b_eq=[1]*6,integrality=['binary']*9);self.assertAlmostEqual(r['objective'],want)
 def test_matrix_refs_artifact_and_prespawn(self):
  def npy(name,shape,values):
   header=("{'descr': '<f8', 'fortran_order': False, 'shape': "+repr(shape)+", }").encode();header+=b' '*((64-(10+len(header)+1)%64)%64)+b'\n'
   (self.cwd/name).write_bytes(b'\x93NUMPY\x01\0'+struct.pack('<H',len(header))+header+struct.pack('<%dd'%len(values),*values))
  npy('a.npy',(400,400),[float(i==j) for i in range(400) for j in range(400)]);npy('b.npy',(400,),[1.]*400);npy('c.npy',(400,),[-1.]*400)
  r=self.call('opt_linprog',c='c.npy',a_ub='a.npy',b_ub='b.npy');self.assertEqual(r['status'],'OK');self.assertIsInstance(r['x'],dict);self.assertLess(len(json.dumps(r)),16000);self.assertTrue(pathlib.Path(r['artifact']['path']).is_file())
  x=self.call('array_inspect',ref=r['x']['ref'],limit=5);self.assertEqual(x['preview'],[1]*5)
  npy('too.npy',(400,501),[0.]*(400*501))
  self.call('opt_linprog',rc=2,env=dict(os.environ,PYCLI_PYTHON='/missing-python'),c=[1],a_ub='too.npy',b_ub=[1])
  for params in ({'c':[1,2],'a_ub':[[1]],'b_ub':[1]},{'c':[1],'integrality':['semi']},{'c':[1],'node_limit':100001},{'c':['NaN']},{'c':[1],'time_limit':31}):self.call('opt_linprog',rc=2,env=dict(os.environ,PYCLI_PYTHON='/missing-python'),**params)
  self.call('opt_linprog',rc=6,env=dict(os.environ,PYCLI_READONLY='1'),c=[1],out='no.safetensors');self.assertFalse((self.cwd/'no.safetensors').exists())
 def test_lp_limits_artifacts_and_policies(self):
  r=self.call('opt_linprog',c=[1],time_limit=1e-9);self.assertEqual(r['status'],'TIME_LIMIT')
  r=self.call('opt_linprog',c=[3,4,5,8],a_ub=[[2,3,4,5]],b_ub=[8],integrality=['binary']*4,maximize=True,node_limit=0);self.assertIn(r['status'],('OK','NODE_LIMIT'))
  self.call('array_import',format='json',data=[-1,-2],out='input.safetensors')
  r=self.call('opt_linprog',c='input.safetensors#x',bounds=[[0,1],[0,1]]);self.assertEqual(r['x'],[1,1]);self.assertEqual(r['objective'],-3)
  self.call('opt_linprog',rc=1,c=[1],a_ub=[[1],[-1]],b_ub=[0,-1],out='none.safetensors');self.assertFalse((self.cwd/'none.safetensors').exists())
  self.ode(rc=6,env=dict(os.environ,PYCLI_READONLY='1'))
  for op,p in [('opt_linprog',{'c':[1]}),('ode_solve',{'rhs':['-y[0]'],'y0':[1],'t_span':[0,1]})]:
   self.call(op,env=dict(os.environ,PYCLI_DRYRUN='1'),out='dry.safetensors',**p);self.assertFalse((self.cwd/'dry.safetensors').exists())
 def test_highs_license_and_primary_fixture(self):
  license=(ROOT/'licenses/scipy/subprojects/highs/LICENSE.txt').read_bytes();self.assertIn(b'MIT License',license);self.assertIn(license,(ROOT/'THIRD_PARTY_NOTICES.md').read_bytes())
  source=json.loads((ROOT/'tests/fixtures/robertson-source.json').read_text());self.assertEqual(source['sha256'],'da31255ff43e3a13c2c0ccaed51c3cd260b8ac9d9082f34b8587c9007d9a2376');self.assertTrue(any(line==3931 and '7.158403e-01' in text for line,text in source['rows']))
 def test_decay_all_methods_and_backward(self):
  for method in ('RK45','RK23','DOP853','Radau','BDF','LSODA'):
   r=self.ode(method=method,rtol=1e-9,atol=1e-12);self.assertEqual(r['status'],'OK');self.assertAlmostEqual(r['final_state'][0],math.exp(-1),delta=2e-8)
  r=self.ode(t_span=[1,0],y0=[math.exp(-1)],t_eval=[1,.5,0],rtol=1e-10,atol=1e-12);self.assertAlmostEqual(r['final_state'][0],1,delta=2e-9)
 def test_harmonic_lotka_conservation(self):
  r=self.ode(rhs=['y[1]','-y[0]'],y0=[1,0],t_span=[0,20],max_output_points=101,method='DOP853',rtol=1e-10,atol=1e-12)
  self.assertEqual(r['status'],'OK')
  for t,ys in zip(r['t'],r['y']):self.assertAlmostEqual(ys[0],math.cos(t),delta=1e-8);self.assertAlmostEqual(sum(v*v for v in ys),1,delta=1e-8)
  r=self.ode(rhs=['a*y[0]-b*y[0]*y[1]','d*y[0]*y[1]-g*y[1]'],params={'a':1.5,'b':1,'d':1,'g':3},y0=[2,1],t_span=[0,20],method='DOP853',rtol=1e-10,atol=1e-12,max_output_points=101)
  def h(y):return y[0]-3*math.log(y[0])+y[1]-1.5*math.log(y[1])
  self.assertLess(max(abs(h(y)-h([2,1])) for y in r['y']),1e-6)
 def test_rhs_event_r_semantics(self):
  r=self.ode(rhs=['ifelse(log(y[0]-2)>0,1,0)']);self.assertEqual(r['status'],'NON_FINITE')
  r=self.ode(events=[{'expr':'ifelse(log(y[0]-2)>0,1,0)'}]);self.assertEqual(r['status'],'NON_FINITE')
  r=self.ode(rhs=['choose(5,2.5)'],y0=[0]);self.assertAlmostEqual(r['final_state'][0],10,places=8);self.assertEqual(sum(w['count'] for w in r['warnings']),1)
 def test_robertson_primary_reference(self):
  # ODEPACK DLSODA example, opkdmain.f line3931, run at RTOL=1e-4. Its displayed digits are not certified accuracy.
  for method in ('BDF','Radau'):
   r=self.ode(rhs=['-0.04*y[0]+10000*y[1]*y[2]','0.04*y[0]-10000*y[1]*y[2]-30000000*y[1]^2','30000000*y[1]^2'],y0=[1,0,0],t_span=[0,40],method=method,rtol=1e-10,atol=1e-13)
   self.assertEqual(r['status'],'OK');self.assertAlmostEqual(r['final_state'][0],.7158403,delta=3e-5);self.assertAlmostEqual(r['final_state'][1],.9186334e-5,delta=2e-9);self.assertAlmostEqual(r['final_state'][2],.2841505,delta=3e-5);self.assertAlmostEqual(sum(r['final_state']),1,delta=1e-11)
 def test_ball_event_and_event_limit(self):
  r=self.ode(rhs=['y[1]','-g'],params={'g':9.81},y0=[10,0],t_span=[0,5],events=[{'expr':'y[0]','direction':-1,'terminal':True}],rtol=1e-10,atol=1e-12)
  self.assertEqual(r['status'],'OK');self.assertEqual(r['termination'],'event');self.assertAlmostEqual(r['final_time'],math.sqrt(20/9.81),places=9);self.assertAlmostEqual(r['final_state'][0],0,delta=1e-9);self.assertEqual(r['event_counts'],[1])
  r=self.ode(rhs=['0'],t_span=[0,100],max_step=.1,events=[{'expr':'0'}]);self.assertEqual(r['status'],'EVENT_LIMIT');self.assertEqual(r['event_counts'],[32])
 def test_budgets_nonfinite_and_step_failure(self):
  for method in ('RK45','RK23','DOP853','Radau','BDF','LSODA'):
   r=self.ode(method=method,max_evals=1);self.assertEqual(r['status'],'EVAL_LIMIT');self.assertLessEqual(r['evals'],1)
  r=self.ode(time_limit=1e-9);self.assertEqual(r['status'],'TIME_LIMIT')
  r=self.ode(rhs=['1/0']);self.assertEqual(r['status'],'NON_FINITE')
  r=self.ode(events=[{'expr':'0/0'}]);self.assertEqual(r['status'],'NON_FINITE')
  r=self.ode(t_span=[0,1000000000000],max_step=.001,max_steps=5);self.assertEqual(r['status'],'STEP_LIMIT');self.assertLessEqual(r['steps'],5)
  r=self.ode(rhs=['y[0]^2'],t_span=[0,2],rtol=1e-10,atol=1e-12);self.assertEqual(r['status'],'STEP_SIZE_TOO_SMALL')
 def test_ode_artifacts_and_named_state(self):
  r=self.ode(max_output_points=1000,out='o.safetensors');self.assertEqual(r['status'],'OK');self.assertIsInstance(r['y'],dict);self.assertTrue((self.cwd/'o.safetensors').exists());self.assertLess(len(json.dumps(r)),16000)
  self.assertAlmostEqual(r['final_state'][0],math.exp(-1),delta=1e-5)
  r=self.ode(rhs={'s':'-k*s'},y0={'s':1},params={'k':2});self.assertAlmostEqual(r['final_state'][0],math.exp(-2),delta=1e-5)
 def test_injection_and_caps_before_spawn(self):
  env=dict(os.environ,PYCLI_PYTHON='/missing-python')
  corpus=json.loads((ROOT/'tests/expr-corpus.json').read_text())
  negative=corpus['reject'];negative=[x['expression'] if isinstance(x,dict) else x for x in negative]
  vector=json.loads((ROOT/'tests/expr-corpus-pycli.json').read_text())
  for case in vector['reject']:
   text=re.sub(r'\b[xy]\b',lambda m:{'x':'y','y':'t'}[m[0]],case['expression'])
   for path in ('rhs','events'):
    p={'rhs':[text,'0']} if path=='rhs' else {'rhs':['0','0'],'events':[{'expr':text}]}
    self.ode(rc=2,env=env,y0=[1,0],**p)
  negative+=['y[-1]','y[1]','y[0].__class__','y[0]**2','t; import os','__import__("os")','lambda:1']
  for text in negative:
   for path in ('rhs','events'):
    p={'rhs':[text]} if path=='rhs' else {'events':[{'expr':text}]}
    self.ode(rc=2,env=env,**p)
  for p in ({'y0':[1]*33,'rhs':['0']*33},{'t_span':[0,1e300]},{'events':[{'expr':'t'}]*5},{'t_eval':[0,.5,.4,1]},{'max_output_points':10001},{'rtol':0},{'atol':-1},{'params':{'t':1}},{'method':'unsafe'}):self.ode(rc=2,env=env,**p)
 def test_atol_type_validation_before_spawn(self):
  for value in ('0.000001',True,{},None):
   with self.subTest(value=value):self.ode(rc=2,env=dict(os.environ,PYCLI_PYTHON='/missing-python'),atol=value)
if __name__=='__main__':unittest.main()
