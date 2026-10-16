"""S4 review regressions: original-data certificates, native statuses, ODE metadata and access twins."""
import itertools,json,math,os,pathlib,random,subprocess,sys,tempfile,unittest
import numpy as np
from unittest.mock import patch
BIN=sys.argv.pop(1);ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tests/oracle'))
from lp_vertices import optimum
from s3 import load_worker
W=load_worker()
class Review(unittest.TestCase):
 def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.cwd=pathlib.Path(self.tmp.name)
 def tearDown(self):self.tmp.cleanup()
 def call(self,op='opt_linprog',rc=0,env=None,**p):
  r=subprocess.run([BIN,op,'--input-json',json.dumps(p)],cwd=self.cwd,env=env,capture_output=True,text=True,timeout=45)
  self.assertEqual(r.returncode,rc,(r.stdout,r.stderr));return json.loads(r.stdout)
 def ode(self,**p):return self.call('ode_solve',**dict({'rhs':['-y[0]'],'y0':[1],'t_span':[0,1]},**p))
 def test_astra_exact_small_coefficients(self):
  for scale in (1e-100,1e-12,1e-10,1e-9,1,1e8,1e12):
   for integer in (False,True):
    for bounded in (False,True):
     p=dict(c=[-1],a_ub=[[scale]],b_ub=[scale])
     if integer:p['integrality']=['integer']
     if bounded:p['bounds']=[[0,2]]
     with self.subTest(scale=scale,integer=integer,bounded=bounded):
      r=self.call(**p);self.assertEqual(r['status'],'OK');self.assertAlmostEqual(r['x'][0],1);self.assertAlmostEqual(r['objective'],-1)
      if not integer:self.assertTrue(r['dual_feasible']);self.assertAlmostEqual(r['duals']['ub'][0]*scale,-1,places=8)
 def test_wide_dynamic_range_property_exact_vertices(self):
  rng=random.Random(410247)
  for k in range(24):
   a=[[rng.choice([-4,-2,-1,1,2,3]) for _ in range(2)] for _ in range(3)];b=[rng.randint(1,8) for _ in a]
   base=dict(c=[rng.choice([-4,-1,1,3]) for _ in range(2)],a_ub=a,b_ub=b,bounds=[[-3,3]]*2,maximize=bool(k%2))
   if k%4==0:base.update(a_eq=[[1,-1]],b_eq=[0])
   want=optimum(base);p=dict(base);scales=[10.**rng.randint(-14,14) for _ in a];objective_scale=10.**rng.randint(-12,12)
   p.update(a_ub=[[s*v for v in row] for row,s in zip(a,scales)],b_ub=[s*v for v,s in zip(b,scales)],c=[v*objective_scale for v in base['c']])
   r=self.call(**p);self.assertEqual(r['status'],'OK',r);self.assertTrue(r['primal_feasible']);self.assertTrue(r['dual_feasible']);self.assertTrue(math.isclose(r['objective']/objective_scale,want,rel_tol=1e-7,abs_tol=1e-8),r)
 def test_opus_huge_inputs_before_spawn(self):
  cases=[dict(c=[1],maximize=True,bounds=[[0,1e25]]),dict(c=[1],maximize=True,a_ub=[[1]],b_ub=[1e25]),dict(c=[1],bounds=[[-1e21,5]]),dict(c=[1e20]),dict(c=[1,1],a_ub=[[-1e16,-1]],b_ub=[-1])]
  for p in cases:
   with self.subTest(p=p):
    r=self.call(rc=2,env=dict(os.environ,PYCLI_PYTHON='/missing-python'),**p);self.assertEqual(r['error']['data']['tool_code'],'VALUE_EXCEEDS_SOLVER_LIMIT')
  self.call(rc=2,env=dict(os.environ,PYCLI_PYTHON='/missing-python'),c=[-1,-1],a_ub=[[1,1e-14]],b_ub=[1])
 def test_native_status_mapping(self):
  for code,want in [(2,'MODEL_ERROR'),(7,'OK'),(8,'INFEASIBLE'),(9,'INFEASIBLE_OR_UNBOUNDED'),(10,'UNBOUNDED'),(13,'TIME_LIMIT'),(14,'ITERATION_LIMIT'),(16,'NODE_LIMIT')]:
   self.assertEqual(W.highs_status('anything (HiGHS Status %d: test)'%code), (want,code))
  r=self.call(c=[-1,-1],integrality=['integer']*2,bounds=[[0,None]]*2);self.assertEqual(r['status'],'INFEASIBLE_OR_UNBOUNDED');self.assertEqual(r['highs_model_status'],9)
  r=self.call(c=[1],a_ub=[[1],[-1]],b_ub=[0,-1]);self.assertEqual(r['status'],'INFEASIBLE');self.assertEqual(r['highs_model_status'],8)
 def test_failed_ode_metadata_and_short_grid(self):
  r=self.ode(rhs=['y[0]^2'],t_span=[0,2],rtol=1e-10,atol=1e-12);self.assertEqual(r['status'],'STEP_SIZE_TOO_SMALL');self.assertEqual(r['termination'],'failed');self.assertTrue(r['partial_trajectory'])
 def test_short_grid(self):
  for span in ([1e9,1e9+.000001],[1e9+.000001,1e9]):
   r=self.ode(rhs=['1'],y0=[0],t_span=span);self.assertEqual(r['status'],'OK');self.assertAlmostEqual(r['final_state'][0],span[1]-span[0],delta=1e-15);self.assertEqual(len(set(r['t'])),len(r['t']));self.assertLess(r['effective_output_points'],101)
 def test_initial_event_and_integer_indices(self):
  for direction in (-1,0,1):
   for terminal in (False,True):
    r=self.ode(events=[{'expr':'y[0]-1','direction':direction,'terminal':terminal}]);self.assertEqual(r['event_counts'],[0]);self.assertEqual(r['final_time'],1)
 def test_integer_event_indices(self):
  r=self.ode(events=[{'expr':'y[0]-0.5','direction':-1}]);self.assertIs(type(r['event_index'][0]),int)
  r=self.ode(rhs=['1'],y0=[0],t_span=[0,8],max_step=.1,events=[{'expr':'sin(t)','direction':1,'terminal':True}]);self.assertAlmostEqual(r['final_time'],2*math.pi,places=8);self.assertEqual(r['event_counts'],[1]);self.assertIs(type(r['event_index'][0]),int)
  r=self.ode(rhs=['y[1]','-9.81'],y0=[10,0],t_span=[0,5],t_eval=[0,1,3,5],events=[{'expr':'y[0]','direction':-1,'terminal':True}]);self.assertAlmostEqual(r['t'][-1],math.sqrt(20/9.81),places=8);self.assertAlmostEqual(r['y'][-1][0],0,delta=1e-8)
 def test_limits_equality_duals_and_integer_output(self):
  rng=random.Random(410247);weights=[[rng.randint(10,90) for _ in range(60)] for _ in range(5)];values=[rng.randint(10,90) for _ in range(60)]
  p=dict(c=values,a_ub=weights,b_ub=[sum(row)//2 for row in weights],integrality=['binary']*60,maximize=True)
  r=self.call(**p,node_limit=0);self.assertEqual(r['status'],'NODE_LIMIT');self.assertEqual(r['highs_model_status'],16);self.assertFalse(r['has_primal'])
  # time_limit 0.001 on a 60-variable binary knapsack: TIME_LIMIT always, by one of two documented paths. Under host load the 1 ms budget
  # can expire before HiGHS is even started (py/s4.py: 'wall budget exhausted before HiGHS', no HiGHS fields); otherwise HiGHS itself
  # stops with model status 13. Both prove that the limit reached the solver layer; neither may answer OK. A fast machine can hold an
  # integer-feasible incumbent at the limit (has_primal true, seen on the GitHub runners): then x must be binary and within the budget.
  r=self.call(**p,time_limit=.001);self.assertEqual(r['status'],'TIME_LIMIT')
  self.assertTrue(r.get('highs_model_status')==13 or r.get('message')=='wall budget exhausted before HiGHS',r)
  if r['has_primal']:self.assertTrue(all(x in (0,1) for x in r['x']) and sum(w*x for w,x in zip(weights[0],r['x']))<=p['b_ub'][0],r)
  r=self.call(**p,node_limit=1);self.assertEqual(r['status'],'NODE_LIMIT');self.assertEqual(r['highs_model_status'],16);self.assertTrue(r['has_primal']);self.assertTrue(all(x==round(x) and (x!=0 or math.copysign(1,x)==1) for x in r['x']))
  p=dict(c=[3,5],a_ub=[[1,0],[0,2]],b_ub=[4,12],a_eq=[[3,2]],b_eq=[18],maximize=True);r=self.call(**p);s=self.call(**dict(p,b_eq=[18.0001]));self.assertAlmostEqual((s['objective']-r['objective'])/.0001,r['duals']['eq'][0],places=7)
 def test_read_twins_and_write_twins(self):
  for op,p in [('opt_linprog',{'c':[1]}),('ode_solve',{'rhs':['-y[0]'],'y0':[1],'t_span':[0,1]})]:
   r=self.call(op,env=dict(os.environ,PYCLI_READONLY='1'),**p);self.assertEqual(r['status'],'OK');self.assertFalse(r['outputs_in_artifact'])
   r=self.call(op+'_to',out='result.safetensors',**p);self.assertTrue(r['outputs_in_artifact']);self.assertTrue((self.cwd/'result.safetensors').exists());(self.cwd/'result.safetensors').unlink()
   self.call(op+'_to',rc=6,env=dict(os.environ,PYCLI_READONLY='1'),out='none.safetensors',**p)
   self.call(op+'_to',env=dict(os.environ,PYCLI_DRYRUN='1'),out='none.safetensors',**p);self.assertFalse((self.cwd/'none.safetensors').exists())
  self.ode(rc=1,max_output_points=1000);self.assertEqual(list(self.cwd.iterdir()),[])
 def test_original_certificate_fault_injection_and_rounding(self):
  from scipy.optimize import OptimizeResult
  def run(x,dual,integer=False):
   np.save(self.cwd/'c.npy',[-1.]);np.save(self.cwd/'a.npy',[[1e-10]]);np.save(self.cwd/'b.npy',[1e-10])
   p=dict(c_file=str(self.cwd/'c.npy'),a_ub_file=str(self.cwd/'a.npy'),b_ub_file=str(self.cwd/'b.npy'),maximize=False,bounds=[[0,2]],integrality=[int(integer)],time_limit=10,time_budget_s=10,maxiter=10000,node_limit=10000,mip_rel_gap=0,force_artifact=False)
   a=OptimizeResult(status=0,message='Optimization terminated (HiGHS Status 7: Optimal)',nit=0,x=np.array([x]),ineqlin={'marginals':np.array([dual])},eqlin={'marginals':np.array([])},lower={'marginals':np.array([0.])},upper={'marginals':np.array([0.])},mip_gap=0)
   with patch('scipy.optimize.linprog',return_value=a):return W.opt_linprog(p)
  for x,dual in [(2,0),(1,0),(1,float('nan'))]:
   with self.subTest(x=x,dual=dual):self.assertEqual(run(x,dual)['status'],'NUMERICAL_FAILURE')
  r=run(.99999999,0,True);self.assertEqual(r['x'],[1]);self.assertGreater(r['integrality_error'],0)
 def test_integer_zero_sign_and_artifact_flag(self):
  r=self.call(c=[3,4,5,8],a_ub=[[2,3,4,5]],b_ub=[8],integrality=['binary']*4,maximize=True)
  self.assertTrue(all(x==round(x) and (x!=0 or math.copysign(1,x)==1) for x in r['x']))
  self.assertFalse(r['outputs_in_artifact'])
if __name__=='__main__':unittest.main()
