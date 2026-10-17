"""Independent table oracles: pandas directly plus hand-computed tables."""
import json, pathlib, subprocess, sys, tempfile, unittest, zipfile, struct, os
from oracle.frames import grouped_sales, spring_hourly
BIN=sys.argv.pop(1)
class Frames(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.cwd=pathlib.Path(self.tmp.name)
 def tearDown(self):self.tmp.cleanup()
 def call(self,op,rc=0,**p):
  r=subprocess.run([BIN,'--input-json',json.dumps({'op':op,'params':p},ensure_ascii=False)],cwd=self.cwd,capture_output=True,text=True)
  self.assertEqual(r.returncode,rc,(op,r.stdout,r.stderr)); j=json.loads(r.stdout);return j.get('result',j)
 def table(self,columns,**p):return self.call('frame_import',columns=columns,out='t.npz',**p)
 def inspect(self,ref='t.npz',**p):return self.call('frame_inspect',frame=ref,head=20,**p)
 def test_hand_and_direct_oracle(self):
  import pandas as pd
  self.table({'region':['E','W','E','W'],'sales':[10,20,30,40]})
  r=self.call('frame_groupby',frame='t.npz',keys=['region'],aggs={'sales':['sum','mean','count']},out='g.npz')
  got=self.inspect('g.npz')['head']['data'];self.assertEqual(got,{'region':['E','W'],'sales__sum':[40,60],'sales__mean':[20.,30.],'sales__count':[2,2]})
  self.assertEqual(got['sales__sum'],grouped_sales())
 def test_dtype_identity(self):
  cols={'i':[9223372036854775807,None,-9223372036854775808],'f':['NaN','Inf',None],'b':[True,None,False],'s':['NA','雪😀',None],'dt':['2026-01-01T00:00:00.000000001',None,'2026-01-03'],'cat':['b','a',None]}
  d={'i':'int64','f':'float64','b':'bool','s':'string','dt':'datetime64[ns]','cat':{'dtype':'category','categories':['b','a','unused'],'ordered':True}}
  self.table(cols,dtypes=d);got=self.inspect();self.assertEqual(got['head']['data']['i'],cols['i']);self.assertEqual(got['head']['data']['f'],cols['f']);self.assertEqual(got['head']['data']['s'],cols['s']);self.assertEqual(got['null_counts']['f'],1)
  self.assertEqual(got['column_meta']['cat']['categories'],['b','a','unused']);self.assertTrue(got['column_meta']['cat']['ordered']);self.assertEqual(got['head']['data']['dt'],[cols['dt'][0],None,'2026-01-03T00:00:00']);self.assertEqual(got['head']['data']['b'],cols['b']);self.assertEqual(got['head']['data']['cat'],cols['cat'])
  self.call('frame_import',frame='t.npz',out='u.npz');self.assertEqual(self.inspect('u.npz')['head'],got['head'])
  import numpy as np
  with np.load(self.cwd/'t.npz',allow_pickle=False) as z:
   self.assertIn('__pycli_table__',z.files)
   for key in z.files:self.assertFalse(z[key].dtype.hasobject)
 def test_csv(self):
  (self.cwd/'a.csv').write_bytes('\ufeff名前,x\r\n"雪,人",1\r\n"a""b",2\r\n'.encode())
  self.call('frame_import',csv='a.csv',out='t.npz');self.assertEqual(self.inspect()['head']['data']['名前'],['雪,人','a"b'])
  for text in ('a,a\n1,2\n',',b\n1,2\n','a,b\n1\n','a\n1e999\n','a\n'+('9'*10000)+'\n'):
   (self.cwd/'bad.csv').write_text(text);self.call('frame_import',rc=2,csv='bad.csv',out='bad.npz')
 def test_predicates_data_and_stable_sort(self):
  payload="__import__('os').system('touch OWNED')"
  self.table({'name':[payload,'a','aa',None],'n':[1,2,2,4]})
  self.call('frame_select',frame='t.npz',where={'col':'name','op':'eq','value':payload},out='p.npz');self.assertEqual(self.inspect('p.npz')['head']['data']['name'],[payload]);self.assertFalse((self.cwd/'OWNED').exists())
  self.call('frame_select',frame='t.npz',where={'and':[{'col':'n','op':'between','value':[2,4]},{'not':{'col':'name','op':'isnull'}}]},sort_by=[{'col':'n','desc':True}],columns=['name'],limit=2,out='s.npz');self.assertEqual(self.inspect('s.npz')['head']['data']['name'],['a','aa'])
  for op,val in [('contains','a.*'),('startswith','__import__')]:
   self.call('frame_select',frame='t.npz',where={'col':'name','op':op,'value':val},out=op+'.npz')
  self.assertEqual(self.inspect('contains.npz')['shape'][0],0)
  self.call('frame_select',rc=2,frame='t.npz',where={'col':'n','op':'query','value':'n>0'},out='bad.npz')
 def test_dst(self):
  import pandas as pd
  dates=['2026-03-08T00:30:00-05:00','2026-03-08T01:30:00-05:00','2026-03-08T03:30:00-04:00','2026-03-08T04:30:00-04:00']
  self.table({'ts':dates,'v':[1,2,3,4]},dtypes={'ts':{'dtype':'datetime64[ns]','tz':'America/New_York'}})
  self.call('frame_resample',frame='t.npz',time='ts',rule='1h',aggs={'v':['sum']},out='r.npz')
  got=self.inspect('r.npz')['head']['data'];self.assertEqual(got['v__sum'],[1,2,3,4]);self.assertFalse(any('T02:' in x for x in got['ts']))
  self.assertEqual(got['v__sum'],spring_hourly());self.assertEqual(got['ts'],['2026-03-08T00:00:00-05:00','2026-03-08T01:00:00-05:00','2026-03-08T03:00:00-04:00','2026-03-08T04:00:00-04:00'])
  for rule in ('pd.offsets','1D; import os','1D\\n','0D','999999999D'):
   self.call('frame_resample',rc=2,frame='t.npz',time='ts',rule=rule,aggs={'v':['sum']},out='bad.npz')
 def test_fall_dst_and_bad_dates(self):
  dates=['2026-11-01T01:15:00-04:00','2026-11-01T01:15:00-05:00',None]
  self.table({'ts':dates,'v':[2,3,4]},dtypes={'ts':{'dtype':'datetime64[ns]','tz':'America/New_York'}})
  self.assertEqual(self.inspect()['null_counts']['ts'],1)
  self.call('frame_resample',rc=2,frame='t.npz',time='ts',rule='1h',aggs={'v':['sum']},out='bad.npz')
  self.call('frame_select',frame='t.npz',where={'not':{'col':'ts','op':'isnull'}},out='clean.npz')
  self.call('frame_resample',frame='clean.npz',time='ts',rule='1h',aggs={'v':['sum']},out='r.npz')
  data=self.inspect('r.npz')['head']['data'];self.assertEqual(data['v__sum'],[2,3]);self.assertIn('-04:00',data['ts'][0]);self.assertIn('-05:00',data['ts'][1])
  for value in ('2026-03-08T02:30:00','2026-11-01T01:30:00'):
   self.call('frame_import',rc=2,columns={'ts':[value]},dtypes={'ts':{'dtype':'datetime64[ns]','tz':'America/New_York'}},out='bad.npz')
  self.call('frame_import',rc=2,columns={'ts':['2026-01-01']},dtypes={'ts':{'dtype':'datetime64[ns]','tz':'../../etc/passwd'}},out='bad.npz')
 def test_records_aggregations_and_csv_roundtrip(self):
  import pandas as pd
  self.call('frame_import',records=[{'k':'a','x':1},{'k':'a','x':3},{'k':'b','x':5}],out='t.npz')
  fns=['sum','mean','min','max','count','std','var','median','nunique','first','last']
  self.call('frame_groupby',frame='t.npz',keys=['k'],aggs={'x':fns},out='g.npz');d=self.inspect('g.npz')['head']['data']
  self.assertEqual(d['x__var'][0],2.);self.assertAlmostEqual(d['x__std'][0],2**0.5);self.assertEqual(d['x__first'],[1,5]);self.assertEqual(d['x__last'],[3,5])
  direct=pd.DataFrame({'k':['a','a','b'],'x':[1,3,5]}).groupby('k')['x'].agg(['sum','mean','min','max','count','median','nunique'])
  for fn in direct:self.assertEqual(d['x__'+fn],direct[fn].tolist())
  self.call('frame_export',frame='t.npz',format='csv',out='e.csv');self.call('frame_import',csv='e.csv',out='copy.npz');self.assertEqual(self.inspect('copy.npz')['head'],self.inspect()['head'])
  self.call('frame_groupby',rc=2,frame='t.npz',keys=['k'],aggs={'x':['__class__']},out='bad.npz')
  self.call('frame_import',rc=2,records=[{'a':1},{'b':2}],out='bad.npz')
 def test_large_artifact_preview_and_prespawn(self):
  import numpy as np
  (self.cwd/'large.csv').write_text('x,y\n'+''.join(f'{i},{i*2}\n' for i in range(700)))
  self.call('frame_import',csv='large.csv',out='t.npz');self.assertEqual(self.inspect()['shape'],[700,2])
  self.call('frame_export',rc=1,frame='t.npz',format='json')
  self.call('frame_select',frame='t.npz',out='selected.npz');self.assertEqual(self.inspect('selected.npz')['shape'],[700,2])
  self.call('frame_inspect',frame='t.npz',head=500)
  self.call('frame_inspect',rc=2,frame='t.npz',head=501)
  self.call('frame_import',rc=2,csv='large.csv',max_rows=699,out='bad.npz')
  # A missing interpreter is an oracle that invalid caps fail before any spawn.
  env=dict(os.environ,PYCLI_PYTHON='/missing-python')
  r=subprocess.run([BIN,'frame_inspect','--input-json',json.dumps({'frame':'t.npz','head':501})],cwd=self.cwd,env=env,capture_output=True);self.assertEqual(r.returncode,2,r.stdout)
  with np.load(self.cwd/'t.npz',allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
  metadata=json.loads(arrays['__pycli_table__'].tobytes());metadata['n_rows']=2000001;arrays['__pycli_table__']=np.frombuffer(json.dumps(metadata).encode(),dtype='uint8');np.savez(self.cwd/'too_many.npz',**arrays)
  r=subprocess.run([BIN,'frame_inspect','--input-json','{"frame":"too_many.npz"}'],cwd=self.cwd,env=env,capture_output=True);self.assertEqual(r.returncode,2,r.stdout)
 def test_export_policy_caps(self):
  self.table({'x':[1,None,3]});self.call('frame_export',frame='t.npz',format='json',out='e.json');self.assertEqual(json.loads((self.cwd/'e.json').read_text())['data']['x'],[1,None,3])
  self.call('frame_export',rc=5,frame='t.npz',format='json',out='e.json')
  self.assertEqual(self.call('frame_export',frame='t.npz',format='json')['table']['data']['x'],[1,None,3])
  for flag,rc in [('--readonly',6),('--dryrun',0)]:
   p=subprocess.run([BIN,flag,'frame_import','--input-json',json.dumps({'columns':{'x':[1]},'out':'no.npz'})],cwd=self.cwd,capture_output=True);self.assertEqual(p.returncode,rc);self.assertFalse((self.cwd/'no.npz').exists())
  self.call('frame_import',rc=2,columns={'x':list(range(1001))},out='big.npz')
  self.call('frame_import',rc=2,columns={'a':[[1]]},out='bad.npz')
  self.call('frame_inspect',rc=2,frame='t.npz',head=1001)
 def test_artifact_dtype_overrides_and_nonfinite_aggregate(self):
  self.table({'k':['a','a'],'x':[1,2],'v':['Inf','Inf']},dtypes={'v':'float64'})
  self.call('frame_import',frame='t.npz',dtypes={'x':'float64'},out='converted.npz')
  self.assertEqual(self.inspect('converted.npz')['dtypes']['x'],'float64')
  self.call('frame_groupby',frame='t.npz',keys=['k'],aggs={'v':['sum']},out='g.npz');self.assertEqual(self.inspect('g.npz')['head']['data']['v__sum'],['Inf'])
  self.call('frame_import',rc=2,columns={'x':[1]},dtypes={'x':{'dtype':[]}},out='bad.npz')
 def test_typed_predicate_nonfinite_tokens(self):
  self.table({'x':['Inf','-Inf','NaN',None],'s':['1','two','three','four']},dtypes={'x':'float64'})
  self.call('frame_select',frame='t.npz',where={'col':'x','op':'eq','value':'Inf'},out='inf.npz');self.assertEqual(self.inspect('inf.npz')['shape'][0],1)
  e=self.call('frame_select',rc=2,frame='t.npz',where={'col':'x','op':'eq','value':'text'},out='bad.npz');self.assertIn('column x',e['error']['message'])  # the message names the column
  self.call('frame_select',rc=2,frame='t.npz',where={'col':'s','op':'eq','value':1},out='bad.npz')
 def test_status_ok_on_inspect_and_inline_export(self):
  self.table({'x':[1,2],'s':['a','b']})
  self.assertEqual(self.inspect()['status'],'OK')
  self.assertEqual(self.call('frame_export',frame='t.npz',format='csv')['status'],'OK')
  self.assertEqual(self.call('frame_export',frame='t.npz',format='json')['status'],'OK')
  e=self.call('frame_select',rc=2,frame='t.npz',where={'col':'x','op':'gt','value':'4'},out='bad.npz');self.assertIn('column x',e['error']['message'])
  (self.cwd/'tiny.csv').write_text('x\n1e-999\n');self.call('frame_import',rc=2,csv='tiny.csv',out='bad.npz')
 def test_missing_mask_predicates(self):
  self.table({'x':['Inf','-Inf','NaN',None]},dtypes={'x':'float64'})
  self.call('frame_select',frame='t.npz',where={'col':'x','op':'isnull'},out='null.npz');self.assertEqual(self.inspect('null.npz')['shape'][0],1)
  self.call('frame_select',frame='t.npz',where={'col':'x','op':'ne','value':0},out='ne.npz');self.assertEqual(self.inspect('ne.npz')['shape'][0],3)
 def test_all_null_aggregate_dtypes(self):
  self.table({'k':['a','a'],'x':[None,None],'when':[None,None],'cat':[None,None]},dtypes={'x':'int64','when':{'dtype':'datetime64[ns]','tz':'UTC'},'cat':{'dtype':'category','categories':['b','a'],'ordered':True}})
  self.call('frame_groupby',frame='t.npz',keys=['k'],aggs={'x':['std','min','count'],'when':['first'],'cat':['last']},out='g.npz')
  got=self.inspect('g.npz');self.assertEqual(got['dtypes']['x__std'],'float64');self.assertEqual(got['dtypes']['x__min'],'int64');self.assertEqual(got['dtypes']['x__count'],'int64');self.assertEqual(got['dtypes']['when__first'],'datetime64[ns]');self.assertEqual(got['column_meta']['when__first']['tz'],'UTC');self.assertEqual(got['column_meta']['cat__last']['categories'],['b','a'])
 def test_int64_wire_overflow_does_not_round_into_range(self):
  self.call('frame_import',rc=2,columns={'x':[-(1<<63)-1]},dtypes={'x':'int64'},out='bad.npz')
  self.call('frame_import',rc=2,columns={'x':[(1<<63)]},dtypes={'x':'int64'},out='bad.npz')
 def test_minimal_dependencies(self):
  import importlib.util
  for name in ('pyarrow','numexpr','bottleneck','openpyxl'):self.assertIsNone(importlib.util.find_spec(name),name)
 def test_string_storage_budget_before_allocation(self):
  import resource
  # Input is only 132KiB, but fixed-width UTF-8 storage would require >256MiB.
  (self.cwd/'wide.csv').write_text('s\n'+('x'*65536)+'\n'+('y\n'*4100))
  def cap():resource.setrlimit(resource.RLIMIT_AS,(400*1024*1024,400*1024*1024))
  r=subprocess.run([BIN,'frame_import','--input-json','{"csv":"wide.csv","out":"wide.npz"}'],cwd=self.cwd,capture_output=True,text=True,preexec_fn=cap)
  self.assertEqual(r.returncode,1,r.stdout);self.assertEqual(json.loads(r.stdout)['error']['data']['tool_code'],'OUTPUT_TOO_LARGE',r.stdout);self.assertFalse((self.cwd/'wide.npz').exists())
 def test_npz_attacks(self):
  import numpy as np
  np.savez(self.cwd/'object.npz',x=np.array([{'evil':1}],dtype=object));self.call('frame_inspect',rc=2,frame='object.npz')
  with zipfile.ZipFile(self.cwd/'bomb.npz','w',compression=zipfile.ZIP_DEFLATED) as z:z.writestr('x.npy',bytes(10_000_000))
  self.call('frame_inspect',rc=2,frame='bomb.npz')
  self.table({'x':[1]});raw=(self.cwd/'t.npz').read_bytes();pos=raw.index(b'PK\x01\x02');mut=bytearray(raw);struct.pack_into('<I',mut,pos+24,0xffffffff);(self.cwd/'oversize.npz').write_bytes(mut);self.call('frame_inspect',rc=2,frame='oversize.npz')
  self.assertEqual(self.call('la_solve',a=[[1]],b=[2])['meta'].get('pandas'),'unloaded')
if __name__=='__main__':unittest.main()
