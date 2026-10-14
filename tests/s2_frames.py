"""Independent table oracles: pandas directly plus hand-computed tables."""
import json, pathlib, subprocess, sys, tempfile, unittest, zipfile, struct, os
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
  ref=pd.DataFrame({'region':['E','W','E','W'],'sales':[10,20,30,40]}).groupby('region')['sales'].sum()
  self.assertEqual(got['sales__sum'],ref.tolist())
 def test_dtype_identity(self):
  cols={'i':[9223372036854775807,None,-9223372036854775808],'f':['NaN','Inf',None],'b':[True,None,False],'s':['NA','雪😀',None],'dt':['2026-01-01T00:00:00.000000001',None,'2026-01-03'],'cat':['b','a',None]}
  d={'i':'int64','f':'float64','b':'bool','s':'string','dt':'datetime64[ns]','cat':{'dtype':'category','categories':['b','a','unused'],'ordered':True}}
  self.table(cols,dtypes=d);got=self.inspect();self.assertEqual(got['head']['data']['i'],cols['i']);self.assertEqual(got['head']['data']['f'],cols['f']);self.assertEqual(got['head']['data']['s'],cols['s']);self.assertEqual(got['null_counts']['f'],1)
  self.assertEqual(got['column_meta']['cat']['categories'],['b','a','unused']);self.assertTrue(got['column_meta']['cat']['ordered']);self.assertEqual(got['head']['data']['dt'][0],cols['dt'][0])
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
  direct=pd.DataFrame({'ts':pd.to_datetime(dates,utc=True).tz_convert('America/New_York'),'v':[1,2,3,4]}).resample('1h',on='ts')['v'].sum();self.assertEqual(got['v__sum'],direct.tolist())
  for rule in ('pd.offsets','1D; import os','1D\\n','0D','999999999D'):
   self.call('frame_resample',rc=2,frame='t.npz',time='ts',rule=rule,aggs={'v':['sum']},out='bad.npz')
 def test_export_policy_caps(self):
  self.table({'x':[1,None,3]});self.call('frame_export',frame='t.npz',format='json',out='e.json');self.assertEqual(json.loads((self.cwd/'e.json').read_text())['data']['x'],[1,None,3])
  self.call('frame_export',rc=5,frame='t.npz',format='json',out='e.json')
  self.assertEqual(self.call('frame_export',frame='t.npz',format='json')['table']['data']['x'],[1,None,3])
  for flag,rc in [('--readonly',6),('--dryrun',0)]:
   p=subprocess.run([BIN,flag,'frame_import','--input-json',json.dumps({'columns':{'x':[1]},'out':'no.npz'})],cwd=self.cwd,capture_output=True);self.assertEqual(p.returncode,rc);self.assertFalse((self.cwd/'no.npz').exists())
  self.call('frame_import',rc=2,columns={'x':list(range(1001))},out='big.npz')
  self.call('frame_import',rc=2,columns={'a':[[1]]},out='bad.npz')
  self.call('frame_inspect',rc=2,frame='t.npz',head=1001)
 def test_npz_attacks(self):
  import numpy as np
  np.savez(self.cwd/'object.npz',x=np.array([{'evil':1}],dtype=object));self.call('frame_inspect',rc=2,frame='object.npz')
  with zipfile.ZipFile(self.cwd/'bomb.npz','w',compression=zipfile.ZIP_DEFLATED) as z:z.writestr('x.npy',bytes(10_000_000))
  self.call('frame_inspect',rc=2,frame='bomb.npz')
  self.table({'x':[1]});raw=(self.cwd/'t.npz').read_bytes();pos=raw.index(b'PK\x01\x02');mut=bytearray(raw);struct.pack_into('<I',mut,pos+24,0xffffffff);(self.cwd/'oversize.npz').write_bytes(mut);self.call('frame_inspect',rc=2,frame='oversize.npz')
  self.assertEqual(self.call('la_solve',a=[[1]],b=[2])['meta'].get('pandas'),'unloaded')
if __name__=='__main__':unittest.main()
