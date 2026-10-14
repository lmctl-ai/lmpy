"""S2 review regressions; expected tables are hand computed, not worker generated."""
import io, json, os, pathlib, struct, subprocess, sys, tempfile, unittest, zipfile
BIN=sys.argv.pop(1)
class Review(unittest.TestCase):
 def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.cwd=pathlib.Path(self.tmp.name)
 def tearDown(self):self.tmp.cleanup()
 def call(self,op,rc=0,env=None,timeout=25,**p):
  r=subprocess.run([BIN,'--input-json',json.dumps({'op':op,'params':p})],cwd=self.cwd,env=env,capture_output=True,text=True,timeout=timeout)
  self.assertEqual(r.returncode,rc,r.stdout);return json.loads(r.stdout)
 def table(self,columns,**p):return self.call('frame_import',columns=columns,out='t.npz',**p)
 def inspect(self,ref='t.npz'):return self.call('frame_inspect',frame=ref)['head']['data']
 def test_csv_prespawn_budgets(self):
  env=dict(os.environ,PYCLI_PYTHON='/missing-python')
  for name,text in [('cells','a,b,c\n'+'1,2,3\n'*166667),('rows','a\n'+'1\n'*200001),('width','a\n'+'x'*1000000+'\n'),('bytes','a\n'+('x'*100+'\n')*170000)]:
   with self.subTest(name=name):
    (self.cwd/'a.csv').write_text(text);self.call('frame_import',rc=2,env=env,csv='a.csv',out=name+'.npz')
 def test_category_linear_and_bounded_summary(self):
  (self.cwd/'a.csv').write_text('c\n'+''.join('v%06d\n'%i for i in range(60000)))
  r=self.call('frame_import',csv='a.csv',dtypes={'c':'category'},out='t.npz',timeout=15)
  self.assertLess(len(json.dumps(r)),65536);self.assertEqual(r['column_meta']['c']['n_categories'],60000);self.assertTrue(r['column_meta']['c']['categories_truncated'])
 def test_wide_preview_and_inline_export(self):
  (self.cwd/'a.csv').write_text('s\n'+('x'*60000+'\n')*10)
  r=self.call('frame_import',csv='a.csv',out='t.npz');self.assertLess(len(json.dumps(r)),65536);self.assertTrue(r['preview_truncated'])
  self.call('frame_export',rc=1,frame='t.npz',format='json')
 def test_int_sum_overflow(self):
  for vals in ([2**62]*3,[2**63-1,1],[-2**63,-1]):
   self.call('frame_import',columns={'k':['a']*len(vals),'ts':['2026-03-01']*len(vals),'x':vals},dtypes={'ts':'datetime64[ns]'},out='t.npz')
   for op,args in [('frame_groupby',{'keys':['k']}),('frame_resample',{'time':'ts','rule':'1ME'})]:
    r=self.call(op,rc=1,frame='t.npz',aggs={'x':['sum']},out='g.npz',**args);self.assertEqual(r['error']['data']['tool_code'],'INT64_OVERFLOW');self.assertFalse((self.cwd/'g.npz').exists())
   (self.cwd/'t.npz').unlink()
 def test_calendar_bins_hand_oracle(self):
  self.table({'ts':['2026-03-01','2026-03-31','2026-05-01'],'x':[1,3,5]},dtypes={'ts':'datetime64[ns]'})
  fixtures={'1ME':(['2026-03-31','2026-04-30','2026-05-31'],[4,0,5]),'1MS':(['2026-03-01','2026-04-01','2026-05-01'],[4,0,5]),'1QS':(['2026-01-01','2026-04-01'],[4,5]),'1YS':(['2026-01-01'],[9])}
  for rule,(labels,values) in fixtures.items():
   self.call('frame_resample',frame='t.npz',time='ts',rule=rule,aggs={'x':['sum']},out=rule+'.npz');d=self.inspect(rule+'.npz');self.assertEqual(d['ts'],[s+'T00:00:00' for s in labels]);self.assertEqual(d['x__sum'],values)
 def test_csv_nonfinite_roundtrip(self):
  cols={'x':[1.5,'NaN','Inf','-Inf',None,-0.0],'i':[1,None,3,4,5,6],'ts':['2026-01-01',None,None,None,None,None]};types={'x':'float64','i':'int64','ts':'datetime64[ns]'}
  self.table(cols,dtypes=types);self.call('frame_export',frame='t.npz',format='csv',out='e.csv');self.call('frame_import',csv='e.csv',dtypes=types,out='u.npz');self.assertEqual(self.inspect('u.npz'),self.inspect())
  self.assertIn('NaN',(self.cwd/'e.csv').read_text());self.assertIn('-Inf',(self.cwd/'e.csv').read_text())
 def test_csv_strict_tokens_and_blank_lines(self):
  for text,dtype in [('1_000','int64'),('1_5.5','float64')]:
   (self.cwd/'a.csv').write_text('x\n'+text+'\n');self.call('frame_import',rc=2,csv='a.csv',dtypes={'x':dtype},out='bad.npz')
  (self.cwd/'a.csv').write_text('x,id\n1,007\nNA,+5\n\n');self.call('frame_import',csv='a.csv',out='t.npz');r=self.call('frame_inspect',frame='t.npz');self.assertEqual(r['dtypes'],{'x':'int64','id':'string'});self.assertEqual(r['head']['data'],{'x':[1,None],'id':['007','+5']})
 def test_special_dates(self):
  self.table({'ts':['2026-01-01']},dtypes={'ts':'datetime64[ns]'})
  for word in ('now','today','yesterday','tomorrow','NaT','20260101','Jan 1 2026'):
   self.call('frame_import',rc=2,columns={'ts':[word]},dtypes={'ts':'datetime64[ns]'},out='bad.npz')
   self.call('frame_select',rc=2,frame='t.npz',where={'col':'ts','op':'lt','value':word},out='bad.npz')
 def test_special_datetime_predicates(self):
  self.table({'ts':['2026-01-01']},dtypes={'ts':'datetime64[ns]'})
  for word in ('now','today','yesterday','tomorrow','NaT','20260101'):
   with self.subTest(word=word):self.call('frame_select',rc=2,frame='t.npz',where={'col':'ts','op':'lt','value':word},out='bad.npz')
 def test_csv_nullable_integer_identifiers_and_blank_line(self):
  (self.cwd/'a.csv').write_text('x,id\n1,007\nNA,+5\n');self.call('frame_import',csv='a.csv',out='t.npz');r=self.call('frame_inspect',frame='t.npz');self.assertEqual(r['dtypes'],{'x':'int64','id':'string'});self.assertEqual(r['head']['data'],{'x':[1,None],'id':['007','+5']})
 def test_csv_trailing_blank(self):
  (self.cwd/'a.csv').write_text('x,y\n1,2\n\n');self.call('frame_import',csv='a.csv',out='t.npz');self.assertEqual(self.inspect(),{'x':[1],'y':[2]})
 def test_npz_shadow_names_prespawn(self):
  self.table({'x':[1,2,3]});raw=(self.cwd/'t.npz').read_bytes()
  # zipfile truncates NUL on writing too: mutate equal-length raw names afterwards.
  for name in ('c0.npyXz.npy','c0.npy-z.npy'):
   with zipfile.ZipFile(io.BytesIO(raw)) as source,zipfile.ZipFile(self.cwd/'bad.npz','w') as dest:
    for info in source.infolist():dest.writestr(info.filename,source.read(info))
    dest.writestr(name,source.read('c0.npy'))
   b=(self.cwd/'bad.npz').read_bytes().replace(b'c0.npyXz.npy',b'c0.npy\0z.npy');(self.cwd/'bad.npz').write_bytes(b)
   self.call('frame_inspect',rc=2,env=dict(os.environ,PYCLI_PYTHON='/missing-python'),frame='bad.npz')
 def test_warnings_and_string_sum_and_integer_predicates(self):
  self.table({'ts':['2026-01-01T00:00:00Z',None],'x':[50,51],'s':['a','b']},dtypes={'ts':{'dtype':'datetime64[ns]','tz':'UTC'}})
  self.assertEqual(self.call('frame_inspect',frame='t.npz')['warnings'],[])
  self.call('frame_groupby',rc=2,frame='t.npz',keys=['x'],aggs={'s':['sum']},out='bad.npz')
  self.call('frame_select',frame='t.npz',where={'col':'x','op':'eq','value':50.0},out='p.npz');self.assertEqual(self.inspect('p.npz')['x'],[50])
 def test_string_sum_rejected(self):
  self.table({'k':['a','a'],'s':['x','y']})
  self.call('frame_groupby',rc=2,frame='t.npz',keys=['k'],aggs={'s':['sum']},out='bad.npz')
 def test_integral_float_predicate(self):
  self.table({'x':[50,51]})
  self.call('frame_select',frame='t.npz',where={'col':'x','op':'eq','value':50.0},out='p.npz');self.assertEqual(self.inspect('p.npz')['x'],[50])
  self.call('frame_select',rc=2,frame='t.npz',where={'col':'x','op':'gt','value':50.5},out='bad.npz')
 def test_dst_daily_and_weekly_right_defaults(self):
  self.table({'ts':['2026-03-07T12:00:00-05:00','2026-03-08T12:00:00-04:00','2026-03-09T12:00:00-04:00'],'x':[1,2,3]},dtypes={'ts':{'dtype':'datetime64[ns]','tz':'America/New_York'}})
  self.call('frame_resample',frame='t.npz',time='ts',rule='1D',aggs={'x':['sum']},out='d.npz');d=self.inspect('d.npz');self.assertEqual(d,{'ts':['2026-03-07T00:00:00-05:00','2026-03-08T00:00:00-05:00','2026-03-09T00:00:00-04:00'],'x__sum':[1,2,3]})
  self.call('frame_resample',frame='t.npz',time='ts',rule='1W',aggs={'x':['sum']},out='w.npz');self.assertEqual(self.inspect('w.npz'),{'ts':['2026-03-08T00:00:00-05:00','2026-03-15T00:00:00-04:00'],'x__sum':[3,3]})
  self.call('frame_resample',frame='t.npz',time='ts',rule='1W',closed='right',label='right',aggs={'x':['sum']},out='explicit.npz');self.assertEqual(self.inspect('explicit.npz'),self.inspect('w.npz'))
 def test_nan_null_group_policy(self):
  self.table({'k':['NaN',None,'NaN',1.5],'x':[1,2,3,4]},dtypes={'k':'float64'})
  self.call('frame_groupby',frame='t.npz',keys=['k'],aggs={'x':['sum','std']},out='g.npz');d=self.inspect('g.npz');self.assertEqual(d['k'],[1.5,None]);self.assertEqual(d['x__sum'],[4,6]);self.assertIsNone(d['x__std'][0])
if __name__=='__main__':unittest.main()
