"""S8a review regressions: hand-computed expectations and independent NPZ decoding."""
import pathlib
from s8_ops import Base, call, rejected, NO_PYTHON, BIN, ROOT, np, json, os, subprocess, unittest

class Review(Base):
    def test_table_scores(self):
        for i, values in enumerate(([2., -1.], [float('nan'), .9], [float('inf'), .9])):
            t = self.npy(np.array([[0., values[0]], [1., values[1]]]))
            f = self.p('scores%d.npz' % i)
            call('array_to_frame', dict(ref=t, columns=['truth','score'], out=f))
            r = call('ml_metrics', dict(frame=f, y_true='truth', y_score='score', metrics=['brier','log_loss','roc_auc']), rc=2)
            self.assertEqual(r['error']['data']['tool_code'], 'BAD_DATA')

    def test_exact_regression_and_large_classification(self):
        t = self.table({'truth':[2**53,2**53+1], 'pred':[2**53,2**53]})
        self.assertEqual(call('ml_metrics', dict(frame=t,y_true='truth',y_pred='pred',metrics=['accuracy']))['metrics']['accuracy'], .5)
        r = call('ml_metrics',dict(frame=t,y_true='truth',y_pred='pred',metrics=['mse','accuracy']),rc=2)
        self.assertEqual(r['error']['data']['tool_code'],'INT64_NOT_EXACT')

    def test_stable_regression(self):
        for scale in (1e200, 1e-200, 1.):
            r = call('ml_metrics',dict(y_true=[scale,-scale],y_pred=[0,0],metrics=['mse','mae','r2']))
            self.assertEqual(r['metrics']['r2'],0.)
            if scale==1e200:
                self.assertEqual(r['status'],'UNDEFINED'); self.assertIn('mse',r['undefined'])
        r=call('ml_metrics',dict(y_true=[1e308,1e308],y_pred=[1e308,1e308],metrics=['r2','mae']))
        self.assertEqual(r['status'],'UNDEFINED')

    def test_pos_label_scalar_before_spawn(self):
        for label in ([1],{'a':1}):
            rejected('ml_metrics',dict(y_true=[0,1],y_pred=[0,1],metrics=['f1'],pos_label=label),'BAD_PARAM')

    def test_pos_label_membership_and_multiclass(self):
        r=call('ml_metrics',dict(y_true=[0,0],y_score=[.1,.2],pos_label=7,metrics=['brier']),rc=2)
        self.assertEqual(r['error']['data']['tool_code'],'BAD_PARAM')
        self.assertEqual(call('ml_metrics',dict(y_true=[0,1,2],y_score=[.1,.5,.9],metrics=['roc_auc']))['status'],'UNDEFINED')

    def test_empty_tables_are_artifacts(self):
        t=self.table({'id':[1,2],'v':[10.,20.]})
        e=self.p('empty.npz'); call('frame_select',dict(frame=t,limit=0,out=e))
        for op,p in [('frame_mutate',dict(frame=e,assign=[dict(name='w',expr='v+1')])),('frame_rolling',dict(frame=e,columns=['v'],window=2,functions=['sum'])),('frame_merge',dict(left=t,right=e,on=['id'],how='inner'))]:
            z=call(op,dict(p,out=self.p(op+'.npz'))); self.assertEqual(z['status'],'OK'); self.assertEqual(z['shape'][0],0)
        z=call('frame_merge',dict(left=t,right=e,on=['id'],how='left',out=self.p('left.npz')))
        self.assertEqual(z['preview']['data']['v_y'],[None,None])

    def test_rolling_wide_time_arithmetic(self):
        for i,(times,w,expected) in enumerate([(['1700-01-01','2200-01-01'],'1D',[1.,2.]),(['1700-01-01','1700-01-02'],'10000D',[1.,1.5]),(['1750-01-01','1750-01-02'],'9999999D',[1.,1.5])]):
            t=self.table({'t':times,'v':[1.,2.]},name='t%d.npz'%i,dtypes={'t':'datetime64[ns]'})
            r=call('frame_rolling',dict(frame=t,columns=['v'],time_window=w,on='t',functions=['mean'],out=self.p('r%d.npz'%i)))
            self.assertEqual(r['preview']['data']['v__mean'],expected)

    def test_rolling_stable_and_count_minimum(self):
        t=self.table({'v':[1e308,1e308]})
        r=call('frame_rolling',dict(frame=t,columns=['v'],window=2,functions=['mean','std','var','count'],out=self.p('r.npz')))
        self.assertEqual(r['preview']['data']['v__mean'],[None,1e308])
        self.assertEqual(r['preview']['data']['v__std'],[None,0.])
        self.assertEqual(r['preview']['data']['v__count'],[None,2])
        t=self.table({'v':[1e200,-1e200]},name='wide.npz')
        r=call('frame_rolling',dict(frame=t,columns=['v'],window=2,functions=['std'],out=self.p('std.npz')))
        self.assertAlmostEqual(r['preview']['data']['v__std'][1]/1e200,2**.5)
        t=self.table({'v':['Inf','-Inf']},name='inf.npz',dtypes={'v':'float64'})
        self.assertEqual(call('frame_rolling',dict(frame=t,columns=['v'],window=2,functions=['mean'],out=self.p('inf-out.npz')),rc=2)['error']['data']['tool_code'],'NON_FINITE')

    def test_missing_cardinality(self):
        l=self.table({'k':[1,None,None]},dtypes={'k':'int64'})
        r=self.table({'k':[1,None]},name='right.npz',dtypes={'k':'int64'})
        z=call('frame_merge',dict(left=l,right=r,on=['k'],validate='one_to_one',out=self.p('out.npz')),rc=2)
        self.assertEqual(z['error']['data']['tool_code'],'MERGE_VALIDATION')

    def test_category_union_preserves_dtype(self):
        l=self.table({'k':['a','b']},dtypes={'k':{'dtype':'category','categories':['a','b','left_only']}})
        r=self.table({'k':['b','c']},name='right.npz',dtypes={'k':{'dtype':'category','categories':['b','c','right_only']}})
        z=call('frame_merge',dict(left=l,right=r,on=['k'],how='outer',out=self.p('out.npz')))
        self.assertEqual(z['column_meta']['k']['dtype'],'category')
        self.assertEqual(z['column_meta']['k']['categories'],['a','b','left_only','c','right_only'])
        z=call('frame_merge',dict(left=l,right=l,on=['k'],out=self.p('same.npz')))
        self.assertEqual(z['column_meta']['k']['categories'],['a','b','left_only'])

    def test_string_join_preallocation_budget(self):
        # Direct NPZ fixture avoids a dangerous baseline allocation: only 5000 x 65000 = 325 MB.
        l=self.csv_table({'k':[1]*5000},name='left.npz')
        r=self.table({'k':[1],'s':['x'*65000]},name='right.npz')
        z=call('frame_merge',dict(left=l,right=r,on=['k'],out=self.p('out.npz')),rc=1)
        self.assertEqual(z['error']['data']['tool_code'],'OUTPUT_TOO_LARGE')

    def test_append_bridge_preserves_source(self):
        t=self.table({'ts':['2026-01-01T00:00:00-05:00','2026-01-02T00:00:00-05:00'],'name':['é',None],'id':[1,2]},dtypes={'ts':{'dtype':'datetime64[ns]','tz':'America/New_York'}})
        z=call('array_to_frame',dict(frame=t,ref=[2.,3.],columns=['fitted'],out=self.p('out.npz')))
        self.assertEqual(z['shape'],[2,4]); self.assertEqual(z['preview']['data']['name'],['é',None])
        self.assertEqual(z['preview']['data']['ts'][0],'2026-01-01T00:00:00-05:00')
        rejected('array_to_frame',dict(frame=t,ref=[1.],columns=['fitted'],out=self.p('bad.npz')),'BAD_DATA')
        rejected('array_to_frame',dict(frame=t,ref=[1.,2.],columns=['id'],out=self.p('bad.npz')),'BAD_PARAM')

    def test_summary_timezone(self):
        t=self.table({'t':['2026-01-01T00:00:00-05:00'],'x':[1.]},dtypes={'t':{'dtype':'datetime64[ns]','tz':'America/New_York'}})
        z=call('frame_mutate',dict(frame=t,assign=[dict(name='y',expr='x+1')],out=self.p('out.npz')))
        self.assertEqual(z['preview']['data']['t'],['2026-01-01T00:00:00-05:00'])

    def test_bridge_finite_default(self):
        t=self.p('t.npz'); call('array_to_frame',dict(ref=[1.,'NaN',3.],columns=['x'],out=t))
        self.assertEqual(call('frame_to_array',dict(frame=t,columns=['x']),rc=2)['error']['data']['tool_code'],'NA_NOT_ALLOWED')
        self.assertEqual(call('frame_to_array',dict(frame=t,columns=['x'],missing='nan'))['source_nan_count'],1)
        t2=self.p('masked.npz'); z=call('array_to_frame',dict(ref=[1.,'NaN'],columns=['x'],nan_policy='missing',out=t2))
        self.assertTrue(z['nan_mask_changed'])

    def test_generated_names_before_spawn(self):
        c='x'*250; t=self.table({'k':[1],c:[2]})
        rejected('frame_merge',dict(left=t,right=t,on=['k'],suffixes=['_1234567','_y'],out=self.p('bad.npz')),'BAD_DATA')
        rejected('frame_merge',dict(left=t,right=t,on=['k'],suffixes=['_x\0','_y'],out=self.p('bad.npz')),'BAD_DATA')
        rejected('frame_rolling',dict(frame=t,columns=[c],window=1,functions=['median'],out=self.p('bad.npz')),'BAD_DATA')

    def test_mixed_metrics_class_cap_before_spawn(self):
        rejected('ml_metrics',dict(y_true=list(range(31)),y_pred=list(range(31)),metrics=['confusion_matrix','mse']),'INPUT_TOO_LARGE')
        t=self.table({'y':list(range(31))})
        rejected('ml_metrics',dict(frame=t,y_true='y',y_pred='y',metrics=['confusion_matrix','mse']),'INPUT_TOO_LARGE')

    def test_pinned_rng_and_nondefault_t(self):
        for dist,params in [('norm',{}),('uniform',{}),('poisson',{'lam':3}),('t',{'df':8,'loc':5,'scale':3})]:
            rng=np.random.Generator(np.random.PCG64(7))
            expected={'norm':lambda:rng.normal(size=5),'uniform':lambda:rng.uniform(size=5),'poisson':lambda:rng.poisson(3,5),'t':lambda:5+3*rng.standard_t(8,5)}[dist]()
            p=dict(dist=dist,fn='rvs',params=params,seed=7,n=5)
            np.testing.assert_array_equal(call('stats_dist',p)['values'],expected)
            z=call('stats_dist_to',dict(p,out=self.p(dist+'.safetensors')))
            from s1 import read_safetensors
            np.testing.assert_array_equal(read_safetensors(z['artifacts']['x']['path'])['x'],expected)

    def test_time_units_and_balanced_accuracy(self):
        for unit,mult in [('min',60),('h',3600),('D',86400)]:
            times=np.datetime64('2026-01-01')+np.array([0,mult//2,mult,2*mult]).astype('timedelta64[s]')
            t=self.table({'t':[str(v) for v in times],'x':[1,2,4,8]},name=unit+'.npz',dtypes={'t':'datetime64[ns]'})
            z=call('frame_rolling',dict(frame=t,columns=['x'],time_window='1'+unit,on='t',functions=['sum'],out=self.p(unit+'-out.npz')))
            self.assertEqual(z['preview']['data']['x__sum'],[1.,3.,6.,8.])
        z=call('ml_metrics',dict(y_true=[0,0,1,1],y_pred=[0,0,0,2],metrics=['balanced_accuracy']))
        self.assertEqual(z['metrics']['balanced_accuracy'],.5)

    def worker(self):
        ns={'__name__':'review_worker'}
        source='\n'.join((pathlib.Path(os.environ['REVIEW_BASELINE_S8']) if name=='s8.py' and os.environ.get('REVIEW_BASELINE_S8') else ROOT/'py'/name).read_text() for name in ('frames.py','numerics.py','s4.py','sm.py','ml.py','s7.py','s8.py','worker.py'))
        exec(compile(source,'embedded-worker','exec'),ns)
        return ns

    def test_join_budget_precedes_gather(self):
        l=self.csv_table({'k':[1]*5000},name='left.npz')
        r=self.table({'k':[1],'s':['x'*65000]},name='right.npz')
        w=self.worker()
        def forbidden(*args): raise AssertionError('gather occurred before byte-budget validation')
        w['s8_take']=forbidden
        with self.assertRaises(w['OutputTooLarge']):
            w['frame_merge'](dict(left_file=l,right_file=r,left_keys=['k'],right_keys=['k'],key_shared=True,how='left',validate='many_to_one',max_rows=200000,output_names=['k','s']))

    def test_registered_budgets_reach_guard(self):
        w=self.worker(); observed=[]
        def timeout(p,body): observed.append(p['time_limit']); return None
        w['s7_guard']=timeout
        for name in ('frame_merge','ml_metrics','array_to_frame','frame_to_array_to','stats_dist_to'):
            self.assertEqual(w['s8_operation'](name)({'time_limit':.125})['status'],'TIME_LIMIT',name)
        self.assertEqual(observed,[.125]*5)

    def test_schema_contract_corrections(self):
        ops={v['name']:v for v in json.loads(subprocess.run([BIN,'--schema'],capture_output=True,text=True).stdout)['operations']}
        self.assertIn('only inside',ops['frame_mutate']['description'])
        params={v['name']:v for v in ops['array_to_frame']['params']}
        self.assertIn('inline JSON',params['dtypes']['description'])
        self.assertIn('zero_division',next(v for v in ops['ml_metrics']['params'] if v['name']=='average')['description'])
        self.assertIn('time_limit',{v['name'] for v in ops['stats_dist_to']['params']})

    def test_density_envelope_and_param_status(self):
        z=call('stats_dist',dict(dist='norm',fn='cdf',x=[0.],params={}))
        self.assertEqual(z['status'],'OK'); self.assertEqual(z['fn'],'cdf')
        rejected('stats_dist',dict(dist='norm',fn='cdf',x=[0.],params={'scale':0}),'BAD_PARAM')

if __name__=='__main__': unittest.main()
