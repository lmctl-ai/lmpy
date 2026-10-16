"""S8b-1 review reproducers and independent residual/statistical references."""
from s8b_ops import Base, call, rejected, ROOT, BIN, np, json, os, subprocess, unittest, read_safetensors
import math, pathlib
from unittest.mock import patch

Y=[1.,2.,1.5,2.5,1.2,2.8,1.9,2.1,1.4,2.6,1.7,2.3]

class Review(Base):
    def tsa(self,test,y=Y,**kw):return call('sm_tsa_tools',dict(test=test,columns={'y':y},series='y',**kw))
    def matrix(self,fn,a,**kw):return call('la_matrix_fn',dict(fn=fn,a=a,**kw))
    def diag(self,task='eigsh',**kw):return dict(task=task,shape=[3,3],row=[0,1,2],col=[0,1,2],val=[1.,2.,3.],**kw)
    def worker(self):
        ns={'__name__':'review_worker'}
        source='\n'.join((pathlib.Path(os.environ['REVIEW_VARIANT_DIR'])/name if os.environ.get('REVIEW_VARIANT_DIR') and (pathlib.Path(os.environ['REVIEW_VARIANT_DIR'])/name).exists() else ROOT/'py'/name).read_text() for name in ('frames.py','numerics.py','s4.py','sm.py','ml.py','s7.py','s8.py','s8b.py','worker.py'))
        exec(compile(source,'embedded-worker','exec'),ns);return ns

    def test_stationarity_scale_invariance(self):
        for test,extra in [('adfuller',{}),('kpss',{}),('ljung_box',{'lags':[1,3,5]})]:
            ref=self.tsa(test,**extra)
            for scale in (1e160,1e-165):
                with self.subTest(test=test,scale=scale):
                    z=self.tsa(test,[scale*v for v in Y],**extra)
                    self.assertEqual(z['status'],'OK')
                    np.testing.assert_allclose(z['statistic'],ref['statistic'],rtol=1e-10,atol=1e-12)
                    np.testing.assert_allclose(z['p_value'],ref['p_value'],rtol=1e-9,atol=1e-12)
                    if test=='adfuller':
                        self.assertEqual(z['used_lag'],ref['used_lag'])
                        self.assertAlmostEqual(z['icbest']-ref['icbest'],2*(len(Y)-1-z['maxlag'])*math.log(scale),places=8)

    def test_time_series_alpha_rejected_before_spawn(self):
        for test in ('adfuller','kpss','ljung_box'):
            with self.subTest(test=test):rejected('sm_tsa_tools',dict(test=test,columns={'y':Y},series='y',alpha=.5),'BAD_PARAM')

    def test_ljung_model_df_and_lag_order(self):
        z=self.tsa('ljung_box',lags=5,model_df=1)
        self.assertEqual(z['lags'],[2,3,4,5])
        # df=2 chi-square survival is exp(-Q/2).
        self.assertAlmostEqual(z['p_value'][1],math.exp(-z['statistic'][1]/2),places=14)
        self.assertEqual(self.tsa('ljung_box',model_df=2)['lags'],list(range(3,11)))
        for lags in ([2,2],[3,2]):
            rejected('sm_tsa_tools',dict(test='ljung_box',columns={'y':Y},series='y',lags=lags),'BAD_PARAM')

    def test_adf_effective_maxlag(self):
        y=[1,2,0,3,1,4,-1,2,0,1,3,2]
        z=self.tsa('adfuller',y,autolag='none')
        self.assertEqual(z['maxlag'],z['used_lag']);self.assertEqual(z['maxlag'],4)
        replay=self.tsa('adfuller',y,autolag='none',maxlag=z['maxlag'])
        self.assertEqual(z['statistic'],replay['statistic'])
        # Noninteger Schwert bound: ceil(12*(160/100)**.25)=14.
        y=(np.sin(np.arange(160)*1.7)+np.cos(np.arange(160)*.3)).tolist()
        self.assertEqual(self.tsa('adfuller',y,autolag='none')['maxlag'],14)

    def test_short_acf_pacf_status(self):
        for test in ('acf','pacf'):
            for n in (2,8):
                z=call('sm_tsa_tools',dict(test=test,columns={'y':list(range(n))},series='y'),rc=2)
                self.assertEqual(z['error']['data']['tool_code'],'TOO_SHORT')

    def test_adjusted_acf_hand_covariances(self):
        for adjusted,want in [(False,[1,.25,-.3,-.45]),(True,[1,1/3,-.6,-1.8])]:
            z=self.tsa('acf',[1,2,3,4],nlags=3,adjusted=adjusted)
            np.testing.assert_allclose(z['values'],want,rtol=1e-14,atol=1e-15)

    def test_logm_conditioning_singular(self):
        for a in ([[1,2],[2,4]],[[1,2,3],[4,5,6],[7,8,9]]):
            with self.subTest(a=a):self.assertEqual(self.matrix('logm',a)['status'],'SINGULAR')
        self.assertEqual(self.matrix('logm',[[1,1],[0,1]])['status'],'OK')

    def test_inverse_relative_residual_and_scales(self):
        for scale in (1.,1e8,1e-8):
            a=np.array([[4.,7.],[2.,6.]])*scale
            z=self.matrix('inv',a.tolist());self.assertEqual(z['status'],'OK')
            y=np.asarray(z['y']);error=np.linalg.norm(a@y-np.eye(2),'fro')
            want=error/(np.linalg.norm(a,'fro')*np.linalg.norm(y,'fro')+math.sqrt(2))
            self.assertAlmostEqual(z['residual']/want,1.,places=10)
        z=self.matrix('inv',[[1e8,1e8],[1e8,100000000.01000001]])
        self.assertEqual(z['status'],'OK')
        h=[[1/(i+j+1) for j in range(10)] for i in range(10)]
        z=self.matrix('inv',h);self.assertEqual(z['status'],'OK');self.assertTrue(z['warnings'])

    def test_pinv_effective_cutoff(self):
        z=self.matrix('pinv',[[1,0],[0,.001],[0,0]],rtol=.01)
        self.assertEqual((z['status'],z['rank']),('OK',1))
        np.testing.assert_array_equal(z['y'],[[1,0,0],[0,0,0]])
        self.assertAlmostEqual(z['truncation'],.001/math.sqrt(1+.001**2),places=14)
        self.assertLess(z['residual'],1e-12)

    def test_tiny_sparse_rhs_scaling_read_and_write(self):
        for task in ('cg','gmres','spsolve'):
            for i,scale in enumerate((1e-200,1.,1e200)):
                p=dict(task=task,shape=[1,1],row=[0],col=[0],val=[2.],b=[scale])
                z=call('la_sparse',p);self.assertEqual(z['status'],'OK')
                self.assertAlmostEqual(z['x'][0]/scale,.5,places=14);self.assertNotIn('k',z)
                out=self.p(task+str(i)+'.safetensors');call('la_sparse_to',dict(p,out=out))
                self.assertAlmostEqual(read_safetensors(out)['x'][0]/scale,.5,places=14)

    def test_unattainable_iterative_tolerance_honest(self):
        n=12;a=np.array([[1/(i+j+1) for j in range(n)] for i in range(n)])
        row,col=np.indices(a.shape)
        z=call('la_sparse',dict(task='cg',shape=[n,n],row=row.ravel().tolist(),col=col.ravel().tolist(),val=a.ravel().tolist(),b=[1.]*n,rtol=1e-12,maxiter=100000))
        if z['relative_residual']>1e-11:
            self.assertNotEqual(z['status'],'OK');self.assertFalse(z['tolerance_met'])

    def test_superlu_preflight_caps_both_paths(self):
        n=20000;p=dict(shape=[n,n],row=self.npy(np.arange(n)),col=self.npy(np.arange(n)),val=self.npy(np.ones(n)))
        rejected('la_sparse_to',dict(p,task='spsolve',b=self.npy(np.ones(n)),out=self.p('out.safetensors')),'INPUT_TOO_LARGE')
        rejected('la_sparse',dict(p,task='eigsh',k=1,sigma=0,seed=1),'INPUT_TOO_LARGE')
        n=1000;r=np.repeat(np.arange(n),21);c=(r+np.tile(np.arange(21),n))%n
        rejected('la_sparse_to',dict(task='spsolve',shape=[n,n],row=self.npy(r),col=self.npy(c),val=self.npy(np.ones(len(r))),b=self.npy(np.ones(n)),out=self.p('fill.safetensors')),'INPUT_TOO_LARGE')

    def test_sparse_factor_shift_singular(self):
        z=call('la_sparse',self.diag(k=1,sigma=2.,seed=1));self.assertEqual(z['status'],'SINGULAR')
        z=call('la_sparse_to',self.diag(k=1,sigma=2.,seed=1,out=self.p('out.safetensors')),rc=1)
        self.assertEqual(z['error']['data']['tool_code'],'SINGULAR');self.assertFalse(os.path.exists(self.p('out.safetensors')))

    def test_ncv_and_unused_direct_options_preflight(self):
        rejected('la_sparse',self.diag('svds',k=1,ncv=3,seed=1),'BAD_PARAM')
        self.assertEqual(call('la_sparse',self.diag('svds',k=1,ncv=2,seed=1))['status'],'OK')
        for opt in ({'rtol':.1},{'maxiter':5}):
            rejected('la_sparse',self.diag('spsolve',b=[1,1,1],**opt),'BAD_PARAM')

    def test_frame_int64_exactness(self):
        out=self.p('table.npz')
        call('frame_import',dict(columns={'r':[0,1],'c':[0,1],'v':[2**53+1,1]},dtypes={'r':'int64','c':'int64','v':'int64'},out=out))
        z=call('la_sparse',dict(task='spsolve',shape=[2,2],frame=out,row='r',col='c',val='v',b=[1,1]),rc=2)
        self.assertEqual(z['error']['data']['tool_code'],'INT64_NOT_EXACT')

    def test_eigen_residual_scales(self):
        p=dict(task='eigsh',shape=[4,4],row=[0,0,1,1,2,3],col=[0,1,0,1,2,3],val=[1e11,-1e11,-1e11,1e11,2e11,3e11],k=2,which='SM',seed=1)
        z=call('la_sparse',p);self.assertEqual(z['status'],'OK');self.assertLess(z['max_residual'],1e-6)

    def test_complex_order_and_svd_vectors(self):
        rows=[];cols=[];vals=[]
        for i in range(5):rows += [2*i,2*i+1];cols += [2*i+1,2*i];vals += [-float(i+1),float(i+1)]
        z=call('la_sparse',dict(task='eigs',shape=[10,10],row=rows,col=cols,val=vals,k=2,which='LI',seed=4,vectors=True))
        self.assertLess(z['w_im'][0],0);self.assertGreater(z['w_im'][1],0)
        a=np.zeros((10,10));a[rows,cols]=vals;v=np.array(z['v_re'])+1j*np.array(z['v_im']);w=np.array(z['w_re'])+1j*np.array(z['w_im'])
        np.testing.assert_allclose(a@v,v*w,atol=1e-12)
        z=call('la_sparse',dict(task='svds',shape=[8,6],row=list(range(6)),col=list(range(6)),val=[1,2,3,4,5,6],k=3,seed=5,vectors=True))
        u,vt,s=np.array(z['u']),np.array(z['vt']),np.array(z['s']);a=np.zeros((8,6));a[np.arange(6),np.arange(6)]=np.arange(1,7)
        np.testing.assert_allclose(u.T@u,np.eye(3),atol=1e-12);np.testing.assert_allclose(vt@vt.T,np.eye(3),atol=1e-12)
        np.testing.assert_allclose(a@vt.T,u*s,atol=1e-12)

    def sparse_worker_params(self):
        return dict(shape=[3,3],task='eigsh',row_file=self.npy([0,1,2]),col_file=self.npy([0,1,2]),val_file=self.npy([1.,2.,3.]),seed=7,k=1,which='LM',rtol=0.,maxiter=100,vectors=True,time_budget_s=10.)

    def test_arpack_start_is_pcg64(self):
        import scipy.sparse.linalg as spla
        w=self.worker();p=self.sparse_worker_params();captured=[]
        def oracle(a,**kw):captured.append(kw['v0'].copy());return np.array([3.]),np.array([[0.],[0.],[1.]])
        with patch.object(spla,'eigsh',side_effect=oracle):w['la_sparse'](p)
        np.testing.assert_array_equal(captured[0],np.random.Generator(np.random.PCG64(7)).uniform(-1,1,3))

    def test_svds_bad_triple_cannot_report_ok(self):
        import scipy.sparse.linalg as spla
        w=self.worker();p=dict(self.sparse_worker_params(),task='svds')
        with patch.object(spla,'svds',return_value=(np.array([[1.],[0.],[0.]]),np.array([10.]),np.array([[1.,0.,0.]]))):
            self.assertEqual(w['la_sparse'](p)['status'],'NUMERICAL_FAILURE')

    def test_root_bracket_ignores_start_and_endpoint_fields(self):
        z=call('opt_root',dict(equations=['log(x)'],start={'x':-1},bracket=[.5,2]))
        self.assertEqual(z['status'],'OK');self.assertAlmostEqual(z['x']['x'],1.,places=11)
        z=call('opt_root',dict(equations=['x-1'],start={'x':999},bracket=[1,2]));self.assertEqual(z['iterations'],0)
        z=call('opt_root',dict(equations=['log(x)-1'],start={'x':1},bracket=[0,4]),rc=2)
        self.assertEqual(z['error']['data']['tool_code'],'NONFINITE_BRACKET')

    def test_root_unused_params_preflight(self):
        p=dict(equations=['x^2-2'],start={'x':1})
        rejected('opt_root',dict(p,bracket=[1,2],method='lm'),'BAD_PARAM')
        rejected('opt_root',dict(p,maxiter=1),'BAD_PARAM')

    def test_special_one_element_broadcast(self):
        for i,arg in enumerate(([2.],self.npy([2.]))):
            p=dict(fn='gammainc',args={'a':arg,'x':[1.,2.,3.]})
            z=call('special_fn',p);want=[1-math.exp(-x)*(1+x) for x in (1,2,3)]
            np.testing.assert_allclose(z['y'],want,rtol=1e-14)
            out=self.p('special%d.safetensors'%i);call('special_fn_to',dict(p,out=out));np.testing.assert_allclose(read_safetensors(out)['y'],want,rtol=1e-14)
        ref=call('special_fn_to',dict(fn='logsumexp',args={'a':[math.log(2)]},out=self.p('scalar.safetensors')))['artifacts']['y']['ref']
        self.assertEqual(call('special_fn',dict(fn='xlogy',args={'x':ref,'y':[1.,math.e]}))['n'],2)

    def test_special_mixed_diagnostics(self):
        z=call('special_fn',dict(fn='logit',args={'p':[0.,-1.]}))
        self.assertEqual((z['status'],z['first_index'],z['n_nan'],z['n_inf']),('DOMAIN_ERROR',1,1,1))
        w=self.worker();import scipy.special as sp
        p=dict(fn='logsumexp',arg_names=['a'],arg_vector={'a':True},arg_a_file=self.npy([1.,2.,3.]),time_budget_s=10.)
        with patch.object(sp,'logsumexp',return_value=np.inf):z=w['special_fn'](p)
        self.assertEqual(z['n_bad'],1)

if __name__=='__main__':unittest.main()
