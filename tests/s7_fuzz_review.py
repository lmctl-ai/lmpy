"""Seed-71015 Astra primary/secondary regressions; independent NumPy oracles.
Honest numerical failures at extreme interpolation scales are explicitly permitted.
Nearest-neighbour identity and radius membership checked as well as distances."""
import os,sys,tempfile,pathlib
BIN=str(pathlib.Path(sys.argv.pop(1)).resolve())
work=tempfile.TemporaryDirectory(dir='.')
os.chdir(work.name)
import json, subprocess, time, numpy as np, math, pathlib
rng=np.random.default_rng(71015); logs=[]; errors=[]
def call(op,p):
 t=time.monotonic(); r=subprocess.run([BIN,'--input-json',json.dumps({'op':op,'params':p})],text=True,capture_output=True,timeout=15); z=json.loads(r.stdout); logs.append({'op':op,'params':p,'rc':r.returncode,'result':z,'wall':time.monotonic()-t}); return z
def check(tag,got,want,atol=1e-9):
 if logs[-1]['result'].get('status')=='NUMERICAL_FAILURE' and tag.startswith('scaled-'):return
 try: np.testing.assert_allclose(got,want,rtol=1e-8,atol=atol)
 except Exception as e: errors.append({'tag':tag,'case':logs[-1],'expected':np.asarray(want).tolist(),'error':str(e)})
for trial in range(5):
 x=np.cumsum(rng.uniform(.2,2,8)); y=rng.normal(size=8); q=rng.uniform(x[0],x[-1],10)
 p={'x':x.tolist(),'y':y.tolist(),'x_new':q.tolist()}
 z=call('interp_eval',dict(p,kind='linear')); check('linear',z.get('values'),np.interp(q,x,y))
 h=np.diff(x); A=np.zeros((8,8)); b=np.zeros(8); A[0,0]=A[-1,-1]=1
 for i in range(1,7): A[i,i-1:i+2]=[h[i-1],2*(h[i-1]+h[i]),h[i]]; b[i]=6*((y[i+1]-y[i])/h[i]-(y[i]-y[i-1])/h[i-1])
 m=np.linalg.solve(A,b); j=np.clip(np.searchsorted(x,q)-1,0,6); a=(x[j+1]-q)/h[j]; c=(q-x[j])/h[j]
 want=a*y[j]+c*y[j+1]+((a**3-a)*m[j]+(c**3-c)*m[j+1])*h[j]**2/6
 z=call('interp_eval',dict(p,kind='cubic',bc_type='natural')); check('natural-cubic',z.get('values'),want)
 z=call('interp_eval',dict(p,kind='cubic',y=(x**3-2*x+3).tolist())); check('polynomial',z.get('values'),q**3-2*q+3)
for family in ['butter','cheby1','cheby2','ellip','bessel']:
 f=np.linspace(0,.99,13); p=dict(family=family,order=int(rng.integers(2,9)),wn=.23,freqs=f.tolist())
 if family in ['cheby1','ellip']: p['rp']=1
 if family in ['cheby2','ellip']: p['rs']=40
 z=call('sig_filter',p); sos=np.asarray(z['sos']); v=np.exp(-1j*np.pi*f); H=np.ones(len(f),complex)
 for row in sos: H*=(row[0]+row[1]*v+row[2]*v*v)/(row[3]+row[4]*v+row[5]*v*v)
 check('filter',z['magnitude'],np.abs(H))
for n in [7,8,13,16,31]:
 x=rng.normal(size=n); k=np.arange(n//2+1); fft=np.exp(-2j*np.pi*k[:,None]*np.arange(n)[None,:]/n)@x; power=np.abs(fft)**2/n; power[1:]*=2
 if n%2==0:power[-1]/=2
 z=call('sig_spectrum',dict(method='periodogram',x=x.tolist(),detrend='none',fs=1));check('dft',z.get('pxx'),power)
 a=rng.integers(-2,3,size=n); peaks=[];i=1
 while i<n-1:
  j=i
  while j+1<n and a[j+1]==a[i]:j+=1
  if a[i]>a[i-1] and j<n-1 and a[j]>a[j+1]:peaks.append((i+j)//2)
  i=j+1
 z=call('sig_peaks',{'x':a.tolist()});check('peaks',z.get('indices'),peaks)
for trial in range(4):
 a=rng.normal(size=(7,3)); q=rng.normal(size=(3,3)); dif=abs(a[None,:,:]-q[:,None,:])
 for p in [1,2,3.5,'inf']:
  ds=dif.max(axis=2) if p=='inf' else (dif**p).sum(axis=2)**(1/p)
  z=call('spatial_knn',dict(points=a.tolist(),queries=q.tolist(),k=3,p=p));check('knn',z.get('distances'),np.sort(ds,axis=1)[:,:3]);check('knn-indices',z.get('indices'),np.argsort(ds,axis=1)[:,:3])
  z=call('spatial_knn',dict(points=a.tolist(),queries=q.tolist(),radius=2,p=p));check('radius-counts',z.get('counts'),(ds<=2).sum(axis=1));check('radius-distance',z.get('distances'),np.concatenate([np.sort(row[row<=2]) for row in ds]));check('radius-membership',z.get('indices'),np.concatenate([np.flatnonzero(row<=2)[np.argsort(row[row<=2])] for row in ds]))
 for metric in ['euclidean','sqeuclidean','cityblock','chebyshev','cosine','correlation','hamming','jaccard']:
  if metric=='euclidean':w=np.sqrt((dif*dif).sum(axis=2))
  elif metric=='sqeuclidean':w=(dif*dif).sum(axis=2)
  elif metric=='cityblock':w=dif.sum(axis=2)
  elif metric=='chebyshev':w=dif.max(axis=2)
  elif metric in ['cosine','correlation']:
   aa=a-a.mean(axis=1,keepdims=True) if metric=='correlation' else a;qq=q-q.mean(axis=1,keepdims=True) if metric=='correlation' else q;w=1-(qq@aa.T)/np.outer(np.linalg.norm(qq,axis=1),np.linalg.norm(aa,axis=1))
  elif metric=='hamming':w=(dif!=0).mean(axis=2)
  else:w=np.zeros((3,7))
  z=call('spatial_dist',dict(a=q.tolist(),b=a.tolist(),metric=metric));check('dist-'+metric,z.get('distances'),w)
for d in range(2,7):
 a=rng.normal(size=(d+1,d)); w=abs(np.linalg.det(a[1:]-a[0]))/math.factorial(d)
 for kind in ['convex_hull','delaunay']:
  z=call('spatial_hull',dict(points=a.tolist(),kind=kind));check('simplex',z.get('volume'),w)
for pts in [[[0,0]],[[0,0],[1,1]],[[0,0],[1,1],[2,2]],[[0,0]]*4,[[0,0,0],[1,0,0],[0,1,0],[1,1,0]]]:
 for kind in ['convex_hull','delaunay','voronoi']:
  z=call('spatial_hull',dict(points=pts,kind=kind))
  if z.get('status')!='DEGENERATE':errors.append({'tag':'degenerate','case':logs[-1]})
for scale in [1e-150,1e-12,1,1e12,1e150]:
 for kind in ['linear','cubic','pchip','akima','makima']:
  z=call('interp_eval',dict(kind=kind,x=[v*scale for v in range(5)],y=[0,1,2,3,4],x_new=[1.5*scale,2.5*scale]));check('scaled-'+kind,z.get('values'),[1.5,2.5])
for x in [[0,1,1,2],[0,2,1,3],[0,1e-16,1,2]]:call('interp_eval',dict(kind='linear',x=x,y=[0,1,2,3],x_new=[.5]))
pathlib.Path('results.json').write_text(json.dumps({'logs':logs,'errors':errors},indent=2)); print('primary calls',len(logs),'errors',errors,flush=True);assert not errors,errors

# Secondary starts from its documented fresh seed.
rng=np.random.default_rng(71015);logs=[];errors=[]
for trial in range(3):
 x=np.cumsum(rng.uniform(.3,1,7)); y=rng.normal(size=7)
 for lam in [0,1e5]:
  z=call('interp_eval',{'kind':'smoothing','x':x.tolist(),'y':(2*x+3).tolist(),'x_new':x.tolist(),'lam':lam});check('smooth-line',z.get('values'),2*x+3,1e-5)
 a=rng.normal(size=(8,2)); q=rng.normal(size=(3,2)); z=call('interp_eval',dict(kind='rbf',x=a.tolist(),y=(a[:,0]+2*a[:,1]+3).tolist(),x_new=q.tolist(),extrapolate='extend'));check('rbf-affine',z.get('values'),q[:,0]+2*q[:,1]+3)
 x=rng.normal(size=15); z=call('sig_filter',dict(family='butter',order=3,wn=.3,x=x.tolist(),mode='sosfilt'))
 y=x.copy()
 for b0,b1,b2,a0,a1,a2 in z['sos']:
  out=np.zeros_like(y)
  for i in range(len(y)):out[i]=(b0*y[i]+(b1*y[i-1]-a1*out[i-1] if i>0 else 0)+(b2*y[i-2]-a2*out[i-2] if i>1 else 0))/a0
  y=out
 check('filter-recurrence',z['y'],y)
 a=rng.normal(size=(4,3)); L=rng.normal(size=(3,3));vi=L.T@L;diff=a[:,None,:]-a[None,:,:];w=np.sqrt(np.einsum('...i,ij,...j->...',diff,vi,diff));z=call('spatial_dist',dict(a=a.tolist(),metric='mahalanobis',vi=vi.tolist()));check('mahalanobis',z.get('distances'),w)
pathlib.Path('secondary.json').write_text(json.dumps({'logs':logs,'errors':errors},indent=2));print('secondary calls',len(logs),'errors',errors,flush=True);assert not errors,errors

os.chdir("..");work.cleanup()
