"""Inside the hardened runtime: S7 operations at their caps, including the worst-case (cyclic polytope / moment-curve) hull inputs.
Usage: docker run --rm --pull=never --network=none --cpus=2 --memory=4g --pids-limit=128 --read-only --cap-drop=ALL --security-opt=no-new-privileges
  --tmpfs /tmp:rw,noexec,nosuid,nodev,size=512m -v $PWD/scripts:/s:ro --entrypoint python3 pycli:runtime -I /s/measure-s7-caps.py rbf hull tri cand misc
Each case prints: operation, exit code, status or error code, wall seconds and the cgroup peak so far."""
import json,subprocess,tempfile,time,pathlib,sys,numpy as np
def peak():
    for f in ('/sys/fs/cgroup/memory.peak','/sys/fs/cgroup/memory/memory.max_usage_in_bytes'):
        if pathlib.Path(f).exists(): return int(pathlib.Path(f).read_text())
d=pathlib.Path(tempfile.mkdtemp(prefix='pycli-s7-caps-',dir='/tmp'))
BIN='/usr/local/bin/pycli'
def run(tag,op,params):
    t=time.time()
    r=subprocess.run([BIN,'--input-json',json.dumps({'op':op,'params':params})],cwd=d,capture_output=True,text=True)
    try: j=json.loads(r.stdout)
    except Exception: j={'raw':r.stdout[:200]}
    st=j.get('status') or (j.get('error') or {}).get('data',{}).get('tool_code') or j.get('raw')
    extra={k:j.get(k) for k in ('n_simplices','n_vertices','n_ridges','interpolation_error') if k in j}
    print(tag,'rc',r.returncode,st,round(time.time()-t,1),'s peak',round(peak()/1e9,2),'GB',extra,flush=True)
rng=np.random.default_rng(1)
which=sys.argv[1:]
def pts(name,a): np.save(d/name,a); return name
if 'rbf' in which:
    for n,dim in ((5000,2),(5000,8),(2000,3)):
        x=rng.uniform(size=(n,dim)); y=np.sin(x.sum(1)); q=rng.uniform(size=(1000,dim))
        for k in ('thin_plate_spline','gaussian'):
            run(f'rbf n={n} d={dim} {k}','interp_eval_to',dict(kind='rbf',kernel=k,epsilon=2.0,x=pts('x.npy',x),y=pts('y.npy',y),x_new=pts('q.npy',q),extrapolate='extend',out=f'r{n}{dim}{k[:2]}.safetensors'))
if 'hull' in which:
    for dim,n in ((2,20000),(3,20000),(4,5000),(5,1000),(6,300)):
        x=rng.normal(size=(n,dim)); x/=np.linalg.norm(x,axis=1)[:,None]   # on the sphere: every point a vertex
        run(f'hull sphere d={dim} n={n}','spatial_hull_to',dict(kind='convex_hull',points=pts('p.npy',x),out=f'h{dim}.safetensors'))
        t=np.linspace(0,1,n); c=np.stack([t**k for k in range(1,dim+1)],1)   # moment curve: cyclic polytope, worst-case facet count
        run(f'hull moment d={dim} n={n}','spatial_hull_to',dict(kind='convex_hull',points=pts('p.npy',c),out=f'hm{dim}.safetensors'))
if 'tri' in which:
    for dim,n in ((2,20000),(3,5000),(4,300),(5,100),(6,50)):
        x=rng.uniform(size=(n,dim))
        run(f'delaunay uniform d={dim} n={n}','spatial_hull_to',dict(kind='delaunay',points=pts('p.npy',x),out=f'd{dim}.safetensors'))
        run(f'voronoi uniform d={dim} n={n}','spatial_hull_to',dict(kind='voronoi',points=pts('p.npy',x),out=f'v{dim}.safetensors'))
        t=np.linspace(0,1,n); c=np.stack([t**k for k in range(1,dim+1)],1)
        run(f'delaunay moment d={dim} n={n}','spatial_hull_to',dict(kind='delaunay',points=pts('p.npy',c),out=f'dm{dim}.safetensors'))
if 'cand' in which:
    for kind,dim,n in (('convex_hull',4,800),('convex_hull',5,400),('convex_hull',6,100),('delaunay',3,800),('delaunay',4,300),('delaunay',5,100),('voronoi',3,800),('voronoi',4,300),('voronoi',5,100),('voronoi',6,50),('delaunay',6,50)):
        t=np.linspace(0,1,n); c=np.stack([t**k for k in range(1,dim+1)],1)
        run(f'{kind} moment d={dim} n={n}','spatial_hull_to',dict(kind=kind,points=pts('p.npy',c),out=f'c{kind[:2]}{dim}{n}.safetensors'))
if 'misc' in which:
    x=rng.normal(size=(60000,3)); run('knn 60000x3 k=10','spatial_knn_to',dict(points=pts('p.npy',x),k=10,exclude_self=True,out='k.safetensors'))
    run('knn radius','spatial_knn_to',dict(points=pts('p.npy',x),radius=0.1,max_pairs=1000000,out='kr.safetensors'))
    a=rng.normal(size=(1000,20)); run('dist 1000x1000','spatial_dist_to',dict(a=pts('a.npy',a),metric='mahalanobis',vi=pts('vi.npy',np.eye(20)),out='dd.safetensors'))
    s=rng.normal(size=200000); run('welch 200000','sig_spectrum_to',dict(x=pts('s.npy',s),method='welch',nperseg=65536,out='w.safetensors'))
    run('filter 200000','sig_filter_to',dict(family='ellip',order=20,btype='bandpass',wn=[0.1,0.3],rp=1,rs=60,x=pts('s.npy',s),mode='sosfiltfilt',out='f.safetensors'))
    run('peaks 200000','sig_peaks_to',dict(x=pts('s.npy',s),out='pk.safetensors',width=1.0))
    run('interp pchip 200000','interp_eval_to',dict(kind='pchip',x=pts('t.npy',np.arange(200000.0)),y=pts('s.npy',s),x_new=pts('q.npy',np.linspace(0,199999,200000)),out='ip.safetensors'))
