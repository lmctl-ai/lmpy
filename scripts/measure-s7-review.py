"""Run one S7 review resource case in a fresh hardened runtime container.
Mount a private writable working directory and this script read-only, use --entrypoint
python3, and pass a case name. See docs/timings.md for the exact container flags.
ru_maxrss is the CLI/worker child high-water mark in KiB, not total cgroup memory.
"""
import json,pathlib,resource,subprocess,sys,time
import numpy as np
case=sys.argv[1]
if case=='knn_preflight':
 np.save('points.npy',np.arange(200000.)[:,None]);op='spatial_knn_to';p=dict(points='points.npy',k=1000,out='out.safetensors')
elif case=='knn_at_output_cap':
 np.save('points.npy',np.arange(1000.)[:,None]);np.save('queries.npy',np.arange(950.)[:,None]+.1);op='spatial_knn_to';p=dict(points='points.npy',queries='queries.npy',k=1000,out='out.safetensors')
elif case.startswith('peaks'):
 np.save('saw.npy',np.arange(200000.)+2*(np.arange(200000)%2));op='sig_peaks' if case=='peaks_inline' else 'sig_peaks_to';p=dict(x='saw.npy')
 if op.endswith('_to'):p['out']='out.safetensors'
 if case=='peaks_windowed':p['wlen']=199
elif case=='radius_dense':
 np.save('points.npy',np.zeros((200000,1)));op='spatial_knn_to';p=dict(points='points.npy',radius=1,max_pairs=100000,out='out.safetensors')
else:raise ValueError(case)
t=time.monotonic();r=subprocess.run(['/usr/local/bin/pycli','--input-json',json.dumps(dict(op=op,params=p))],capture_output=True,text=True);seconds=time.monotonic()-t
z=json.loads(r.stdout);print(json.dumps(dict(case=case,seconds=seconds,exit_code=r.returncode,maxrss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,result=z)))
for name in ['points.npy','queries.npy','saw.npy','out.safetensors']:pathlib.Path(name).unlink(missing_ok=True)
