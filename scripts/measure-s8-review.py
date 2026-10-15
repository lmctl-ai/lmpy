"""Offline hardened-container review probes; run in a private writable cwd."""
import json, pathlib, resource, subprocess, time
binary='/usr/local/bin/pycli'
results=[]
def call(op,p,rc=0):
 start=time.monotonic();r=subprocess.run([binary,'--input-json',json.dumps(dict(op=op,params=p))],capture_output=True,text=True)
 z=json.loads(r.stdout);assert r.returncode==rc,(op,r.returncode,z)
 results.append(dict(op=op,exit_code=r.returncode,seconds=round(time.monotonic()-start,4),status=z.get('status',z.get('error',{}).get('data',{}).get('tool_code')),peak_children_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss))
 return z
pathlib.Path('left.csv').write_text('k\n'+'1\n'*200000)
call('frame_import',dict(csv='left.csv',out='left.npz'))
call('frame_import',dict(columns={'k':[1],'s':['x'*65000]},out='right.npz'))
for how in ('left','right','outer'):
 call('frame_merge',dict(left='left.npz',right='right.npz',on=['k'],how=how,validate='many_to_one',out=how+'.npz'),1)
# Actual allowed work cap, with high-magnitude data and no float64 overflow.
pathlib.Path('roll.csv').write_text('x\n'+'1e200\n'*200000)
call('frame_import',dict(csv='roll.csv',out='roll.npz'))
call('frame_rolling',dict(frame='roll.npz',columns=['x'],window=500,functions=['mean'],out='rolled.npz',time_limit=30))
print(json.dumps(results,indent=2))
