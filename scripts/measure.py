"""Offline timing harness; wall times include subprocess/container startup."""
import json, os, statistics, subprocess, sys, time
images=sys.argv[1:]
def summary(values):
    values=sorted(values)
    return {'p50_ms':round(statistics.median(values)*1000,3),'p95_ms':round(values[min(len(values)-1,int(.95*len(values)))]*1000,3),'n':len(values)}
results={}
for image in images:
    row={'image_bytes':int(subprocess.check_output(['docker','image','inspect',image,'--format','{{.Size}}']))}
    for name, code in [('boot','pass'),('numpy','import numpy'),('scipy_stats','import scipy.stats')]:
        inner='import json,subprocess,time,statistics; values=[]\nfor _ in range(20):\n t=time.monotonic(); subprocess.run(["python3","-P","-s","-B","-c",'+repr(code)+'],check=True); values.append(time.monotonic()-t)\nprint(json.dumps(values))'
        values=json.loads(subprocess.check_output(['docker','run','--rm','--pull=never','--network=none','-e','OPENBLAS_NUM_THREADS=1','-e','PYTHONHASHSEED=0','-e','OPENBLAS_CORETYPE=ARMV8','--entrypoint','python3',image,'-P','-s','-B','-c',inner]))
        row[name]=summary(values)
    values=[]
    for _ in range(10):
        start=time.monotonic()
        subprocess.run(['docker','run','--rm','--pull=never','--network=none','--entrypoint','/bin/true',image],check=True)
        values.append(time.monotonic()-start)
    row['docker_run']=summary(values)
    results[image]=row
print(json.dumps(results,indent=2))
