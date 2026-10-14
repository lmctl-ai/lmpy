"""Measure complete per-call latency with the production shim's flags."""
import json, pathlib, subprocess, sys, tempfile, time
def summary(values):
    import statistics
    values=sorted(values)
    return {'p50_ms':round(statistics.median(values)*1000,3),'p95_ms':round(values[min(len(values)-1,int(.95*len(values)))]*1000,3),'n':len(values)}
results={}
for image in sys.argv[1:]:
    row={'image_bytes':int(subprocess.check_output(['docker','image','inspect',image,'--format','{{.Size}}']))}
    with tempfile.TemporaryDirectory(dir='.') as directory:
        shim=pathlib.Path(directory)/'pycli'
        shim.write_text(pathlib.Path('bin/pycli').read_text().replace('pycli:runtime',image));shim.chmod(0o755)
        for op,params in [('py_info',{}),('la_solve',{'a':[[1,2,3],[0,1,4],[5,6,0]],'b':[14,14,17]}),('stats_dist',{'dist':'norm','fn':'ppf','x':[.975]})]:
            values=[]
            for _ in range(10):
                start=time.monotonic()
                result=json.loads(subprocess.check_output([str(shim),'--input-json',json.dumps({'op':op,'params':params})]))
                assert 'error' not in result,result
                values.append(time.monotonic()-start)
            row[op]=summary(values)
    results[image]=row
print(json.dumps(results,indent=2))
