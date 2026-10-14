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
        work=pathlib.Path(directory)
        import random, struct
        rng=random.Random(1)
        def save_npy(path,shape):
            # Host Python has no NumPy: write a v1 little-endian float64 .npy by hand.
            dims=''.join(f'{d}, ' for d in shape)
            header=("{'descr': '<f8', 'fortran_order': False, 'shape': ("+dims.rstrip()+"), }").encode()
            header+=b' '*((-(10+len(header)+1))%64)+b'\n'
            count=1
            for d in shape: count*=d
            path.write_bytes(b'\x93NUMPY\x01\x00'+struct.pack('<H',len(header))+header+struct.pack(f'<{count}d',*(rng.gauss(0,1) for _ in range(count))))
        save_npy(work/'tall.npy',(200,5)); save_npy(work/'rhs.npy',(200,)); save_npy(work/'sq.npy',(100,100))
        counter=[0]
        def unique():
            counter[0]+=1
            return str(work/f'out{counter[0]}.safetensors')
        # (name, op, params; a callable receives nothing and returns fresh params for write ops)
        cases=[('py_info','py_info',{}),('la_solve','la_solve',{'a':[[1,2,3],[0,1,4],[5,6,0]],'b':[14,14,17]}),
               ('stats_dist','stats_dist',{'dist':'norm','fn':'ppf','x':[.975]}),
               ('array_import','array_import',lambda:{'format':'json','data':[[1,2],[3,4]],'out':unique()}),
               ('array_inspect','array_inspect',{'ref':str(work/'sq.npy')}),
               ('la_lstsq_200x5','la_lstsq',{'a':str(work/'tall.npy'),'b':str(work/'rhs.npy')}),
               ('la_props_100x100','la_props',{'a':str(work/'sq.npy')}),
               ('la_decomp_svd_100x100','la_decomp',lambda:{'a':str(work/'sq.npy'),'kind':'svd','out':unique()}),
               ('la_decomp_eig_100x100','la_decomp',lambda:{'a':str(work/'sq.npy'),'kind':'eig','out':unique()})]
        for name,op,params in cases:
            values=[]
            for _ in range(10):
                call=params() if callable(params) else params
                start=time.monotonic()
                result=json.loads(subprocess.check_output([str(shim),'--input-json',json.dumps({'op':op,'params':call})]))
                assert 'error' not in result,result
                values.append(time.monotonic()-start)
            row[name]=summary(values)
    results[image]=row
print(json.dumps(results,indent=2))
