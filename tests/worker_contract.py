"""Independent codec data and audit probes, with no production dispatch imports."""
import hashlib, importlib.util, json, math, pathlib, random, struct, sys
root=pathlib.Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('fixed_worker',root/'py/worker.py')
w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
rng=random.Random(2026)
values=[0.,-0.,5e-324,sys.float_info.max,sys.float_info.min]
while len(values)<100000:
    value=struct.unpack('d',rng.randbytes(8))[0]
    if math.isfinite(value): values.append(value)
encoded=json.loads(w.codec(values))
assert all(struct.pack('d',x)==struct.pack('d',y) for x,y in zip(values,encoded))
assert json.loads(w.codec([None,float('nan'),float('inf'),-float('inf')]))==['NA','NaN','Inf','-Inf']
for name in ('pickle','socket','subprocess','urllib','http'):
    try: w.audit('import',(name,None,None,None,None))
    except RuntimeError: pass
    else: raise AssertionError('audit allowed '+name)
print('100000 binary64 values, nonfinite tokens and audit blocklist passed')
