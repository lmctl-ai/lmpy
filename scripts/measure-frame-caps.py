"""Run inside the runtime container under production memory/pids/tmpfs limits."""
import json, pathlib, resource, subprocess, tempfile, time
with tempfile.TemporaryDirectory(prefix='pycli-caps-') as tmp:
 root=pathlib.Path(tmp)
 cases={
  'numeric_cell_cap':('a,b,c,d,e\n'+'1,2,3,4,5\n'*100000,{}),
  'row_cap':('x\n'+'1\n'*200000,{}),
  'string_cell_cap':(','.join('c'+str(i) for i in range(50))+'\n'+(','.join('abcd' for i in range(50))+'\n')*10000,{}),
  'category_200k_rows_100k_levels':('c\n'+''.join(str(i%100000)+'\n' for i in range(200000)),{'dtypes':{'c':'category'}}),
  'csv_byte_near_cap':('s\n'+('x'*82+'\n')*200000,{}),
 }
 results={}
 for name,(text,options) in cases.items():
  (root/'input.csv').write_text(text);p={'csv':'input.csv','out':name+'.npz',**options}
  start=time.monotonic();r=subprocess.run(['/usr/local/bin/pycli','frame_import','--input-json',json.dumps(p)],cwd=root,capture_output=True,text=True)
  elapsed=time.monotonic()-start;assert r.returncode==0,(name,r.returncode,r.stdout,r.stderr)
  j=json.loads(r.stdout);assert len(r.stdout.encode())<65536
  results[name]={'seconds':elapsed,'csv_bytes':len(text.encode()),'shape':j['shape'],'worker_ms':j['meta']['elapsed_ms'],'peak_child_rss_kib_upper_bound':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss}
  (root/(name+'.npz')).unlink();print(name,json.dumps(results[name]),flush=True)
 print(json.dumps(results,indent=2))
