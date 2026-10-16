"""Run Kimi's five deliberately incorrect S8b-2 workers in private scratch copies.
In pycli:test: python scripts/check-s8b2-review-mutations.py /build-pycli/bin/pycli_dev [results.json]
Every mutant must produce a failing regression in tests/s8b2_review.py; production files are never changed.
"""
import json, os, pathlib, subprocess, sys, tempfile
root=pathlib.Path(__file__).resolve().parents[1]
variants=[
 ('hops_plus_100','s8g.py',"hop_raw = np.asarray([hops.get(u, 0) for u in nodes], dtype='<i8')","hop_raw = np.asarray([hops.get(u, 0) + 100 if u in dist else 0 for u in nodes], dtype='<i8')",'GraphReview.test_twin_hops_are_exact_and_zero_row_tables_round_trip'),
 ('reject_zero_rows','s8.py',"    s8_cap(len(names) <= TABLE_COLS and n <= TABLE_ROWS and n * len(names) <= TABLE_CELLS, 'table size')","    s8_cap(len(names) <= TABLE_COLS and 0 < n <= TABLE_ROWS and n * len(names) <= TABLE_CELLS, 'table size')",'GraphReview.test_twin_hops_are_exact_and_zero_row_tables_round_trip'),
 ('logsf_from_sf','s8.py',"'sf': frozen.sf, 'logsf': frozen.logsf, 'ppf': frozen.ppf, 'isf': frozen.isf}","'sf': frozen.sf, 'logsf': (lambda q, _np=__import__('numpy'): _np.log(frozen.sf(q))), 'ppf': frozen.ppf, 'isf': frozen.isf}",'DistReview.test_far_tails_and_isf_are_not_derived_from_their_complements'),
 ('isf_from_ppf','s8.py',"'sf': frozen.sf, 'logsf': frozen.logsf, 'ppf': frozen.ppf, 'isf': frozen.isf}","'sf': frozen.sf, 'logsf': frozen.logsf, 'ppf': frozen.ppf, 'isf': (lambda q: frozen.ppf(1 - q))}",'DistReview.test_far_tails_and_isf_are_not_derived_from_their_complements'),
 ('rvs_mt19937','s8.py',"    rng = np.random.Generator(np.random.PCG64(p['seed']))\n    with np.errstate(all='ignore'):\n        if dist == 'norm':","    rng = np.random.Generator(np.random.MT19937(p['seed']))\n    with np.errstate(all='ignore'):\n        if dist == 'norm':",'DistReview.test_pcg64_golden_streams')]
results=[]
for name,file,old,new,test in variants:
    source=(root/'py'/file).read_text()
    assert old in source,(name,old)
    with tempfile.TemporaryDirectory(prefix='pycli-mutant-') as tmp:
        d=pathlib.Path(tmp);(d/file).write_text(source.replace(old,new))
        (d/'worker.py').write_text('\n'.join((d/n if (d/n).exists() else root/'py'/n).read_text() for n in ('frames.py','numerics.py','s4.py','sm.py','ml.py','s7.py','s8.py','s8b.py','s8g.py','worker.py')))
        wrapper=d/'python';wrapper.write_text('#!/bin/sh\nshift 6\nexec python3 -P -s -B -X faulthandler "'+str(d/'worker.py')+'" "$@"\n');wrapper.chmod(0o755)
        env=dict(os.environ,REVIEW_VARIANT_DIR=str(d),PYCLI_PYTHON=str(wrapper))
        r=subprocess.run([sys.executable,str(root/'tests/s8b2_review.py'),sys.argv[1],test],env=env,text=True,capture_output=True)
        results.append(dict(variant=name,test=test,exit_code=r.returncode,output=(r.stdout+r.stderr)[-3000:]))
        print(name,'exit',r.returncode,flush=True)
        assert r.returncode==1,results[-1]
if len(sys.argv)>2:pathlib.Path(sys.argv[2]).write_text(json.dumps(results,indent=2)+'\n')
