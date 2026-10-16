"""Run Kimi's five deliberately incorrect workers in private scratch copies.
In pycli:test: python scripts/check-s8b-review-mutations.py /build-pycli/bin/pycli_dev
Every mutant must produce a failing regression; production files are never changed.
"""
import json, os, pathlib, subprocess, sys, tempfile
root=pathlib.Path(__file__).resolve().parents[1]
variants=[
 ('inverse_residual_zero','s8b.py',"result['residual'] = float(residual)","residual = 0.0; result['residual'] = float(residual)",'test_inverse_relative_residual_and_scales'),
 ('arpack_mt19937','s8b.py','np.random.PCG64(p[\'seed\'])','np.random.MT19937(p[\'seed\'])','test_arpack_start_is_pcg64'),
 ('conjugate_eigs','s8b.py',"return 'OK', '', {'w': w, 'v': v}","return 'OK', '', {'w': w.conjugate(), 'v': v}",'test_complex_order_and_svd_vectors'),
 ('zero_svd_vectors','s8b.py',"return 'OK', '', {'s': s, 'u': u, 'vt': vt}","return 'OK', '', {'s': s, 'u': u*0, 'vt': vt*0}",'test_complex_order_and_svd_vectors'),
 ('acf_unadjusted','sm.py',"adjusted=p['adjusted']","adjusted=False",'test_adjusted_acf_hand_covariances')]
results=[]
for name,file,old,new,test in variants:
    source=(root/'py'/file).read_text()
    assert old in source,(name,old)
    with tempfile.TemporaryDirectory(prefix='pycli-mutant-') as tmp:
        d=pathlib.Path(tmp);(d/file).write_text(source.replace(old,new))
        (d/'worker.py').write_text('\n'.join((d/n if (d/n).exists() else root/'py'/n).read_text() for n in ('frames.py','numerics.py','s4.py','sm.py','ml.py','s7.py','s8.py','s8b.py','worker.py')))
        wrapper=d/'python';wrapper.write_text('#!/bin/sh\nshift 6\nexec python3 -P -s -B -X faulthandler "'+str(d/'worker.py')+'" "$@"\n');wrapper.chmod(0o755)
        env=dict(os.environ,REVIEW_VARIANT_DIR=str(d),PYCLI_PYTHON=str(wrapper))
        r=subprocess.run([sys.executable,str(root/'tests/s8b_review.py'),sys.argv[1],'Review.'+test],env=env,text=True,capture_output=True)
        results.append(dict(variant=name,test=test,exit_code=r.returncode,output=r.stdout+r.stderr))
        print(name,'exit',r.returncode,flush=True)
        assert r.returncode==1,results[-1]
if len(sys.argv)>2:pathlib.Path(sys.argv[2]).write_text(json.dumps(results,indent=2)+'\n')
