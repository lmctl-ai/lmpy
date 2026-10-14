"""Kernel termination during output must leave no named partial artifact."""
import pathlib,subprocess,sys,tempfile,signal
binary=str(pathlib.Path(sys.argv[1]).resolve())
with tempfile.TemporaryDirectory(dir='.') as directory:
    out=pathlib.Path(directory).resolve()/'cancel.safetensors'
    p=subprocess.run([binary,str(out)],cwd=directory,capture_output=True)
    assert p.returncode == -signal.SIGXFSZ,p.returncode
    assert not list(out.parent.iterdir()),'terminated writer leaked '+str(list(out.parent.iterdir()))
print('kernel-terminated writer leaves no artifact or named partial file')
