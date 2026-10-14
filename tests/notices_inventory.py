"""The shipped notice inventory must exactly regenerate from pinned inputs."""
import pathlib,subprocess,sys
root=pathlib.Path(__file__).resolve().parents[1]
p=subprocess.run([sys.executable,str(root/'scripts/notices.py')],capture_output=True,check=True)
assert p.stdout==(root/'THIRD_PARTY_NOTICES.md').read_bytes(),'notice generation/inventory drift'
print('hash-verified source notices and wheel metadata regenerate shipped inventory exactly')
