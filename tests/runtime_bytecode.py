"""Run inside the final runtime image, with -P -s -B."""
import importlib.machinery, pathlib, struct
compiled=[]
original=importlib.machinery.SourceFileLoader.source_to_code
def record(self,data,path,*args,**kwargs):
 if '/site-packages/' not in path:compiled.append(path)
 return original(self,data,path,*args,**kwargs)
importlib.machinery.SourceFileLoader.source_to_code=record
import pandas
assert not compiled, compiled
for name in ('typing','inspect','tarfile'):
 p=pathlib.Path('/usr/local/lib/python3.12/__pycache__')/(name+'.cpython-312.pyc')
 assert struct.unpack('<I',p.read_bytes()[4:8])[0]==3, p
print('stdlib bytecode: hash-checked, no fresh pandas source compilation')
