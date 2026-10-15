"""Exact rational vertex enumeration for tiny bounded LPs; no solver dependency."""
from fractions import Fraction as F
from itertools import combinations

def optimum(p):
 n=len(p['c']);a=list(p.get('a_ub',[]));b=list(p.get('b_ub',[]));eq=p.get('a_eq',[]);be=p.get('b_eq',[])
 for i,(lo,hi) in enumerate(p['bounds']):
  for v,s in [(lo,-1),(hi,1)]:
   if v is not None:a.append([s*int(j==i) for j in range(n)]);b.append(s*v)
 a=[[F(str(v)) for v in row] for row in a];b=list(map(lambda x:F(str(x)),b));eq=[[F(str(v)) for v in row] for row in eq];be=list(map(lambda x:F(str(x)),be));out=[]
 for indices in combinations(range(len(a)+len(eq)),n):
  aa=a+eq;bb=b+be;m=[list(aa[i])+[bb[i]] for i in indices]
  for j in range(n):
   k=next((k for k in range(j,n) if m[k][j]),None)
   if k is None:break
   m[j],m[k]=m[k],m[j];v=m[j][j];m[j]=[x/v for x in m[j]]
   for k in range(n):
    if k!=j:
     v=m[k][j];m[k]=[x-v*y for x,y in zip(m[k],m[j])]
  else:
   x=[row[-1] for row in m]
   if all(sum(u*v for u,v in zip(row,x))<=r for row,r in zip(a,b)) and all(sum(u*v for u,v in zip(row,x))==r for row,r in zip(eq,be)):
    out.append(sum(F(str(c))*v for c,v in zip(p['c'],x)))
 return float((max if p.get('maximize') else min)(out))
