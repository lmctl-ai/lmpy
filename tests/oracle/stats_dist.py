"""Direct distribution calls; no worker imports or argument translation."""
from scipy.stats import norm, t
FIXTURES = [
    ({'dist':'norm','fn':'ppf','x':[0.975]}, [norm.ppf(0.975)]),
    ({'dist':'norm','fn':'pdf','x':[-2.,0.,3.]}, norm.pdf([-2.,0.,3.])),
    ({'dist':'norm','fn':'cdf','x':[-2.,0.,3.],'params':{'loc':1.,'scale':2.}}, norm.cdf([-2.,0.,3.], loc=1., scale=2.)),
    ({'dist':'t','fn':'sf','x':[-3.,0.,5.],'params':{'df':7.}}, t.sf([-3.,0.,5.],7.)),
    ({'dist':'t','fn':'ppf','x':[0.01,0.5,0.99],'params':{'df':3.}}, t.ppf([0.01,0.5,0.99],3.)),
]
assert norm.ppf(0.975) == 1.959963984540054
