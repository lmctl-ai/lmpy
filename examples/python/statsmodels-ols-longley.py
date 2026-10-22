# Source: the statsmodels OLS documentation example on the Longley data
#   (statsmodels.regression.linear_model.OLS docstring and the "Ordinary Least Squares" example notebook,
#   https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html):
#     data = sm.datasets.longley.load(); data.exog = sm.add_constant(data.exog); results = sm.OLS(data.endog, data.exog).fit()
#   Longley is also the NIST StRD linear-regression benchmark with certified values; the .lms checks those as well.
# Prints the answers as one JSON object; examples/python/statsmodels-ols-longley.lms pins them.
import json
import numpy as np
import statsmodels.api as sm

data = sm.datasets.longley.load()
exog = sm.add_constant(data.exog)
results = sm.OLS(data.endog, exog).fit()

print(json.dumps({
    "params": results.params.tolist(),
    "bse": results.bse.tolist(),
    "tvalues": results.tvalues.tolist(),
    "pvalues": results.pvalues.tolist(),
    "conf_int": np.asarray(results.conf_int()).tolist(),
    "rsquared": results.rsquared, "rsquared_adj": results.rsquared_adj,
    "fvalue": results.fvalue, "f_pvalue": results.f_pvalue,
    "llf": results.llf, "aic": results.aic, "bic": results.bic,
    "ssr": results.ssr, "scale_sqrt": float(np.sqrt(results.scale)),
    "df_model": results.df_model, "df_resid": results.df_resid,
    "condition_number": results.condition_number,
    "durbin_watson": float(sm.stats.durbin_watson(results.resid)),
}))
