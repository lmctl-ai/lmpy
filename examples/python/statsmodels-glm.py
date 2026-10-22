# Source: statsmodels GLM examples on public-domain bundled datasets.
#   - RAND HIE doctor visits, Poisson: the statsmodels "Discrete Choice Models Overview" count example
#     (https://www.statsmodels.org/stable/examples/notebooks/generated/discrete_choice_overview.html):
#       rand_data = sm.datasets.randhie.load(); rand_exog = sm.add_constant(rand_data.exog, prepend=False)
#       sm.Poisson(rand_data.endog, rand_exog)
#     fitted here as the identical GLM, sm.GLM(endog, exog, family=sm.families.Poisson()) (the GLM docs' form; same MLE).
#   - ANES 1996, binomial/logit: the binary vote (1 = Dole) on the regressors of the same notebook's anes96 model
#     (logpopul, selfLR, age, educ, income, constant last), sm.GLM(vote, exog, family=sm.families.Binomial()).
#   - Engel food expenditure, gamma with a log link: sm.GLM(foodexp, add_constant(income),
#     family=sm.families.Gamma(link=sm.families.links.Log())) (the GLM docs' gamma family on the positive response
#     of statsmodels' Engel example data).
# Prints the answers as one JSON object; examples/python/statsmodels-glm.lms pins them.
import json
import statsmodels.api as sm

rand_data = sm.datasets.randhie.load()
rand_exog = sm.add_constant(rand_data.exog, prepend=False)
pois = sm.GLM(rand_data.endog, rand_exog, family=sm.families.Poisson()).fit()

anes = sm.datasets.anes96.load_pandas().data
anes_exog = sm.add_constant(anes[["logpopul", "selfLR", "age", "educ", "income"]], prepend=False)
logit = sm.GLM(anes["vote"], anes_exog, family=sm.families.Binomial()).fit()

engel = sm.datasets.engel.load_pandas().data
gamma = sm.GLM(engel["foodexp"], sm.add_constant(engel["income"]),
               family=sm.families.Gamma(link=sm.families.links.Log())).fit()

out = {}
for name, r in [("randhie", pois), ("anes96", logit), ("engel", gamma)]:
    out.update({
        f"{name}_params": r.params.tolist(), f"{name}_bse": r.bse.tolist(), f"{name}_pvalues": r.pvalues.tolist(),
        f"{name}_deviance": r.deviance, f"{name}_null_deviance": r.null_deviance, f"{name}_pearson_chi2": r.pearson_chi2,
        f"{name}_scale": r.scale, f"{name}_llf": r.llf, f"{name}_aic": r.aic, f"{name}_df_resid": r.df_resid,
    })
print(json.dumps(out, default=float))
