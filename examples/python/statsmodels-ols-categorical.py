# Source: the statsmodels formula documentation ("Fitting models using R-style formulas",
#   https://www.statsmodels.org/stable/example_formulas.html: categorical regressors with C(), smf.ols, and the ANOVA
#   comparison of nested models with sm.stats.anova_lm), applied to statsmodels' bundled Grunfeld investment panel
#   (11 firms, 1935-1954; public domain) instead of the docs' Guerry data, which statsmodels downloads from the R datasets
#   site. The model is the classic fixed-effects (LSDV) regression invest ~ value + capital + C(firm).
# Prints the answers as one JSON object; examples/python/statsmodels-ols-categorical.lms pins them.
import json
import statsmodels.api as sm
import statsmodels.formula.api as smf

df = sm.datasets.grunfeld.load_pandas().data
mod = smf.ols("invest ~ value + capital + C(firm)", data=df).fit()
restricted = smf.ols("invest ~ value + capital", data=df).fit()
cmp = sm.stats.anova_lm(restricted, mod)
typ2 = sm.stats.anova_lm(mod, typ=2)
# t test of one linear hypothesis, as in the formula docs' t_test / f_test section
tt = mod.t_test("value = capital")

print(json.dumps({
    "param_names": list(mod.params.index), "params": mod.params.tolist(), "bse": mod.bse.tolist(),
    "tvalues": mod.tvalues.tolist(), "pvalues": mod.pvalues.tolist(),
    "rsquared": mod.rsquared, "rsquared_adj": mod.rsquared_adj, "fvalue": mod.fvalue, "f_pvalue": mod.f_pvalue,
    "ssr": mod.ssr, "llf": mod.llf, "aic": mod.aic, "bic": mod.bic, "df_resid": mod.df_resid,
    "restricted_ssr": restricted.ssr,
    "anova_F": float(cmp["F"].iloc[1]), "anova_p": float(cmp["Pr(>F)"].iloc[1]), "anova_df_diff": float(cmp["df_diff"].iloc[1]),
    "anova_ss_diff": float(cmp["ss_diff"].iloc[1]),
    "typ2_sum_sq": typ2["sum_sq"].tolist(), "typ2_F": typ2["F"].iloc[:3].tolist(), "typ2_p": typ2["PR(>F)"].iloc[:3].tolist(),
    "ttest_value_eq_capital": float(tt.tvalue[0][0]), "ttest_p": float(tt.pvalue),
}))
