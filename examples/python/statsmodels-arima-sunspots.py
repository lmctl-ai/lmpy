# Source: statsmodels time-series examples on the yearly sunspots data (bundled, public domain).
#   "Autoregressive Moving Average (ARMA): Sunspots data" (https://www.statsmodels.org/stable/examples/notebooks/generated/tsa_arma_0.html):
#     ARIMA(dta, order=(2, 0, 0)).fit() and order (3, 0, 0): params, aic/bic/hqic, Durbin-Watson of the residuals,
#     plot_acf(dta, lags=40) / plot_pacf(dta, lags=40) (here the acf / pacf values those plots draw: plot_pacf's default
#     method is "ywm"), sm.tsa.acf(resid, qstat=True); out-of-sample forecasts of the AR(3) model.
#   "Stationarity and detrending (ADF/KPSS)" (https://www.statsmodels.org/stable/examples/notebooks/generated/stationarity_detrending_adf_kpss.html):
#     adfuller(df["SUNACTIVITY"], autolag="AIC") and kpss(df["SUNACTIVITY"], regression="c", nlags="auto").
#   statsmodels.stats.diagnostic.acorr_ljungbox docstring: ARIMA(data["SUNACTIVITY"], order=(1, 0, 1)) residuals, lags=[10].
# Prints the answers as one JSON object; examples/python/statsmodels-arima-sunspots.lms pins them.
import json
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss

warnings.simplefilter("ignore")
dta = sm.datasets.sunspots.load_pandas().data
dta.index = pd.Index(sm.tsa.datetools.dates_from_range("1700", "2008"))
dta.index.freq = dta.index.inferred_freq
del dta["YEAR"]

arma_mod20 = ARIMA(dta, order=(2, 0, 0)).fit()
arma_mod30 = ARIMA(dta, order=(3, 0, 0)).fit()
resid = arma_mod30.resid
r, q, p = sm.tsa.acf(resid.values.squeeze(), fft=True, qstat=True)
x = dta.values.squeeze()
acf40 = sm.tsa.acf(x, nlags=40)
pacf40 = sm.tsa.pacf(x, nlags=40, method="ywm")

adf = adfuller(dta["SUNACTIVITY"], autolag="AIC")
kp = kpss(dta["SUNACTIVITY"], regression="c", nlags="auto")

res101 = ARIMA(dta["SUNACTIVITY"], order=(1, 0, 1)).fit()
lb = sm.stats.acorr_ljungbox(res101.resid, lags=[10], return_df=True)

print(json.dumps({
    "ar2_params": arma_mod20.params.tolist(), "ar2_aic": arma_mod20.aic, "ar2_bic": arma_mod20.bic,
    "ar2_hqic": arma_mod20.hqic, "ar2_llf": arma_mod20.llf,
    "ar3_params": arma_mod30.params.tolist(), "ar3_aic": arma_mod30.aic, "ar3_bic": arma_mod30.bic,
    "ar3_hqic": arma_mod30.hqic, "ar3_llf": arma_mod30.llf,
    "ar3_durbin_watson": float(sm.stats.durbin_watson(resid.values)),
    "ar3_forecast": arma_mod30.forecast(steps=5).tolist(),
    "ar3_resid_acf": r[:11].tolist(), "ar3_resid_q": q[:10].tolist(), "ar3_resid_q_p": p[:10].tolist(),
    "acf40": acf40.tolist(), "pacf40": pacf40.tolist(),
    "adf_stat": adf[0], "adf_pvalue": adf[1], "adf_usedlag": adf[2], "adf_nobs": adf[3],
    "adf_crit": [adf[4]["1%"], adf[4]["5%"], adf[4]["10%"]], "adf_icbest": adf[5],
    "kpss_stat": kp[0], "kpss_pvalue": kp[1], "kpss_lags": kp[2],
    "lb_stat": float(lb["lb_stat"].iloc[0]), "lb_pvalue": float(lb["lb_pvalue"].iloc[0]),
}))
