# Source: pandas 2.x user guide, "Time series / date functionality" (https://pandas.pydata.org/docs/user_guide/timeseries.html)
#   Resampling section: rng = pd.date_range("1/1/2012", periods=100, freq="s"); ts = pd.Series(randint(0, 500, 100), rng);
#   ts.resample("5Min").sum() / .ohlc() / .max(); and frequency conversion of a daily series (resample "W" and "MS",
#   upsampled 6-hourly data downsampled with "D"); plus the "Windowing operations" guide
#   (https://pandas.pydata.org/docs/user_guide/window.html): rolling(7).mean() / .std(), and shift / diff / pct_change.
# The docs draw with np.random.randint / randn; here np.random.default_rng(7) supplies the same kinds of draws in a
# reproducible stream (lmpy's rng(7) is bit-identical).
# Prints the answers as one JSON object; examples/python/pandas-timeseries.lms pins them.
import json
import numpy as np
import pandas as pd

rng = np.random.default_rng(7)

# 100 seconds of integers: the user guide's 5-minute bins (one bin), and 1-minute bins (two bins) for ohlc / max
idx = pd.date_range("1/1/2012", periods=100, freq="s")
ts = pd.Series(rng.integers(0, 500, len(idx)), index=idx)
r5 = ts.resample("5Min").sum()
ohlc = ts.resample("1Min").ohlc()
mx = ts.resample("1Min").max()

# a daily random walk over 120 days
days = pd.date_range("2024-01-01", periods=120, freq="D")
walk = pd.Series(100 + rng.standard_normal(120).cumsum(), index=days)
weekly = walk.resample("W").mean()          # weeks ending Sunday, labelled by the Sunday
monthly = walk.resample("MS").sum()         # calendar months, labelled by the first day
monthly_last = walk.resample("MS").last()
roll_mean = walk.rolling(7).mean()
roll_std = walk.rolling(7).std()
lag = walk.shift(1)
change = walk.diff()
pct = walk.pct_change()

# 6-hourly readings downsampled to days
q = pd.date_range("2024-03-01", periods=40, freq="6h")
six = pd.Series(rng.normal(20.0, 3.0, 40), index=q)
daily = six.resample("D").agg(["mean", "min", "max", "count"])

nan = lambda s: [None if pd.isna(x) else float(x) for x in s]
print(json.dumps({
    "sec_5min_sum": r5.tolist(), "sec_1min_open": ohlc["open"].tolist(), "sec_1min_high": ohlc["high"].tolist(),
    "sec_1min_low": ohlc["low"].tolist(), "sec_1min_close": ohlc["close"].tolist(), "sec_1min_max": mx.tolist(),
    "weekly_labels": [d.strftime("%Y-%m-%d") for d in weekly.index], "weekly_mean": weekly.tolist(),
    "monthly_labels": [d.strftime("%Y-%m-%d") for d in monthly.index], "monthly_sum": monthly.tolist(),
    "monthly_last": monthly_last.tolist(),
    "rolling7_mean": nan(roll_mean), "rolling7_std": nan(roll_std),
    "shift1": nan(lag), "diff": nan(change), "pct_change": nan(pct),
    "daily_mean": daily["mean"].tolist(), "daily_min": daily["min"].tolist(), "daily_max": daily["max"].tolist(),
    "daily_count": daily["count"].tolist(),
}))
