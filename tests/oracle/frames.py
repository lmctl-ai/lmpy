"""Direct pandas oracle, independent of the worker and its serialization."""
def grouped_sales():
    import pandas as pd
    return pd.DataFrame({'region':['E','W','E','W'],'sales':[10,20,30,40]}).groupby('region')['sales'].sum().tolist()

def spring_hourly():
    import pandas as pd
    dates=['2026-03-08T00:30:00-05:00','2026-03-08T01:30:00-05:00','2026-03-08T03:30:00-04:00','2026-03-08T04:30:00-04:00']
    df=pd.DataFrame({'ts':pd.to_datetime(dates,utc=True).tz_convert('America/New_York'),'v':[1,2,3,4]})
    return df.resample('1h',on='ts')['v'].sum().tolist()
