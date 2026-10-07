import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor

def _agg_count_by_time(d, time_col, target_col):
    work=d.copy(); work[time_col]=pd.to_datetime(work[time_col],errors='coerce'); work=work.dropna(subset=[time_col]).sort_values(time_col)
    if work.empty: raise ValueError('No valid timestamps found for forecasting.')
    daily=work.set_index(time_col).assign(_c=1)['_c'].resample('D').sum().interpolate().dropna()
    return daily, 'records'

def execute(df,target,time_col,horizon=7):
    if time_col not in df.columns: raise ValueError('Forecasting requires a time column.')
    numeric_cols=[c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if target and target not in df.columns:
        if numeric_cols: target=numeric_cols[0]
        else: raise ValueError(f"Target '{target}' not found.")
    if target and not pd.api.types.is_numeric_dtype(df[target]):
        if numeric_cols: target=numeric_cols[0]
    daily=None; yname=target
    if target and target in df.columns and pd.api.types.is_numeric_dtype(df[target]):
        d=df[[time_col,target]].copy(); d[time_col]=pd.to_datetime(d[time_col],errors='coerce'); d[target]=pd.to_numeric(d[target],errors='coerce'); d=d.dropna().sort_values(time_col)
        daily=d.set_index(time_col)[target].resample('D').mean().interpolate().dropna()
    if len(daily)<30: raise ValueError('At least 30 daily observations are recommended for forecasting.')
    vals=daily.values.astype(float); lags=7; rows=[]; ys=[]
    for i in range(lags,len(vals)): rows.append(vals[i-lags:i]); ys.append(vals[i])
    X=np.array(rows); y=np.array(ys); split=max(10,int(len(X)*.8)); model=RandomForestRegressor(n_estimators=300,max_depth=10,min_samples_leaf=2,random_state=42,n_jobs=2); model.fit(X[:split],y[:split]); pred=model.predict(X[split:]); rmse=float(np.sqrt(np.mean((pred-y[split:])**2)))
    history=list(vals[-lags:]); future=[]; last_date=daily.index[-1]
    for i in range(horizon):
        pv=float(model.predict(np.array(history[-lags:]).reshape(1,-1))[0]); future.append({'date':str(last_date+pd.Timedelta(days=i+1)).split(' ')[0],'forecast':pv}); history.append(pv)
    return {'success':True,'operation':'forecast','summary':f"Forecasted {yname} for the next {horizon} day(s) using a lag-based Random Forest time-series model.",'metrics':{'backtest_RMSE':rmse,'target_used':yname},'table':future,'chart':{'type':'forecast','x':'date','y':'forecast','history':[{'date':str(i).split(' ')[0],'actual':float(v)} for i,v in daily.tail(60).items()],'forecast':future},'model':model,'evaluation':{'backtest_rows':len(y[split:])},'limitations':['Forecast quality depends on stability of historical patterns; this is a practical baseline rather than a guarantee.']}
