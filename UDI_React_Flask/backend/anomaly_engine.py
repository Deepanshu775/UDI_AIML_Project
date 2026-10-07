import numpy as np, pandas as pd
from sklearn.ensemble import IsolationForest

def execute(df, plan):
    t=plan.target_column
    numeric_cols=[c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if t and t not in df.columns: raise ValueError(f"Column '{t}' not found.")
    if t and not pd.api.types.is_numeric_dtype(df[t]):
        if numeric_cols:
            t=numeric_cols[0]
        else:
            return _categorical_count_anomaly(df, t)
    if not t:
        if numeric_cols:
            t=numeric_cols[0]
        else:
            raise ValueError('At least one numeric column is required for anomaly detection.')
    x=pd.to_numeric(df[t],errors='coerce'); valid=x.dropna();
    if len(valid)<12: raise ValueError('At least 12 valid observations are recommended for anomaly detection.')
    q1,q3=valid.quantile(.25),valid.quantile(.75); iqr=q3-q1; lo,hi=q1-1.5*iqr,q3+1.5*iqr
    z=(valid-valid.mean())/(valid.std() or 1); iqr_mask=(valid<lo)|(valid>hi); z_mask=z.abs()>3
    iso=IsolationForest(random_state=42,contamination='auto',n_estimators=250); labels=iso.fit_predict(valid.to_numpy().reshape(-1,1)); scores=-iso.score_samples(valid.to_numpy().reshape(-1,1))
    out=df.loc[valid.index].copy(); out['iqr_anomaly']=iqr_mask.values; out['zscore_anomaly']=z_mask.values; out['isolation_anomaly']=labels==-1; out['anomaly_score']=scores
    anomalies=out[out[['iqr_anomaly','zscore_anomaly','isolation_anomaly']].sum(axis=1)>=2].sort_values('anomaly_score',ascending=False)
    return {'success':True,'operation':'detect_anomalies','summary':f"Found {len(anomalies):,} records flagged by at least two anomaly methods on '{t}'.",'metrics':{'iqr_count':int(iqr_mask.sum()),'zscore_count':int(z_mask.sum()),'isolation_count':int((labels==-1).sum()),'consensus_count':int(len(anomalies)),'target_used':str(t)},'table':anomalies.head(25).replace({np.nan:None}).to_dict('records'),'chart':{'type':'scatter','x':'row_index','y':t,'data':[{'row_index':int(i),'value':float(v),'anomaly':bool(i in anomalies.index)} for i,v in valid.head(500).items()]},'limitations':['Anomaly flags are statistical indicators and require domain review.']}

def _categorical_count_anomaly(df, cat_column):
    s=df[cat_column].dropna().astype(str)
    counts=s.value_counts()
    counts_df=counts.reset_index(); counts_df.columns=['Category','Count']
    numeric_counts=pd.to_numeric(counts_df['Count'])
    q1,q3=numeric_counts.quantile(.25),numeric_counts.quantile(.75); iqr=q3-q1; lo,hi=q1-1.5*iqr,q3+1.5*iqr
    iqr_mask=(numeric_counts<lo)|(numeric_counts>hi); z=(numeric_counts-numeric_counts.mean())/(numeric_counts.std() or 1); z_mask=z.abs()>3
    iso=IsolationForest(random_state=42,contamination='auto',n_estimators=250)
    X=numeric_counts.to_numpy().reshape(-1,1); labels=iso.fit_predict(X); scores=-iso.score_samples(X)
    counts_df['iqr_anomaly']=iqr_mask.values; counts_df['zscore_anomaly']=z_mask.values; counts_df['isolation_anomaly']=labels==-1; counts_df['anomaly_score']=scores
    anomalies=counts_df[counts_df[['iqr_anomaly','zscore_anomaly','isolation_anomaly']].sum(axis=1)>=2].sort_values('anomaly_score',ascending=False)
    return {'success':True,'operation':'detect_anomalies','summary':f"Category counts on '{cat_column}' flagged {len(anomalies):,} categories as anomalous by at least two methods (very high/low frequency).",'metrics':{'iqr_count':int(iqr_mask.sum()),'zscore_count':int(z_mask.sum()),'isolation_count':int((labels==-1).sum()),'consensus_count':int(len(anomalies)),'target_used':f'counts of {cat_column}'},'table':anomalies.head(25).replace({np.nan:None}).to_dict('records'),'chart':{'type':'scatter','x':'Category','y':'Count','data':[{'row_index':i,'Category':str(r['Category']),'value':float(r['Count']),'anomaly':bool((r[['iqr_anomaly','zscore_anomaly','isolation_anomaly']].sum()>=2))} for i,r in counts_df.head(500).iterrows()]},'limitations':['Categorical anomaly detection uses per-category counts; rare categories are flagged when they stand out from the overall frequency distribution.']}
