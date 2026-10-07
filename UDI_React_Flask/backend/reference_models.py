"""Load domain reference models only when the uploaded schema supports them.
For arbitrary real-world schemas the runtime ML engine trains fresh models instead.
"""
from pathlib import Path
import json, joblib, numpy as np, pandas as pd
from config import MODEL_DIR

def _meta(domain):
    p=MODEL_DIR/domain/'metadata.json'
    if not p.exists(): return {}
    return json.loads(p.read_text(encoding='utf8'))

def can_use(domain, task, df, target):
    m=_meta(domain)
    if not m or target!=m.get({'regression':'regression_target','classification':'classification_target','forecast':'forecast_target','anomaly':'regression_target'}.get(task)): return False
    if task=='anomaly': needed=m.get('anomaly_features',[])
    elif task=='forecast': needed=[m.get('forecast_target'), 'Date']
    else: needed=m.get('training_feature_columns' if task=='regression' else 'classification_feature_columns',[])
    return bool(needed) and all(c in df.columns for c in needed if c)

def load_selected(domain,task):
    m=_meta(domain); key={'regression':'selected_regression','classification':'selected_classification','forecast':'forecasting_model','anomaly':'anomaly_model'}[task]
    name=m.get(key)
    if not name: raise ValueError(f'No reference {task} model is available for {domain}.')
    obj=joblib.load(MODEL_DIR/domain/name)
    return obj,m

def regression(df,domain,target):
    obj,m=load_selected(domain,'regression'); pred=obj.predict(df[m['training_feature_columns']]); return {'model':obj,'name':Path(m['selected_regression']).stem,'pred':pred,'target':target,'metadata':m}

def classification(df,domain,target):
    obj,m=load_selected(domain,'classification'); pipe=obj['pipeline']; pred=pipe.predict(df[m['classification_feature_columns']]); labels=obj['classes']; decoded=[labels[int(v)] for v in pred]; return {'model':pipe,'name':Path(m['selected_classification']).stem,'pred':decoded,'target':target,'metadata':m}
