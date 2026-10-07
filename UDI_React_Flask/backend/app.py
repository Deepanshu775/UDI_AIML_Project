import uuid, traceback
from pathlib import Path
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from config import PORT, DOMAINS, UPLOAD_DIR, MOCK_MODE, GEMINI_API_KEY, GEMINI_MODEL
from data_profiler import load_dataset, profile_dataframe
from schemas import AnalysisPlan
from validator import validate_plan
from analytics_engine import execute as analytics_execute
from anomaly_engine import execute as anomaly_execute
from forecasting_engine import execute as forecast_execute
from ml_engine import regression, classification
from planner import create_plan, explain
from result_formatter import clean_result
from session_manager import new_session, get, reset

app=Flask(__name__, static_folder='../frontend/dist', static_url_path='')
CORS(app)
app.config['MAX_CONTENT_LENGTH']=50*1024*1024

def run_plan(df,plan,domain=None):
    if plan.operation in ('get_summary','get_current_state','compare_groups','analyze_trend','calculate_correlation'): return analytics_execute(df,plan)
    if plan.operation=='detect_anomalies': return anomaly_execute(df,plan)
    if plan.operation=='forecast': return forecast_execute(df,plan.target_column,plan.time_column,plan.forecast_horizon or 7)
    if plan.operation=='predict_value':
        return regression(df, plan.target_column, plan.feature_columns)

    if plan.operation=='predict_class':
        return classification(df, plan.target_column, plan.feature_columns)

    if plan.operation=='run_scenario':
        if not plan.target_column: raise ValueError('Scenario analysis requires a numeric target column.')
        if not plan.scenario_changes: raise ValueError('Scenario analysis requires at least one numeric scenario change.')
        base=regression(df,plan.target_column)
        model=base.get('model')
        if model is None: raise ValueError('Scenario model could not be prepared.')
        target=plan.target_column
        feature_columns=(base.get('evaluation') or {}).get('feature_columns')
        if feature_columns:
            X=df[[c for c in feature_columns if c in df.columns]].copy()
        else:
            X=df.drop(columns=[target],errors='ignore').copy()
        for c in list(X.columns):
            if pd.api.types.is_datetime64_any_dtype(X[c]): X=X.drop(columns=[c])
        baseline=float(np.nanmean(model.predict(X)))
        scenario=X.copy()
        applied=[]
        for col,pct in plan.scenario_changes.items():
            if col not in scenario.columns: raise ValueError(f"Scenario column '{col}' does not exist in the dataset.")
            if not pd.api.types.is_numeric_dtype(scenario[col]): raise ValueError(f"Scenario column '{col}' must be numeric.")
            scenario[col]=scenario[col]*(1+float(pct)/100.0); applied.append(f"{col}: {pct:+g}%")
        scenario_value=float(np.nanmean(model.predict(scenario))); change=scenario_value-baseline; pct=(change/baseline*100) if baseline else None
        return {'success':True,'operation':'run_scenario','summary':f"Scenario estimate for '{target}' changes from {baseline:,.2f} to {scenario_value:,.2f} after applying {', '.join(applied)}.",'metrics':{'baseline':baseline,'scenario':scenario_value,'absolute_change':change,'percentage_change':pct},'table':[{'state':'Baseline','value':baseline},{'state':'Scenario','value':scenario_value}], 'chart':{'type':'bar','x':'state','y':'value','data':[{'state':'Baseline','value':baseline},{'state':'Scenario','value':scenario_value}]},'evaluation':base.get('evaluation',{}),'limitations':['Scenario results are model-based estimates, not observed outcomes.']}

    raise ValueError('The requested question cannot be supported from the current dataset.')

@app.get('/api/health')
def health():
    from config import MOCK_MODE, GEMINI_API_KEY
    gemini_ok = bool(GEMINI_API_KEY) and not MOCK_MODE
    return jsonify({
        'ok':True,
        'gemini_enabled': gemini_ok,
        'fallback_engine': True,
        'model':GEMINI_MODEL,
        'mock_mode':MOCK_MODE,
        'status': 'ready'
    })

@app.get('/api/domains')
def domains():
    out=[]
    for k,v in DOMAINS.items():
        meta={**v,'id':k}
        out.append(meta)
    return jsonify(out)

@app.post('/api/session')
def session():
    data=request.get_json(silent=True) or {}; sid=new_session(data.get('domain')); return jsonify({'session_id':sid})

@app.post('/api/new-chat')
def new_chat():
    data=request.get_json(silent=True) or {}; sid=data.get('session_id'); domain=data.get('domain')
    if not sid: sid=new_session(domain)
    else: reset(sid,domain)
    return jsonify({'session_id':sid,'reset':True})

@app.post('/api/upload')
def upload():
    sid=request.form.get('session_id'); domain=request.form.get('domain')
    if not sid: sid=new_session(domain)
    s=get(sid)
    if domain not in DOMAINS: return jsonify({'code':'INVALID_DOMAIN','error':'Select a valid domain.','retryable':False}),400
    file=request.files.get('file')
    if not file: return jsonify({'code':'DATASET_PARSE_ERROR','error':'No dataset file supplied.','retryable':False}),400
    ext=Path(file.filename).suffix.lower()
    if ext not in ('.csv','.xlsx','.xls'): return jsonify({'code':'DATASET_PARSE_ERROR','error':'Only CSV, XLSX and XLS files are supported.','retryable':False}),400
    path=UPLOAD_DIR/f'{uuid.uuid4().hex}{ext}'; file.save(path)
    try:
        df=load_dataset(path)
    except Exception as e:
        return jsonify({'code':'DATASET_PARSE_ERROR','error':f'Could not parse dataset: {e}','retryable':False}),400
    finally:
        try: path.unlink(missing_ok=True)
        except Exception: pass
    if df.empty: return jsonify({'code':'DATASET_PARSE_ERROR','error':'The uploaded dataset is empty.','retryable':False}),400
    profile=profile_dataframe(df,file.filename)
    s.update({'domain':domain,'df':df,'profile':profile,'messages':[],'last_plan':None,'last_result':None})
    return jsonify({'session_id':sid,'profile':profile,'preview':df.head(10).replace({pd.NA:None}).to_dict('records')})

@app.post('/api/chat')
def chat():
    data=request.get_json(force=True); sid=data.get('session_id'); q=(data.get('message') or '').strip()
    s=get(sid) if sid else None
    if not s or s.get('df') is None: return jsonify({'code':'DATASET_PARSE_ERROR','error':'Upload a dataset first.','retryable':False}),400
    if not q: return jsonify({'code':'INVALID_PLAN','error':'Please enter a question.','retryable':False}),400
    try:
        plan=create_plan(q,s['profile'],s['domain'],s['messages'])
        vr=validate_plan(plan,s['df'])
        if not vr['valid']:
            result={'success':False,'operation':plan.operation,'summary':'The requested analysis cannot be performed reliably with the current dataset.','metrics':{},'table':[],'chart':None,'limitations':vr['errors']+vr['warnings'],'error_code':'INVALID_PLAN','retryable':False}
            answer='I cannot run that analysis reliably with the available data. ' + ' '.join(vr['errors'])
        else:
            result=run_plan(s['df'],plan,s['domain']); result['validation_warnings']=vr['warnings']; result=clean_result(result); answer=explain(q,plan,result,s['domain'])
        s['messages'].append({'role':'user','content':q}); s['messages'].append({'role':'assistant','content':answer,'plan':plan.model_dump(),'result':result}); s['last_plan']=plan.model_dump(); s['last_result']=result
        return jsonify({'answer':answer,'plan':plan.model_dump(),'result':result})
    except Exception as e:
        traceback.print_exc()
        msg=str(e)
        if any(token in msg.lower() for token in ['column','target','dataset','not found','does not exist']):
            friendly=msg if len(msg)<200 else 'The requested analysis cannot be performed with the current dataset.'
        elif 'scenario' in msg.lower():
            friendly=msg
        else:
            friendly=f'Something went wrong while processing your question: {msg[:150]}'
        code='MODEL_TRAINING_ERROR' if any(k in msg.lower() for k in ['fit','train','model','xgboost','classification','regression']) else ('DATASET_PARSE_ERROR' if 'dataset' in msg.lower() else 'BACKEND_ERROR')
        return jsonify({'code':code,'error':friendly,'retryable':False}),422

@app.route('/',defaults={'path':''})
@app.route('/<path:path>')
def frontend(path):
    dist=Path(app.static_folder)
    if path and (dist/path).exists(): return send_from_directory(dist,path)
    if (dist/'index.html').exists(): return send_from_directory(dist,'index.html')
    return jsonify({'message':'UDI API is running. Build frontend with npm run build.'})

if __name__=='__main__': app.run(host='0.0.0.0',port=PORT,debug=False)
