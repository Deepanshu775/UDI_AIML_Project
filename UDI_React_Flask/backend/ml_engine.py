import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
try:
    from xgboost import XGBRegressor, XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

def _prep(X):
    nums=X.select_dtypes(include=np.number).columns.tolist(); cats=[c for c in X.columns if c not in nums]
    return ColumnTransformer([('num',Pipeline([('imputer',SimpleImputer(strategy='median')),('scale',StandardScaler())]),nums),('cat',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))]),cats)],remainder='drop')

def _features(df,target,features=None):
    X=df.drop(columns=[target]).copy()
    if features: X=X[[c for c in features if c in X.columns]]
    for c in list(X.columns):
        if pd.api.types.is_datetime64_any_dtype(X[c]): X=X.drop(columns=[c])
    # Identifier columns are normally not useful predictors and can explode one-hot cardinality.
    drop=[]
    for c in X.columns:
        high_cardinality = X[c].nunique(dropna=True) >= max(50, int(len(X)*.9))
        name_hint = str(c).lower().replace('-', '_')
        id_like = any(tok in name_hint.split('_') for tok in ['id','uuid','key']) or name_hint.endswith('_id')
        if high_cardinality and (id_like or not pd.api.types.is_numeric_dtype(X[c])):
            drop.append(c)
    return X.drop(columns=drop,errors='ignore')

def regression(df,target,features=None):
    if target not in df.columns: raise ValueError(f"Target '{target}' not found.")
    if not pd.api.types.is_numeric_dtype(df[target]): raise ValueError(f"Regression target '{target}' must be numeric. No unrelated column will be substituted.")
    d=df.dropna(subset=[target]).copy(); X=_features(d,target,features); y=pd.to_numeric(d[target],errors='coerce'); valid=y.notna(); X=X.loc[valid]; y=y.loc[valid]
    if len(d)<30: raise ValueError('At least 30 valid rows are required for regression.')
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
    candidates={'Linear Regression':LinearRegression(),'Decision Tree Regressor':DecisionTreeRegressor(max_depth=8,min_samples_leaf=3,random_state=42),'Random Forest Regressor':RandomForestRegressor(n_estimators=300,max_depth=12,min_samples_leaf=2,random_state=42,n_jobs=2),'Gradient Boosting Regressor':GradientBoostingRegressor(n_estimators=250,max_depth=3,learning_rate=.04,random_state=42),'XGBoost Regressor':XGBRegressor(n_estimators=300,max_depth=5,learning_rate=.04,subsample=.85,colsample_bytree=.85,objective='reg:squarederror',random_state=42,n_jobs=2)} if XGB_AVAILABLE else {}
    rows=[]; best=None; best_score=-np.inf
    for name,est in candidates.items():
        pipe=Pipeline([('prep',_prep(X)),('model',est)]); pipe.fit(Xtr,ytr); pred=pipe.predict(Xte); r2=float(r2_score(yte,pred)); mae=float(mean_absolute_error(yte,pred)); rmse=float(mean_squared_error(yte,pred)**.5); rows.append({'model':name,'R2':r2,'MAE':mae,'RMSE':rmse})
        if r2>best_score: best_score=r2; best=(name,pipe,pred)
    name,pipe,pred=best
    return {'success':True,'operation':'predict_value','summary':f"{name} was selected after evaluating {len(candidates)} regression models for target '{target}'.",'metrics':{'MAE':float(mean_absolute_error(yte,pred)),'RMSE':float(mean_squared_error(yte,pred)**.5),'R2':float(r2_score(yte,pred)),'target_used':target},'table':[{'actual':float(a),'predicted':float(b)} for a,b in zip(yte.head(20),pred[:20])],'chart':{'type':'scatter','x':'actual','y':'predicted','data':[{'actual':float(a),'predicted':float(b)} for a,b in zip(yte,pred)]},'model':pipe,'evaluation':{'selected_model':name,'train_rows':len(Xtr),'test_rows':len(Xte),'candidate_models':rows,'feature_columns':X.columns.tolist()},'limitations':[]}

def classification(df,target,features=None):
    if target not in df.columns: raise ValueError(f"Target '{target}' not found.")
    d=df.dropna(subset=[target]).copy(); X=_features(d,target,features); labels=LabelEncoder(); y=labels.fit_transform(d[target].astype(str)); nclasses=len(labels.classes_)
    if nclasses<2: raise ValueError(f"Classification target '{target}' has fewer than two classes.")
    if len(d)<30: raise ValueError('At least 30 valid rows are required for classification.')
    if d[target].value_counts().min()<3: raise ValueError('Each classification class needs at least 3 observations.')
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    candidates={'Logistic Regression':LogisticRegression(max_iter=1000),'Decision Tree Classifier':DecisionTreeClassifier(max_depth=8,min_samples_leaf=3,random_state=42),'Random Forest Classifier':RandomForestClassifier(n_estimators=300,max_depth=12,min_samples_leaf=2,random_state=42,n_jobs=2),'Gradient Boosting Classifier':GradientBoostingClassifier(n_estimators=250,max_depth=3,learning_rate=.04,random_state=42),'XGBoost Classifier':XGBClassifier(n_estimators=300,max_depth=4,learning_rate=.05,subsample=.85,colsample_bytree=.85,eval_metric='logloss',random_state=42,n_jobs=2)} if XGB_AVAILABLE else {}
    rows=[]; best=None; best_score=-np.inf
    for name,est in candidates.items():
        pipe=Pipeline([('prep',_prep(X)),('model',est)]); pipe.fit(Xtr,ytr); pred=pipe.predict(Xte); proba=pipe.predict_proba(Xte); f1=float(f1_score(yte,pred,average='weighted',zero_division=0)); auc=float(roc_auc_score(yte,proba[:,1])) if nclasses==2 else None; rows.append({'model':name,'Accuracy':float(accuracy_score(yte,pred)),'Precision':float(precision_score(yte,pred,average='weighted',zero_division=0)),'Recall':float(recall_score(yte,pred,average='weighted',zero_division=0)),'F1':f1,'ROC_AUC':auc});
        if f1>best_score: best_score=f1; best=(name,pipe,pred)
    name,pipe,pred=best
    return {'success':True,'operation':'predict_class','summary':f"{name} was selected after evaluating {len(candidates)} classification models for target '{target}'.",'metrics':next(r for r in rows if r['model']==name),'table':[{'actual':labels.inverse_transform([a])[0],'predicted':labels.inverse_transform([b])[0]} for a,b in zip(yte[:25],pred[:25])],'chart':None,'model':pipe,'evaluation':{'selected_model':name,'classes':labels.classes_.tolist(),'train_rows':len(Xtr),'test_rows':len(Xte),'candidate_models':rows,'feature_columns':X.columns.tolist()},'limitations':[]}
