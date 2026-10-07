import sys
from pathlib import Path
import pandas as pd
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DOMAINS
from data_profiler import profile_dataframe
from schemas import AnalysisPlan
from validator import validate_plan
from analytics_engine import execute
from anomaly_engine import execute as anomaly
from ml_engine import regression, classification
from forecasting_engine import execute as forecast_execute

SAMPLES_EXIST = all((Path(__file__).resolve().parents[1] / 'sample_data' / f'{d}.csv').exists() for d in DOMAINS)
ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = ROOT / 'sample_data'

def _make_company(n=100, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2025-01-01', periods=n, freq='D')
    categories = ['Electronics','Apparel','Home','Food']
    df = pd.DataFrame({
        'Date': dates,
        'Product_Category': rng.choice(categories, size=n),
        'Region': rng.choice(['North','South','East','West'], size=n),
        'Sales': rng.integers(500, 10000, size=n).astype(float),
        'Profit': rng.integers(50, 3000, size=n).astype(float),
        'Demand': rng.integers(20, 500, size=n).astype(int),
        'Inventory': rng.integers(100, 2000, size=n).astype(int),
        'Customers': rng.integers(50, 1500, size=n).astype(int),
        'Promo_Flag': rng.integers(0, 2, size=n).astype(int),
        'High_Return_Risk': rng.integers(0, 2, size=n).astype(int),
    })
    return df

def _load_or_build(domain, n=100):
    path = SAMPLE_DIR / f'{domain}.csv'
    if path.exists():
        from data_profiler import load_dataset
        return load_dataset(path)
    if domain == 'company':
        return _make_company(n)
    rng = np.random.default_rng(abs(hash(domain)) % (2**32))
    numeric_target = DOMAINS[domain]['target']
    forecast_target = DOMAINS[domain]['forecast_target']
    dates = pd.date_range('2025-01-01', periods=n, freq='D')
    groups = ['A','B','C','D']
    out = pd.DataFrame({
        'Date': dates,
        'Group': rng.choice(groups, size=n),
        forecast_target: rng.integers(50, 2000, size=n).astype(float),
        numeric_target: rng.integers(100, 5000, size=n).astype(float),
        'Measure_X': rng.uniform(0.5, 5.0, size=n),
        'Measure_Y': rng.integers(20, 200, size=n).astype(int),
        'Binary_Y': rng.integers(0, 2, size=n).astype(int),
        'Capacity': rng.integers(100, 1000, size=n).astype(int),
        'Utilization': rng.uniform(0.3, 0.95, size=n),
        'Missing_Flag': rng.integers(0, 2, size=n).astype(int),
    })
    return out

def test_all_domains_exist_in_config():
    assert len(DOMAINS) == 7
    for domain in DOMAINS:
        assert 'label' in DOMAINS[domain]
        assert 'target' in DOMAINS[domain]
        assert 'forecast_target' in DOMAINS[domain]

def test_profile_and_summary():
    df = _load_or_build('company')
    p = profile_dataframe(df, 'company.csv')
    if SAMPLES_EXIST:
        assert p['rows'] == 900
    numeric_col = 'Sales' if 'Sales' in df.columns else DOMAINS['company']['target']
    if numeric_col not in df.columns:
        numeric_col = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])][0]
    plan = AnalysisPlan(operation='get_summary', target_column=numeric_col)
    r = execute(df, plan)
    assert r['success'] and r['metrics']['sum'] > 0

def test_validation_rejects_unknown_column():
    df = _load_or_build('hospital')
    plan = AnalysisPlan(operation='get_summary', target_column='DoesNotExist')
    assert not validate_plan(plan, df)['valid']

def test_anomaly_engine():
    df = _load_or_build('company')
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    target = 'Sales' if 'Sales' in df.columns else numeric_cols[0]
    plan = AnalysisPlan(operation='detect_anomalies', target_column=target)
    r = anomaly(df, plan)
    assert r['success']

def test_all_domains_run_via_generic_engines():
    ops = ['get_summary', 'compare_groups', 'detect_anomalies', 'forecast']
    for domain in DOMAINS:
        df = _load_or_build(domain, n=150)
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        datetime_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        category_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= min(20, max(3, df.shape[0] // 20)) and not pd.api.types.is_numeric_dtype(df[c])]
        target = numeric_cols[0]
        if datetime_cols and numeric_cols:
            plan = AnalysisPlan(operation='get_summary', target_column=target)
            assert validate_plan(plan, df)['valid']
            r = execute(df, plan)
            assert r['success'], f"get_summary failed for {domain}: {r}"
        if category_cols:
            group = category_cols[0]
            plan = AnalysisPlan(operation='compare_groups', target_column=target, group_by=[group])
            if validate_plan(plan, df)['valid']:
                r = execute(df, plan)
                assert r['success'], f"compare_groups failed for {domain}: {r}"
        if numeric_cols and df.shape[0] >= 30:
            plan = AnalysisPlan(operation='detect_anomalies', target_column=target)
            r = anomaly(df, plan)
            assert r['success'], f"anomaly failed for {domain}: {r}"
        if datetime_cols and numeric_cols:
            plan = AnalysisPlan(operation='forecast', target_column=numeric_cols[0], time_column=datetime_cols[0], forecast_horizon=7)
            if validate_plan(plan, df)['valid']:
                r = forecast_execute(df, plan.target_column, plan.time_column, plan.forecast_horizon)
                assert r['success'], f"forecast failed for {domain}: {r}"
