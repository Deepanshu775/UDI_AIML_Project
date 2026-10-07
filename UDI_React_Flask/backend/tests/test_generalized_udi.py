import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from schemas import AnalysisPlan
from validator import validate_plan
from analytics_engine import execute
from anomaly_engine import execute as anomaly_execute
from ml_engine import regression, classification
from forecasting_engine import execute as forecast_execute
from gemini_client import _rule_based_plan

def make_df(n=90):
    rng=np.random.default_rng(7)
    dates=pd.date_range("2025-01-01", periods=n, freq="D")
    region=rng.choice(["North","South","Goa"], n)
    product=rng.choice(["A","B","C"], n)
    sales=rng.normal(1000,150,n)
    cost=rng.normal(600,90,n)
    profit=sales-cost+rng.normal(0,35,n)
    demand=rng.normal(250,35,n)
    churn=rng.choice(["Yes","No"], n, p=[.25,.75])
    return pd.DataFrame({"Date":dates,"Region":region,"Product":product,"Sales":sales,"Cost":cost,"Profit":profit,"Demand":demand,"Churn":churn})

def test_same_dataset_routes_to_different_operations():
    df=make_df()
    p=__import__("data_profiler",fromlist=["profile_dataframe"]).profile_dataframe(df,"x.csv")
    questions=[
        "What is the average profit?",
        "Which product has the highest profit?",
        "How has profit changed over time?",
        "What variables are related to profit?",
        "Are there unusual profit values?",
        "Predict profit.",
        "Will this customer churn?",
        "What will demand be next month?",
        "What happens if sales increase by 10%?",
    ]
    plans=[_rule_based_plan(q,p,"company") for q in questions]
    ops=[x.operation for x in plans]
    assert ops[0]=="get_summary"
    assert ops[1]=="compare_groups"
    assert ops[2]=="analyze_trend"
    assert ops[3]=="calculate_correlation"
    assert ops[4]=="detect_anomalies"
    assert ops[5]=="predict_value"
    assert ops[6]=="predict_class"
    assert ops[7]=="forecast"
    assert ops[8]=="run_scenario"
    assert len(set(ops)) >= 8

def test_summary_respects_requested_aggregation():
    df=make_df()
    plan=AnalysisPlan(operation="get_summary",target_column="Profit",aggregation="mean")
    r=execute(df,plan)
    assert abs(r["metrics"]["value"]-df["Profit"].mean()) < 1e-8
    plan=AnalysisPlan(operation="get_summary",target_column="Profit",aggregation="sum")
    r=execute(df,plan)
    assert abs(r["metrics"]["value"]-df["Profit"].sum()) < 1e-8

def test_compare_respects_group_and_aggregation():
    df=make_df()
    plan=AnalysisPlan(operation="compare_groups",target_column="Sales",group_by=["Region"],aggregation="mean")
    r=execute(df,plan)
    assert r["success"] and r["table"]
    assert r["table"][0]["Region"] in set(df["Region"])

def test_ml_is_trained_on_uploaded_schema():
    df=make_df()
    r=regression(df,"Profit",["Sales","Cost","Demand"])
    assert r["success"] and r["evaluation"]["feature_columns"] == ["Sales","Cost","Demand"]
    c=classification(df,"Churn",["Sales","Cost","Demand"])
    assert c["success"] and "selected_model" in c["evaluation"]

def test_validation_rejects_invented_target():
    df=make_df()
    plan=AnalysisPlan(operation="predict_value",target_column="Revenue")
    v=validate_plan(plan,df)
    assert not v["valid"] and any("Revenue" in e for e in v["errors"])

def test_forecast_requires_real_time_series():
    df=make_df()
    plan=AnalysisPlan(operation="forecast",target_column="Demand",time_column="Date",forecast_horizon=7)
    assert validate_plan(plan,df)["valid"]
    r=forecast_execute(df,"Demand","Date",7)
    assert r["success"] and len(r["table"])==7

def test_anomaly_is_statistical_not_fake():
    df=make_df()
    df.loc[0,"Profit"]=10000
    plan=AnalysisPlan(operation="detect_anomalies",target_column="Profit")
    r=anomaly_execute(df,plan)
    assert r["success"] and r["metrics"]["target_used"]=="Profit"
