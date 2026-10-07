from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field

OPS = ["get_summary","get_current_state","compare_groups","analyze_trend","detect_anomalies","calculate_correlation","predict_class","predict_value","forecast","run_scenario","unsupported"]

class AnalysisPlan(BaseModel):
    """Validated, tool-level plan. Gemini chooses parameters; Python validates and executes."""
    intent: str = "general_analysis"
    operation: Literal["get_summary","get_current_state","compare_groups","analyze_trend","detect_anomalies","calculate_correlation","predict_class","predict_value","forecast","run_scenario","unsupported"]
    target_column: Optional[str] = None
    group_by: Optional[List[str]] = None
    aggregation: Optional[Literal["sum","mean","median","min","max","count","std","percentage"]] = None
    filters: Optional[Dict[str, Any]] = None
    time_column: Optional[str] = None
    time_frequency: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    forecast_horizon: Optional[int] = Field(default=None, ge=1, le=365)
    entity_column: Optional[str] = None
    entity_values: Optional[List[str]] = None
    model_type: Optional[str] = None
    feature_columns: Optional[List[str]] = None
    scenario_changes: Optional[Dict[str, float]] = None
    reasoning: Optional[str] = None
