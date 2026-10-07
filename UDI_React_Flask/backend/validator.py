import pandas as pd
from schemas import AnalysisPlan

def validate_plan(plan: AnalysisPlan, df: pd.DataFrame):
    errors, warnings = [], []
    cols = set(df.columns)
    numeric_cols = {c for c in cols if pd.api.types.is_numeric_dtype(df[c])}
    datetime_cols = {c for c in cols if pd.api.types.is_datetime64_any_dtype(df[c])}

    referenced = [plan.target_column, plan.time_column, plan.entity_column]
    referenced += (plan.group_by or []) + (plan.feature_columns or [])
    for c in referenced:
        if c and c not in cols:
            errors.append(f"Column '{c}' does not exist in the uploaded dataset.")

    if plan.entity_values and plan.entity_column in cols:
        vals = set(df[plan.entity_column].dropna().astype(str).unique())
        missing = [v for v in plan.entity_values if str(v) not in vals]
        if missing:
            warnings.append(f"Requested values not found: {missing}")

    if plan.operation in ("predict_value", "run_scenario", "forecast"):
        if not plan.target_column:
            errors.append("A numeric target column must be identified for this operation.")
        elif plan.target_column not in numeric_cols:
            errors.append(f"Target '{plan.target_column}' must be numeric; no unrelated column will be substituted.")

    if plan.operation == "predict_class":
        if not plan.target_column:
            errors.append("A classification target column must be identified.")
        elif plan.target_column not in cols:
            errors.append(f"Classification target '{plan.target_column}' does not exist.")
        else:
            y = df[plan.target_column].dropna()
            if y.nunique() < 2:
                errors.append(f"Classification target '{plan.target_column}' has fewer than two classes.")
            elif y.value_counts().min() < 3:
                errors.append(f"Classification target '{plan.target_column}' has too few observations in its minority class.")

    if plan.operation == "calculate_correlation" and len(numeric_cols) < 2:
        errors.append("Correlation requires at least two numeric columns.")

    if plan.operation in ("analyze_trend", "get_current_state", "forecast"):
        if plan.time_column and plan.time_column not in cols:
            errors.append(f"Time column '{plan.time_column}' does not exist.")
        elif plan.operation == "forecast" and not plan.time_column:
            errors.append("A time column is required for forecasting.")
        if plan.time_column in cols:
            d = pd.to_datetime(df[plan.time_column], errors="coerce").dropna()
            if plan.operation == "forecast":
                if len(d) < 30:
                    errors.append("Forecasting requires at least 30 valid historical timestamps.")
                elif d.nunique() < 12:
                    errors.append("Forecasting requires at least 12 distinct time points.")

    if plan.operation == "compare_groups" and not (plan.group_by or plan.entity_column):
        errors.append("A group column is required for group comparison.")

    if plan.operation == "run_scenario":
        if not plan.scenario_changes:
            errors.append("Scenario analysis requires at least one requested numeric change.")
        else:
            for c in plan.scenario_changes:
                if c not in numeric_cols:
                    errors.append(f"Scenario column '{c}' must be a numeric column in the uploaded dataset.")

    if plan.feature_columns:
        for c in plan.feature_columns:
            if c not in cols:
                errors.append(f"Feature column '{c}' does not exist in the uploaded dataset.")
            elif c == plan.target_column:
                errors.append(f"Feature column '{c}' cannot also be the target.")

    if len(df) < 10 and plan.operation not in ("get_summary", "get_current_state", "compare_groups"):
        errors.append("The dataset is too small for this analysis.")

    if plan.operation == "unsupported":
        errors.append(plan.reasoning or "The requested analysis is not supported by the current dataset.")

    return {"valid": not errors, "errors": errors, "warnings": warnings}
