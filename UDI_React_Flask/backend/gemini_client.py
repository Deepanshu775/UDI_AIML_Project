import json, re
from config import GEMINI_API_KEY, GEMINI_MODEL, MOCK_MODE
from schemas import AnalysisPlan

try:
    from google import genai
    from google.genai import types
    _GEMINI_AVAILABLE = True
except Exception:
    _GEMINI_AVAILABLE = False


def _clean_schema_for_gemini(schema):
    if isinstance(schema, dict):
        out={}
        for k,v in schema.items():
            if k in {'additionalProperties','title','$schema'}:
                continue
            out[k]=_clean_schema_for_gemini(v)
        if 'anyOf' in out:
            non_null=[x for x in out['anyOf'] if not (isinstance(x,dict) and x.get('type')=='null')]
            if len(non_null)==1:
                merged={k:v for k,v in out.items() if k!='anyOf'}
                merged.update(non_null[0])
                out=merged
        return out
    if isinstance(schema,list): return [_clean_schema_for_gemini(x) for x in schema]
    return schema

_ANALYSIS_PLAN_SCHEMA=_clean_schema_for_gemini(AnalysisPlan.model_json_schema())


def _numeric_cols(profile):
    return [c['name'] for c in profile.get('columns',[]) if c.get('dtype') in ('numeric','integer','float')]

def _categorical_cols(profile):
    return [c['name'] for c in profile.get('columns',[]) if c.get('dtype') in ('categorical','string')]

def _datetime_cols(profile):
    return [c['name'] for c in profile.get('columns',[]) if c.get('dtype')=='datetime']

def _fuzzy_find_col(keyword, all_cols):
    kw = keyword.lower().replace(' ', '_').replace('-','_')
    for c in all_cols:
        cname = c.lower().replace(' ','_').replace('-','_')
        if kw in cname:
            return c
    exact_kw = [k.lower() for k in all_cols]
    if kw in exact_kw:
        return all_cols[exact_kw.index(kw)]
    return None

def _pick_target(domain, profile, preferred=None):
    all_cols=[c['name'] for c in profile.get('columns',[])]
    if preferred:
        hit = _fuzzy_find_col(preferred, all_cols)
        if hit: return hit
    domain_hints={
        'company':['profit','revenue','sales','demand','cost','customers','inventory'],
        'hospital':['patients','admissions','waiting_time','capacity','staff','beds'],
        'college':['attendance','gpa','enrollment','faculty','graduation','fees'],
        'municipality':['waste_collected','complaints','service_demand','zones','population'],
        'ngo':['beneficiaries','donations','aid_distributed','volunteers','funds'],
        'community':['resource_requests','participants','shortage_count','requests','needs'],
        'event_organizer':['attendance','registrations','tickets_sold','capacity','revenue'],
    }
    for hint in domain_hints.get(domain,[]):
        hit = _fuzzy_find_col(hint, all_cols)
        if hit: return hit
    nums=_numeric_cols(profile)
    if nums: return nums[0]
    return all_cols[0] if all_cols else None

def _extract_target_from_question(question, profile):
    q = question.lower()
    all_cols=[c['name'] for c in profile.get('columns',[])]
    words = re.split(r'[\s_\-?.,!]+', q)
    col_scores = {}
    for c in all_cols:
        c_parts = set(re.split(r'[\s_\-]+', c.lower()))
        overlap = sum(1 for w in words if w and len(w) > 2 and any(w in p for p in c_parts))
        if overlap: col_scores[c] = overlap
        cnorm = c.lower().replace('_',' ').replace('-',' ')
        qnorm = q.replace('_',' ').replace('-',' ')
        if cnorm and (f' {cnorm} ' in f' {qnorm} ' or qnorm.startswith(cnorm+' ') or qnorm.endswith(' '+cnorm) or qnorm==cnorm):
            col_scores[c] = col_scores.get(c, 0) + 50
        if len(c) >= 5 and re.search(r'\b'+re.escape(c.lower())+r'\b', q):
            col_scores[c] = col_scores.get(c, 0) + 100
    if col_scores:
        return max(col_scores.items(), key=lambda x: x[1])[0]
    return None


def _rule_based_plan(question, profile, domain):
    q=question.lower()
    nums=_numeric_cols(profile)
    cats=_categorical_cols(profile)
    dates=_datetime_cols(profile)
    cols=[c['name'] for c in profile.get('columns',[])]

    def has(*keywords):
        return any(k in q for k in keywords)

    def has_word(*keywords):
        return any(re.search(r'\b'+re.escape(k)+r'\b', q) for k in keywords)

    correlation_words = ('correlation','correlate','relation between','relationship between','affect on','depends on','link between','connected to','association between','drives the','driver of','correlated with','related to','association with','related with','relationship with','related?','related,','correlates')
    forecast_kws = ('forecast','future','horizon','weeks','months','quarter','project','projected','expected','q1','q2','q3','q4','projection','predict future','what will','for next','year ahead','annual forecast','12 month','12-month','forecasted','next week','next month','next quarter','next days','upcoming')
    descriptive_kws = ('average of','mean of','total of','sum of','median of','minimum of','maximum of','what is the min of','what is the max of','value of the','amount of the','count of the','percentage of','percent of','ratio of','stats for','statistic for','mean value','average value','total value','sum value')
    descriptive_prefixes = ('what is average ','what is the average','what is mean','what is the mean','what is total ','what is the total','what is sum','what is the sum','what is median','what is the median','what is minimum','what is the minimum','what is maximum','what is the maximum','what is the value of','what is the amount of','how much is the total','how much is the average','how many is the total')
    is_descriptive = False
    has_correlation = any(cw in q for cw in correlation_words)
    has_scenario_markers = has('scenario','what if','if we ','suppose that','assume that','what-if','simulate','simulation','hypothetical')
    has_predict_markers = has('predict value','prediction of','estimate the value','demand forecast','cost estimate','how much will be','what will be the value','predicted value','regress on','build a model for','machine learning model','ml model to predict','regression for','using a regression','using a model')
    has_forecast_markers = any(fk in q for fk in forecast_kws)

    if not has_correlation and not has_scenario_markers and not has_predict_markers and not has_forecast_markers:
        if any(p in q for p in descriptive_prefixes): is_descriptive = True
        if has(*descriptive_kws): is_descriptive = True

    explicit_target_full = _extract_target_from_question(question, profile)
    numeric_mentions = [c for c in nums if c.lower() in q]
    categorical_mentions = [c for c in cats if c.lower() in q]
    if is_descriptive:
        explicit_target = explicit_target_full if explicit_target_full in nums else (numeric_mentions[0] if numeric_mentions else explicit_target_full)
    elif has_scenario_markers:
        outcome_candidates = [c for c in numeric_mentions if any(h in c.lower() for h in ['profit','revenue','sales','demand','patients','attendance','beneficiaries','cost','spend','count','total','value'])]
        outcome_priority = [c for c in outcome_candidates if any(h in c.lower() for h in ['profit','revenue','sales','demand','patients','attendance','beneficiaries'])]
        if outcome_priority: explicit_target = outcome_priority[0]
        elif outcome_candidates: explicit_target = _pick_target(domain, profile, None)
        else: explicit_target = _pick_target(domain, profile, None)
    else:
        explicit_target = explicit_target_full

    if is_descriptive:
        target = explicit_target or (numeric_mentions[0] if numeric_mentions else None)
        group_hint = categorical_mentions[0] if categorical_mentions else None
        if group_hint and has('compare','versus',' vs','rank','best','worst','ranking','breakdown','grouped by','by category','top 10','top 5','grouped','highest','lowest','which is'):
            target_for_compare = target or (nums[0] if nums else None)
            if target_for_compare and target_for_compare in nums:
                return AnalysisPlan(
                    intent='compare_groups',
                    operation='compare_groups',
                    target_column=target_for_compare,
                    group_by=[group_hint],
                    reasoning='Rule-based: descriptive comparison with explicit grouping column.'
                )
        return AnalysisPlan(
            intent='get_summary',
            operation='get_summary',
            target_column=target,
            reasoning='Rule-based: descriptive / statistic keywords (avg/sum/total/etc) detected.'
        )

    if has('scenario','what if','if we ','suppose that','assume that','what-if','simulate','simulation','hypothetical','sensitivity analysis','what happens if'):
        target = explicit_target or _pick_target(domain, profile)
        if target and target in nums:
            scenario_changes = {}
            mentioned_numeric = None
            qnorm = q.replace('_',' ').replace('-',' ')
            for c in nums:
                if c == target: continue
                cnorm = c.lower().replace('_',' ').replace('-',' ')
                if cnorm and (cnorm in qnorm or f' {cnorm} ' in f' {qnorm} '):
                    mentioned_numeric = c
                    break
            if mentioned_numeric:
                m = re.search(r'(\d+(?:\.\d+)?)\s*%', q)
                pct = float(m.group(1)) if m else 10.0
                scenario_changes[mentioned_numeric] = pct
            else:
                for c in nums:
                    if c != target:
                        scenario_changes[c] = 10.0
                        break
            return AnalysisPlan(
                intent='run_scenario',
                operation='run_scenario',
                target_column=target,
                scenario_changes=scenario_changes,
                reasoning='Rule-based: scenario keywords.'
            )

    if has('anomaly','outlier','abnormal','unusual','fraud detection','strange value','deviation','anomalies','outliers','irregular value','weird value','outside the norm'):
        target = explicit_target or (nums[0] if nums else None)
        return AnalysisPlan(
            intent='detect_anomalies',
            operation='detect_anomalies',
            target_column=target,
            reasoning='Rule-based: anomaly keywords.'
        )

    if has(*forecast_kws):
        time_col = dates[0] if dates else None
        target = explicit_target or _pick_target(domain, profile)
        if time_col and target and target in nums:
            horizon = 365 if ('year' in q and 'day' not in q) else 90 if 'quarter' in q else 30 if ('month' in q or 'annual' in q) else 7
            return AnalysisPlan(
                intent='forecast_values',
                operation='forecast',
                target_column=target,
                time_column=time_col,
                forecast_horizon=horizon,
                reasoning='Rule-based: forecast keywords detected.'
            )
        elif target and target in nums:
            return AnalysisPlan(
                intent='get_summary',
                operation='get_summary',
                target_column=target,
                reasoning='Rule-based: forecast requested but no time column; falling back to summary of target.'
            )

    if has('trend','over time','increase over','decrease over','growth of','tendency','upward trend','downward trend','rising over','falling over','dropping over','time series','pattern over time','trending','is it increasing','is it decreasing'):
        time_col = dates[0] if dates else None
        target = explicit_target or _pick_target(domain, profile)
        if time_col and target:
            return AnalysisPlan(
                intent='analyze_trend',
                operation='analyze_trend',
                target_column=target,
                time_column=time_col,
                reasoning='Rule-based: trend/time keywords.'
            )

    if has('correlation','correlate','relation between','relationship between','affect on','depends on','link between','connected to','association between','drives the','driver of','correlated with','related to','association with'):
        if len(nums)>=2:
            return AnalysisPlan(
                intent='calculate_correlation',
                operation='calculate_correlation',
                target_column=explicit_target if explicit_target in nums else nums[0],
                reasoning='Rule-based: correlation keywords.'
            )

    if has('classify','classify into','classes of','classification of','classification for','categorize','categorized into','which category','segment into','what type of','flag the','label the','class assignment','risk classification'):
        target = None
        for c in ['Risk_Flag','Capacity_Risk','Attendance_Risk','Service_Risk','Coverage_Risk','Shortage_Risk']:
            if c in cols: target=c; break
        if not target:
            target = explicit_target if explicit_target in cats else (cats[0] if cats else None)
        if target and target in cols:
            return AnalysisPlan(
                intent='predict_class',
                operation='predict_class',
                target_column=target,
                reasoning='Rule-based: classification keywords.'
            )

    compare_strict = ('compare ','comparison','versus',' vs ','ranking','rank the','top 10','top 5','best performing','worst performing','best group','worst group','between groups','breakdown by','grouped by','per category','which is highest','which is lowest','sorted by','by group')
    compare_by_x = ('by region','by product','by zone','by category','by department','by country','by state','by city','by type','by segment')
    compare_target_hint = None
    if explicit_target in cats:
        pass
    for cat in cats:
        if any(b in q for b in [f'by {cat.lower()}',f'per {cat.lower()}',f'for each {cat.lower()}',f'{cat.lower()} breakdown',f'group by {cat.lower()}']):
            compare_target_hint = cat
            break
    if has(*compare_strict) or has(*compare_by_x) or compare_target_hint:
        group = compare_target_hint or (cats[0] if cats else None)
        target = explicit_target if explicit_target in nums else (nums[0] if nums else None)
        if group and target:
            return AnalysisPlan(
                intent='compare_groups',
                operation='compare_groups',
                target_column=target,
                group_by=[group],
                reasoning='Rule-based: comparison/grouping keywords.'
            )

    if has('current state','now','today status','present state','latest data','recent snapshot','right now','as of now','current snapshot','live data','up to date status','current value'):
        return AnalysisPlan(
            intent='get_current_state',
            operation='get_current_state',
            reasoning='Rule-based: current-state keywords.'
        )

    summary_kws = ('summary','overview of','overall picture','summary statistics','describe the','snapshot of','full report','summary of','high level overview','quick stats','tell me about the data','data overview','everything about','all stats')
    if has(*summary_kws):
        target = explicit_target or None
        return AnalysisPlan(
            intent='get_summary',
            operation='get_summary',
            target_column=target,
            reasoning='Rule-based: summary / overview keywords.'
        )

    predict_kws_full = ('predict value','prediction of','estimate the value','demand forecast','cost estimate','how much will be','what will be the value','predicted value','regress on','build a model for','machine learning model','ml model to predict','regression for','using a regression','using a model','predict second','predict first','ml prediction','predict future value','model to predict')
    if has(*predict_kws_full) or has_predict_markers:
        target = explicit_target or _pick_target(domain, profile)
        if target and target in nums:
            return AnalysisPlan(
                intent='predict_value',
                operation='predict_value',
                target_column=target,
                reasoning='Rule-based: prediction / regression model keywords.'
            )

    # Generic local fallback: semantic operation detection without domain-specific targets.
    if not any(k in q for k in ['will this','yes or no','which class','classify','classification','churn','high-risk','low-risk','risk ']) and any(k in q for k in ['predict ', 'predicts ', 'prediction', 'estimate ', 'estimated ', 'will ', 'expected value']):
        target = explicit_target if explicit_target in nums else (nums[0] if nums else None)
        if target:
            return AnalysisPlan(intent='predict_value', operation='predict_value', target_column=target,
                                feature_columns=[c for c in nums if c != target] + [c for c in cats if c != target],
                                reasoning='Local fallback: predictive language detected; target matched to the uploaded schema.')
    if any(k in q for k in ['will this', 'yes or no', 'which class', 'classify', 'classification', 'churn', 'risk', 'high-risk', 'low-risk', 'category will']):
        target = explicit_target if explicit_target in cols else (cats[0] if cats else None)
        if target:
            return AnalysisPlan(intent='predict_class', operation='predict_class', target_column=target,
                                reasoning='Local fallback: classification language detected; target matched to the uploaded schema.')
    if any(k in q for k in ['current state','latest','most recent','recent value','current value']):
        return AnalysisPlan(intent='get_current_state', operation='get_current_state', time_column=dates[0] if dates else None,
                            reasoning='Local fallback: current-state language detected.')
    if any(k in q for k in ['highest','lowest','best','worst','compare','comparison','ranking','rank ','top ','bottom ','by ']) and cats and nums:
        target = explicit_target if explicit_target in nums else nums[0]
        group = categorical_mentions[0] if categorical_mentions else cats[0]
        return AnalysisPlan(intent='compare_groups', operation='compare_groups', target_column=target, group_by=[group], aggregation='mean',
                            reasoning='Local fallback: comparison/ranking language detected.')
    if explicit_target:
        return AnalysisPlan(
            intent='get_summary',
            operation='get_summary',
            target_column=explicit_target,
            aggregation='mean',
            reasoning=f'Local fallback: explicitly mentioned column {explicit_target}.'
        )

    if has('profit','revenue amount','sales amount','patients total','attendance total','beneficiaries total','how many total','number of total','quantity of total'):
        target = explicit_target or _pick_target(domain, profile)
        if target and target in nums:
            return AnalysisPlan(
                intent='get_summary',
                operation='get_summary',
                target_column=target,
                reasoning='Rule-based: business metric mentioned; showing summary stats.'
            )

    return AnalysisPlan(
        intent='get_summary',
        operation='get_summary',
        reasoning='Rule-based: default fallback to summary.'
    )


def _rule_based_explain(question, plan, result, domain):
    q = question.lower()
    op = plan.operation if hasattr(plan, 'operation') else (plan.get('operation') if isinstance(plan,dict) else None)
    summary = result.get('summary') or "The analysis completed successfully."
    parts = []

    if op == 'get_summary':
        metrics = result.get('metrics') or {}
        target = plan.target_column if hasattr(plan,'target_column') else (plan.get('target_column') if isinstance(plan,dict) else None)
        stat_types = []
        if any(k in q for k in ['average','mean']): stat_types.append('mean')
        if any(k in q for k in ['total','sum']): stat_types.append('sum')
        if any(k in q for k in ['median']): stat_types.append('median')
        if any(k in q for k in ['min','minimum','smallest']): stat_types.append('min')
        if any(k in q for k in ['max','maximum','largest','highest']): stat_types.append('max')
        if not stat_types: stat_types = ['mean','sum','median','min','max']

        if target:
            lead_parts = []
            for st in stat_types:
                if st in metrics and metrics[st] is not None:
                    v = metrics[st]
                    label = {'mean':'Average','sum':'Total','median':'Median','min':'Minimum','max':'Maximum'}.get(st, st.replace('_',' ').title())
                    if isinstance(v, float): lead_parts.append(f"{label} {target} = {v:,.2f}")
                    else: lead_parts.append(f"{label} {target} = {v:,}")
            if lead_parts:
                parts.append(f"For your question \"{question}\": " + "; ".join(lead_parts) + ".")
            else:
                parts.append(f"For your question \"{question}\": {summary}")
            if metrics.get('count'):
                parts.append(f"Based on {metrics['count']:,} records.")
        else:
            parts.append(f"For your question \"{question}\": {summary}")

        chart = result.get('chart')
        if chart:
            parts.append(f"A {chart.get('type','data')} chart has been prepared.")
        table = result.get('table') or []
        if table:
            parts.append(f"Detailed statistics for {min(len(table),10)} columns are shown in the table.")
        limitations = result.get('limitations') or []
        if limitations:
            parts.append("Note: " + " ".join(limitations[:2]))
        parts.append("Answer generated by the UDI local analytics engine.")
        return " ".join(parts)

    if op == 'forecast':
        metrics = result.get('metrics') or {}
        target = plan.target_column if hasattr(plan,'target_column') else (plan.get('target_column') if isinstance(plan,dict) else None)
        horizon = None
        if hasattr(plan, 'forecast_horizon'): horizon = plan.forecast_horizon
        elif isinstance(plan, dict): horizon = plan.get('forecast_horizon')
        parts.append(f"For your question \"{question}\": {summary}")
        if horizon and target:
            periods = f"{horizon}-day" if horizon <= 90 else f"{horizon//30}-month" if horizon >= 30 else f"{horizon}-period"
            parts.append(f"This forecast covers the next {periods} for {target}.")
        if metrics.get('backtest_RMSE'):
            parts.append(f"Forecast backtest RMSE = {metrics['backtest_RMSE']:,.2f} (lower is better).")
        chart = result.get('chart')
        if chart: parts.append(f"A {chart.get('type','forecast')} chart is ready showing the projection.")
        table = result.get('table') or []
        if table:
            rows_mentioned = []
            for r in table[:5]:
                row_items = [f"{k}: {v:,.2f}" if isinstance(v,float) else f"{k}: {v}" for k,v in list(r.items())[:3]]
                rows_mentioned.append("(" + ", ".join(row_items) + ")")
            if rows_mentioned: parts.append("Sample forecast points: " + ", ".join(rows_mentioned) + ".")
        limitations = result.get('limitations') or []
        if limitations: parts.append("Important: " + " ".join(limitations[:2]))
        parts.append("Generated by UDI rule-based forecasting engine.")
        return " ".join(parts)

    if op == 'predict_value':
        metrics = result.get('metrics') or {}
        target = plan.target_column if hasattr(plan,'target_column') else (plan.get('target_column') if isinstance(plan,dict) else None)
        eval_ = result.get('evaluation') or {}
        parts.append(f"For your question \"{question}\": {summary}")
        if eval_.get('model'): parts.append(f"Model selected: {eval_['model']}.")
        if metrics:
            bits = []
            for k in ['R2','r2_score','MAE','mae','RMSE','rmse']:
                if k in metrics and metrics[k] is not None:
                    v = metrics[k]; label = k.upper() if len(k)<=4 else k.title()
                    bits.append(f"{label} = {v:,.3f}" if isinstance(v,float) else f"{label} = {v}")
            if bits: parts.append("Performance: " + ", ".join(bits) + ".")
        if target: parts.append(f"Target column analyzed: {target}.")
        chart = result.get('chart')
        if chart: parts.append(f"A {chart.get('type','data')} chart is provided.")
        limitations = result.get('limitations') or []
        if limitations: parts.append("Limitations: " + " ".join(limitations[:2]))
        parts.append("Generated by UDI local ML engine.")
        return " ".join(parts)

    if op == 'compare_groups':
        metrics = result.get('metrics') or {}
        parts.append(f"For your question \"{question}\": {summary}")
        if metrics:
            top = metrics.get('top_group_sum') or metrics.get('top_group_count')
            top_val = metrics.get('top_group_value')
            if top and top_val is not None:
                if isinstance(top_val,float): parts.append(f"Leading group: {top} = {top_val:,.2f}.")
                else: parts.append(f"Leading group: {top} = {top_val:,}.")
        chart = result.get('chart')
        if chart: parts.append(f"A {chart.get('type','bar')} chart is ready with group comparisons.")
        table = result.get('table') or []
        if table: parts.append(f"Top {min(len(table),8)} groups are shown in the table.")
        parts.append("Generated by UDI local analytics engine.")
        return " ".join(parts)

    if op == 'analyze_trend':
        metrics = result.get('metrics') or {}
        parts.append(f"For your question \"{question}\": {summary}")
        if metrics.get('change_pct') is not None:
            parts.append(f"Overall change: {metrics['change_pct']:+.2f}% across the time window.")
        chart = result.get('chart')
        if chart: parts.append(f"A {chart.get('type','line')} chart visualizes the trend over time.")
        limitations = result.get('limitations') or []
        if limitations: parts.append("Note: " + " ".join(limitations[:2]))
        parts.append("Generated by UDI local time-series engine.")
        return " ".join(parts)

    if op == 'detect_anomalies':
        metrics = result.get('metrics') or {}
        parts.append(f"For your question \"{question}\": {summary}")
        if metrics.get('anomaly_count') is not None:
            parts.append(f"Total anomalies detected: {metrics['anomaly_count']:,}.")
        table = result.get('table') or []
        if table: parts.append(f"Top {min(len(table),10)} anomalous rows are listed in the table.")
        limitations = result.get('limitations') or []
        if limitations: parts.append("Note: " + " ".join(limitations[:2]))
        parts.append("Generated by UDI local anomaly engine.")
        return " ".join(parts)

    if op == 'run_scenario':
        metrics = result.get('metrics') or {}
        parts.append(f"For your question \"{question}\": {summary}")
        base = metrics.get('baseline'); scen = metrics.get('scenario')
        change = metrics.get('absolute_change'); pct = metrics.get('percentage_change')
        if base is not None and scen is not None:
            if isinstance(base,float): parts.append(f"Baseline = {base:,.2f}, Scenario = {scen:,.2f}.")
            else: parts.append(f"Baseline = {base:,}, Scenario = {scen:,}.")
        if change is not None and pct is not None:
            if isinstance(change,float): parts.append(f"Absolute change = {change:+,.2f} ({pct:+.2f}%).")
            else: parts.append(f"Absolute change = {change:+,} ({pct:+.2f}%).")
        chart = result.get('chart')
        if chart: parts.append(f"A {chart.get('type','bar')} chart compares baseline vs scenario.")
        limitations = result.get('limitations') or []
        if limitations: parts.append("Important: " + " ".join(limitations[:2]))
        parts.append("Generated by UDI local scenario engine.")
        return " ".join(parts)

    if op == 'calculate_correlation':
        parts.append(f"For your question \"{question}\": {summary}")
        table = result.get('table') or []
        if table:
            top = table[0]
            if 'correlation' in top:
                parts.append(f"Strongest association: {top.get('feature_1')} ↔ {top.get('feature_2')} = {top.get('correlation'):+.3f}.")
            parts.append(f"The top {min(len(table),10)} pairs are ranked in the table.")
        limitations = result.get('limitations') or []
        if limitations: parts.append("Remember: " + " ".join(limitations[:2]))
        parts.append("Generated by UDI local correlation engine.")
        return " ".join(parts)

    if op == 'predict_class':
        metrics = result.get('metrics') or {}
        target = plan.target_column if hasattr(plan,'target_column') else (plan.get('target_column') if isinstance(plan,dict) else None)
        eval_ = result.get('evaluation') or {}
        parts.append(f"For your question \"{question}\": {summary}")
        if eval_.get('model'): parts.append(f"Model: {eval_['model']}.")
        if target: parts.append(f"Classified column: {target}.")
        table = result.get('table') or []
        if table: parts.append(f"Recent predictions: {min(len(table),10)} rows shown.")
        limitations = result.get('limitations') or []
        if limitations: parts.append("Limitations: " + " ".join(limitations[:2]))
        parts.append("Generated by UDI local classification engine.")
        return " ".join(parts)

    if op == 'get_current_state':
        parts.append(f"For your question \"{question}\": {summary}")
        table = result.get('table') or []
        if table:
            row = table[0]
            items = [f"{k}: {v:,.2f}" if isinstance(v,float) else f"{k}: {v}" for k,v in list(row.items())[:8]]
            parts.append("Latest record: " + ", ".join(items) + ".")
        parts.append("Generated by UDI local analytics engine.")
        return " ".join(parts)

    parts.append(f"For your question \"{question}\": {summary}")
    metrics = result.get('metrics') or {}
    if metrics:
        bits = [f"{k.replace('_',' ').title()}: {v:,.2f}" if isinstance(v,float) else f"{k.replace('_',' ').title()}: {v}" for k,v in list(metrics.items())[:4]]
        if bits: parts.append("Key metrics: " + ", ".join(bits) + ".")
    chart = result.get('chart')
    if chart: parts.append(f"A {chart.get('type','data')} chart has been prepared to visualize the results.")
    table = result.get('table') or []
    if table: parts.append(f"Results are shown for {min(len(table),8)} representative rows.")
    limitations = result.get('limitations') or []
    if limitations: parts.append("Important limitations: " + " ".join(limitations[:2]))
    parts.append("Generated by UDI local analytics engine.")
    return " ".join(parts)


class GeminiService:
    def __init__(self):
        self._client = None
        self._force_mock = MOCK_MODE or not _GEMINI_AVAILABLE or not GEMINI_API_KEY
        if not self._force_mock and _GEMINI_AVAILABLE and GEMINI_API_KEY:
            try:
                self._client = genai.Client(api_key=GEMINI_API_KEY)
            except Exception:
                self._client = None
                self._force_mock = True

    def plan(self, question, profile, domain, history):
        if self._force_mock or self._client is None:
            return _rule_based_plan(question, profile, domain)
        system=f"""You are UDI's analytical planning layer. You do not calculate results and you do not write or execute Python. Select exactly one approved operation: get_summary, get_current_state, compare_groups, analyze_trend, detect_anomalies, calculate_correlation, predict_class, predict_value, forecast, run_scenario, unsupported. Use only exact column names from the supplied dataset profile. The dataset determines what is feasible. Never substitute an unrelated column for a requested target. If a required target, grouping column, time column, or outcome is unavailable, choose unsupported and explain the missing requirement in reasoning. CRITICAL PRIORITY RULES: (1) Any question asking for average, mean, sum, total, min, max of a column = MUST use get_summary with target_column set to the exact matching column from the profile. Never use predict_value or forecast for descriptive statistics. (2) Questions using "first year" or "next year" + a descriptive statistic like "average" or "total" = get_summary, NOT forecast. (3) Only use forecast when the user explicitly wants PROJECTIONS INTO THE FUTURE (not descriptive stats like averages). (4) Always choose a target_column that matches what column the user is asking about (e.g. "First Year Profit" -> fuzzy match any column name containing "Profit"). Domain: {domain}."""
        context={'domain':domain,'profile':profile,'history':history[-6:],'question':question}
        try:
            resp=self._client.models.generate_content(
                model=GEMINI_MODEL,
                contents=json.dumps(context,default=str),
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    response_mime_type='application/json',
                    response_schema=_ANALYSIS_PLAN_SCHEMA,
                    temperature=0.1,
                    max_output_tokens=1200,
                ),
            )
            parsed=getattr(resp,'parsed',None)
            if isinstance(parsed,AnalysisPlan): return parsed
            if isinstance(parsed,dict): return AnalysisPlan.model_validate(parsed)
            return AnalysisPlan.model_validate_json(resp.text)
        except Exception:
            return _rule_based_plan(question, profile, domain)

    def explain(self, question, plan, result, domain):
        if self._force_mock or self._client is None:
            return _rule_based_explain(question, plan, result, domain)
        prompt={'domain':domain,'question':question,'plan':plan.model_dump(),'verified_result':result}
        system="""You are UDI's explanation layer. Your job is to produce a crisp, specific, question-driven answer. RULES: (1) Lead with the EXACT STATISTIC the user asked for (e.g. if they asked "Average X", say "Average X = $Y" first, using the number from verified_result.metrics.mean). (2) Only use verified numbers and columns that appear inside verified_result. Never invent numbers. (3) For get_summary, DO NOT mention ML models, regression, or prediction — this is a descriptive statistic. Mention only summary.mean/sum/median/min/max and the record count. (4) For forecast — lead with horizon, target, and backtest RMSE only if available. (5) For predict_value — mention model name and R2/MAE/RMSE metrics. (6) Distinguish past descriptive stats (mean/sum/etc) vs future predictions (forecasts/scenarios/predict). Keep it short and specific. Always end with the source of the answer (Gemini vs "UDI local engine fallback"). If this explanation itself is running because Gemini call failed, do not mention that error — just present the numbers clearly."""
        try:
            resp=self._client.models.generate_content(
                model=GEMINI_MODEL,
                contents=json.dumps(prompt,default=str),
                config=types.GenerateContentConfig(system_instruction=system,temperature=0.2,max_output_tokens=700),
            )
            text=(getattr(resp,'text',None) or '').strip()
            if not text: raise RuntimeError('Gemini returned an empty explanation.')
            return text
        except Exception:
            return _rule_based_explain(question, plan, result, domain)
