import pandas as pd
import numpy as np
import re
import warnings

_UNNAMED_RE = re.compile(r'^(unnamed|unknown|column)\s*[:_\-]?\s*\d*$', re.IGNORECASE)

def _detect_header_row(raw):
    n = min(len(raw), 8)
    if n <= 1:
        return 0
    scores = []
    for i in range(n):
        row = raw.iloc[i]
        non_null = int(row.notna().sum())
        all_count = len(row)
        non_null_ratio = non_null / max(1, all_count)
        text_count = 0
        numeric_count = 0
        dup_names = 0
        seen = set()
        empty_count = 0
        unnamed_count = 0
        for v in row:
            if pd.isna(v):
                empty_count += 1
                continue
            s = str(v).strip()
            if not s:
                empty_count += 1
                continue
            if _UNNAMED_RE.match(s):
                unnamed_count += 1
            if s.lower() in seen:
                dup_names += 1
            seen.add(s.lower())
            try:
                float(s.replace(',', ''))
                numeric_count += 1
            except Exception:
                text_count += 1
        score = (
            + 3.0 * non_null_ratio
            + 2.0 * (text_count / max(1, all_count))
            - 2.5 * (numeric_count / max(1, all_count))
            - 1.5 * (empty_count / max(1, all_count))
            - 2.0 * (dup_names / max(1, all_count))
            - 3.0 * (unnamed_count / max(1, all_count))
        )
        scores.append((score, i))
    scores.sort(reverse=True)
    best_idx = scores[0][1]
    best_row = raw.iloc[best_idx]
    best_unnamed = sum(1 for v in best_row if pd.isna(v) or _UNNAMED_RE.match(str(v).strip() or 'x'))
    if best_unnamed / max(1, len(best_row)) >= 0.5:
        return 0
    return best_idx

def _promote_first_row_if_needed(df):
    if df.empty:
        return df
    cols = list(df.columns)
    n = len(cols)
    unnamed_ratio = sum(1 for c in cols if _UNNAMED_RE.match(str(c))) / max(1, n)
    if unnamed_ratio < 0.4:
        return df
    row0 = df.iloc[0]
    non_null = int(row0.notna().sum())
    if non_null / max(1, n) < 0.5:
        return df
    try:
        numeric_count = 0
        text_count = 0
        for v in row0:
            if pd.isna(v):
                continue
            s = str(v).strip()
            if not s:
                continue
            try:
                float(s.replace(',', ''))
                numeric_count += 1
            except Exception:
                text_count += 1
        if numeric_count / max(1, (numeric_count + text_count)) > 0.5:
            return df
        new_cols = [str(v).strip() if pd.notna(v) and str(v).strip() else f'col_{j}' for j, v in enumerate(row0)]
        dedup = {}
        clean_cols = []
        for j, c in enumerate(new_cols):
            base = c or f'col_{j}'
            if base in dedup:
                dedup[base] += 1
                clean_cols.append(f'{base}_{dedup[base]}')
            else:
                dedup[base] = 0
                clean_cols.append(base)
        out = df.iloc[1:].copy()
        out.columns = clean_cols
        out = out.reset_index(drop=True)
        out = out.dropna(how='all')
        return out
    except Exception:
        return df

def _coerce_types(df):
    for c in df.columns:
        if df[c].dtype == 'object':
            s = df[c].dropna()
            if len(s) == 0:
                continue
            try:
                parsed = pd.to_numeric(s.astype(str).str.replace(',', '', regex=False), errors='coerce')
                if parsed.notna().mean() >= 0.8 and parsed.nunique(dropna=True) > 0:
                    df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', '', regex=False), errors='coerce')
                    continue
            except Exception:
                pass
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    parsed_dt = pd.to_datetime(s, errors='coerce')
                if parsed_dt.notna().mean() >= 0.8 and parsed_dt.nunique(dropna=True) > 3:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        df[c] = pd.to_datetime(df[c], errors='coerce')
            except Exception:
                pass
    return df

def _read_with_detection(read_fn, suffix, path):
    raw = read_fn(path, header=None)
    if raw.empty:
        return raw
    header_idx = _detect_header_row(raw)
    if header_idx == 0:
        try:
            default_df = read_fn(path)
        except Exception:
            default_df = None
        if default_df is not None and not default_df.empty:
            cols = list(default_df.columns)
            unnamed_ratio = sum(1 for c in cols if _UNNAMED_RE.match(str(c))) / max(1, len(cols))
            if unnamed_ratio < 0.4:
                default_df = _promote_first_row_if_needed(default_df)
                default_df = _coerce_types(default_df.copy())
                return default_df
    new_cols = [str(v).strip() if pd.notna(v) else f'col_{j}' for j, v in enumerate(raw.iloc[header_idx])]
    dedup = {}
    clean_cols = []
    for j, c in enumerate(new_cols):
        base = c or f'col_{j}'
        if base in dedup:
            dedup[base] += 1
            clean_cols.append(f'{base}_{dedup[base]}')
        else:
            dedup[base] = 0
            clean_cols.append(base)
    data_start = header_idx + 1
    if data_start >= len(raw):
        df = raw.iloc[:0].copy()
        df.columns = clean_cols
        return df
    df = raw.iloc[data_start:].copy()
    df.columns = clean_cols
    df = df.reset_index(drop=True)
    df = df.dropna(how='all')
    df = _promote_first_row_if_needed(df)
    df = _coerce_types(df.copy())
    return df

def load_dataset(path):
    suffix=str(path).lower().split('.')[-1]
    if suffix=='csv':
        return _read_with_detection(pd.read_csv, suffix, path)
    if suffix in ('xlsx','xls'):
        return _read_with_detection(pd.read_excel, suffix, path)
    raise ValueError('Only CSV and Excel files are supported.')

def infer_kind(s):
    if pd.api.types.is_datetime64_any_dtype(s): return 'datetime'
    if pd.api.types.is_numeric_dtype(s): return 'numeric'
    nun=s.nunique(dropna=True); n=len(s)
    if nun <= min(30, max(5,int(n*.05))): return 'categorical'
    return 'text'

def profile_dataframe(df, name='dataset'):
    work=df.copy()
    date_candidates=[]
    for c in work.columns:
        if not pd.api.types.is_datetime64_any_dtype(work[c]):
            if work[c].dtype=='object':
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    parsed=pd.to_datetime(work[c], errors='coerce')
                if parsed.notna().mean() >= .8 and parsed.nunique()>3:
                    work[c]=parsed
        if pd.api.types.is_datetime64_any_dtype(work[c]): date_candidates.append(c)
    cols=[]
    for c in work.columns:
        s=work[c]; kind=infer_kind(s)
        identifier=bool(
            len(s) > 30 and s.nunique(dropna=True) >= max(20, .9*len(s))
            and (
                any(token in str(c).lower() for token in ['id','uuid','key','code'])
                or not pd.api.types.is_numeric_dtype(s)
            )
        )
        cols.append({'name':c,'dtype':kind,'unique_values':int(s.nunique(dropna=True)),'missing':int(s.isna().sum()),'missing_pct':round(float(s.isna().mean()*100),2),'identifier_like':identifier})
    date_range=None
    if date_candidates:
        d=work[date_candidates[0]].dropna()
        if len(d): date_range={'column':date_candidates[0],'start':str(d.min()),'end':str(d.max())}
    numeric=work.select_dtypes(include=np.number)
    stats={}
    for c in numeric.columns:
        x=numeric[c].dropna()
        stats[c]={'mean':float(x.mean()) if len(x) else None,'median':float(x.median()) if len(x) else None,'min':float(x.min()) if len(x) else None,'max':float(x.max()) if len(x) else None,'std':float(x.std()) if len(x)>1 else None}
    return {'dataset_name':name,'rows':int(len(work)),'columns':cols,'column_names':list(work.columns),'numeric_columns':list(numeric.columns),'categorical_columns':[c['name'] for c in cols if c['dtype']=='categorical'],'datetime_columns':date_candidates,'text_columns':[c['name'] for c in cols if c['dtype']=='text'],'identifier_columns':[c['name'] for c in cols if c['identifier_like']],'missing_total':int(work.isna().sum().sum()),'duplicate_rows':int(work.duplicated().sum()),'date_range':date_range,'numeric_stats':stats}
