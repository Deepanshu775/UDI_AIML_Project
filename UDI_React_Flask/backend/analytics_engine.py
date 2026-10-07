import pandas as pd, numpy as np

def _filter(df, plan):
    out=df.copy()
    if plan.filters:
        for c,v in plan.filters.items():
            if c in out.columns:
                vals=v if isinstance(v,list) else [v]; out=out[out[c].astype(str).isin([str(x) for x in vals])]
    if plan.start_date and plan.time_column in out.columns:
        d=pd.to_datetime(out[plan.time_column],errors='coerce'); out=out[d>=pd.to_datetime(plan.start_date)]
    if plan.end_date and plan.time_column in out.columns:
        d=pd.to_datetime(out[plan.time_column],errors='coerce'); out=out[d<=pd.to_datetime(plan.end_date)]
    return out

def _categorical_summary(d, target):
    s=d[target].dropna().astype(str)
    counts=s.value_counts()
    nunique=int(counts.shape[0])
    total=int(len(s))
    top=counts.head(10)
    rows=[{'Category':str(k),'Count':int(v),'Percent':round(float(v)/max(1,total)*100,2)} for k,v in top.items()]
    metrics={'total_records':total,'unique_categories':nunique,'top_category':str(counts.index[0]) if len(counts) else None,'top_category_count':int(counts.iloc[0]) if len(counts) else 0}
    if len(counts)==0:
        summary=f"{target} has no non-null values in the current filter."
    else:
        top_cat=str(counts.index[0]); top_count=int(counts.iloc[0])
        summary=(f"{target}: {nunique:,} unique categories across {total:,} records. "
                 f"Top category is '{top_cat}' with {top_count:,} records "
                 f"({round(top_count/max(1,total)*100,2)}% of the dataset).")
    return {'success':True,'operation':'get_summary','summary':summary,'metrics':metrics,'table':rows,'chart':{'type':'bar','x':'Category','y':'Count','data':rows[:15]},'limitations':['Counts are based on exact string equality; similar values with different spelling or casing are counted separately.']}

def _categorical_compare(d, target, g):
    if target == g:
        counts = d[target].dropna().astype(str).value_counts().reset_index()
        counts.columns=['Category','Count']
        total=int(counts['Count'].sum())
        counts['Percent']=(counts['Count']/max(1,total)*100).round(2)
        rows=counts.head(20).to_dict('records')
        summary=f"Compared {counts.shape[0]} categories in '{target}' by record counts."
        return {'success':True,'operation':'compare_groups','summary':summary,'metrics':{'groups':int(counts.shape[0]),'total_records':total},'table':rows,'chart':{'type':'bar','x':'Category','y':'Count','data':rows[:15]},'limitations':[f"Target and group are the same column ('{target}'); showing value counts instead of cross-group comparison."]}
    grouped = d.groupby([g, target], dropna=False).size().reset_index(name='count')
    pivot = grouped.pivot_table(index=g, columns=target, values='count', aggfunc='sum', fill_value=0)
    totals = pivot.sum(axis=1).sort_values(ascending=False)
    rows=[]
    for g_val, total in totals.head(20).items():
        row={'Group':g_val,'Total Records':int(total)}
        sub = grouped[grouped[g]==g_val].sort_values('count',ascending=False).head(5)
        for _, r in sub.iterrows():
            row[f"Top: {r[target]}"] = int(r['count'])
        rows.append(row)
    summary=f"Compared {target} distribution across {totals.shape[0]} groups."
    return {'success':True,'operation':'compare_groups','summary':summary,'metrics':{},'table':rows,'chart':{'type':'bar','x':'Group','y':'Total Records','data':[{'Group':str(k),'Total Records':int(v)} for k,v in totals.head(15).items()]},'limitations':['Compare groups on categorical target shows record counts per group and top sub-categories.']}

def execute(df, plan):
    d=_filter(df,plan); op=plan.operation
    if op=='get_summary':
        target=plan.target_column
        if target and target not in d.columns:
            raise ValueError(f"Column '{target}' does not exist in the uploaded dataset.")
        if not target:
            numeric_cols=d.select_dtypes(include=np.number).columns.tolist()
            if len(numeric_cols)==1:
                target=numeric_cols[0]
            elif len(numeric_cols)>1:
                rows=[]
                for c in numeric_cols[:20]:
                    s=d[c].dropna()
                    rows.append({'Column':c,'Mean':float(s.mean()) if len(s) else None,'Total':float(s.sum()) if len(s) else None,'Median':float(s.median()) if len(s) else None,'Min':float(s.min()) if len(s) else None,'Max':float(s.max()) if len(s) else None,'Count':int(len(s))})
                return {'success':True,'operation':op,'summary':f"Summary statistics for {len(numeric_cols)} numeric columns.","metrics":{},'table':rows,'chart':None,'limitations':[]}
            else:
                cats=[c for c in d.columns if not pd.api.types.is_datetime64_any_dtype(d[c])]
                if not cats: raise ValueError("No analyzable columns are available.")
                return _categorical_summary(d,cats[0])
        s=d[target].dropna()
        if pd.api.types.is_numeric_dtype(d[target]):
            agg=plan.aggregation or 'mean'
            if agg=='sum': value=float(s.sum()); label='Total'
            elif agg=='median': value=float(s.median()); label='Median'
            elif agg=='min': value=float(s.min()); label='Minimum'
            elif agg=='max': value=float(s.max()); label='Maximum'
            elif agg=='std': value=float(s.std()); label='Standard deviation'
            elif agg=='count': value=int(s.size); label='Count'
            elif agg=='percentage':
                value=float(s.mean()); label='Percentage/mean'
            else: value=float(s.mean()); label='Average'
            metrics={'aggregation':agg,'value':value,'count':int(s.size),'mean':float(s.mean()),'sum':float(s.sum()),'median':float(s.median()),'min':float(s.min()),'max':float(s.max()),'std':float(s.std()) if len(s)>1 else None}
            return {'success':True,'operation':op,'summary':f"{label} {target} = {value:,.2f} based on {len(s):,} valid records.",'metrics':metrics,'table':[{'Statistic':k,'Value':v} for k,v in [('mean',metrics['mean']),('sum',metrics['sum']),('median',metrics['median']),('min',metrics['min']),('max',metrics['max']),('std',metrics['std'])]],'chart':None,'limitations':[]}
        return _categorical_summary(d,target)

    if op=='get_current_state':
        if plan.time_column and plan.time_column in d.columns:
            dd=pd.to_datetime(d[plan.time_column],errors='coerce'); idx=dd.idxmax(); row=d.loc[idx].to_dict() if pd.notna(idx) else d.iloc[-1].to_dict()
        else: row=d.iloc[-1].to_dict()
        return {'success':True,'operation':op,'summary':'Latest available record selected as the current state.','metrics':{},'table':[row],'chart':None,'limitations':[]}
    if op=='compare_groups':
        g=(plan.group_by or [plan.entity_column])[0]; t=plan.target_column
        if not g or g not in d.columns: raise ValueError('A valid group column is required.')
        if t and t not in d.columns: raise ValueError(f"Column '{t}' does not exist in the uploaded dataset.")
        if t and pd.api.types.is_numeric_dtype(d[t]):
            agg=plan.aggregation or 'mean'
            func={'sum':'sum','mean':'mean','median':'median','min':'min','max':'max','std':'std','count':'count'}.get(agg,'mean')
            tab=d.groupby(g,dropna=False)[t].agg(func).reset_index(name=agg).sort_values(agg,ascending=False)
            summary=f"Compared {t} across {tab[g].nunique()} groups using {agg}."
            rows=tab.head(20).replace({np.nan:None}).to_dict('records')
            return {'success':True,'operation':op,'summary':summary,'metrics':{'groups':int(tab.shape[0]),'aggregation':agg,'top_group':str(tab.iloc[0][g]) if len(tab) else None,'top_value':float(tab.iloc[0][agg]) if len(tab) else None},'table':rows,'chart':{'type':'bar','x':g,'y':agg,'data':tab.head(15).replace({np.nan:None}).to_dict('records')},'limitations':[]}
        counts=d.groupby(g,dropna=False).size().reset_index(name='count').sort_values('count',ascending=False)
        rows=counts.head(20).replace({np.nan:None}).to_dict('records')
        return {'success':True,'operation':op,'summary':f"Compared record counts across {counts[g].nunique()} groups.","metrics":{'groups':int(counts.shape[0]),'top_group':str(counts.iloc[0][g]) if len(counts) else None,'top_value':int(counts.iloc[0]['count']) if len(counts) else None},'table':rows,'chart':{'type':'bar','x':g,'y':'count','data':counts.head(15).replace({np.nan:None}).to_dict('records')},'limitations':['No numeric target was supplied, so comparison uses record counts.']}

    if op=='analyze_trend':
        t=plan.target_column; tc=plan.time_column
        if not t or not tc: raise ValueError('Trend analysis requires target and time columns.')
        x=d[[tc,t]].copy(); x[tc]=pd.to_datetime(x[tc],errors='coerce'); x=x.dropna(subset=[tc]).sort_values(tc)
        categorical_target = not pd.api.types.is_numeric_dtype(x[t]) if t in x.columns else False
        if categorical_target or not pd.api.types.is_numeric_dtype(x[t]):
            x[t]=pd.to_numeric(x[t],errors='coerce')
        if pd.api.types.is_numeric_dtype(x[t]):
            numeric = x[t].notna().sum()
            if numeric < 4:
                x = x.dropna(subset=[tc]).copy()
                x['_count']=1
                freq='ME' if x[tc].nunique()>60 else 'W'
                agg=x.set_index(tc)['_count'].resample(freq).sum().reset_index(); agg.columns=[tc, 'records']
                xname='records'
                if len(agg) < 4: raise ValueError('Not enough valid observations for a trend.')
                direction='increasing' if agg[xname].iloc[-1]>agg[xname].iloc[0] else 'decreasing'
                base=agg[xname].iloc[0] if agg[xname].iloc[0] else None
                return {'success':True,'operation':op,'summary':f"Record-count trend for {t} is {direction} over time.",'metrics':{'change_pct':float((agg[xname].iloc[-1]/base-1)*100) if base else None},'table':agg.tail(24).replace({np.nan:None}).to_dict('records'),'chart':{'type':'line','x':tc,'y':xname,'data':agg.tail(48).replace({np.nan:None}).to_dict('records')},'limitations':[f'{t} is not numeric, so records-per-time-unit counts are shown instead of numeric aggregates.']}
            x=x.dropna(subset=[t])
        if len(x)<4: raise ValueError('Not enough valid observations for a trend.')
        freq='ME' if x[tc].nunique()>60 else 'W'
        agg=x.set_index(tc)[t].resample(freq).mean().reset_index(); agg['pct_change']=agg[t].pct_change()*100
        direction='increasing' if agg[t].iloc[-1]>agg[t].iloc[0] else 'decreasing'
        return {'success':True,'operation':op,'summary':f"The {t} series is generally {direction} over the observed period.",'metrics':{'change_pct':float((agg[t].iloc[-1]/agg[t].iloc[0]-1)*100) if agg[t].iloc[0] else None},'table':agg.tail(24).replace({np.nan:None}).to_dict('records'),'chart':{'type':'line','x':tc,'y':t,'data':agg.tail(48).replace({np.nan:None}).to_dict('records')},'limitations':['Trend direction is descriptive and does not establish causation.']}
    if op=='calculate_correlation':
        nums=d.select_dtypes(include=np.number)
        if nums.shape[1] < 2:
            return {'success':True,'operation':op,'summary':'At least two numeric columns are required for correlation; showing categorical value counts instead.','metrics':{},'table':[],'chart':None,'limitations':['No pairwise correlations available because fewer than two numeric columns exist.']}
        corr=nums.corr(method='spearman' if len(d)>500 else 'pearson'); rows=[]
        for a in corr.columns:
            for b in corr.columns:
                if a<b: rows.append({'feature_1':a,'feature_2':b,'correlation':float(corr.loc[a,b])})
        rows=sorted(rows,key=lambda r:abs(r['correlation']),reverse=True)[:20]
        return {'success':True,'operation':op,'summary':'Correlation highlights statistical association, not causation.','metrics':{},'table':rows,'chart':None,'limitations':['Correlation does not prove causation.']}
    raise ValueError(f'Unsupported analytics operation: {op}')
