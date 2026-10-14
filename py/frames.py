"""Fixed table operations; embedded with the main worker, never imported by path."""
TABLE_ROWS = 2_000_000
TABLE_COLS = 256
TABLE_BYTES = 256 * 1024 * 1024
TABLE_TZ = {'UTC', 'America/New_York', 'America/Chicago', 'America/Los_Angeles', 'Europe/London', 'Europe/Berlin', 'Asia/Tokyo', 'Asia/Kolkata', 'Australia/Sydney'}


def frame_columns(names):
    if not names or len(names) > TABLE_COLS or len(set(names)) != len(names):
        raise BadData('need 1..256 unique columns')
    if any(not isinstance(n, str) or not n or len(n.encode('utf-8')) > 256 or '\x00' in n for n in names):
        raise BadData('column names must be nonempty UTF-8, <=256 bytes')


def frame_dtype(spec):
    spec = {'dtype': spec} if isinstance(spec, str) else spec
    if not isinstance(spec, dict) or set(spec) - {'dtype', 'tz', 'categories', 'ordered'}:
        raise BadData('invalid dtype policy')
    if not isinstance(spec.get('dtype'),str) or spec.get('dtype') not in {'int64', 'float64', 'bool', 'string', 'datetime64[ns]', 'category'}:
        raise BadData('unsupported dtype policy')
    if 'tz' in spec and (spec['dtype'] != 'datetime64[ns]' or not isinstance(spec['tz'],str) or spec['tz'] not in TABLE_TZ):
        raise BadData('unsupported timezone')
    if set(spec) & {'categories', 'ordered'} and spec['dtype'] != 'category':
        raise BadData('category options require category dtype')
    if 'ordered' in spec and not isinstance(spec['ordered'],bool):
        raise BadData('ordered must be boolean')
    return dict(spec)


def frame_build(data, policies, inferred, from_csv=False, na_values=()):
    import pandas as pd
    import numpy as np
    import pytz
    frame_columns(list(data))
    if set(policies) - set(data):
        raise BadData('dtype override names an unknown column')
    lengths = {len(v) for v in data.values()}
    if len(lengths) != 1 or next(iter(lengths)) > TABLE_ROWS:
        raise BadData('ragged columns or more than 2M rows')
    series, meta, masks = {}, {}, {}
    for col, original in data.items():
        vals = list(original)
        if any(isinstance(v, (dict, list)) for v in vals):
            raise BadData('nested cells unsupported')
        if from_csv:
            vals = [None if v == '' or v in na_values else v for v in vals]
        spec = policies.get(col)
        if spec is None:
            nonnull = [v for v in vals if v is not None]
            if from_csv:
                import re
                intpat = re.compile(r'[+-]?[0-9]+\Z')
                floatpat = re.compile(r'[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z')
                if nonnull and all(intpat.fullmatch(v) for v in nonnull):
                    spec = 'int64'
                elif nonnull and all(floatpat.fullmatch(v) or v in ('NA','NaN','Inf','-Inf') for v in nonnull):
                    spec = 'float64'
                elif nonnull and all(v in ('true', 'false', 'True', 'False') for v in nonnull):
                    spec = 'bool'
                else:
                    spec = 'string'
            elif nonnull and all(isinstance(v, bool) for v in nonnull):
                spec = 'bool'
            elif nonnull and all(isinstance(v, int) and not isinstance(v, bool) for v in nonnull):
                spec = 'int64'
            elif nonnull and all(isinstance(v, (int, float)) and not isinstance(v, bool) or v in ('NA','NaN','Inf','-Inf') for v in nonnull):
                spec = 'float64'
            else:
                spec = 'string'
            inferred[col] = spec
        spec = frame_dtype(spec)
        dtype = spec['dtype']
        mask = [v is None or (dtype not in ('string','category') and v == 'NA') for v in vals]
        try:
            if dtype == 'int64':
                parsed = []
                for v, missing in zip(vals, mask):
                    if missing:
                        parsed.append(0);continue
                    if isinstance(v, bool) or isinstance(v, float) and not v.is_integer():
                        raise BadData('int64 requires integral cells')
                    n = int(v)
                    if not -(1 << 63) <= n < (1 << 63):
                        raise BadData('int64 cell out of range')
                    parsed.append(n)
                s = pd.Series(pd.array(parsed, dtype='Int64'));s[mask] = pd.NA
            elif dtype == 'float64':
                parsed = []
                for v, missing in zip(vals, mask):
                    if missing:
                        parsed.append(np.nan);continue
                    if isinstance(v, bool):
                        raise BadData('float64 does not accept bool')
                    n = float({'NaN': 'nan', 'Inf': 'inf', '-Inf': '-inf'}.get(v, v) if isinstance(v,str) else v)
                    if not math.isfinite(n) and v not in ('NaN', 'Inf', '-Inf'):
                        raise BadData('float64 overflow; use explicit nonfinite tokens')
                    parsed.append(n)
                s = pd.Series(parsed, dtype='float64')
            elif dtype == 'bool':
                parsed = []
                for v, missing in zip(vals, mask):
                    if missing:
                        parsed.append(False)
                    elif isinstance(v,bool):
                        parsed.append(v)
                    elif from_csv and v in ('true','false','True','False'):
                        parsed.append(v in ('true','True'))
                    else:
                        raise BadData('bool requires boolean cells')
                s = pd.Series(pd.array(parsed,dtype='boolean'));s[mask] = pd.NA
            elif dtype in ('string','category'):
                if any(v is not None and not isinstance(v,str) for v in vals):
                    raise BadData('string/category requires strings')
                if dtype == 'string':
                    s = pd.Series(vals,dtype='string')
                else:
                    levels = spec.get('categories', sorted({v for v in vals if v is not None}))
                    if not isinstance(levels,list) or any(not isinstance(v,str) for v in levels) or len(set(levels)) != len(levels) or any(v is not None and v not in levels for v in vals):
                        raise BadData('invalid category levels')
                    spec.update(categories=levels,ordered=bool(spec.get('ordered',False)))
                    s = pd.Series(pd.Categorical(vals,categories=levels,ordered=spec['ordered']))
            else:
                vals = [None if m else v for v,m in zip(vals,mask)]
                if any(v is not None and not isinstance(v,str) for v in vals):
                    raise BadData('datetime cells require ISO strings')
                # ISO only. Mixed naive/aware is rejected; naive local DST ambiguity raises.
                aware = [pd.Timestamp(v).tzinfo is not None for v in vals if v is not None]
                if aware and any(aware) != all(aware):
                    raise BadData('cannot mix naive and aware datetimes')
                tz = spec.get('tz')
                if aware and all(aware):
                    if not tz:
                        raise BadData('aware datetimes require explicit tz')
                    s = pd.Series(pd.to_datetime(vals,format='ISO8601',utc=True).tz_convert(tz))
                else:
                    s = pd.Series(pd.to_datetime(vals,format='ISO8601'))
                    if tz:
                        s = s.dt.tz_localize(tz,ambiguous='raise',nonexistent='raise')
                s = s.astype(pd.DatetimeTZDtype(unit='ns',tz=tz) if tz else 'datetime64[ns]')
                mask = s.isna().tolist()
        except (ValueError, TypeError, OverflowError, pytz.AmbiguousTimeError, pytz.NonExistentTimeError) as exc:
            raise BadData('column '+col+': '+str(exc)) from exc
        series[col], meta[col], masks[col] = s, spec, np.asarray(mask,dtype=bool)
    return pd.DataFrame(series), meta, masks


def frame_load(p):
    import numpy as np
    import pandas as pd
    if 'frame_file' in p:
        with np.load(p['frame_file'], allow_pickle=False) as z:
            try:
                a=z['__pycli_table__']
                if a.dtype != np.dtype('uint8') or a.ndim != 1 or a.size > 1024*1024:
                    raise BadData('invalid table metadata array')
                metadata=json.loads(a.tobytes().decode('utf-8'))
                if set(metadata) != {'version','columns','n_rows','column_meta','na_masks'} or metadata['version'] != 1:
                    raise BadData('invalid table metadata')
                names=metadata['columns'];frame_columns(names);n=metadata['n_rows']
                if not isinstance(n,int) or not 0<=n<=TABLE_ROWS or set(metadata['column_meta']) != set(names):
                    raise BadData('invalid table rows/column metadata')
                expected={'__pycli_table__'} | {'c'+str(i) for i in range(len(names))} | {'m'+str(i) for i in range(len(names))}
                if set(z.files) != expected or metadata['na_masks'] != {col:'m'+str(i) for i,col in enumerate(names)}:
                    raise BadData('unexpected table members/masks')
                series, meta, masks = {},{},{}
                for i,col in enumerate(names):
                    spec=frame_dtype(metadata['column_meta'][col]);dtype=spec['dtype'];raw=z['c'+str(i)];mask=z['m'+str(i)]
                    if raw.ndim!=1 or raw.size!=n or mask.dtype!=np.dtype('bool') or mask.shape!=(n,):
                        raise BadData('table column/mask shape mismatch')
                    expected_dtype={'int64':'<i8','float64':'<f8','bool':'|b1','datetime64[ns]':'<i8','category':'<i8'}
                    if dtype=='string':
                        if raw.dtype.kind!='S':raise BadData('strings require UTF-8 bytes arrays')
                        vals=[bytes(v).decode('utf-8') for v in raw]
                        if any('\x00' in v for v in vals):raise BadData('NUL in strings unsupported')
                        s=pd.Series(vals,dtype='string');s[mask]=pd.NA
                    elif raw.dtype!=np.dtype(expected_dtype[dtype]):
                        raise BadData('table dtype mismatch')
                    elif dtype=='datetime64[ns]':
                        s=pd.Series(pd.to_datetime(raw,unit='ns',utc=bool(spec.get('tz'))))
                        if spec.get('tz'):s=s.dt.tz_convert(spec['tz'])
                        s[mask]=pd.NaT
                        if not np.array_equal(s.isna().to_numpy(),mask):raise BadData('NaT/mask mismatch')
                    elif dtype=='category':
                        levels=spec.get('categories');ordered=spec.get('ordered')
                        if not isinstance(levels,list) or any(not isinstance(v,str) for v in levels) or len(set(levels))!=len(levels) or not isinstance(ordered,bool):raise BadData('invalid categories')
                        if ((raw < -1)|(raw>=len(levels))).any() or not np.array_equal(raw==-1,mask):raise BadData('invalid category codes/mask')
                        s=pd.Series(pd.Categorical.from_codes(raw,categories=levels,ordered=ordered))
                    elif dtype=='float64':
                        s=pd.Series(raw);s[mask]=np.nan
                    else:
                        s=pd.Series(pd.array(raw,dtype={'int64':'Int64','bool':'boolean'}[dtype]));s[mask]=pd.NA
                    series[col],meta[col],masks[col]=s,spec,mask.copy()
                return pd.DataFrame(series),meta,masks,{}
            except (KeyError, ValueError, TypeError, UnicodeError) as exc:
                raise BadData('invalid table artifact: '+str(exc)) from exc
    inferred={}
    if 'csv_file' in p:
        import csv
        with open(p['csv_file'],encoding='utf-8-sig',newline='') as f:
            rows=csv.reader(f,delimiter=p.get('delimiter',','),strict=True)
            names=next(rows);frame_columns(names);data={col:[] for col in names}
            for row in rows:
                if len(row)!=len(names):raise BadData('ragged CSV')
                for col,v in zip(names,row):data[col].append(v)
        iscsv=True
    elif 'columns' in p:
        data=p['columns'];iscsv=False
    else:
        records=p['records'];names=list(records[0]) if records else p.get('names',[])
        frame_columns(names)
        if any(set(row)!=set(names) for row in records):raise BadData('records must have identical columns')
        data={col:[row[col] for row in records] for col in names};iscsv=False
    policies=dict(p.get('dtypes',{}))
    for col in p.get('parse_dates',[]):
        if col in policies and frame_dtype(policies[col])['dtype']!='datetime64[ns]':raise BadData('conflicting parse_dates/dtypes')
        policies.setdefault(col,'datetime64[ns]')
    df,meta,masks=frame_build(data,policies,inferred,iscsv,p.get('na_values',[]))
    return df,meta,masks,inferred


def frame_preview(df,meta,masks,indices):
    import pandas as pd
    data={}
    for col in df.columns:
        dtype=meta[col]['dtype'];vals=[]
        for i in indices:
            value=df[col].iloc[i]
            if masks[col][i]:
                vals.append(FRAME_NULL)
            elif dtype=='datetime64[ns]':
                vals.append(value.isoformat())
            elif dtype in ('string','category'):
                vals.append(str(value))
            elif dtype=='bool':
                vals.append(bool(value))
            elif dtype=='int64':
                vals.append(int(value))
            else:
                vals.append(float(value))
        data[col]=vals
    return {'columns':list(df.columns),'data':data,'n_rows':len(indices)}


def frame_save(df,meta,masks,p,inferred):
    import numpy as np
    if len(df)>TABLE_ROWS or len(df.columns)>TABLE_COLS:raise OutputTooLarge('table output cap exceeded')
    frame_columns(list(df.columns));arrays={}
    # Fixed-width byte strings can expand a tiny CSV into gigabytes. Budget every
    # column and mask before constructing any storage array (including siblings).
    projected=1024*1024
    string_values={}
    for col in df:
        dtype=meta[col]['dtype']
        if dtype=='string':
            values=[('' if m else str(v)).encode('utf-8') for v,m in zip(df[col],masks[col])]
            if any(b'\x00' in v for v in values):raise BadData('NUL in strings unsupported')
            width=max(1,max(map(len,values),default=0));string_values[col]=(values,width)
        else:
            width=1 if dtype=='bool' else 8
        projected+=len(df)*(width+1)
        if projected>TABLE_BYTES:raise OutputTooLarge('table storage exceeds 256MiB before allocation')
    for i,col in enumerate(df.columns):
        dtype=meta[col]['dtype'];s=df[col];mask=np.asarray(masks[col],dtype=bool)
        if dtype=='string':
            values,width=string_values[col]
            raw=np.asarray(values,dtype='S'+str(width))
        elif dtype=='category':raw=s.cat.codes.to_numpy(dtype='<i8')
        elif dtype=='datetime64[ns]':raw=s.array.asi8.astype('<i8')
        elif dtype=='float64':raw=s.to_numpy(dtype='<f8')
        else:raw=s.to_numpy(dtype={'int64':'<i8','bool':'|b1'}[dtype],na_value=0)
        arrays['c'+str(i)]=raw;arrays['m'+str(i)]=mask
    metadata={'version':1,'columns':list(df.columns),'n_rows':len(df),'column_meta':meta,'na_masks':{c:'m'+str(i) for i,c in enumerate(df.columns)}}
    body=json.dumps(metadata,ensure_ascii=False,allow_nan=False).encode('utf-8')
    if len(body)>1024*1024:raise OutputTooLarge('table metadata exceeds 1MiB')
    arrays['__pycli_table__']=np.frombuffer(body,dtype='uint8')
    if sum(a.nbytes for a in arrays.values())>TABLE_BYTES-1024*1024:raise OutputTooLarge('table artifact exceeds 256MiB')
    np.savez(p['output_file'],allow_pickle=False,**arrays)
    if os.path.getsize(p['output_file'])>TABLE_BYTES:raise OutputTooLarge('table artifact exceeds 256MiB')
    return {'status':'OK','shape':[len(df),len(df.columns)],'column_meta':meta,'inferred':inferred,'preview':frame_preview(df,meta,masks,range(min(len(df),min(10,1000//len(df.columns)))))}


def frame_predicate(df,node):
    import pandas as pd
    if 'and' in node or 'or' in node:
        key='and' if 'and' in node else 'or';parts=[frame_predicate(df,n) for n in node[key]];result=parts[0]
        for part in parts[1:]:result=result & part if key=='and' else result | part
        return result
    if 'not' in node:return ~frame_predicate(df,node['not'])
    col=node['col']
    if col not in df:raise BadData('unknown predicate column')
    s=df[col];op=node['op'];v=node.get('value')
    try:
        if op=='isnull':return s.isna()
        if op=='isin':return s.isin(v).fillna(False)
        if op=='between':return s.between(v[0],v[1]).fillna(False)
        if op=='startswith':return s.str.startswith(v,na=False)
        if op=='contains':return s.str.contains(v,regex=False,na=False)
        functions={'eq':s.eq,'ne':s.ne,'lt':s.lt,'le':s.le,'gt':s.gt,'ge':s.ge}
        return functions[op](v).fillna(False)
    except (TypeError, ValueError, AttributeError) as exc:raise BadData('invalid predicate for column dtype') from exc


def frame_aggregations(df,aggs):
    import pandas as pd
    # Caller names select our function objects, never pandas string dispatch.
    functions={'sum':pd.Series.sum,'mean':pd.Series.mean,'min':pd.Series.min,'max':pd.Series.max,'count':pd.Series.count,'std':pd.Series.std,'var':pd.Series.var,'median':pd.Series.median,'nunique':pd.Series.nunique,'first':lambda s:s.dropna().iloc[0] if len(s.dropna()) else None,'last':lambda s:s.dropna().iloc[-1] if len(s.dropna()) else None}
    out={}
    for col,ops in aggs.items():
        if col not in df:raise BadData('unknown aggregation column')
        for op in ops:
            name=col+'__'+op
            if name in df or name in out:raise BadData('flattened aggregation name collision')
            out[name]=pd.NamedAgg(column=col,aggfunc=functions[op])
    return out


def frame_operation(op,p):
    import numpy as np
    import pandas as pd
    df,meta,masks,inferred=frame_load(p)
    if op=='frame_import' and 'frame_file' in p and (p.get('dtypes') or p.get('parse_dates')):
        policies=dict(meta);policies.update(p.get('dtypes',{}))
        for col in p.get('parse_dates',[]):policies[col]='datetime64[ns]'
        table=frame_preview(df,meta,masks,range(len(df)))
        data={c:[None if isinstance(v,FrameNull) else 'NaN' if isinstance(v,float) and math.isnan(v) else 'Inf' if isinstance(v,float) and v==math.inf else '-Inf' if isinstance(v,float) and v==-math.inf else v for v in vals] for c,vals in table['data'].items()}
        df,meta,masks=frame_build(data,policies,inferred)
    if op=='frame_inspect':
        head=p.get('head',10);tail=p.get('tail',0)
        if (min(head,len(df))+min(tail,len(df)))*len(df.columns)>1000:raise BadData('head+tail exceeds 1000 cells')
        result={'shape':[len(df),len(df.columns)],'dtypes':{c:m['dtype'] for c,m in meta.items()},'column_meta':meta,'null_counts':{c:int(m.sum()) for c,m in masks.items()},'inferred':inferred,'head':frame_preview(df,meta,masks,range(min(head,len(df)))),'tail':frame_preview(df,meta,masks,range(max(0,len(df)-tail),len(df)))}
        if p.get('describe',False):
            numeric=[c for c in df if meta[c]['dtype'] in ('int64','float64')]
            if len(numeric)*8+ (min(head,len(df))+min(tail,len(df)))*len(df.columns)>1000:raise OutputTooLarge('describe plus preview exceeds 1000 cells')
            result['describe']={c:{str(k):float(v) for k,v in df[c].describe().items()} for c in numeric}
        return result
    if op=='frame_export':
        if not p.get('output_file') and df.size>1000:raise OutputTooLarge('inline export exceeds 1000 cells; pass out')
        if p['format']=='json':
            table=frame_preview(df,meta,masks,range(len(df)))
            if p.get('output_file'):
                with open(p['output_file'],'w',encoding='utf-8') as f:f.write(codec(table))
            else:return {'table':table}
        else:
            import csv
            def rows():
                yield list(df.columns)
                for i in range(len(df)):
                    yield [('' if masks[c][i] else format(float(df[c].iloc[i]),'.17g') if meta[c]['dtype']=='float64' else df[c].iloc[i].isoformat() if meta[c]['dtype']=='datetime64[ns]' else str(df[c].iloc[i])) for c in df]
            if p.get('output_file'):
                with open(p['output_file'],'w',encoding='utf-8',newline='') as f:csv.writer(f,lineterminator='\n').writerows(rows())
            else:
                import io
                f=io.StringIO();csv.writer(f,lineterminator='\n').writerows(rows());return {'csv':f.getvalue()}
        return {'status':'OK','shape':[len(df),len(df.columns)]}
    if op=='frame_select':
        if 'where' in p:df=df.loc[frame_predicate(df,p['where'])]
        if p.get('sort_by'):
            names=[v['col'] for v in p['sort_by']]
            if any(c not in df for c in names):raise BadData('unknown sort column')
            df=df.sort_values(names,ascending=[not v.get('desc',False) for v in p['sort_by']],kind='stable')
        if 'limit' in p:df=df.head(p['limit'])
        if 'select_columns' in p:
            names=p['select_columns'];frame_columns(names)
            if any(c not in df for c in names):raise BadData('unknown selected column')
            df=df[names]
        masks={c:masks[c][df.index.to_numpy()] for c in df};meta={c:meta[c] for c in df};df=df.reset_index(drop=True)
    elif op in ('frame_groupby','frame_resample'):
        aggs=frame_aggregations(df,p['aggs'])
        try:
            if op=='frame_groupby':
                keys=p['keys']
                if not keys or len(set(keys))!=len(keys) or any(c not in df for c in keys):raise BadData('invalid group keys')
                if set(keys)&set(aggs):raise BadData('group key/flattened output collision')
                out=df.groupby(keys,sort=True,dropna=p.get('dropna',False),observed=True).agg(**aggs).reset_index()
            else:
                timecol=p['time']
                if timecol not in meta or meta[timecol]['dtype']!='datetime64[ns]':raise BadData('resample requires datetime column')
                if timecol in aggs:raise BadData('time/output collision')
                if df[timecol].isna().any():raise BadData('resample time contains NaT')
                if len(df):
                    # Bound bins before pandas allocates a dense time grid; units are checked by C++.
                    import re
                    count,unit=re.fullmatch(r'([1-9][0-9]*)(min|h|D|W|MS|ME|QS|YS)',p['rule']).groups()
                    seconds={'min':60,'h':3600,'D':82800,'W':7*82800,'MS':28*82800,'ME':28*82800,'QS':84*82800,'YS':365*82800}[unit]*int(count)
                    if (df[timecol].max()-df[timecol].min()).total_seconds()/seconds+3>TABLE_ROWS:raise BadData('resample bin count exceeds cap')
                out=df.resample(p['rule'],on=timecol,closed=p.get('closed','left'),label=p.get('label','left')).agg(**aggs).reset_index()
        except (ValueError, TypeError) as exc:raise BadData('invalid aggregation: '+str(exc)) from exc
        data={c:[None if pd.isna(v) else ('Inf' if v>0 else '-Inf') if isinstance(v,(float,np.floating)) and math.isinf(float(v)) else v.isoformat() if isinstance(v,pd.Timestamp) else v.item() if isinstance(v,np.generic) else v for v in out[c]] for c in out}
        policies={c:meta[c] for c in (p['keys'] if op=='frame_groupby' else [p['time']])}
        # Preserve categorical and timezone policies of keys; infer aggregate result types explicitly.
        df,meta,masks=frame_build(data,policies,inferred)
    return frame_save(df,meta,masks,p,inferred)
