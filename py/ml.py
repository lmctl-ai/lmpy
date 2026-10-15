"""S6 fixed estimator factories and numeric-only model attributes; no caller-selected constructors."""
def ml_import():
    import sklearn.linear_model, sklearn.svm, sklearn.naive_bayes, sklearn.discriminant_analysis
    import sklearn.neighbors, sklearn.cluster, sklearn.decomposition, sklearn.preprocessing
    import sklearn.ensemble, sklearn.model_selection, sklearn.metrics, sklearn.exceptions
    import joblib, threadpoolctl,ctypes
    # Linux/glibc is pinned. Resolve symbols from the loaded process: no ldconfig child.
    threadpoolctl.ThreadpoolController._system_libraries["libc"]=ctypes.CDLL(None)


def ml_estimator(family, params, seed):
    import sklearn.linear_model as linear
    import sklearn.svm as svm
    import sklearn.naive_bayes as nb
    import sklearn.discriminant_analysis as da
    import sklearn.neighbors as neighbors
    import sklearn.cluster as cluster
    import sklearn.decomposition as decomp
    import sklearn.preprocessing as prep
    import sklearn.ensemble as ens
    p = dict(params)
    if 'class_weight' in p and p['class_weight'] == 'none': p['class_weight'] = None
    if family == 'linear': return linear.LinearRegression(n_jobs=1, **p)
    if family == 'ridge': return linear.Ridge(solver='svd', **p)
    if family == 'lasso': return linear.Lasso(selection='cyclic', **p)
    if family == 'elastic_net': return linear.ElasticNet(selection='cyclic', **p)
    if family == 'logistic': return linear.LogisticRegression(solver='lbfgs', n_jobs=1, **p)
    if family == 'linear_svc': return svm.LinearSVC(dual='auto', random_state=seed, **p)
    if family == 'gaussian_nb': return nb.GaussianNB(**p)
    if family == 'lda':
        if p['n_components'] == 0: p['n_components'] = None
        return da.LinearDiscriminantAnalysis(solver='svd', **p)
    if family == 'knn_classifier': return neighbors.KNeighborsClassifier(algorithm='brute', metric='euclidean', n_jobs=1, **p)
    if family == 'knn_regressor': return neighbors.KNeighborsRegressor(algorithm='brute', metric='euclidean', n_jobs=1, **p)
    if family == 'kmeans': return cluster.KMeans(algorithm='lloyd', random_state=seed, **p)
    if family == 'pca': return decomp.PCA(svd_solver='full', **p)
    if family == 'standard_scaler': return prep.StandardScaler()
    if family.startswith('random_forest_'):
        p['max_features'] = None if p['max_features'] == 'all' else p['max_features']
        return (ens.RandomForestClassifier if family.endswith('classifier') else ens.RandomForestRegressor)(n_jobs=1, random_state=seed, **p)
    if family.startswith('hist_gradient_boosting_'):
        return (ens.HistGradientBoostingClassifier if family.endswith('classifier') else ens.HistGradientBoostingRegressor)(early_stopping=False, random_state=seed, **p)
    raise BadData('unknown estimator')


def ml_matrix(df, features, prep, learned=None):
    import numpy as np
    pieces=[]; categories={} if learned is None else learned['categories']
    for name in features:
        if name not in df: raise BadData('missing feature: '+name)
        col=df[name]
        if col.isna().any(): raise BadData('ML rejects missing values: '+name)
        if name in prep['one_hot']:
            vals=col.map(str).to_numpy()
            if learned is None: categories[name]=sorted(set(vals))
            if not set(vals).issubset(categories[name]): raise BadData('unseen category: '+name)
            pieces.append(np.column_stack([vals==level for level in categories[name]]).astype(float))
        else:
            try: a=np.asarray(col,dtype=float)
            except (ValueError,TypeError): raise BadData('numeric feature required: '+name)
            if not np.isfinite(a).all() or (np.abs(a)>1e12).any(): raise BadData('nonfinite/huge feature: '+name)
            pieces.append(a[:,None])
    x=np.column_stack(pieces)
    if x.shape[1]>64 or x.size>200000: raise BadData('encoded feature cap')
    return x,dict(scale=prep['scale'],one_hot=prep['one_hot'],categories=categories)


def ml_target(df,p,task):
    import numpy as np
    import re
    if task not in ('classification','regression'): return None,[]
    col=df[p['target']]
    if col.isna().any(): raise BadData('target missing')
    if task=='regression':
        a=np.asarray(col,dtype=float)
        if not np.isfinite(a).all(): raise BadData('target nonfinite')
        return a,[]
    values=col.tolist()
    # Match the C++ source-independent policy; leading zeros and '+' stay strings.
    if all(isinstance(v,str) and re.fullmatch(r'0|-?[1-9][0-9]*',v) and len(v)<=11 and abs(int(v))<=10**9 for v in values):values=[int(v) for v in values]
    labels=sorted(set(values))
    if len(labels)<2 or len(labels)>32: raise BadData('classification needs 2..32 classes')
    codes={label:i for i,label in enumerate(labels)}
    return np.array([codes[v] for v in values],dtype='int64'),labels


def ml_fit_one(df,p):
    import numpy as np
    import sklearn.preprocessing, sklearn.exceptions
    task=p['ml_spec']['families'][p['family']]['task']
    x,prep=ml_matrix(df,p['features'],p['preprocess']); y,labels=ml_target(df,p,task)
    scaler=None
    if prep['scale']:
        scaler=sklearn.preprocessing.StandardScaler();x=scaler.fit_transform(x)
    if p['family']=='kmeans' and len(np.unique(x,axis=0))<p['params']['n_clusters']:
        return None,None,None,None,None,'DEGENERATE_CLUSTERS'
    if p['family']=='lda':
        within=x-np.array([x[y==i].mean(axis=0) for i in range(len(labels))])[y]
        if not np.any(within):return None,None,None,None,None,'DEGENERATE_DATA'
    est=ml_estimator(p['family'],p['params'],p.get('seed',0))
    with warnings.catch_warnings(record=True) as notes:
        warnings.simplefilter('always');est.fit(x,y)
    for note in notes: warnings.warn(str(note.message), note.category)
    if p['family']=='kmeans' and len(np.unique(est.labels_))<p['params']['n_clusters']:return None,None,None,None,None,'DEGENERATE_CLUSTERS'
    if (p['family']=='kmeans' and est.n_iter_>=p['params']['max_iter']) or any(issubclass(v.category,sklearn.exceptions.ConvergenceWarning) for v in notes): return None,None,None,None,None,'NO_CONVERGENCE'
    return est,x,prep,labels,scaler,'RANK_DEFICIENT' if p['family']=='linear' and est.rank_<x.shape[1] else 'OK'


def ml_result(est,x,kind,family,labels):
    import numpy as np
    classification=bool(labels)
    if not np.isfinite(x).all() or (np.abs(x)>1e150).any(): raise NumericalFailure('transformed features exceed safe arithmetic range')
    # A class label is always finite, even when argmax sees only invalid scores.
    # Check the underlying public scores before accepting a classification.
    if family in ('logistic','linear_svc','lda'):
        if not np.isfinite(est.decision_function(x)).all(): raise NumericalFailure('nonfinite classification score')
    elif family in ('gaussian_nb','knn_classifier'):
        if not np.isfinite(est.predict_proba(x)).all(): raise NumericalFailure('nonfinite classification probability')
        if est.predict_proba(x).shape != (len(x),len(labels)): raise NumericalFailure('probability width disagrees with stored classes')
    elif family=='kmeans' and kind=='predict':
        if not np.isfinite(est.transform(x)).all(): raise NumericalFailure('nonfinite centroid distance')
    if kind=='predict':
        if family in ('pca','standard_scaler'): raise BadData('use transform for '+family)
        a=est.predict(x)
        if classification: a=np.asarray([labels[int(v)] for v in a])
    elif kind=='predict_proba':
        if family not in ('logistic','gaussian_nb','lda','knn_classifier','random_forest_classifier','hist_gradient_boosting_classifier'): raise BadData('predict_proba unavailable')
        a=est.predict_proba(x)
        if a.shape != (len(x),len(labels)): raise NumericalFailure('probability width disagrees with stored classes')
    elif kind=='decision_function':
        if family not in ('logistic','linear_svc','lda','hist_gradient_boosting_classifier'): raise BadData('decision_function unavailable')
        a=est.decision_function(x)
    elif kind=='transform':
        if family not in ('pca','standard_scaler','kmeans','lda'): raise BadData('transform unavailable')
        a=est.transform(x)
    else: raise BadData('unknown prediction kind')
    a=np.asarray(a)
    if a.dtype.kind in 'fiu' and not np.isfinite(a).all(): raise NumericalFailure('nonfinite prediction')
    if a.size>200000: raise BadData('prediction output cell cap')
    if a.ndim==1: return {'prediction':a.tolist()}
    prefix={'predict_proba':'probability','decision_function':'decision','transform':'component'}[kind]
    return {prefix+'_'+str(i):a[:,i].tolist() for i in range(a.shape[1])}


def ml_preview(data):
    n=min(20,800//max(1,len(data)))
    out={k:v[:n] for k,v in data.items()}
    while len(json.dumps(out,ensure_ascii=False).encode())>24000 and n:
        n//=2;out={k:v[:n] for k,v in data.items()}
    return out


def ml_save_table(data,p):
    inferred={}
    df,meta,masks=frame_build(data,{},inferred,False)
    result=frame_save(df,meta,masks,p,inferred)
    result.pop('preview',None)  # one bounded preview; never duplicate a wide output
    result['outputs_in_artifact']=True
    result['prediction_preview']=ml_preview(data)
    result['preview_truncated']=len(df)>len(next(iter(result['prediction_preview'].values())))
    return result


def ml_inline_parameters(result,tensors):
    # Training arrays/class codes are never echoed. A single small fitted table
    # must fit both the parameter budget and the whole-response array budget.
    eligible={k:v for k,v in tensors if k not in ('train_x_','train_y_','classes_','pp_mean_','pp_scale_')}
    result['parameters_omitted']=True
    if not eligible or sum(v.size for v in eligible.values())>100:return
    candidate={k:v.tolist() for k,v in eligible.items()}
    public={k:v for k,v in result.items() if k not in ('outputs','model_metadata')}
    public['parameters']=candidate
    def entries(value):
        if isinstance(value,list):return sum(entries(v) if isinstance(v,(list,dict)) else 1 for v in value)
        if isinstance(value,dict):return sum(entries(v) for v in value.values())
        return 0
    if entries(public)>980 or len(json.dumps(public,ensure_ascii=False).encode())>60000:return
    result['parameters']=candidate;result['parameters_omitted']=False


def ml_fit(p):
    import numpy as np
    import sklearn,hashlib
    df,meta,masks,inferred=frame_load(p)
    est,x,prep,labels,scaler,status=ml_fit_one(df,p)
    if status not in ('OK','RANK_DEFICIENT'): return {'status':status}
    family=p['family'];attrs=p['ml_spec']['families'][family]['attributes']
    kind='transform' if family in ('pca','standard_scaler') else 'predict'
    prediction=ml_result(est,x[:20],kind,family,labels)
    if not attrs:
        future,_,_,_=frame_load({'frame_file':p['predict_file']})
        xx,_=ml_matrix(future,p['features'],p['preprocess'],prep)
        if scaler is not None:xx=scaler.transform(xx)
        result=ml_save_table(ml_result(est,xx,'predict',family,labels),p)
        result['classes']=labels;result['features']=p['features'];result['model_persisted']=False;result['seed']=p['seed'];result['generator']='sklearn RandomState(MT19937)';result['family']=family
        return result
    tensors=[]
    for name in attrs:
        if name=='train_x_': value=x
        elif name=='train_y_': value=est._y
        else: value=np.asarray(getattr(est,name))
        value=np.asarray(value)
        if value.ndim==0:value=value.reshape(1)
        tensors.append((name,value))
    if scaler is not None:tensors.extend([('pp_mean_',scaler.mean_),('pp_scale_',scaler.scale_)])
    # The LDA SVD can retain more discriminant scalings than requested; persist only
    # the public transform's components (predictions use coef_, not scalings_).
    if family=='lda':
        r=est._max_components
        tensors=[(k,v[:,:r] if k=='scalings_' else v[:r] if k=='explained_variance_ratio_' else v) for k,v in tensors]
    elif family=='pca':r=est.n_components_
    else:r=1
    digest=hashlib.sha256(json.dumps(p.get('columns',p.get('records',{})),sort_keys=True,ensure_ascii=False).encode())
    for key in ('frame_file','csv_file'):
        if key in p:digest=hashlib.sha256(open(p[key],'rb').read())
    info={'pycli_model':'1','family':family,'sklearn_version':sklearn.__version__,'features':p['features'],'feature_dtypes':{f:'string' if f in prep['one_hot'] else 'numeric-f64' for f in p['features']},'preprocess':prep,'params':p['params'],'labels':labels,'width':x.shape[1],'train_rows':len(x),'seed':p.get('seed'),'components':int(r),'target':p.get('target',''),'train_sha256':digest.hexdigest(),'lock_id':p['lock_id']}
    metadata={k:json.dumps(v,separators=(',',':'),ensure_ascii=False) if not isinstance(v,str) else v for k,v in info.items()}
    if sum(len(k)+len(v) for k,v in metadata.items())>16384:raise BadData('model metadata exceeds 16KiB')
    result={'status':status,'rank':int(est.rank_) if family=='linear' else None,'rank_deficient':bool(est.rank_<x.shape[1]) if family=='linear' else False,'classes':labels,'features':p['features'],'family':family,'model_metadata':metadata,'outputs':save_outputs(p,tensors),'prediction_preview':ml_preview(prediction),'preview_truncated':len(x)>len(next(iter(ml_preview(prediction).values()))),'seed':p.get('seed'),'preprocess':p['preprocess']}
    ml_inline_parameters(result,tensors)
    if p.get('seed') is None:result.pop('seed')
    if p['ml_spec']['families'][family]['stochastic']:result['generator']='sklearn RandomState(MT19937)'
    return result


def ml_load(p):
    import numpy as np
    import sklearn.preprocessing
    m=p['model_info'];family=m['family'];est=ml_estimator(family,m['params'],0)
    attrs={name:np.load(p[key],allow_pickle=False) for name,key in p['model_attrs'].items()}
    for key in ('classes_','train_y_'):
        if key in attrs and (key=='classes_' or family=='knn_classifier'):attrs[key]=attrs[key].astype('int64')
    if family.startswith('knn_'):
        est.fit(attrs['train_x_'],attrs['train_y_'])
        if family=='knn_classifier' and (not np.array_equal(est.classes_,attrs['classes_']) or not np.array_equal(est.classes_,np.arange(len(m['labels'])))): raise BadData('BAD_MODEL reconstructed classes disagree with stored classes','BAD_MODEL')
    else:
        for key in p['ml_spec']['families'][family]['attributes']:
            value=attrs[key]
            if key in ('intercept_','noise_variance_') and (key=='noise_variance_' or family in ('linear','ridge','lasso','elastic_net')):value=float(value[0])
            setattr(est,key,value)
    est.n_features_in_=m['width']
    if family=='pca':est.n_components_=m['components'];est.n_samples_=m['train_rows']
    if family=='lda':est._max_components=m['components']
    if family=='kmeans':est._n_threads=1;est._n_features_out=len(est.cluster_centers_)
    if family=='standard_scaler':est.n_samples_seen_=m['train_rows']
    scaler=None
    if m['preprocess']['scale']:
        scaler=sklearn.preprocessing.StandardScaler();scaler.mean_=attrs['pp_mean_'];scaler.scale_=attrs['pp_scale_'];scaler.n_features_in_=m['width']
    # Smoke the reconstructed numeric state before processing caller rows.
    ml_result(est,np.zeros((1,m['width'])),'transform' if family in ('pca','standard_scaler') else 'predict',family,m['labels'])
    return est,scaler


def ml_predict(p):
    df,_,_,_=frame_load(p);m=p['model_info'];est,scaler=ml_load(p)
    x,_=ml_matrix(df,m['features'],p['preprocess'],m['preprocess'])
    if scaler is not None:x=scaler.transform(x)
    result=ml_save_table(ml_result(est,x,p['kind'],m['family'],m['labels']),p)
    result['version_skew_accepted']=m['sklearn_version']!='1.7.2'
    result['classes']=m['labels'];result['features']=m['features'];result['family']=m['family']
    return result


def ml_cv(p):
    import numpy as np
    import sklearn.model_selection as split,sklearn.metrics as metric,hashlib
    df,_,_,_=frame_load(p);task=p['ml_spec']['families'][p['family']]['task'];y,labels=ml_target(df,p,task)
    if task=='classification':df=df.copy();df[p['target']]=[labels[int(code)] for code in y]
    n=p['n_splits'];shuffle=p['shuffle'];seed=p['seed'];cv=p['cv'];groups=df[p['groups']].to_numpy() if cv=='group' else None
    if cv=='timeseries':splitter=split.TimeSeriesSplit(n_splits=n)
    elif cv=='group':splitter=split.GroupKFold(n_splits=n,shuffle=shuffle,random_state=seed if shuffle else None)
    elif cv=='stratified':splitter=split.StratifiedKFold(n_splits=n,shuffle=shuffle,random_state=seed if shuffle else None)
    else:splitter=split.KFold(n_splits=n,shuffle=shuffle,random_state=seed if shuffle else None)
    if cv=='group' and len(set(groups))<n:raise BadData('group CV needs at least n_splits distinct groups')
    if cv=='stratified' and max(np.bincount(y))<n:raise BadData('n_splits exceeds every class count')
    if 'roc_auc' in p['metrics'] and (len(labels)!=2 or p['family']=='linear_svc'):raise BadData('roc_auc requires binary predict_proba family')
    if 'log_loss' in p['metrics'] and p['family']=='linear_svc':raise BadData('log_loss requires predict_proba')
    folds=[]
    def undefined(i,train,message,metric_name=None,status='UNDEFINED_METRIC'):
        result={'status':status,'fold_index':i,'folds':folds,'completed_folds':len(folds),'message':message+'; consider stratified CV or shuffle','missing_classes':[v for v in labels if v not in set(df.iloc[train][p['target']])]}
        if metric_name is not None:result['metric']=metric_name
        return result
    for i,(train,test) in enumerate(splitter.split(df,y,groups)):
        if task=='classification':
            present=set(df.iloc[train][p['target']])
            if len(present)<2:return undefined(i,train,'training fold has fewer than two classes',status='INSUFFICIENT_CLASSES')
            if len(present)!=len(labels) and 'log_loss' in p['metrics']:return undefined(i,train,'training fold lacks classes','log_loss')
        est,x,prep,fold_labels,scaler,status=ml_fit_one(df.iloc[train],p)
        if status not in ('OK','RANK_DEFICIENT'):return {'status':status,'fold_index':i,'folds':folds,'completed_folds':len(folds)}
        xx,_=ml_matrix(df.iloc[test],p['features'],p['preprocess'],prep)
        if scaler is not None:xx=scaler.transform(xx)
        pred=np.asarray(ml_result(est,xx,'predict',p['family'],fold_labels)['prediction'])
        true=df.iloc[test][p['target']].to_numpy()
        scores={}
        for name in p['metrics']:
            if name=='accuracy':score=metric.accuracy_score(true,pred)
            elif name=='f1_macro':score=metric.f1_score(true,pred,average='macro',zero_division=0)
            elif name=='roc_auc':
                if len(set(true))<2:return undefined(i,train,'held-out fold has only one class','roc_auc')
                probabilities=ml_result(est,xx,'predict_proba',p['family'],fold_labels)
                score=metric.roc_auc_score((true==labels[1]).astype(int),probabilities['probability_1'])
            elif name=='log_loss':
                probabilities=ml_result(est,xx,'predict_proba',p['family'],fold_labels)
                score=metric.log_loss(true,np.column_stack(list(probabilities.values())),labels=labels)
            elif name=='mse':score=metric.mean_squared_error(true,pred)
            elif name=='mae':score=metric.mean_absolute_error(true,pred)
            else:score=metric.r2_score(true,pred,force_finite=False)
            if not np.isfinite(score):return undefined(i,train,'metric is not finite',name)
            scores[name]=float(score)
        item={'fold':i,'status':status,'train_rows':len(train),'test_rows':len(test),'scores':scores}
        if len(df)*n<=600:item['test_indices']=test.tolist()
        else:item.update(test_indices_sha256=hashlib.sha256(test.astype('<i8').tobytes()).hexdigest(),indices_omitted=True)
        folds.append(item)
    result={'status':'RANK_DEFICIENT' if any(f['status']=='RANK_DEFICIENT' for f in folds) else 'OK','completed_folds':len(folds),'folds':folds,'mean':{k:float(np.mean([f['scores'][k] for f in folds])) for k in p['metrics']},'std':{k:float(np.std([f['scores'][k] for f in folds])) for k in p['metrics']},'seed':seed,'preprocessing_fitted_per_fold':True}
    if shuffle or p['ml_spec']['families'][p['family']]['stochastic']:result['generator']='sklearn RandomState(MT19937)'
    return result


def ml_operation(op,p):
    import signal,joblib,threadpoolctl
    old=signal.getsignal(signal.SIGALRM)
    def timeout(sig,frame):raise _SmAlarm()
    signal.signal(signal.SIGALRM,timeout)
    try:
        signal.setitimer(signal.ITIMER_REAL,min(p['time_budget_s'],p['time_limit']))
        with joblib.parallel_backend('sequential',n_jobs=1),threadpoolctl.threadpool_limits(limits=1):
            result={'ml_fit':ml_fit,'ml_predict':ml_predict,'ml_cv':ml_cv}[op](p)
            result['threads']=[{'api':v['internal_api'],'num_threads':v['num_threads']} for v in threadpoolctl.threadpool_info()]
            return result
    except _SmAlarm:return {'status':'TIME_LIMIT'}
    except NumericalFailure as e:return {'status':'NUMERICAL_FAILURE','message':str(e)}
    except BadData:raise
    except ValueError as e:return {'status':'NUMERICAL_FAILURE','message':str(e)}
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
