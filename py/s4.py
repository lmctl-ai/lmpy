"""S4 fixed HiGHS and budgeted solve_ivp paths. Embedded, never imported from caller paths."""
class StepLimit(SolverStop):
    status = 'STEP_LIMIT'


def s4_payload(result, vectors, p):
    import numpy as np
    # Leave room for final state, event counts, names and endpoint additions.
    items=[(key,np.asarray(value,dtype=np.int64 if key=='event_index' else float)) for key,value in vectors.items()]
    if sum(value.size for _,value in items)>200000:raise OutputTooLarge('S4 output number budget exceeded')
    large=sum(value.size for _,value in items)>850 or len(codec({k:v.tolist() for k,v in items}).encode())>24000
    result['outputs_in_artifact']=False
    if large and not p.get('force_artifact'):raise OutputTooLarge('use the _to operation with out for large S4 results')
    if p.get('force_artifact'):
        stored=[(key,value) for key,value in items if value.size]
        if stored:
            result['outputs']=save_outputs(p,stored);result['outputs_in_artifact']=True
            for key,value in items:result[key]={'tensor':key} if value.size else value.tolist()
    else:
        result.update({key:value.tolist() for key,value in items})
    if len(codec(result).encode())>32768:raise OutputTooLarge('S4 summary exceeds 32KiB')
    return result


def highs_status(message):
    # The pinned public adapter preserves the native enum in this fixed suffix.
    # SciPy status 2 alone is ambiguous between model error and infeasibility.
    try:code=int(message.split('HiGHS Status ',1)[1].split(':',1)[0])
    except (IndexError,ValueError):return 'NUMERICAL_FAILURE',None
    return {2:'MODEL_ERROR',7:'OK',8:'INFEASIBLE',9:'INFEASIBLE_OR_UNBOUNDED',10:'UNBOUNDED',13:'TIME_LIMIT',14:'ITERATION_LIMIT',16:'NODE_LIMIT'}.get(code,'NUMERICAL_FAILURE'),code


def lp_certificate(c,aub,bub,aeq,beq,bounds,x,duals=None):
    """Relative residuals and KKT certificate in ORIGINAL units, no unit-sized floor."""
    import numpy as np
    wide=np.longdouble;z=np.asarray(x,dtype=wide);cost=np.asarray(c,dtype=wide)
    def error(value,scale):
        value=np.asarray(value,dtype=wide);scale=np.asarray(scale,dtype=wide)
        ratio=np.abs(value)/np.maximum(scale,np.finfo(wide).tiny)
        return float(np.max(ratio,initial=0)) if np.isfinite(ratio).all() else math.inf
    primal=0.;raw=0.;products={};scales={}
    for key,a,b in [('ub',aub,bub),('eq',aeq,beq)]:
        if a is None:continue
        a=np.asarray(a,dtype=wide);b=np.asarray(b,dtype=wide);product=a@z;slack=b-product;scale=np.abs(b)+np.abs(a)@np.abs(z)
        violation=np.maximum(-slack,0) if key=='ub' else np.abs(slack)
        primal=max(primal,error(violation,scale));raw=max(raw,float(np.max(violation,initial=0)));products[key]=slack;scales[key]=scale
    for index,key in [(0,'lower'),(1,'upper')]:
        mask=np.asarray([b[index] is not None for b in bounds]);values=np.asarray([b[index] if b[index] is not None else 0 for b in bounds],dtype=wide)
        slack=z-values if index==0 else values-z;violation=np.maximum(-slack[mask],0)
        primal=max(primal,error(violation,np.abs(values[mask])+np.abs(z[mask])));raw=max(raw,float(np.max(violation,initial=0)))
        products[key]=slack;scales[key]=np.abs(values)+np.abs(z)
    result={'primal_relative_violation':primal,'primal_violation':raw}
    if duals is None:return result
    if not all(np.isfinite(v).all() for v in duals.values()):return dict(result,dual_relative_violation=math.inf,relative_duality_gap=math.inf,dual_feasible=False)
    d={k:np.asarray(v,dtype=wide) for k,v in duals.items()};station=cost.copy();station_scale=np.abs(cost);dual_objective=wide(0);gap_scale=np.sum(np.abs(cost*z));dual_error=0.
    for key,a,b in [('ub',aub,bub),('eq',aeq,beq)]:
        if a is None:continue
        a=np.asarray(a,dtype=wide);b=np.asarray(b,dtype=wide);station-=a.T@d[key];station_scale+=np.abs(a).T@np.abs(d[key]);dual_objective+=b@d[key];gap_scale+=np.sum(np.abs(b*d[key]))
        if key=='ub':
            if np.any(d[key]>0):dual_error=math.inf
            dual_error=max(dual_error,error(d[key]*products[key],np.abs(d[key])*scales[key]))
    for index,key in [(0,'lower'),(1,'upper')]:
        mask=np.asarray([b[index] is not None for b in bounds]);values=np.asarray([b[index] if b[index] is not None else 0 for b in bounds],dtype=wide)
        station-=d[key];station_scale+=np.abs(d[key]);dual_objective+=values@d[key];gap_scale+=np.sum(np.abs(values*d[key]))
        if np.any(d[key][~mask]!=0) or np.any(d[key]<0 if index==0 else d[key]>0):dual_error=math.inf
        dual_error=max(dual_error,error(d[key][mask]*products[key][mask],np.abs(d[key][mask])*scales[key][mask]))
    dual_error=max(dual_error,error(station,station_scale));gap=error(cost@z-dual_objective,gap_scale)
    return dict(result,dual_relative_violation=dual_error,relative_duality_gap=gap,dual_feasible=dual_error<=1e-7 and gap<=1e-7)


def opt_linprog(p):
    import numpy as np
    from scipy import optimize
    started=time.monotonic();budget=min(p['time_limit'],p['time_budget_s']);deadline=started+budget
    c=array(p,'c');maximize=p['maximize'];sign=-1 if maximize else 1;cost_scale=float(np.abs(c).max()) or 1.
    aub=array(p,'a_ub') if 'a_ub_file' in p else None;bub=array(p,'b_ub') if aub is not None else None
    aeq=array(p,'a_eq') if 'a_eq_file' in p else None;beq=array(p,'b_eq') if aeq is not None else None
    def scaled(a,b):
        if a is None:return None,None,np.asarray([])
        scale=np.max(np.abs(a),axis=1);scale=np.where(scale==0,1,scale)
        return a/scale[:,None],b/scale,scale
    ua,ub,us=scaled(aub,bub);ea,eb,es=scaled(aeq,beq)
    if time.monotonic()>=deadline:return {'status':'TIME_LIMIT','message':'wall budget exhausted before HiGHS','has_primal':False,'outputs_in_artifact':False}
    options={'time_limit':max(1e-12,deadline-time.monotonic()),'maxiter':p['maxiter'],'mip_max_nodes':p['node_limit'],'mip_rel_gap':p['mip_rel_gap'],'threads':1,'parallel':False,'small_matrix_value':1e-12,'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9,'ipm_optimality_tolerance':1e-10}
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message=r"Unrecognized options detected: \{'threads': 1, 'parallel': False, 'small_matrix_value': 1e-12\}.*",category=optimize.OptimizeWarning)
        answer=optimize.linprog(sign*c/cost_scale,A_ub=ua,b_ub=ub,A_eq=ea,b_eq=eb,bounds=p['bounds'],integrality=p['integrality'],method='highs',options=options)
    status,model_status=highs_status(answer.message)
    if status=='OK' and time.monotonic()>deadline:status='TIME_LIMIT'
    result={'status':status,'message':answer.message,'scipy_status':int(answer.status),'highs_model_status':model_status,'iterations_available':answer.get('nit',-1) is not None and answer.get('nit',-1)>=0,'has_primal':False,'highs_threads':1,'outputs_in_artifact':False,'row_scaling':True,'certificate_tolerance':1e-7}
    if result['iterations_available']:result['iterations']=int(answer.nit)
    mixed=any(p['integrality'])
    if mixed:
        for name in ('mip_gap','mip_node_count','mip_dual_bound'):
            value=answer.get(name)
            if value is not None and np.isfinite(value):result[name]=float(sign*cost_scale*value if name=='mip_dual_bound' else value) if name!='mip_node_count' else int(value)
    x=answer.get('x')
    if x is None:return result
    x=np.asarray(x,dtype=float).copy();integer_mask=np.asarray(p['integrality'],dtype=bool)
    if not np.isfinite(x).all():result.update(status='NUMERICAL_FAILURE',validation_message='non-finite primal result');return result
    integrality_error=float(np.abs(x[integer_mask]-np.rint(x[integer_mask])).max(initial=0))
    if integrality_error<=1e-6:x[integer_mask]=np.rint(x[integer_mask])
    x[x==0]=0.
    with np.errstate(all='ignore'):
        objective=float(c@x);slack=np.asarray([]) if aub is None else bub-aub@x;eq=np.asarray([]) if aeq is None else beq-aeq@x
    if not math.isfinite(objective) or not np.isfinite(slack).all() or not np.isfinite(eq).all():
        result.update(status='NUMERICAL_FAILURE',validation_message='non-finite primal result');return result
    duals=None
    if not mixed and model_status==7:
        duals={key:cost_scale*np.asarray(answer[section]['marginals'])/(us if key=='ub' else es if key=='eq' else 1) for key,section in (('ub','ineqlin'),('eq','eqlin'),('lower','lower'),('upper','upper'))}
    with np.errstate(all='ignore'):certificate=lp_certificate(sign*c,aub,bub,aeq,beq,p['bounds'],x,duals)
    result.update(certificate);result.update(has_primal=True,objective=objective,integrality_error=integrality_error,primal_feasible=certificate['primal_relative_violation']<=1e-7 and integrality_error<=1e-6,solution_kind='optimal' if status=='OK' and (not mixed or result.get('mip_gap')==0) else 'incumbent',optimality_verified=bool(duals is not None and certificate['dual_feasible'] and certificate['primal_relative_violation']<=1e-7))
    if not result['primal_feasible'] or duals is not None and not certificate['dual_feasible']:
        result.update(status='NUMERICAL_FAILURE',solution_kind='candidate',validation_message='original-data primal/KKT certificate failed')
    vectors={'x':x,'slack':slack,'equality_residual':eq}
    if duals is not None and result['status']!='NUMERICAL_FAILURE':
        for key,value in duals.items():vectors['dual_'+key]=sign*value
        result['dual_keys']=['ub','eq','lower','upper']
    return s4_payload(result,vectors,p)


def ode_solve(p):
    import numpy as np
    from scipy import integrate
    spec,digest=load_spec(p);tables=(function_table(np),operator_table(np));check_bindings(spec,tables[0])
    initial=np.asarray(p['initial'],dtype=float);n=initial.size;labels=p['state_names'];names={'y','t'}|set(labels)|set(p['params'])
    guard=Guard(p['max_evals'],min(p['time_limit'],p['time_budget_s']))
    programs=[compile_expression(ast,names,{'y':n},spec,tables) for ast in p['rhs_asts']]
    event_programs=[compile_expression(e['ast'],names,{'y':n},spec,tables) for e in p['events']]
    track={'rhs_calls':0,'event_calls':0,'steps':0,'time':float(p['t_span'][0]),'state':initial.copy(),'nfev':0,'njev':0,'nlu':0}
    def bind(t,y):
        if not np.isfinite(y).all():
            guard.tick();guard.nonfinite+=1;raise NonFinite('non-finite trial state')
        env=dict(p['params']);env.update(t=np.float64(t),y=y);env.update({name:y[i] for i,name in enumerate(labels)});return env
    def scalar(program,env):
        guard.tick()
        with np.errstate(all='ignore'):value=evaluate(program,env)
        if np.ndim(value)!=0:raise ExprMismatch('ODE expression must be scalar')
        value=float(value)
        if not math.isfinite(value):guard.nonfinite+=1;raise NonFinite('non-finite RHS/event value')
        return value
    def rhs(t,y):
        track['rhs_calls']+=1;env=bind(t,y)
        return np.asarray([scalar(program,env) for program in programs])
    callbacks=[]
    for i,program in enumerate(event_programs):
        def event(t,y,program=program):
            track['event_calls']+=1;value=scalar(program,bind(t,y))
            # A finite zero exactly at the initial point has no preceding crossing.
            # The internal NaN suppresses only that comparison in solve_ivp; actual
            # nonfinite caller expressions are rejected by scalar() above.
            return np.nan if t==p['t_span'][0] and value==0 else value
        event.direction=p['events'][i]['direction'];event.terminal=1 if p['events'][i]['terminal'] else 32
        callbacks.append(event)
    classes={'RK45':integrate.RK45,'RK23':integrate.RK23,'DOP853':integrate.DOP853,'Radau':integrate.Radau,'BDF':integrate.BDF,'LSODA':integrate.LSODA}
    base=classes[p['method']]
    class BoundedSolver(base):
        def __init__(self,*args,**kwargs):
            try:super().__init__(*args,**kwargs)
            finally:track.update(nfev=self.__dict__.get('nfev',0),njev=self.__dict__.get('njev',0),nlu=self.__dict__.get('nlu',0))
        def step(self):
            if track['steps']>=p['max_steps']:raise StepLimit('accepted-step cap reached')
            try:message=super().step()
            finally:track.update(nfev=self.nfev,njev=self.njev,nlu=self.nlu)
            if not np.isfinite(self.y).all():guard.nonfinite+=1;raise NonFinite('non-finite accepted state')
            if self.status!='failed':track.update(steps=track['steps']+1,time=float(self.t),state=self.y.copy())
            return message
    event_t=[];event_y=[];event_index=[];counts=[0]*len(callbacks);termination='end';partial=False
    try:
        with np.errstate(all='ignore'):
            answer=integrate.solve_ivp(rhs,p['t_span'],initial,method=BoundedSolver,t_eval=p['grid'],rtol=p['rtol'],atol=p['atol'],max_step=p.get('max_step',np.inf),events=callbacks or None,dense_output=False)
        times=np.asarray(answer.t);states=np.asarray(answer.y).T
        status='OK' if answer.success else 'STEP_SIZE_TOO_SMALL' if 'step size' in answer.message.lower() else 'NO_CONVERGENCE'
        message=answer.message;last_time=track['time'];last_state=track['state']
        if not answer.success:termination='failed';partial=True
        for i,values in enumerate(answer.t_events or []):
            counts[i]=len(values)
            for t,y in zip(values,answer.y_events[i]):event_t.append(float(t));event_y.append(y);event_index.append(i)
        if answer.status==1:
            terminal=[(float(values[-1]),answer.y_events[i][-1],i) for i,values in enumerate(answer.t_events) if len(values) and (p['events'][i]['terminal'] or len(values)>=32)]
            # First terminal root along the integration direction; accepted solver step may extend beyond it.
            last_time,last_state,index=min(terminal,key=lambda x:(x[0]-p['t_span'][0])*(1 if p['t_span'][1]>p['t_span'][0] else -1))
            termination='event';status='OK' if p['events'][index]['terminal'] else 'EVENT_LIMIT'
            if status!='OK':termination='failed';partial=True
        if not len(times) or times[-1]!=last_time:
            times=np.append(times,last_time);states=np.vstack((states.reshape(-1,n),last_state))
    except SolverStop as exc:
        status=exc.status;message=exc.message;partial=True;termination='failed';last_time=track['time'];last_state=track['state']
        times=np.asarray([p['t_span'][0]] if last_time==p['t_span'][0] else [p['t_span'][0],last_time]);states=np.asarray([initial] if times.size==1 else [initial,last_state])
    result={'status':status,'message':message,'termination':termination,'final_time':float(last_time),'final_state':last_state.tolist(),'state_names':labels,'nfev':track['nfev'],'njev':track['njev'],'nlu':track['nlu'],'rhs_calls':track['rhs_calls'],'event_calls':track['event_calls'],'evals':guard.count,'nonfinite_evals':guard.nonfinite,'steps':track['steps'],'partial_trajectory':partial,'event_counts':counts,'effective_output_points':len(p['grid']),'endpoints_added':p['endpoints_added'],'expr_spec_sha256':digest}
    vectors={'t':times,'y':states}
    if event_t:vectors.update(event_t=np.asarray(event_t),event_y=np.asarray(event_y),event_index=np.asarray(event_index,dtype=np.int64))
    return s4_payload(result,vectors,p)
