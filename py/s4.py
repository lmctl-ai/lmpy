"""S4 fixed HiGHS and budgeted solve_ivp paths. Embedded, never imported from caller paths."""
class StepLimit(SolverStop):
    status = 'STEP_LIMIT'


def s4_payload(result, vectors, p):
    import numpy as np
    # Leave room for final state, event counts, names and endpoint additions.
    items=[(key,np.asarray(value,dtype=float)) for key,value in vectors.items()]
    if sum(value.size for _,value in items)>200000:raise OutputTooLarge('S4 output number budget exceeded')
    large=sum(value.size for _,value in items)>850 or len(codec({k:v.tolist() for k,v in items}).encode())>24000
    if p.get('force_artifact') or large:
        stored=[(key,value) for key,value in items if value.size]
        if stored:
            result['outputs']=save_outputs(p,stored)
            for key,value in items:result[key]={'tensor':key} if value.size else value.tolist()
    else:
        result.update({key:value.tolist() for key,value in items})
    if len(codec(result).encode())>32768:raise OutputTooLarge('S4 summary exceeds 32KiB')
    return result


def opt_linprog(p):
    import numpy as np
    from scipy import optimize
    started=time.monotonic();budget=min(p['time_limit'],p['time_budget_s']);deadline=started+budget
    c=array(p,'c');maximize=p['maximize'];sign=-1 if maximize else 1
    aub=array(p,'a_ub') if 'a_ub_file' in p else None; bub=array(p,'b_ub') if aub is not None else None
    aeq=array(p,'a_eq') if 'a_eq_file' in p else None; beq=array(p,'b_eq') if aeq is not None else None
    if time.monotonic()>=deadline:return {'status':'TIME_LIMIT','message':'wall budget exhausted before HiGHS','has_primal':False}
    options={'time_limit':max(1e-12,deadline-time.monotonic()),'maxiter':p['maxiter'],'mip_max_nodes':p['node_limit'],'mip_rel_gap':p['mip_rel_gap'],'threads':1,'parallel':False}
    # The pinned adapter forwards these two fixed native HiGHS controls, despite
    # warning that they are not in its public option table. No caller options pass.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message=r"Unrecognized options detected: \{'threads': 1, 'parallel': False\}.*",category=optimize.OptimizeWarning)
        answer=optimize.linprog(sign*c,A_ub=aub,b_ub=bub,A_eq=aeq,b_eq=beq,bounds=p['bounds'],integrality=p['integrality'],method='highs',options=options)
    message=answer.message.lower()
    status={0:'OK',2:'INFEASIBLE',3:'UNBOUNDED'}.get(answer.status,'NUMERICAL_FAILURE')
    if answer.status in (1,4):
        if 'time limit' in message:status='TIME_LIMIT'
        elif 'node limit' in message or any(p['integrality']) and 'solution limit' in message:status='NODE_LIMIT'
        elif 'iteration limit' in message:status='ITERATION_LIMIT'
    if status=='OK' and time.monotonic()>deadline:status='TIME_LIMIT'
    result={'status':status,'message':answer.message,'scipy_status':int(answer.status),'iterations_available':answer.get('nit',-1) is not None and answer.get('nit',-1)>=0,'has_primal':False,'highs_threads':1}
    if result['iterations_available']:result['iterations']=int(answer.nit)
    mixed=any(p['integrality'])
    if mixed:
        for name in ('mip_gap','mip_node_count','mip_dual_bound'):
            value=answer.get(name)
            if value is not None and np.isfinite(value):result[name]=float(sign*value if name=='mip_dual_bound' else value) if name!='mip_node_count' else int(value)
    x=answer.get('x')
    if x is None:return result
    x=np.asarray(x,dtype=float)
    with np.errstate(all='ignore'):
        objective=float(c@x);slack=np.asarray([]) if aub is None else bub-aub@x;eq=np.asarray([]) if aeq is None else beq-aeq@x
        lows=np.array([-np.inf if b[0] is None else b[0] for b in p['bounds']]);highs=np.array([np.inf if b[1] is None else b[1] for b in p['bounds']])
        violation=max(float(np.maximum(-slack,0).max(initial=0)),float(np.abs(eq).max(initial=0)),float(np.maximum(lows-x,0).max(initial=0)),float(np.maximum(x-highs,0).max(initial=0)))
        integrality_error=float(np.abs(x[np.asarray(p['integrality'],dtype=bool)]-np.rint(x[np.asarray(p['integrality'],dtype=bool)])).max(initial=0))
    if not np.isfinite(x).all() or not math.isfinite(objective) or not np.isfinite(slack).all() or not np.isfinite(eq).all():
        result.update(status='NUMERICAL_FAILURE',message='non-finite primal result');return result
    result.update(has_primal=True,objective=objective,primal_violation=violation,integrality_error=integrality_error,solution_kind='optimal' if status=='OK' and (not mixed or result.get('mip_gap')==0) else 'incumbent')
    scale=max(1.0,float(np.abs(bub).max(initial=0)) if bub is not None else 0,float(np.abs(beq).max(initial=0)) if beq is not None else 0)
    result['primal_feasible']=violation<=1e-7+1e-8*scale and integrality_error<=1e-6
    if not result['primal_feasible']:
        result['solution_kind']='candidate'
        if status=='OK':result.update(status='NUMERICAL_FAILURE',message='independent primal feasibility check failed')
    vectors={'x':x,'slack':slack,'equality_residual':eq}
    if not mixed and answer.status==0:
        duals={key:sign*np.asarray(answer[section]['marginals']) for key,section in (('ub','ineqlin'),('eq','eqlin'),('lower','lower'),('upper','upper'))}
        if all(np.isfinite(v).all() for v in duals.values()):
            for key,value in duals.items():vectors['dual_'+key]=value
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
            track['event_calls']+=1;return scalar(program,bind(t,y))
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
        for i,values in enumerate(answer.t_events or []):
            counts[i]=len(values)
            for t,y in zip(values,answer.y_events[i]):event_t.append(float(t));event_y.append(y);event_index.append(i)
        if answer.status==1:
            terminal=[(float(values[-1]),answer.y_events[i][-1],i) for i,values in enumerate(answer.t_events) if len(values) and (p['events'][i]['terminal'] or len(values)>=32)]
            # First terminal root along the integration direction; accepted solver step may extend beyond it.
            last_time,last_state,index=min(terminal,key=lambda x:(x[0]-p['t_span'][0])*(1 if p['t_span'][1]>p['t_span'][0] else -1))
            termination='event';status='OK' if p['events'][index]['terminal'] else 'EVENT_LIMIT'
        if not len(times) or times[-1]!=last_time:
            times=np.append(times,last_time);states=np.vstack((states.reshape(-1,n),last_state))
    except SolverStop as exc:
        status=exc.status;message=exc.message;partial=True;termination='budget' if status!='NON_FINITE' else 'nonfinite';last_time=track['time'];last_state=track['state']
        times=np.asarray([p['t_span'][0]] if last_time==p['t_span'][0] else [p['t_span'][0],last_time]);states=np.asarray([initial] if times.size==1 else [initial,last_state])
    result={'status':status,'message':message,'termination':termination,'final_time':float(last_time),'final_state':last_state.tolist(),'state_names':labels,'nfev':track['nfev'],'njev':track['njev'],'nlu':track['nlu'],'rhs_calls':track['rhs_calls'],'event_calls':track['event_calls'],'evals':guard.count,'nonfinite_evals':guard.nonfinite,'steps':track['steps'],'partial_trajectory':partial,'event_counts':counts,'endpoints_added':p['endpoints_added'],'expr_spec_sha256':digest}
    vectors={'t':times,'y':states}
    if event_t:vectors.update(event_t=np.asarray(event_t),event_y=np.asarray(event_y),event_index=np.asarray(event_index))
    return s4_payload(result,vectors,p)
