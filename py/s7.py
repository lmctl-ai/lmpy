"""S7 interpolation, signal processing and spatial operations on fixed scipy.interpolate / scipy.signal / scipy.spatial paths.
Every enum, option string and cap was validated by the C++ front end; the checks here repeat the ones a corrupted request could break.
Assembled after ml.py (shares codec/array/save_outputs/BadData/OutputTooLarge and the SIGALRM budget helpers of sm.py)."""
S7_INLINE = 900
S7_ARTIFACT_NUMBERS = 1900000  # the C++ side refuses worker outputs above 2M numbers; fail cleanly first
S7_RBF_POINTS, S7_RBF_DIM, S7_ORDER, S7_NPERSEG, S7_NFFT, S7_KNN_DIM, S7_KNN_K, S7_DIST_PAIRS = 5000, 8, 20, 65536, 131072, 32, 1000, 1000000
S7_HULL_CAP = {'convex_hull': {2: 20000, 3: 20000, 4: 800, 5: 400, 6: 100}, 'delaunay': {2: 20000, 3: 800, 4: 300, 5: 100, 6: 50}, 'voronoi': {2: 20000, 3: 800, 4: 300, 5: 100, 6: 50}}
S7_WINDOWS = ('hann', 'hamming', 'blackman', 'boxcar', 'bartlett', 'flattop', 'parzen', 'bohman', 'blackmanharris', 'nuttall', 'barthann', 'cosine', 'triang')


def s7_vector(np, p, key):
    return np.asarray(array(p, key), dtype=float)


def s7_emit(np, p, result, named, op, nan_masks=None):
    """Inline arrays (<= 900 numbers in total) or, for the _to twins, raw tensors for the C++ side to store; empty arrays are never stored."""
    if p.get('output_prefix'):
        stored = [(key, np.asarray(value)) for key, value in named if np.size(value)]
        total = sum(int(v.size) for _, v in stored)
        if total > S7_ARTIFACT_NUMBERS:
            raise OutputTooLarge('the %s result holds %d numbers (artifact limit %d): use fewer points or a smaller problem' % (op, total, S7_ARTIFACT_NUMBERS))
        result['empty_outputs'] = [key for key, value in named if not np.size(value)]
        if stored:
            result['outputs'] = save_outputs(p, stored, nan_masks=nan_masks)
        return result
    total = sum(int(np.size(value)) for _, value in named)
    if total > S7_INLINE:
        raise OutputTooLarge('the %s result holds %d numbers (inline limit 1000 including scalars); use %s_to' % (op, total, op))
    for key, value in named:
        result[key] = np.asarray(value).tolist()
    return result


def s7_cap(ok, what):
    """The C++ front end enforces every cap before spawn; a request that got here without that check is refused again, not computed."""
    if not ok:
        raise BadData('%s exceeds the operation cap (this request bypassed the front-end validation)' % what, 'INPUT_TOO_LARGE')


def s7_guard(p, body):
    """Cooperative wall budget (SIGALRM) around one library call; None means the budget expired."""
    return sm_fit_guard(dict(p,time_budget_s=min(p.get('time_budget_s',40),p.get('time_limit',10))), body)


def interp_eval(p):
    import numpy as np
    from scipy import interpolate as itp
    kind, nu, extrapolate = p['kind'], p['nu'], p['extrapolate']
    y = s7_vector(np, p, 'y')
    x = s7_vector(np, p, 'x')
    n = len(y)
    s7_cap(kind != 'rbf' or (n <= S7_RBF_POINTS and x.size // max(n, 1) <= S7_RBF_DIM), 'rbf nodes x dimensions')
    q = s7_vector(np, p, 'x_new') if 'x_new_file' in p else None
    result = {'status': 'OK', 'kind': kind, 'n_nodes': n, 'nu': nu, 'extrapolate': extrapolate}
    lo = hi = None
    if kind != 'rbf':
        if x.ndim != 1 or len(x) != n or not (np.diff(x) > 0).all():
            raise BadData('x must be a strictly increasing vector with one value per y', 'NOT_INCREASING')
        lo, hi = float(x[0]), float(x[-1])

    def build():
        if kind == 'linear':
            spl = itp.make_interp_spline(x, y, k=1)
            return (lambda t, d: spl(t, nu=d, extrapolate=True)), (lambda a, b: float(spl.integrate(a, b, extrapolate=True)))
        if kind == 'cubic':
            cs = itp.CubicSpline(x, y, bc_type=p['bc_type'], extrapolate=True)
            return (lambda t, d: cs(t, d)), (lambda a, b: float(cs.integrate(a, b, extrapolate=True)))
        if kind == 'pchip':
            f = itp.PchipInterpolator(x, y, extrapolate=True)
            return (lambda t, d: f(t, d)), (lambda a, b: float(f.integrate(a, b, extrapolate=True)))
        if kind in ('akima', 'makima'):
            f = itp.Akima1DInterpolator(x, y, method=kind)
            return (lambda t, d: f(t, d, extrapolate=True)), (lambda a, b: float(f.integrate(a, b, extrapolate=True)))
        spl = itp.make_smoothing_spline(x, y, lam=p.get('lam'))
        return (lambda t, d: spl(t, nu=d, extrapolate=True)), (lambda a, b: float(spl.integrate(a, b, extrapolate=True)))

    if kind == 'rbf':
        pts = x.reshape(n, -1)
        options = {'kernel': p['kernel'], 'epsilon': p['epsilon'], 'smoothing': p['smoothing']}
        if 'degree' in p:
            options['degree'] = p['degree']

        def body():
            try:
                rbf = itp.RBFInterpolator(pts, y, **options)
                fit = float(np.max(np.abs(rbf(pts) - y))) if p['smoothing'] == 0 else None
                out = rbf(q.reshape(len(q), -1)) if q is not None else None
            except (ValueError,np.linalg.LinAlgError):
                return 'SINGULAR'
            return rbf, fit, out
        got = s7_guard(p, body)
        if got is None:
            return {'status': 'TIME_LIMIT', 'kind': kind, 'n_nodes': n}
        if got == 'SINGULAR':
            return {'status': 'SINGULAR', 'kind': kind, 'n_nodes': n, 'message': 'the RBF system is singular (duplicate or degenerate points for this kernel and degree)'}
        _, fit, values = got
        scale = max(1.0, float(np.max(np.abs(y))))
        if fit is not None:
            result['interpolation_error'] = fit
            if not fit <= 1e-6 * scale:
                result.update(status='ILL_CONDITIONED', message='the RBF system is too ill-conditioned: the fit misses its own nodes by %.3g' % fit)
                return result
        if values is not None:
            lo_box, hi_box = pts.min(axis=0), pts.max(axis=0)
            outside = ~((q.reshape(len(q), -1) >= lo_box) & (q.reshape(len(q), -1) <= hi_box)).all(axis=1)
            result['n_extrapolated'] = int(outside.sum())
            if not np.isfinite(values[~outside] if extrapolate=='nan' else values).all():return dict(result,status='NUMERICAL_FAILURE')
            if extrapolate == 'nan':
                values = np.where(outside, np.nan, values)
            return s7_emit(np, p, result, [('values', values)], 'interp_eval', {'values':outside} if extrapolate=='nan' else None)
        return result

    def body():
        f, integral = build()
        out = {}
        if q is not None:
            inside = (q >= lo) & (q <= hi)
            with np.errstate(all='ignore'):
                values = np.asarray(f(q, nu), dtype=float)
            out['values'], out['outside'] = values, ~inside
        if 'integrate' in p:
            a, b = p['integrate']
            out['integral'] = integral(a, b)
        return out
    try:
        got = s7_guard(p, body)
    except ValueError as exc:
        raise BadData('the interpolant could not be built: %s' % str(exc)[:200], 'BAD_DATA')
    if got is None:
        return {'status': 'TIME_LIMIT', 'kind': kind, 'n_nodes': n}
    named = []
    if 'values' in got:
        values, outside = got['values'], got['outside']
        result['n_extrapolated'] = int(outside.sum())
        if not np.isfinite(values[~outside] if extrapolate=='nan' else values).all():
            result['status'] = 'NUMERICAL_FAILURE'
            return result
        if extrapolate=='nan':values=np.where(outside,np.nan,values)
        named.append(('values', values))
    if 'integral' in got:
        if not math.isfinite(got['integral']):
            result['status'] = 'NUMERICAL_FAILURE'
            return result
        result['integral'] = got['integral']
    return s7_emit(np, p, result, named, 'interp_eval', {'values':got['outside']} if extrapolate=='nan' and 'values' in got else None)


def sig_filter(p):
    import numpy as np
    from scipy import signal
    family, order, btype = p['family'], p['order'], p['btype']
    s7_cap(1 <= order <= S7_ORDER, 'filter order')
    wn = p['wn']
    kwargs = {'rp': p.get('rp'), 'rs': p.get('rs'), 'btype': btype, 'analog': False, 'ftype': family, 'output': 'sos'}
    if p.get('fs') is not None:
        kwargs['fs'] = p['fs']
    with np.errstate(all='ignore'):
        sos = np.asarray(signal.iirfilter(order, wn, **kwargs), dtype=float)
    result = {'status': 'OK', 'family': family, 'order': order, 'btype': btype, 'n_sections': int(sos.shape[0]), 'sos': sos.tolist()}
    if not np.isfinite(sos).all():
        return {'status': 'NUMERICAL_FAILURE', 'family': family, 'message': 'the design produced non-finite second-order sections'}
    radius = 0.0
    for row in sos:
        a,b,c=np.asarray(row[3:6],dtype=np.longdouble)
        disc=b*b-4*a*c
        if disc<0:section=np.sqrt(c/a)
        else:
            q=-np.longdouble(.5)*(b+np.copysign(np.sqrt(disc),b))
            section=max(abs(q/a),abs(c/q)) if q else np.longdouble(0)
        radius=max(radius,float(section))
    result['max_pole_radius'] = radius
    result['stable'] = radius < 1.0
    # Stable but resonant: poles within 1e-6 of the unit circle ring for more than a million samples; flagged, not refused.
    result['marginal'] = bool(radius >= 1.0 - 1e-6)
    if not result['stable']:
        result['status'] = 'UNSTABLE'
        result['message'] = 'a pole lies on or outside the unit circle: the filter is not applied (lower the order or widen the band)'
        return result
    if 'freqs_file' in p:
        f = s7_vector(np, p, 'freqs')
        worn = f * math.pi if p.get('fs') is None else f
        with np.errstate(all='ignore'):
            _, h = signal.sosfreqz(sos, worN=worn, fs=2 * math.pi if p.get('fs') is None else p['fs'])
        result['magnitude'] = np.abs(h).tolist()
        result['phase'] = np.angle(h).tolist()
    named = []
    if 'x_file' in p:
        x = s7_vector(np, p, 'x')
        with np.errstate(all='ignore'):
            y = signal.sosfiltfilt(sos, x) if p['mode'] == 'sosfiltfilt' else signal.sosfilt(sos, x)
        if not np.isfinite(y).all():
            return {'status': 'NUMERICAL_FAILURE', 'family': family, 'message': 'the filtered signal is not finite'}
        result['mode'] = p['mode']
        named.append(('y', y))
    return s7_emit(np, p, result, named, 'sig_filter') if named else result


def sig_spectrum(p):
    import numpy as np
    from scipy import signal
    x = s7_vector(np, p, 'x')
    method, fs = p['method'], p['fs']
    window = p.get('window') or ('boxcar' if method == 'periodogram' else 'hann')
    detrend = False if p['detrend'] == 'none' else p['detrend']
    n = len(x)
    nperseg = p.get('nperseg')
    nfft = p.get('nfft')
    s7_cap((nperseg or 0) <= S7_NPERSEG and (nfft or 0) <= S7_NFFT, 'nperseg / nfft')
    common = {'window': window, 'detrend': detrend}
    with np.errstate(all='ignore'):
        if method == 'periodogram':
            f, pxx = signal.periodogram(x, fs=fs, nfft=nfft, scaling=p['scaling'], **common)
            effective, segments = nfft or n, 1
        else:
            overlap = p['noverlap']
            effective, segments = nfft or nperseg, 1 + (n - nperseg) // (nperseg - overlap)
            spectral = dict(nperseg=nperseg, noverlap=overlap, nfft=nfft, **common)
            if method == 'welch':
                f, pxx = signal.welch(x, fs=fs, scaling=p['scaling'], average=p['average'], **spectral)
            else:
                y = s7_vector(np, p, 'y')
                if method == 'csd':
                    f, pxx = signal.csd(x, y, fs=fs, scaling=p['scaling'], average=p['average'], **spectral)
                else:
                    f, pxx = signal.coherence(x, y, fs=fs, **spectral)
    result = {'status': 'OK', 'method': method, 'window': window, 'n_samples': n, 'n_segments': int(segments), 'n_bins': len(f), 'df': fs / effective}
    named = [('f', f)]
    if method == 'csd':
        pxx = np.asarray(pxx, dtype=complex)
        magnitude = np.abs(pxx)
        named += [('pxy_re', pxx.real), ('pxy_im', pxx.imag)]
    elif method == 'coherence':
        magnitude = np.asarray(pxx, dtype=float)
        named.append(('cxy', magnitude))
    else:
        magnitude = np.asarray(pxx, dtype=float)
        named.append(('pxx', magnitude))
        # Rectangle-rule integral of the one-sided estimate: the Parseval quantity (mean square of the detrended signal for boxcar windows).
        window_values=signal.get_window(window,n if method=='periodogram' else nperseg)
        enbw=fs*float(np.sum(window_values**2))/float(abs(np.sum(window_values))**2)
        result['total_power']=float(np.sum(magnitude)*result['df']/(enbw if p['scaling']=='spectrum' else 1.0))
        result['enbw']=enbw
    if not all(np.isfinite(np.asarray(v)).all() for _, v in named):
        return {'status': 'NUMERICAL_FAILURE', 'method': method, 'message': 'the spectral estimate is not finite'}
    k = int(np.argmax(magnitude))
    result['peak_index'], result['peak_freq'] = k, float(f[k])
    return s7_emit(np, p, result, named, 'sig_spectrum')


def sig_peaks(p):
    import numpy as np
    from scipy import signal
    x=s7_vector(np,p,'x')
    cheap={key:(tuple(p[key]) if isinstance(p[key],list) else p[key]) for key in ('height','threshold','distance') if p.get(key) is not None}
    peaks,_=signal.find_peaks(x,**cheap)
    count=len(peaks)
    if not p.get('output_prefix') and 4*count>S7_INLINE:raise OutputTooLarge('peak candidates exceed inline output budget; use sig_peaks_to')
    if 4*count>S7_ARTIFACT_NUMBERS:raise OutputTooLarge('peak candidates exceed artifact output budget')
    wlen=p.get('wlen');used=min(len(x),int(wlen)|1) if wlen is not None else len(x)
    if count*used>20000000:raise BadData('prominence search exceeds 20000000 candidate*window work; specify a smaller wlen','BAD_PARAM')
    if count:
        prominences,left,right=signal.peak_prominences(x,peaks,wlen=wlen)
        widths=signal.peak_widths(x,peaks,rel_height=p['rel_height'],prominence_data=(prominences,left,right))[0]
        keep=np.ones(count,dtype=bool)
        for key,values in [('prominence',prominences),('width',widths)]:
            bounds=p.get(key)
            if bounds is None:continue
            lo,hi=bounds if isinstance(bounds,list) else (bounds,None)
            if lo is not None:keep &= values>=lo
            if hi is not None:keep &= values<=hi
        peaks,prominences,widths=peaks[keep],prominences[keep],widths[keep]
    else:prominences=widths=np.zeros(0)
    result={'status':'OK','n_peaks':len(peaks),'n_candidates':count,'n_samples':len(x),'rel_height':p['rel_height'],'wlen_used':used}
    if not len(peaks) and p.get('output_prefix'):return dict(result,status='NO_PEAKS')
    return s7_emit(np,p,result,[('indices',peaks.astype(np.int64)),('heights',x[peaks]),('prominences',prominences),('widths',widths)],'sig_peaks')


def s7_stable_distances(np,points,point,norm):
    delta=np.abs(points.astype(np.longdouble)-np.asarray(point,dtype=np.longdouble))
    maximum=delta.max(axis=-1)
    if norm==np.inf:out=maximum
    else:
        scaled=delta/np.where(maximum==0,1,maximum)[...,None]
        out=maximum*np.sum(scaled**norm,axis=-1)**(1/norm)
    out=np.asarray(out,dtype=float)
    if not np.isfinite(out).all():raise NumericalFailure('distances exceed finite binary64 range')
    return out


def spatial_knn(p):
    import numpy as np
    from scipy.spatial import cKDTree
    points=s7_vector(np,p,'points');q=s7_vector(np,p,'queries') if 'queries_file' in p else points
    n,m=len(points),len(q);norm=np.inf if p['p']=='inf' else float(p['p']);exclude=p['exclude_self']
    s7_cap(points.shape[1]<=S7_KNN_DIM and p.get('k',1)<=S7_KNN_K,'knn dimensions / k')
    brute=norm>2 and norm!=np.inf
    if brute:s7_cap(n*m*points.shape[1]<=20000000,'stable p-norm work')
    tree=None;scale=1.
    if not brute:
        scale=max(float(np.max(np.abs(points))),float(np.max(np.abs(q)))) or 1.
        pts=points/scale;qq=q/scale
        # Refuse unsafe dynamic ranges rather than merge points or underflow powers.
        for original,scaled in [(points,pts),(q,qq)]:
            nz=original!=0
            if (scaled[nz]==0).any() or (np.abs(scaled[nz])<1e-140).any():raise NumericalFailure('unsafe dynamic range for exact tree search')
            for j in range(original.shape[1]):
                if len(np.unique(original[:,j]))!=len(np.unique(scaled[:,j])):raise NumericalFailure('tree scaling loses distinct coordinates')
        tree=cKDTree(pts)
    result={'status':'OK','n_points':n,'n_queries':m,'p':p['p'],'search':'stable_brute' if brute else 'scaled_tree'}
    def nearest(row,want,radius=np.inf):
        if brute:
            distances=s7_stable_distances(np,points,q[row],norm);ids=np.arange(n)
            if exclude:distances[row]=np.inf
            ids=ids[distances<=radius];order=np.lexsort((ids,distances[ids]));ids=ids[order[:want]]
            return distances[ids],ids
        upper=np.nextafter(radius/scale,np.inf) if math.isfinite(radius) else np.inf
        distances,ids=getattr(tree,'qu'+'ery')(qq[row],k=want,p=norm,distance_upper_bound=upper,workers=1)
        distances,ids=np.atleast_1d(distances),np.atleast_1d(ids)
        valid=np.isfinite(distances)&(ids>=0)&(ids<n)
        ids=ids[valid]
        if exclude:ids=ids[ids!=row]
        actual=s7_stable_distances(np,points[ids],q[row],norm) if len(ids) else np.zeros(0)
        valid=actual<=radius;ids,actual=ids[valid],actual[valid];order=np.lexsort((ids,actual))
        return actual[order],ids[order]
    if 'k' in p:
        k=p['k'];s7_cap(2*m*(k+int(exclude))<=S7_ARTIFACT_NUMBERS,'knn query allocation')
        all_i=np.empty((m,k),dtype=np.int64);all_d=np.empty((m,k))
        if not brute:
            want=k+int(exclude);batch=max(1,min(1024,16384//(want*points.shape[1])))
            for start in range(0,m,batch):
                end=min(m,start+batch)
                raw,idx=getattr(tree,'qu'+'ery')(qq[start:end],k=want,p=norm,workers=1)
                raw,idx=raw.reshape(-1,want),idx.reshape(-1,want)
                if not (np.isfinite(raw).all() and ((idx>=0)&(idx<n)).all()):raise NumericalFailure('tree returned missing/out-of-range nearest indices')
                dist=s7_stable_distances(np,points[idx],q[start:end,None,:],norm)
                if exclude:
                    own=idx==np.arange(start,end)[:,None];keep=~own;keep[~own.any(axis=1),-1]=False
                    idx,dist=idx[keep].reshape(-1,k),dist[keep].reshape(-1,k)
                order=np.lexsort((idx,dist),axis=1)
                all_i[start:end],all_d[start:end]=np.take_along_axis(idx,order,1),np.take_along_axis(dist,order,1)
        else:
            for i in range(m):
                dist,idx=nearest(i,k)
                if len(idx)<k:raise NumericalFailure('missing nearest indices')
                all_i[i],all_d[i]=idx[:k],dist[:k]
        return s7_emit(np,p,dict(result,k=k),[('indices',all_i),('distances',all_d)],'spatial_knn')
    radius=p['radius'];limit=min(p['max_pairs'],(S7_ARTIFACT_NUMBERS-m)//2)
    if not p.get('output_prefix'):limit=min(limit,(S7_INLINE-m)//2)
    counts=np.empty(m,dtype=np.int64);all_i=[];all_d=[];total=0
    for i in range(m):
        room=limit-total
        if room<0:raise OutputTooLarge('radius output exceeds pair budget')
        want=min(n,room+1+int(exclude)) if brute else min(n,32,room+1+int(exclude))
        while True:
            dist,idx=nearest(i,want,radius)
            if len(idx)>room:raise OutputTooLarge('radius exceeds max_pairs/output budget (stopped early)')
            if brute or want==n or len(idx)<want-int(exclude):break
            want=min(n,want*2,room+1+int(exclude))
        counts[i]=len(idx);total+=len(idx);all_i.append(idx);all_d.append(dist)
    result.update(radius=radius,n_pairs=total)
    return s7_emit(np,p,result,[('counts',counts),('indices',np.concatenate(all_i).astype(np.int64)),('distances',np.concatenate(all_d))],'spatial_knn')


def spatial_hull(p):
    import numpy as np
    from scipy.spatial import ConvexHull, Delaunay, Voronoi, QhullError
    points = s7_vector(np, p, 'points')
    n, d = points.shape
    kind = p['kind']
    s7_cap(2 <= d <= 6 and n <= S7_HULL_CAP[kind][d], 'hull points for this dimension')
    options = p['qhull_options']  # selected by the C++ table, never by the caller
    result = {'status': 'OK', 'kind': kind, 'n_points': n, 'dim': d, 'qhull_options': options}
    if n < d + 1:
        return {'status': 'DEGENERATE', 'kind': kind, 'n_points': n, 'dim': d, 'message': 'need at least dim + 1 points, got %d' % n}

    def body():
        try:
            if kind == 'convex_hull':
                return ConvexHull(points, qhull_options=options)
            if kind == 'delaunay':
                return Delaunay(points, qhull_options=options)
            return Voronoi(points, qhull_options=options)
        except QhullError as exc:
            return str(exc)
    got = s7_guard(p, body)
    if got is None:
        return {'status': 'TIME_LIMIT', 'kind': kind, 'n_points': n}
    if isinstance(got, str):
        lines = [line.strip() for line in got.splitlines() if line.strip()]
        return {'status': 'DEGENERATE', 'kind': kind, 'n_points': n, 'dim': d, 'message': ' '.join(lines)[:300]}
    if kind == 'convex_hull':
        result.update(volume=float(got.volume), area=float(got.area), n_vertices=int(len(got.vertices)), n_simplices=int(len(got.simplices)))
        named = [('vertices', got.vertices.astype(np.int64)), ('simplices', got.simplices.astype(np.int64))]
    elif kind == 'delaunay':
        simplices = got.simplices.astype(np.int64)
        edges = points[simplices[:, 1:]] - points[simplices[:, :1]]
        volume = float(np.sum(np.abs(np.linalg.det(edges))) / math.factorial(d))
        result.update(volume=volume, n_simplices=int(len(simplices)), n_coplanar=int(len(got.coplanar)))
        named = [('simplices', simplices)]
    else:
        def ragged(rows):
            offsets = np.zeros(len(rows) + 1, dtype=np.int64)
            offsets[1:] = np.cumsum([len(r) for r in rows])
            flat = np.fromiter((v for r in rows for v in r), dtype=np.int64, count=int(offsets[-1]))
            return offsets, flat
        region_offsets, regions = ragged(got.regions)
        ridge_offsets, ridge_vertices = ragged(got.ridge_vertices)
        result.update(n_vertices=int(len(got.vertices)), n_regions=int(len(got.regions)), n_ridges=int(len(got.ridge_points)),
                      n_unbounded_regions=int(sum(1 for r in got.regions if -1 in r)))
        named = [('vertices', got.vertices), ('point_region', got.point_region.astype(np.int64)), ('region_offsets', region_offsets), ('regions', regions),
                 ('ridge_points', got.ridge_points.astype(np.int64)), ('ridge_offsets', ridge_offsets), ('ridge_vertices', ridge_vertices)]
    if not math.isfinite(result.get('volume', 0.0)):
        return {'status': 'NUMERICAL_FAILURE', 'kind': kind, 'message': 'the volume is not finite'}
    return s7_emit(np, p, result, named, 'spatial_hull')


def spatial_dist(p):
    import numpy as np
    from scipy.spatial.distance import cdist
    a = s7_vector(np, p, 'a')
    b = s7_vector(np, p, 'b') if 'b_file' in p else a
    metric = p['metric']
    s7_cap(a.shape[0] * b.shape[0] <= S7_DIST_PAIRS, 'distance pairs')
    kwargs = {'VI': s7_vector(np, p, 'vi')} if metric == 'mahalanobis' else {}

    def body():
        with np.errstate(all='ignore'):
            return cdist(a, b, metric=metric, **kwargs)
    d = s7_guard(p, body)
    if d is None:
        return {'status': 'TIME_LIMIT', 'metric': metric}
    undefined = int((~np.isfinite(d)).sum())
    if undefined:
        return {'status': 'UNDEFINED', 'metric': metric, 'shape': list(d.shape), 'n_undefined': undefined, 'undefined_a_rows':np.flatnonzero((np.ptp(a,axis=1)==0) if metric=='correlation' else np.all(a==0,axis=1))[:100].tolist(), 'undefined_b_rows':np.flatnonzero((np.ptp(b,axis=1)==0) if metric=='correlation' else np.all(b==0,axis=1))[:100].tolist(),
                'message': 'the metric is undefined for some pairs (zero vectors for cosine/correlation, an indefinite VI, or overflow)'}
    result = {'status': 'OK', 'metric': metric, 'shape': list(d.shape),'vi_symmetrised':p.get('vi_symmetrised',False)}
    return s7_emit(np, p, result, [('distances', d)], 'spatial_dist')


def s7_operation(name):
    base = name[:-3] if name.endswith('_to') else name
    operation={'interp_eval':interp_eval,'sig_filter':sig_filter,'sig_spectrum':sig_spectrum,'sig_peaks':sig_peaks,'spatial_knn':spatial_knn,'spatial_hull':spatial_hull,'spatial_dist':spatial_dist}[base]
    def guarded(p):
        try:
            result=s7_guard(p,lambda:operation(p)) if base in ('sig_filter','sig_spectrum','sig_peaks','spatial_knn') else operation(p)
            return {'status':'TIME_LIMIT'} if result is None else result
        except NumericalFailure as exc:return {'status':'NUMERICAL_FAILURE','message':str(exc)}
    return guarded
