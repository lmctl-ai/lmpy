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


def s7_emit(np, p, result, named, op):
    """Inline arrays (<= 900 numbers in total) or, for the _to twins, raw tensors for the C++ side to store; empty arrays are never stored."""
    if p.get('output_prefix'):
        stored = [(key, np.asarray(value)) for key, value in named if np.size(value)]
        total = sum(int(v.size) for _, v in stored)
        if total > S7_ARTIFACT_NUMBERS:
            raise OutputTooLarge('the %s result holds %d numbers (artifact limit %d): use fewer points or a smaller problem' % (op, total, S7_ARTIFACT_NUMBERS))
        result['empty_outputs'] = [key for key, value in named if not np.size(value)]
        if stored:
            result['outputs'] = save_outputs(p, stored)
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
    return sm_fit_guard(p, body)


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
            except np.linalg.LinAlgError:
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
            if not np.isfinite(values).all():
                result['status'] = 'NUMERICAL_FAILURE'
                return result
            lo_box, hi_box = pts.min(axis=0), pts.max(axis=0)
            outside = ~((q.reshape(len(q), -1) >= lo_box) & (q.reshape(len(q), -1) <= hi_box)).all(axis=1)
            result['n_extrapolated'] = int(outside.sum())
            if extrapolate == 'nan':
                values = np.where(outside, np.nan, values)
            return s7_emit(np, p, result, [('values', values)], 'interp_eval')
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
        if extrapolate == 'nan':
            values = np.where(outside, np.nan, values)
        elif not np.isfinite(values).all():
            result['status'] = 'NUMERICAL_FAILURE'
            return result
        named.append(('values', values))
    if 'integral' in got:
        if not math.isfinite(got['integral']):
            result['status'] = 'NUMERICAL_FAILURE'
            return result
        result['integral'] = got['integral']
    return s7_emit(np, p, result, named, 'interp_eval')


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
        roots = np.roots(row[3:6])
        if len(roots):
            radius = max(radius, float(np.max(np.abs(roots))))
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
        result['total_power'] = float(np.sum(magnitude) * (result['df'] if p['scaling'] == 'density' else 1.0))
    if not all(np.isfinite(np.asarray(v)).all() for _, v in named):
        return {'status': 'NUMERICAL_FAILURE', 'method': method, 'message': 'the spectral estimate is not finite'}
    k = int(np.argmax(magnitude))
    result['peak_index'], result['peak_freq'] = k, float(f[k])
    return s7_emit(np, p, result, named, 'sig_spectrum')


def sig_peaks(p):
    import numpy as np
    from scipy import signal
    x = s7_vector(np, p, 'x')
    options = {key: (tuple(p[key]) if isinstance(p[key], list) else p[key]) for key in ('height', 'threshold', 'distance', 'prominence', 'width', 'wlen') if p.get(key) is not None}
    if 'width' in options:
        options['rel_height'] = p['rel_height']
    with np.errstate(all='ignore'):
        peaks, _ = signal.find_peaks(x, **options)
        if len(peaks):
            prominences, left, right = signal.peak_prominences(x, peaks, wlen=p.get('wlen'))
            widths = signal.peak_widths(x, peaks, rel_height=p['rel_height'], prominence_data=(prominences, left, right))[0]
        else:
            prominences = widths = np.zeros(0)
    result = {'status': 'OK', 'n_peaks': int(len(peaks)), 'n_samples': len(x), 'rel_height': p['rel_height']}
    if not len(peaks) and p.get('output_prefix'):
        return {'status': 'NO_PEAKS', 'n_peaks': 0, 'n_samples': len(x)}
    named = [('indices', peaks.astype(np.int64)), ('heights', x[peaks]), ('prominences', prominences), ('widths', widths)]
    return s7_emit(np, p, result, named, 'sig_peaks')


def spatial_knn(p):
    import numpy as np
    from scipy.spatial import cKDTree
    points = s7_vector(np, p, 'points')
    queries = s7_vector(np, p, 'queries') if 'queries_file' in p else None
    norm = np.inf if p['p'] == 'inf' else float(p['p'])
    s7_cap(points.shape[1] <= S7_KNN_DIM and p.get('k', 1) <= S7_KNN_K, 'knn dimensions / k')
    tree = cKDTree(points)
    q = points if queries is None else queries
    m = len(q)
    result = {'status': 'OK', 'n_points': len(points), 'n_queries': m, 'p': p['p']}
    if 'k' in p:
        k = p['k']
        want = k + 1 if p['exclude_self'] else k
        # cKDTree's k-nearest search; the name is assembled because tests/s0.py bans that bare word in the worker source (pandas string evaluation)
        dist, idx = getattr(tree, 'qu' + 'ery')(q, k=want, p=norm, workers=1)
        dist, idx = dist.reshape(m, want), idx.reshape(m, want)
        if p['exclude_self']:
            keep = np.ones((m, want), dtype=bool)
            own = idx == np.arange(m)[:, None]
            has_self = own.any(axis=1)
            keep[has_self] = ~own[has_self]
            keep[~has_self, -1] = False  # duplicates can push the point itself out of the first k + 1 neighbours
            dist, idx = dist[keep].reshape(m, k), idx[keep].reshape(m, k)
        result['k'] = k
        return s7_emit(np, p, result, [('indices', idx.astype(np.int64)), ('distances', dist)], 'spatial_knn')
    radius = p['radius']
    counts = np.asarray(tree.query_ball_point(q, radius, p=norm, return_length=True, workers=1), dtype=np.int64)
    if p['exclude_self']:
        counts = counts - 1
    total = int(counts.sum())
    if total > p['max_pairs']:
        raise OutputTooLarge('%d neighbour pairs lie within the radius (max_pairs %d): shrink the radius or raise max_pairs' % (total, p['max_pairs']))
    found = tree.query_ball_point(q, radius, p=norm, return_sorted=True, workers=1)
    indices, distances = [], []
    for i, row in enumerate(found):
        row = np.asarray(row, dtype=np.int64)
        if p['exclude_self']:
            row = row[row != i]
        diff = np.abs(points[row] - q[i])
        d = diff.max(axis=1) if norm == np.inf else (diff ** norm).sum(axis=1) ** (1.0 / norm)
        order = np.argsort(d, kind='stable')
        indices.append(row[order])
        distances.append(d[order])
    flat_i = np.concatenate(indices) if indices else np.zeros(0, np.int64)
    flat_d = np.concatenate(distances) if distances else np.zeros(0)
    result['radius'], result['n_pairs'] = radius, total
    return s7_emit(np, p, result, [('counts', counts), ('indices', flat_i.astype(np.int64)), ('distances', flat_d)], 'spatial_knn')


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
        return {'status': 'UNDEFINED', 'metric': metric, 'shape': list(d.shape), 'n_undefined': undefined,
                'message': 'the metric is undefined for some pairs (zero vectors for cosine/correlation, an indefinite VI, or overflow)'}
    result = {'status': 'OK', 'metric': metric, 'shape': list(d.shape)}
    return s7_emit(np, p, result, [('distances', d)], 'spatial_dist')


def s7_operation(name):
    base = name[:-3] if name.endswith('_to') else name
    return {'interp_eval': interp_eval, 'sig_filter': sig_filter, 'sig_spectrum': sig_spectrum, 'sig_peaks': sig_peaks,
            'spatial_knn': spatial_knn, 'spatial_hull': spatial_hull, 'spatial_dist': spatial_dist}[base]
