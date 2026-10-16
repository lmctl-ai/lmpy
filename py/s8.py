"""S8a: table/array bridge, frame_mutate / frame_merge / frame_rolling, ml_metrics and seeded sampling. numpy only (pandas is never imported):
tables are read and written member by member in the NPZ layout of py/frames.py, so these operations pay no pandas import and keep every
column's dtype, mask and categories exactly. Assembled after s7.py; the C++ front end (src/s8.hpp) has already enforced every cap and name rule."""
S8_EXACT = 2 ** 53
S8_ARTIFACT_VALUES = 1_000_000
S8_WORK = 100_000_000


def s8_cap(ok, what):
    if not ok:
        raise BadData('%s exceeds the operation cap (this request bypassed the front-end validation)' % what, 'INPUT_TOO_LARGE')


# ------------------------------------------------------------------------------------------------------------ table I/O
def s8_read(path):
    import numpy as np
    with np.load(path, allow_pickle=False) as z:
        try:
            metadata = json.loads(z['__pycli_table__'].tobytes().decode('utf-8'))
            names, n, cmeta = metadata['columns'], metadata['n_rows'], metadata['column_meta']
            frame_columns(names)
            raws, masks = [], []
            for i, name in enumerate(names):
                raw, mask = z['c%d' % i], z['m%d' % i]
                if raw.ndim != 1 or raw.shape != (n,) or mask.dtype != np.dtype('bool') or mask.shape != (n,):
                    raise BadData('table column/mask shape mismatch')
                dtype = cmeta[name]['dtype']
                expected = {'int64': '<i8', 'float64': '<f8', 'bool': '|b1', 'datetime64[ns]': '<i8', 'category': '<i8'}
                if dtype == 'string':
                    if raw.dtype.kind != 'S':
                        raise BadData('strings require UTF-8 bytes arrays')
                elif raw.dtype != np.dtype(expected[dtype]):
                    raise BadData('table dtype mismatch')
                raws.append(raw.copy())
                masks.append(mask.copy())
        except (KeyError, ValueError, TypeError, UnicodeError) as exc:
            raise BadData('invalid table artifact: ' + str(exc)) from exc
    return names, cmeta, n, raws, masks


def s8_write(p, names, cmeta, n, raws, masks):
    import numpy as np
    s8_cap(len(names) <= TABLE_COLS and n <= TABLE_ROWS and n * len(names) <= TABLE_CELLS, 'table size')
    frame_columns(names)
    arrays = {}
    total = 1024 * 1024
    for i, name in enumerate(names):
        raw = np.ascontiguousarray(raws[i])
        total += raw.nbytes + n
        if total > TABLE_BYTES:
            raise OutputTooLarge('table storage exceeds 256MiB')
        arrays['c%d' % i], arrays['m%d' % i] = raw, np.asarray(masks[i], dtype=bool)
    metadata = {'version': 1, 'columns': list(names), 'n_rows': n, 'column_meta': {c: cmeta[c] for c in names}, 'na_masks': {c: 'm%d' % i for i, c in enumerate(names)}}
    body = json.dumps(metadata, ensure_ascii=False, allow_nan=False).encode('utf-8')
    if len(body) > 1024 * 1024:
        raise OutputTooLarge('table metadata exceeds 1MiB')
    arrays['__pycli_table__'] = np.frombuffer(body, dtype='uint8')
    np.savez(p['output_file'], allow_pickle=False, **arrays)


def s8_cell(cmeta, raw, k):
    dtype = cmeta['dtype']
    if dtype == 'string':
        return bytes(raw[k]).decode('utf-8')
    if dtype == 'category':
        return cmeta['categories'][int(raw[k])]
    if dtype == 'bool':
        return bool(raw[k])
    if dtype in ('int64', 'datetime64[ns]'):
        return int(raw[k])
    return float(raw[k])


def s8_datetime(value, tz):
    from datetime import datetime, timedelta, timezone
    from zoneinfo import ZoneInfo
    seconds, nanos = divmod(value, 1_000_000_000)
    dt = datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=seconds)
    dt = dt.astimezone(ZoneInfo(tz)) if tz else dt.replace(tzinfo=None)
    text = dt.isoformat()
    if nanos:
        suffix = ('.%09d' % nanos) if nanos % 1000 else ('.%06d' % (nanos // 1000))
        text = text[:19] + suffix + text[19:]
    return text


def s8_summary(names, cmeta, n, raws, masks, extra):
    rows = min(n, 10, 1000 // max(len(names), 1))
    data = {}
    for i, name in enumerate(names):
        column = []
        for k in range(rows):
            if masks[i][k]:
                column.append(FRAME_NULL)
                continue
            value = s8_cell(cmeta[name], raws[i], k)
            if cmeta[name]['dtype'] == 'datetime64[ns]':
                value = s8_datetime(value, cmeta[name].get('tz'))
            column.append(value)
        data[name] = column
    result = {'status': 'OK', 'shape': [n, len(names)], 'column_meta': {c: cmeta[c] for c in names}, 'preview': {'columns': list(names), 'data': data, 'n_rows': rows}}
    result.update(extra)
    return frame_summary(result)


def s8_save_tensor(p, key, array):
    """One raw little-endian F64 tensor for the C++ side to store. Unlike save_outputs, NaN is allowed (it is an explicit, counted choice)."""
    import numpy as np
    array = np.ascontiguousarray(array, dtype='<f8')
    s8_cap(array.size <= S8_ARTIFACT_VALUES, 'artifact values')
    array.tofile(p['output_prefix'] + '0.bin')
    return [{'key': key, 'dtype': 'F64', 'shape': list(array.shape)}]


def s8_float_column(np, cmeta, raw, mask, name, lossy_ok=False):
    """Exact float64 view of an int64 / float64 / bool column; (values, number of int64 values that are not exactly representable)."""
    dtype = cmeta['dtype']
    if dtype == 'float64':
        return raw.astype(float), 0
    if dtype == 'bool':
        return raw.astype(float), 0
    inexact = ((raw > S8_EXACT) | (raw < -S8_EXACT)) & ~mask
    count = int(inexact.sum())
    if count and not lossy_ok:
        first = int(np.flatnonzero(inexact)[0])
        raise BadData('column %s holds %d int64 value(s) beyond +-2^53 that binary64 cannot represent exactly (first at row %d)' % (name, count, first), 'INT64_NOT_EXACT')
    return raw.astype(float), count


# ---------------------------------------------------------------------------------------------------------------- bridge
def frame_to_array(p):
    import numpy as np
    names, cmeta, n, raws, masks = s8_read(p['frame_file'])
    index = {c: i for i, c in enumerate(names)}
    selected = p['columns']
    offset = p['offset']
    stop = n if p.get('limit') is None else min(n, offset + p['limit'])
    rows = stop - offset
    s8_cap(rows * len(selected) <= S8_ARTIFACT_VALUES // 5, 'array values')
    out = np.empty((rows, len(selected)))
    na_total = nan_total = lossy = 0
    for j, column in enumerate(selected):
        i = index[column]
        raw, mask = raws[i][offset:stop], masks[i][offset:stop]
        values, inexact = s8_float_column(np, cmeta[column], raw, mask, column, p['allow_lossy_int'])
        lossy += inexact
        if mask.any():
            if p['missing'] == 'error':
                raise BadData('column %s has %d masked missing value(s); pass missing=nan to convert them to NaN' % (column, int(mask.sum())), 'NA_NOT_ALLOWED')
            na_total += int(mask.sum())
            values = np.where(mask, np.nan, values)
        if p['missing'] == 'error' and not np.isfinite(values).all():
            raise BadData('column %s has numeric NaN/Inf; missing=error guarantees finite output; use missing=nan explicitly' % column, 'NA_NOT_ALLOWED')
        nan_total += int(np.isnan(values[~mask]).sum())
        out[:, j] = values
    if p['squeeze']:
        out = out[:, 0]
    result = {'status': 'OK', 'shape': list(out.shape), 'columns': list(selected), 'source_dtypes': {c: cmeta[c]['dtype'] for c in selected},
              'source_na_count': na_total, 'source_nan_count': nan_total, 'missing_mask_lost': bool(p['missing'] == 'nan' and na_total > 0),
              'n_rows_total': n, 'offset': offset}
    if p['allow_lossy_int']:
        result['lossy_int_count'] = lossy
    if p.get('output_prefix'):
        result['outputs'] = s8_save_tensor(p, p['key'], out)
    else:
        s8_cap(out.size <= 1000, 'inline values')
        result['values'] = out.tolist()
    return result


def array_to_frame(p):
    import numpy as np
    arr = array(p, 'ref')
    arr = arr.reshape(len(arr), -1)
    n, k = arr.shape
    names = p['columns']
    s8_cap(k <= TABLE_COLS and n <= TABLE_ROWS and n * k <= TABLE_CELLS, 'table size')
    cmeta, raws, masks, na_counts, nan_count = {}, [], [], {}, 0
    for j, name in enumerate(names):
        dtype = p['dtypes'][name]
        v = arr[:, j]
        mask = np.zeros(n, dtype=bool)
        if dtype == 'float64':
            if p['nan_policy'] == 'missing':
                mask = np.isnan(v)
            else:
                nan_count += int(np.isnan(v).sum())
            raw = v.astype('<f8')
        elif dtype == 'int64':
            if not (np.isfinite(v).all() and (v == np.floor(v)).all() and (np.abs(v) <= S8_EXACT).all()):
                raise BadData('column %s is not exactly representable as int64' % name)
            raw = v.astype('<i8')
        else:
            if not np.isin(v, (0.0, 1.0)).all():
                raise BadData('column %s is not exactly 0/1' % name)
            raw = v.astype(bool)
        cmeta[name] = {'dtype': dtype}
        raws.append(raw)
        masks.append(mask)
        na_counts[name] = int(mask.sum())
    if 'frame_file' in p:
        old_names, old_meta, old_n, old_raws, old_masks = s8_read(p['frame_file'])
        if old_n != n:
            raise BadData('frame and ref need identical row counts')
        for j, name in enumerate(names):
            if name in old_names:
                i = old_names.index(name)
                old_raws[i], old_masks[i] = raws[j], masks[j]
            else:
                old_names.append(name); old_raws.append(raws[j]); old_masks.append(masks[j])
            old_meta[name] = cmeta[name]
        names, cmeta, raws, masks = old_names, old_meta, old_raws, old_masks
    s8_write(p, names, cmeta, n, raws, masks)
    return s8_summary(names, cmeta, n, raws, masks, {'nan_mask_changed': bool(p['nan_policy'] == 'missing' and any(na_counts.values())), 'dtypes': {c: cmeta[c]['dtype'] for c in names}, 'na_counts': na_counts, 'nan_count': nan_count})


# ---------------------------------------------------------------------------------------------------------------- mutate
def frame_mutate(p):
    import numpy as np
    names, cmeta, n, raws, masks = s8_read(p['frame_file'])
    spec, _ = load_spec(p)
    tables = (function_table(np), operator_table(np))
    state = {name: (raws[i], masks[i]) for i, name in enumerate(names)}
    meta = dict(cmeta)
    order = list(names)
    numeric = {}  # identifier-named numeric columns and earlier assignments: name -> (float64 values, mask)
    assigned = []

    def numeric_view(name):
        if name not in numeric:
            raw, mask = state[name]
            values, _ = s8_float_column(np, meta[name], raw, mask, name)
            numeric[name] = (values, mask)
        return numeric[name]

    def body():
        for step in p['plan']:
            name, refs = step['name'], step['refs']
            env, mask = {}, np.zeros(n, dtype=bool)
            for ref in refs:
                values, ref_mask = numeric_view(ref)
                env[ref] = values
                mask = mask | ref_mask
            program = compile_expression(step['ast'], set(env), {}, spec, tables)
            with np.errstate(all='ignore'):
                value = np.broadcast_to(np.asarray(evaluate(program, env), dtype=float), (n,)).copy()
            bad = ~np.isfinite(value) & ~mask
            count = int(bad.sum())
            if count and p['nonfinite'] == 'error':
                raise BadData('assignment %s produced %d non-finite value(s) (first at row %d); set nonfinite to mask or keep' % (name, count, int(np.flatnonzero(bad)[0])), 'NON_FINITE')
            if count and p['nonfinite'] == 'mask':
                mask = mask | bad
            value = np.where(mask, np.nan, value)
            if name not in state:
                order.append(name)
            state[name] = (value, mask)
            numeric[name] = (value, mask)
            meta[name] = {'dtype': 'float64'}
            assigned.append({'name': name, 'replaced': bool(step['replaced']), 'n_masked': int(mask.sum()), 'n_nonfinite': count})
        return True
    if s7_guard(p, body) is None:
        return {'status': 'TIME_LIMIT'}
    columns_raw = [state[c][0] for c in order]
    columns_mask = [state[c][1] for c in order]
    s8_write(p, order, meta, n, columns_raw, columns_mask)
    return s8_summary(order, meta, n, columns_raw, columns_mask, {'assigned': assigned})


# ----------------------------------------------------------------------------------------------------------------- merge
def s8_keys(np, cmeta, raws, masks, index, columns, uniqueness=False):
    """Per-row key tuples (None where any key is missing). Strings and categories compare by label, everything else by integer value."""
    parts = []
    for c in columns:
        i = index[c]
        dtype = cmeta[c]['dtype']
        if dtype == 'string':
            parts.append([bytes(v) for v in raws[i]])
        elif dtype == 'category':
            levels = [s.encode('utf-8') for s in cmeta[c]['categories']]
            parts.append([levels[v] if v >= 0 else b'' for v in raws[i].tolist()])
        else:
            parts.append(raws[i].tolist())
    miss = np.zeros(len(masks[index[columns[0]]]), dtype=bool)
    for c in columns:
        miss |= masks[index[c]]
    if uniqueness:
        return [tuple(None if masks[index[c]][r] else parts[j][r] for j, c in enumerate(columns)) for r in range(len(miss))]
    keys = [None if miss[r] else tuple(part[r] for part in parts) for r in range(len(miss))]
    return keys


def s8_take(np, cmeta, raw, mask, take):
    """Gather rows by index (-1 = absent -> masked) keeping dtype, categories and masks."""
    absent = take < 0
    safe = np.where(absent, 0, take)
    if len(raw) == 0:
        if not absent.all():
            raise BadData('invalid row in an empty table')
        out = np.zeros(len(take), dtype=raw.dtype)
        out_mask = np.ones(len(take), dtype=bool)
    else:
        out = raw[safe]
        out_mask = mask[safe] | absent
    dtype = cmeta['dtype']
    if dtype == 'category':
        out = np.where(out_mask, -1, out).astype('<i8')
    elif dtype == 'string':
        out = out.copy()
        out[out_mask] = b''
    elif dtype == 'float64':
        out = np.where(out_mask, np.nan, out)
    else:
        out = out.copy()
        out[out_mask] = 0
    return out, out_mask


def frame_merge(p):
    import numpy as np
    ln, lmeta, lrows, lraw, lmask = s8_read(p['left_file'])
    rn, rmeta, rrows, rraw, rmask = s8_read(p['right_file'])
    lindex, rindex = {c: i for i, c in enumerate(ln)}, {c: i for i, c in enumerate(rn)}
    lk, rk, how = p['left_keys'], p['right_keys'], p['how']
    lkeys = s8_keys(np, lmeta, lraw, lmask, lindex, lk)
    rkeys = s8_keys(np, rmeta, rraw, rmask, rindex, rk)
    left_of, right_of = {}, {}
    for i, key in enumerate(lkeys):
        if key is not None:
            left_of.setdefault(key, []).append(i)
    for j, key in enumerate(rkeys):
        if key is not None:
            right_of.setdefault(key, []).append(j)
    # validation and the row count come from the key counts alone: nothing is built before the cap is checked
    mode = p['validate']
    def duplicates(meta, raw, mask, index, columns):
        counts = {}
        for key in s8_keys(np, meta, raw, mask, index, columns, uniqueness=True):
            counts[key] = counts.get(key, 0) + 1
        return sum(count > 1 for count in counts.values())
    left_dup = duplicates(lmeta, lraw, lmask, lindex, lk)
    right_dup = duplicates(rmeta, rraw, rmask, rindex, rk)
    if mode in ('one_to_one', 'one_to_many') and left_dup:
        raise BadData('validate=%s: %d key value(s) are repeated in the left table' % (mode, left_dup), 'MERGE_VALIDATION')
    if mode in ('one_to_one', 'many_to_one') and right_dup:
        raise BadData('validate=%s: %d key value(s) are repeated in the right table' % (mode, right_dup), 'MERGE_VALIDATION')
    matched = sum(len(v) * len(right_of.get(key, ())) for key, v in left_of.items())
    left_only = sum(1 for key in lkeys if key is None or key not in right_of)
    right_only = sum(1 for key in rkeys if key is None or key not in left_of)
    total = {'inner': matched, 'left': matched + left_only, 'right': matched + right_only, 'outer': matched + left_only + right_only}[how]
    names = p['output_names']
    if total > p['max_rows']:
        raise BadData('the join would produce %d rows (max_rows %d)' % (total, p['max_rows']), 'ROW_EXPLOSION')
    if total * len(names) > TABLE_CELLS:
        raise BadData('the join would produce %d cells (limit %d)' % (total * len(names), TABLE_CELLS), 'ROW_EXPLOSION')
    # Account for fixed-width payloads before index lists, gathers or label unification.
    projected = 1024 * 1024
    for c in ln:
        width = lraw[lindex[c]].dtype.itemsize
        if p['key_shared'] and c in lk:
            partner = rk[lk.index(c)]
            lm, rm = lmeta[c], rmeta[partner]
            width = max(width, rraw[rindex[partner]].dtype.itemsize)
            if lm['dtype'] != rm['dtype'] and {lm['dtype'], rm['dtype']} <= {'category', 'string'}:
                for meta in (lm, rm):
                    if meta['dtype'] == 'category':
                        width = max(width, max((len(v.encode('utf-8')) for v in meta['categories']), default=1))
        projected += total * (width + 1)
    for c in rn:
        if not (p['key_shared'] and c in rk):
            projected += total * (rraw[rindex[c]].dtype.itemsize + 1)
    if projected > TABLE_BYTES:
        raise OutputTooLarge('join fixed-width output would require %d bytes (limit 256MiB)' % projected)
    li, ri = [], []

    def left_driven(unmatched_left):
        for i, key in enumerate(lkeys):
            hits = right_of.get(key, ()) if key is not None else ()
            if hits:
                for j in hits:
                    li.append(i)
                    ri.append(j)
            elif unmatched_left:
                li.append(i)
                ri.append(-1)
    if how == 'right':
        for j, key in enumerate(rkeys):
            hits = left_of.get(key, ()) if key is not None else ()
            if hits:
                for i in hits:
                    li.append(i)
                    ri.append(j)
            else:
                li.append(-1)
                ri.append(j)
    else:
        left_driven(how in ('left', 'outer'))
        if how == 'outer':
            for j, key in enumerate(rkeys):
                if key is None or key not in left_of:
                    li.append(-1)
                    ri.append(j)
    li, ri = np.asarray(li, dtype=np.int64), np.asarray(ri, dtype=np.int64)
    assert len(li) == total
    out_raw, out_mask, out_meta, out_names = [], [], {}, []
    shared = p['key_shared']
    plan = list(p['output_names'])
    position = 0
    for c in ln:
        name = plan[position]
        position += 1
        raw, mask = s8_take(np, lmeta[c], lraw[lindex[c]], lmask[lindex[c]], li)
        meta = dict(lmeta[c])
        if shared and c in lk:
            partner = rk[lk.index(c)]
            rraw_c, rmask_c = s8_take(np, rmeta[partner], rraw[rindex[partner]], rmask[rindex[partner]], ri)
            lm, rm = lmeta[c], rmeta[partner]
            same_kind = lm['dtype'] == rm['dtype'] and (lm['dtype'] != 'category' or lm.get('categories') == rm.get('categories'))
            use_left = li >= 0
            if lm['dtype'] == rm['dtype'] == 'category':
                levels = list(lm['categories'])
                seen = set(levels)
                for level in rm['categories']:
                    if level not in seen:
                        seen.add(level); levels.append(level)
                codes = {v: i for i, v in enumerate(levels)}
                remap = np.asarray([codes[v] for v in rm['categories']], dtype=np.int64)
                mapped = remap[np.maximum(rraw_c, 0)] if len(remap) else np.full(total, -1, dtype=np.int64)
                raw = np.where(use_left, raw, mapped)
                meta = dict(lm, categories=levels, ordered=bool(lm.get('ordered') and rm.get('ordered') and lm['categories'] == rm['categories']))
            elif same_kind and lm['dtype'] != 'string':
                raw = np.where(use_left, raw, rraw_c)
            else:
                labels = []
                for r in range(total):
                    side = (lm, raw, mask) if use_left[r] else (rm, rraw_c, rmask_c)
                    labels.append(b'' if side[2][r] else (side[0]['categories'][int(side[1][r])].encode('utf-8') if side[0]['dtype'] == 'category' else bytes(side[1][r])))
                width = max(1, max((len(v) for v in labels), default=1))
                raw = np.asarray(labels, dtype='S%d' % width)
                meta = {'dtype': 'string'}
            mask = np.where(use_left, mask, rmask_c)
        out_names.append(name)
        out_raw.append(raw)
        out_mask.append(mask)
        out_meta[name] = meta
    for c in rn:
        if shared and c in rk:
            continue
        name = plan[position]
        position += 1
        raw, mask = s8_take(np, rmeta[c], rraw[rindex[c]], rmask[rindex[c]], ri)
        out_names.append(name)
        out_raw.append(raw)
        out_mask.append(mask)
        out_meta[name] = dict(rmeta[c])
    assert out_names == plan
    s8_write(p, out_names, out_meta, total, out_raw, out_mask)
    return s8_summary(out_names, out_meta, total, out_raw, out_mask, {'how': how, 'n_pairs': matched, 'n_left_only': left_only, 'n_right_only': right_only, 'columns': out_names})


# --------------------------------------------------------------------------------------------------------------- rolling
def s8_windows(np, p, n, raws, masks, names, cmeta):
    """Inclusive [start, end] row range of every window."""
    rows = np.arange(n)
    if 'time_window_seconds' in p:
        on = names.index(p['on'])
        if masks[on].any():
            raise BadData('the datetime column %s holds missing values' % p['on'])
        times = raws[on].astype(np.int64)
        if (times[1:] < times[:-1]).any():
            raise BadData('the datetime column %s must be sorted ascending' % p['on'], 'NOT_SORTED')
        span = int(round(p['time_window_seconds'] * 1e9))
        lower = np.iinfo(np.int64).min
        thresholds = np.asarray([max(lower, int(t) - span) for t in times], dtype=np.int64)
        start = np.searchsorted(times, thresholds, side='right')
        # NaT (INT64_MIN) cannot occur in this complete datetime column.
        start = np.minimum(start, rows)
        return start.astype(np.int64), rows.astype(np.int64)
    w = p['window']
    if p['center']:
        return np.maximum(0, rows - w // 2).astype(np.int64), np.minimum(n - 1, rows + (w - 1) // 2).astype(np.int64)
    return np.maximum(0, rows - w + 1).astype(np.int64), rows.astype(np.int64)


def frame_rolling(p):
    import numpy as np
    names, cmeta, n, raws, masks = s8_read(p['frame_file'])
    index = {c: i for i, c in enumerate(names)}
    start, end = s8_windows(np, p, n, raws, masks, names, cmeta)
    widths = end - start + 1
    functions = p['functions']
    work = int(widths.sum()) * len(p['columns']) * len(functions)
    if work > S8_WORK:
        raise BadData('the rolling work is %d window-elements (limit %d)' % (work, S8_WORK), 'INPUT_TOO_LARGE')
    minimum = p['min_periods']
    out_names, out_meta, out_raw, out_mask = list(names), dict(cmeta), list(raws), list(masks)
    added = []

    def body():
        wmax = max(1, int(widths.max()) if n else 1)
        step = max(1, 2_000_000 // wmax)
        for column in p['columns']:
            i = index[column]
            values, _ = s8_float_column(np, cmeta[column], raws[i], masks[i], column)
            if np.isinf(values[~masks[i]]).any():
                raise BadData('rolling refuses unmasked infinite values', 'NON_FINITE')
            values = np.where(masks[i] | np.isnan(values), np.nan, values)  # a missing value and a NaN are both skipped
            padded = np.concatenate([values, [np.nan]])
            results = {f: np.full(n, np.nan) for f in functions}
            counts = np.zeros(n, dtype=np.int64)
            for a in range(0, n, step):
                b = min(n, a + step)
                s, e = start[a:b], end[a:b]
                width = int((e - s).max()) + 1
                pos = s[:, None] + np.arange(width)[None, :]
                valid = pos <= e[:, None]
                window = padded[np.where(valid, pos, n)]  # invalid slots read the trailing NaN
                present = ~np.isnan(window)
                c = present.sum(axis=1)
                counts[a:b] = c
                with np.errstate(all='ignore'):
                    zero = np.where(present, window, 0.0).astype(np.longdouble)
                    total = zero.sum(axis=1)
                    mean = total / np.maximum(c, 1)
                    for f in functions:
                        if f == 'sum':
                            r = total
                        elif f == 'mean':
                            r = mean
                        elif f in ('std', 'var'):
                            dev = np.where(present, window.astype(np.longdouble) - mean[:, None], 0.0)
                            var = (dev * dev).sum(axis=1) / np.maximum(c - 1, 1)
                            r = np.sqrt(var) if f == 'std' else var
                        elif f == 'min':
                            r = np.where(present, window, np.inf).min(axis=1)
                        elif f == 'max':
                            r = np.where(present, window, -np.inf).max(axis=1)
                        elif f == 'median':
                            import warnings
                            with warnings.catch_warnings():
                                warnings.simplefilter('ignore')
                                r = np.nanmedian(np.where(present, window, np.nan), axis=1)
                        else:
                            continue
                        need = max(minimum, 2 if f in ('std', 'var') else 1)
                        valid_result = c >= need
                        if not np.isfinite(np.asarray(r[valid_result], dtype=float)).all():
                            raise BadData('rolling %s exceeds representable float64 range' % f, 'NON_FINITE')
                        results[f][a:b] = np.where(valid_result, r, np.nan)
            for f in functions:
                name = '%s__%s' % (column, f)
                added.append(name)
                out_names.append(name)
                if f == 'count':
                    out_raw.append(counts.astype('<i8'))
                    out_mask.append(widths < minimum)
                    out_meta[name] = {'dtype': 'int64'}
                else:
                    r = results[f]
                    out_raw.append(r.astype('<f8'))
                    out_mask.append(np.isnan(r))
                    out_meta[name] = {'dtype': 'float64'}
        return True
    if s7_guard(p, body) is None:
        return {'status': 'TIME_LIMIT'}
    s8_write(p, out_names, out_meta, n, out_raw, out_mask)
    return s8_summary(out_names, out_meta, n, out_raw, out_mask, {'columns_added': added})


# --------------------------------------------------------------------------------------------------------------- metrics
def s8_labels(np, cmeta, raw, mask, name):
    """Canonical label list: bool/int -> int, float -> int when integral, strings and categories -> str."""
    if mask.any():
        raise BadData('column %s holds %d masked missing value(s); metrics need complete data' % (name, int(mask.sum())), 'NA_NOT_ALLOWED')
    dtype = cmeta['dtype']
    if dtype == 'string':
        return [bytes(v).decode('utf-8') for v in raw]
    if dtype == 'category':
        return [cmeta['categories'][int(v)] for v in raw]
    if dtype == 'float64':
        if not np.isfinite(raw).all():
            raise BadData('column %s holds NaN/Inf' % name, 'NA_NOT_ALLOWED')
        return [int(v) if v == int(v) else float(v) for v in raw]
    return [int(v) for v in raw]


def s8_array_labels(np, values):
    return [int(v) if v == int(v) else float(v) for v in values.tolist()]


def s8_order(labels):
    return sorted(set(labels), key=lambda v: (isinstance(v, str), v))


def ml_metrics(p):
    import numpy as np
    wanted = p['metrics']
    cols = {}
    if 'frame_file' in p:
        names, cmeta, n, raws, masks = s8_read(p['frame_file'])
        index = {c: i for i, c in enumerate(names)}
        for key in ('y_true', 'y_pred', 'y_score'):
            if p.get(key) is not None:
                c, i = p[key], index[p[key]]
                if key == 'y_score':
                    if masks[i].any():
                        raise BadData('column %s holds masked missing values' % c, 'NA_NOT_ALLOWED')
                    cols[key] = s8_float_column(np, cmeta[c], raws[i], masks[i], c)[0]
                else:
                    cols[key] = s8_labels(np, cmeta[c], raws[i], masks[i], c)
    else:
        for key in ('y_true', 'y_pred', 'y_score'):
            if key + '_file' in p:
                values = array(p, key)
                cols[key] = s8_array_labels(np, values) if key != 'y_score' else values.astype(float)
    n = len(cols['y_true'])
    s8_cap(0 < n <= 200000, 'sample size')
    for key, v in cols.items():
        if len(v) != n:
            raise BadData('y_true, y_pred and y_score must have the same length')
    label_metrics = [m for m in wanted if m in ('accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1', 'confusion_matrix')]
    regression = [m for m in wanted if m in ('r2', 'mse', 'mae', 'mape')]
    scored = [m for m in wanted if m in ('roc_auc', 'log_loss', 'brier')]
    metrics, undefined = {}, {}
    result = {'n': n}
    if label_metrics:
        yt, yp = cols['y_true'], cols['y_pred']
        labels = s8_order(yt + yp)
        if len(labels) > 30:
            raise BadData('classification metrics allow at most 30 distinct labels (found %d)' % len(labels), 'INPUT_TOO_LARGE')
        position = {l: k for k, l in enumerate(labels)}
        conf = np.zeros((len(labels), len(labels)), dtype=np.int64)
        np.add.at(conf, ([position[v] for v in yt], [position[v] for v in yp]), 1)
        tp = np.diag(conf).astype(float)
        fp = conf.sum(axis=0) - np.diag(conf)
        fn = conf.sum(axis=1) - np.diag(conf)
        support = conf.sum(axis=1)
        zd = {'undefined': None, '0': 0.0, '1': 1.0}[p['zero_division']]
        average = p['average']
        pos = p.get('pos_label')

        def ratio(a, b):
            return float(a) / float(b) if b else zd

        def prf(t, f_p, f_n):
            return ratio(t, t + f_p), ratio(t, t + f_n), ratio(2 * t, 2 * t + f_p + f_n)
        if 'accuracy' in wanted:
            metrics['accuracy'] = float(tp.sum()) / n
        if 'balanced_accuracy' in wanted:
            present = support > 0
            metrics['balanced_accuracy'] = float(np.mean(tp[present] / support[present]))
        if {'precision', 'recall', 'f1'} & set(wanted):
            if average == 'binary':
                if pos is None:
                    if not set(labels) <= {0, 1}:
                        raise BadData('average=binary needs pos_label for labels other than 0/1', 'BAD_PARAM')
                    pos = 1
                k = position.get(pos)
                if k is None:
                    raise BadData('pos_label %r does not occur among the labels %s' % (pos, labels), 'BAD_PARAM')
                triple = prf(tp[k], fp[k], fn[k])
            elif average == 'micro':
                triple = prf(tp.sum(), fp.sum(), fn.sum())
            else:
                rows = [prf(tp[k], fp[k], fn[k]) for k in range(len(labels))]
                weights = support.astype(float) if average == 'weighted' else np.ones(len(labels))
                triple = tuple(None if any(r[c] is None for r in rows) else float(sum(w * r[c] for w, r in zip(weights, rows)) / weights.sum()) for c in range(3))
            for name, value in zip(('precision', 'recall', 'f1'), triple):
                if name in wanted:
                    if value is None:
                        undefined[name] = 'zero denominator (no predicted or no actual positives); set zero_division to 0 or 1 to define it'
                    else:
                        metrics[name] = value
        if 'confusion_matrix' in wanted:
            result['confusion_matrix'] = conf.tolist()
        result['labels'] = labels
        result['average'] = average
        if pos is not None and average == 'binary':
            result['pos_label'] = pos
    if scored:
        yt, score = cols['y_true'], np.asarray(cols['y_score'], dtype=float)
        classes = s8_order(yt)
        if not np.isfinite(score).all():
            raise BadData('y_score must contain only finite values', 'BAD_DATA')
        if ('log_loss' in scored or 'brier' in scored) and ((score < 0) | (score > 1)).any():
            raise BadData('y_score must be a probability in [0,1]', 'BAD_DATA')
        pos = p.get('pos_label')
        if pos is not None and pos not in classes:
            raise BadData('explicit pos_label does not occur among true labels', 'BAD_PARAM')
        if pos is None:
            if len(classes) <= 2 and not set(classes) <= {0, 1}:
                raise BadData('scored metrics need pos_label for labels other than 0/1', 'BAD_PARAM')
            pos = 1
        result['pos_label'] = pos
        if len(classes) == 2 and pos not in classes:
            raise BadData('pos_label %r does not occur among the labels %s' % (pos, classes), 'BAD_PARAM')
        target = np.asarray([1.0 if v == pos else 0.0 for v in yt])
        if len(classes) > 2:
            for m in scored:
                undefined[m] = 'binary scores need y_true with at most two classes (found %d)' % len(classes)
        else:
            if 'roc_auc' in scored:
                n_pos, n_neg = int(target.sum()), int(n - target.sum())
                if n_pos == 0 or n_neg == 0:
                    undefined['roc_auc'] = 'y_true holds only one class'
                else:
                    _, inverse, counts = np.unique(score, return_inverse=True, return_counts=True)
                    last = np.cumsum(counts)
                    average_rank = (last - counts + 1 + last) / 2.0
                    ranks = average_rank[inverse]
                    metrics['roc_auc'] = float((ranks[target == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
            if 'log_loss' in scored:
                eps = 1e-15
                q = np.clip(score, eps, 1 - eps)
                metrics['log_loss'] = float(-np.mean(target * np.log(q) + (1 - target) * np.log1p(-q)))
                result['log_loss_eps'] = eps
            if 'brier' in scored:
                metrics['brier'] = float(np.mean((score - target) ** 2))
    if regression:
        if 'frame_file' in p:
            for key in ('y_true', 'y_pred'):
                c, i = p[key], index[p[key]]
                s8_float_column(np, cmeta[c], raws[i], masks[i], c)
        yt = np.asarray(cols['y_true'], dtype=np.longdouble)
        yp = np.asarray(cols['y_pred'], dtype=np.longdouble)
        if not (np.isfinite(yt).all() and np.isfinite(yp).all()):
            raise BadData('regression metrics need finite values', 'NA_NOT_ALLOWED')
        error = yt - yp
        if 'mse' in regression:
            metrics['mse'] = np.mean(error * error)
        if 'mae' in regression:
            metrics['mae'] = np.mean(np.abs(error))
        if 'r2' in regression:
            total = np.sum((yt - yt.mean()) ** 2)
            if (yt == yt[0]).all():
                undefined['r2'] = 'y_true is constant (zero total variance)'
            else:
                metrics['r2'] = 1.0 - np.sum(error * error) / total
        if 'mape' in regression:
            if (yt == 0).any():
                undefined['mape'] = 'y_true contains %d zero(s)' % int((yt == 0).sum())
            else:
                metrics['mape'] = np.mean(np.abs(error / yt))
    for name, value in list(metrics.items()):
        converted = float(value)
        if not np.isfinite(converted) or (converted == 0 and value != 0):
            undefined[name] = 'result is outside representable float64 range'
            del metrics[name]
        else:
            metrics[name] = converted
    result['metrics'] = {m: metrics[m] for m in wanted if m in metrics and m != 'confusion_matrix'}
    result['undefined'] = {m: undefined[m] for m in wanted if m in undefined}
    result['status'] = 'UNDEFINED' if undefined else 'OK'
    return result


# ------------------------------------------------------------------------------------------------------------------ rvs
DIST_NUMPY_RVS = ('norm', 't', 'uniform', 'expon', 'gamma', 'beta', 'lognorm', 'chi2', 'poisson', 'binom')
DIST_SHAPES = {'t': ('df',), 'genextreme': ('c',), 'genpareto': ('c',), 'skewnorm': ('a',), 'levy_stable': ('alpha', 'beta'), 'johnsonsu': ('a', 'b'),
               'nct': ('df', 'nc'), 'ncf': ('dfn', 'dfd', 'nc'), 'vonmises': ('kappa',), 'truncnorm': ('a', 'b'), 'weibull_min': ('c',), 'pareto': ('b',),
               'f': ('dfn', 'dfd'), 'zipf': ('a',), 'betabinom': ('n', 'a', 'b'), 'nbinom': ('n', 'p'), 'geom': ('p',)}
DIST_DISCRETE = ('poisson', 'binom', 'zipf', 'betabinom', 'nbinom', 'geom')


def dist_frozen(dist, prm):
    """scipy.stats frozen distribution from the validated parameter table (py names for the original ten families, scipy names for the rest)."""
    from scipy import stats
    if dist == 'uniform':
        return stats.uniform(loc=prm['low'], scale=prm['high'] - prm['low'])
    if dist == 'expon':
        return stats.expon(scale=prm['scale'])
    if dist == 'gamma':
        return stats.gamma(prm['shape'], scale=prm['scale'])
    if dist == 'beta':
        return stats.beta(prm['a'], prm['b'])
    if dist == 'lognorm':
        return stats.lognorm(prm['sigma'], scale=math.exp(prm['mu']))
    if dist == 'chi2':
        return stats.chi2(prm['df'])
    if dist == 'poisson':
        return stats.poisson(prm['lam'])
    if dist == 'binom':
        return stats.binom(int(prm['n']), prm['p'])
    families = {'norm': stats.norm, 't': stats.t, 'genextreme': stats.genextreme, 'genpareto': stats.genpareto, 'skewnorm': stats.skewnorm,
                'levy_stable': stats.levy_stable, 'johnsonsu': stats.johnsonsu, 'nct': stats.nct, 'ncf': stats.ncf, 'vonmises': stats.vonmises,
                'truncnorm': stats.truncnorm, 'weibull_min': stats.weibull_min, 'cauchy': stats.cauchy, 'laplace': stats.laplace,
                'logistic': stats.logistic, 'gumbel_r': stats.gumbel_r, 'pareto': stats.pareto, 'rayleigh': stats.rayleigh, 'f': stats.f,
                'zipf': stats.zipf, 'betabinom': stats.betabinom, 'nbinom': stats.nbinom, 'geom': stats.geom}
    shapes = [prm[k] for k in DIST_SHAPES.get(dist, ())]
    if dist == 'betabinom':
        shapes[0] = int(shapes[0])
    if dist in DIST_DISCRETE:
        return families[dist](*shapes, loc=prm['loc'])
    return families[dist](*shapes, loc=prm['loc'], scale=prm['scale'])


def dist_methods(frozen, discrete):
    mass = discrete
    return {'pdf': frozen.pmf if mass else frozen.pdf, 'logpdf': frozen.logpmf if mass else frozen.logpdf, 'cdf': frozen.cdf, 'logcdf': frozen.logcdf,
            'sf': frozen.sf, 'logsf': frozen.logsf, 'ppf': frozen.ppf, 'isf': frozen.isf}


def dist_levy_warning(dist, fn):
    if dist == 'levy_stable' and fn != 'rvs':
        warnings.warn('levy_stable pdf/cdf are numerical integrations: slow, and accurate to roughly 1e-6 relative at best (worse in the far tails); compare with an independent oracle before relying on digits', NumericalWarning)


# Cheap canonical parameters for the trusted-phase warm-up: scipy's lazy imports happen for the requested function only, never on the caller's
# parameters (those are evaluated under the cooperative budget, where an unbounded quantile search is TIME_LIMIT and not a hang or a kill).
DIST_WARM_PARAMS = {'t': {'df': 5.0}, 'uniform': {'low': 0.0, 'high': 1.0}, 'gamma': {'shape': 2.0, 'scale': 1.0}, 'beta': {'a': 2.0, 'b': 3.0}, 'lognorm': {'mu': 0.0, 'sigma': 1.0},
                    'chi2': {'df': 3.0}, 'poisson': {'lam': 2.0}, 'binom': {'n': 5, 'p': 0.5}, 'genextreme': {'c': 0.2}, 'genpareto': {'c': 0.2}, 'skewnorm': {'a': 2.0},
                    'levy_stable': {'alpha': 1.5, 'beta': 0.3}, 'johnsonsu': {'a': 1.0, 'b': 2.0}, 'nct': {'df': 5.0, 'nc': 1.0}, 'ncf': {'dfn': 3.0, 'dfd': 7.0, 'nc': 1.0},
                    'vonmises': {'kappa': 2.0}, 'truncnorm': {'a': -1.0, 'b': 2.0}, 'weibull_min': {'c': 2.0}, 'pareto': {'b': 3.0}, 'f': {'dfn': 3.0, 'dfd': 7.0}, 'zipf': {'a': 3.0},
                    'betabinom': {'n': 5, 'a': 2.0, 'b': 3.0}, 'nbinom': {'n': 3.0, 'p': 0.5}, 'geom': {'p': 0.5}}


def dist_warm(p):
    """Trusted-phase warm-up with canonical parameters: the requested function (plus the cdf behind a zipf quantile guard), nothing else."""
    import numpy as np
    dist, fn = p['dist'], p['fn']
    prm = dict(DIST_WARM_PARAMS.get(dist, {}))
    prm.setdefault('loc', 0.0)
    prm.setdefault('scale', 1.0)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with np.errstate(all='ignore'):
                frozen = dist_frozen(dist, prm)
                points = np.array([1.0, 2.0]) if dist in DIST_DISCRETE else np.array([0.5, 0.6])
                if fn == 'moments':
                    frozen.stats('mvsk')
                    from scipy import integrate
                    integrate.quad(lambda s: s, 0.0, 1.0)
                elif fn == 'rvs':
                    frozen.rvs(size=2, random_state=np.random.Generator(np.random.PCG64(1)))
                else:
                    dist_methods(frozen, dist in DIST_DISCRETE)[fn](np.array([0.3, 0.6]) if fn in ('ppf', 'isf') else points)
                    if fn in ('ppf', 'isf'):
                        frozen.cdf(points)
                        frozen.sf(points)
                        frozen.support()
                    if dist == 'nct' and fn in ('pdf', 'logpdf'):
                        from scipy import integrate
                        from scipy.special import gammaln
                        integrate.quad(lambda s: s, 0.0, 1.0)
                        frozen.cdf(points)
    except Exception:
        pass


def nct_density_by_quadrature(np, x, df, nc):
    """f(x) = integral over s > 0 of s * phi(x s - nc) * f_S(s) ds with S = sqrt(V / df), V ~ chi2(df): the defining integral, used only where scipy's
    closed form underflows to 0 inside the body of the distribution (large nc); adaptive quadrature split at the peak s = nc / x."""
    from scipy import integrate
    from scipy.special import gammaln
    half = df / 2.0
    log_norm = math.log(2.0) + math.log(df) - half * math.log(2.0) - gammaln(half)

    def integrand(s, t):
        if s <= 0:
            return 0.0
        z = t * s - nc
        return math.exp(math.log(s) + log_norm + math.log(s) + (df - 2.0) * math.log(s * math.sqrt(df)) - s * s * df / 2.0 - 0.5 * z * z - 0.5 * math.log(2 * math.pi))
    out = np.empty(len(x))
    for i, t in enumerate(x):
        peak = nc / t if t != 0 and nc / t > 0 else None
        pieces = [(0.0, peak), (peak, np.inf)] if peak else [(0.0, np.inf)]
        total = 0.0
        for a, b in pieces:
            total += integrate.quad(integrand, a, b, args=(t,), limit=200, epsabs=0.0, epsrel=1e-10)[0]
        out[i] = max(total, 0.0)
    return out


def weibull_moments(np, c, loc, scale):
    """Mean from Gamma(1 + 1/c); central moments by quadrature of (t^(1/c) - mu)^k e^-t, which keeps its digits where the raw-moment differences lose
    them (sharp shapes c >> 1 tend to the Gumbel-min limits: skewness -1.1395, excess kurtosis 2.4)."""
    from scipy import integrate
    from scipy.special import gamma
    mu = float(gamma(1.0 + 1.0 / c))
    shift = mu - 1.0

    def central(k):
        f = lambda t: (math.expm1(math.log(t) / c) - shift) ** k * math.exp(-t) if t > 0 else ((-shift) ** k if t == 0 else 0.0)
        return integrate.quad(f, 0.0, np.inf, limit=400, epsabs=0.0, epsrel=1e-11)[0]
    m2, m3, m4 = central(2), central(3), central(4)
    return {'mean': loc + scale * mu, 'var': scale * scale * m2, 'skew': m3 / m2 ** 1.5, 'kurtosis': m4 / (m2 * m2) - 3.0}


def dist_wrap_circle(np, x, loc, scale):
    """vonmises lives on loc + scale * [-pi, pi]: points outside are wrapped back onto the circle (the end points stay where they are)."""
    z = (x - loc) / scale
    outside = np.isfinite(z) & ((z < -math.pi) | (z > math.pi))
    if outside.any():
        z = z.copy()
        z[outside] = np.mod(z[outside] + math.pi, 2 * math.pi) - math.pi
    return loc + scale * z


def dist_evaluate(p):
    import numpy as np
    dist, prm, fn = p['dist'], p['params'], p['fn']
    frozen = dist_frozen(dist, prm)
    base = {'dist': dist, 'fn': fn, 'params': prm}
    dist_levy_warning(dist, fn)
    discrete = dist in DIST_DISCRETE
    out_of_range = np.zeros(0, dtype=bool)
    with np.errstate(all='ignore'):
        if fn == 'moments':
            names = ('mean', 'var', 'skew', 'kurtosis')
            if dist == 'weibull_min':
                out = weibull_moments(np, prm['c'], prm['loc'], prm['scale'])
            else:
                out = dict(zip(names, [float(v) for v in frozen.stats('mvsk')]))
            undefined = [k for k in names if not math.isfinite(out[k])]
            return dict(base, status='OK', moments={k: (None if k in undefined else v) for k, v in out.items()}, moments_undefined=undefined,
                        note='kurtosis is the excess kurtosis; NA means infinite or undefined')
        x = np.atleast_1d(np.load(p['x_file'], allow_pickle=False).astype(float))
        points = x
        if dist == 'zipf' and fn in ('ppf', 'isf'):
            # scipy searches the heavy tail by summing the mass function: quantiles beyond loc + 1e5 are not computed (NA, status OUT_OF_RANGE)
            edge = float(frozen.cdf(prm['loc'] + 1e5)) if fn == 'ppf' else float(frozen.sf(prm['loc'] + 1e5))
            out_of_range = np.isfinite(x) & ((x > edge) if fn == 'ppf' else (x < edge))
            points = np.where(out_of_range, np.nan, x)
        if dist == 'vonmises' and fn in ('cdf', 'logcdf', 'sf', 'logsf'):
            points = dist_wrap_circle(np, x, prm['loc'], prm['scale'])
        try:
            values = np.atleast_1d(np.asarray(dist_methods(frozen, discrete)[fn](points), dtype=float))
            if dist == 'nct' and fn in ('pdf', 'logpdf'):
                dead = (values == 0.0) if fn == 'pdf' else np.isneginf(values)
                dead &= np.isfinite(x)
                if dead.any():
                    c = frozen.cdf(x[dead])
                    body = (c > 1e-300) & (c < 1 - 1e-16)
                    if body.any():
                        idx = np.flatnonzero(dead)[body]
                        density = nct_density_by_quadrature(np, (x[idx] - prm['loc']) / prm['scale'], prm['df'], prm['nc']) / prm['scale']
                        values[idx] = density if fn == 'pdf' else np.log(density)
        except (RuntimeError, ValueError, OverflowError, ZeroDivisionError, ArithmeticError) as exc:
            return dict(base, status='NUMERICAL_FAILURE', message='scipy could not evaluate %s for these parameters: %s' % (fn, exc))
        if dist == 'vonmises' and fn in ('ppf', 'isf'):
            low, high = prm['loc'] - math.pi * prm['scale'], prm['loc'] + math.pi * prm['scale']
            values = np.where(np.isneginf(values), low, np.where(np.isposinf(values), high, values))
        if discrete and fn in ('ppf', 'isf'):
            first = float(frozen.support()[0])
            ends = (x == 0.0) if fn == 'ppf' else (x == 1.0)
            values = np.where(ends & (values < first), first, values)  # scipy's lower - 1 convention never leaves the support
    missing = p.get('missing', [])
    for i in range(len(x)):
        if not math.isnan(float(x[i])) and math.isnan(values[i]) and i not in missing and not (out_of_range.size and out_of_range[i]):
            warnings.warn('SciPy returned NaN for a finite/domain-valid distribution input', NumericalWarning)
            break
    n_out = int(out_of_range.sum()) if out_of_range.size else 0
    status = 'OUT_OF_RANGE' if n_out else 'OK'
    if n_out:
        base = dict(base, n_out_of_range=n_out, message='%d requested zipf quantile(s) lie beyond loc + 1e5 (the cdf there is %r): reported as NA' % (n_out, edge if fn == 'ppf' else 1 - edge))
    if p.get('output_prefix'):
        if n_out:
            return dict(base, status=status)
        values = values.copy()
        values[list(missing)] = np.nan
        n_nan = int(sum(1 for i in range(len(x)) if i not in missing and math.isnan(float(x[i]))))
        return dict(base, status='OK', n_values=int(values.size), n_missing=len(missing), n_nan=n_nan, outputs=s8_save_tensor(p, p['key'], values))
    out = values.tolist()
    for i in missing:
        out[i] = 'NA'
    if n_out:
        for i in np.flatnonzero(out_of_range).tolist():
            out[i] = 'NA'
    return dict(base, status=status, values=out)


def stats_rvs(p):
    if p['fn'] != 'rvs':
        return dist_evaluate(p)
    return stats_sample(p)


def stats_sample_compound(np, dist, prm, n, seed):
    """Compound families drawn from two PCG64 streams (seed, and seed jumped once) so that a shorter n is a prefix of a longer one: scipy draws the
    two full-length arrays of these families one after the other from one stream, which breaks the prefix property."""
    a = np.random.Generator(np.random.PCG64(seed))
    b = np.random.Generator(np.random.PCG64(seed).jumped())
    if dist == 'nct':
        z = a.standard_normal(n) + prm['nc']
        v = b.chisquare(prm['df'], n)
        return prm['loc'] + prm['scale'] * z / np.sqrt(v / prm['df'])
    if dist == 'skewnorm':
        u0, v = a.standard_normal(n), b.standard_normal(n)
        d = prm['a'] / math.sqrt(1.0 + prm['a'] ** 2)
        u1 = d * u0 + v * math.sqrt(1.0 - d * d)
        return prm['loc'] + prm['scale'] * np.where(u0 >= 0, u1, -u1)
    if dist == 'betabinom':
        pr = a.beta(prm['a'], prm['b'], n)
        return (b.binomial(int(prm['n']), pr) + prm['loc']).astype(float)
    # levy_stable, S1 parameterisation: Chambers-Mallows-Stuck as in scipy's _rvs_Z1 (uniform angle from stream a, exponential from stream b)
    alpha, beta = prm['alpha'], prm['beta']
    TH = -math.pi / 2 + math.pi * a.random(n)
    W = b.standard_exponential(n)
    aTH, bTH, cosTH, tanTH = alpha * TH, beta * TH, np.cos(TH), np.tan(TH)
    if alpha == 1:
        z = 2 / math.pi * ((math.pi / 2 + bTH) * tanTH - beta * np.log((math.pi / 2 * W * cosTH) / (math.pi / 2 + bTH)))
    elif beta == 0:
        z = W / (cosTH / np.tan(aTH) + np.sin(TH)) * ((np.cos(aTH) + np.sin(aTH) * tanTH) / W) ** (1.0 / alpha)
    else:
        val0 = beta * math.tan(math.pi * alpha / 2)
        th0 = math.atan(val0) / alpha
        val3 = W / (cosTH / np.tan(alpha * (th0 + TH)) + np.sin(TH))
        z = val3 * ((np.cos(aTH) + np.sin(aTH) * tanTH - val0 * (np.sin(aTH) - np.cos(aTH) * tanTH)) / W) ** (1.0 / alpha)
    return prm['loc'] + prm['scale'] * z


DIST_TWO_STREAMS = ('nct', 'skewnorm', 'betabinom', 'levy_stable')


def stats_sample(p):
    import numpy as np
    dist, prm, n = p['dist'], p['params'], p['n']
    s8_cap(n <= S8_ARTIFACT_VALUES, 'sample size')
    rng = np.random.Generator(np.random.PCG64(p['seed']))
    with np.errstate(all='ignore'):
        if dist == 'norm':
            x = rng.normal(prm['loc'], prm['scale'], n)
        elif dist == 't':
            x = prm['loc'] + prm['scale'] * rng.standard_t(prm['df'], n)
        elif dist == 'uniform':
            x = rng.uniform(prm['low'], prm['high'], n)
        elif dist == 'expon':
            x = rng.exponential(prm['scale'], n)
        elif dist == 'gamma':
            x = rng.gamma(prm['shape'], prm['scale'], n)
        elif dist == 'beta':
            x = rng.beta(prm['a'], prm['b'], n)
        elif dist == 'lognorm':
            x = rng.lognormal(prm['mu'], prm['sigma'], n)
        elif dist == 'chi2':
            x = rng.chisquare(prm['df'], n)
        elif dist == 'poisson':
            x = rng.poisson(prm['lam'], n).astype(float)
        elif dist == 'binom':
            x = rng.binomial(int(prm['n']), prm['p'], n).astype(float)
        elif dist in DIST_TWO_STREAMS:
            x = stats_sample_compound(np, dist, prm, n, p['seed'])
        else:
            try:
                x = dist_frozen(dist, prm).rvs(size=n, random_state=rng)
            except (ValueError, OverflowError) as exc:  # numpy refuses integer draws whose mean overflows ("n too large or p too small")
                x, refused = None, str(exc)
    base = {'dist': dist, 'fn': 'rvs', 'params': prm, 'n': n, 'seed': p['seed'], 'generator': 'PCG64'}
    if x is None:
        return dict(base, status='NUMERICAL_FAILURE', message='the sampler cannot draw these parameters: %s' % refused)
    if dist in ('geom', 'nbinom') and (np.asarray(x) >= 2 ** 63 - 1).any():
        return dict(base, status='NUMERICAL_FAILURE', message='the integer sampler saturated the int64 range for these parameters (p is too small): no values')
    x = np.asarray(x, dtype=float)
    if not np.isfinite(x).all():
        return dict(base, status='NUMERICAL_FAILURE', message='the sampler produced non-finite values for these parameters')
    result = dict(base, status='OK')
    if dist in DIST_TWO_STREAMS:
        result['generator'] = 'PCG64'
        result['streams'] = 2
    if p.get('output_prefix'):
        result['outputs'] = s8_save_tensor(p, p['key'], x)
    else:
        s8_cap(n <= 1000, 'inline values')
        result['values'] = x.tolist()
    return result


def s8_operation(name):
    if name in S8B_OPS:
        return s8b_operation(name)
    if name in S8G_OPS:
        return s8g_operation(name)
    operation = {'frame_to_array': frame_to_array, 'frame_to_array_to': frame_to_array, 'array_to_frame': array_to_frame, 'frame_mutate': frame_mutate,
            'frame_merge': frame_merge, 'frame_rolling': frame_rolling, 'ml_metrics': ml_metrics, 'stats_dist': stats_rvs, 'stats_dist_to': stats_rvs}[name]
    if name in ('frame_mutate', 'frame_rolling'):
        return operation  # these bodies already establish the budget once
    def guarded(p):
        result = s7_guard(p, lambda: operation(p))
        if result is None:
            if p.get('output_file') and os.path.exists(p['output_file']):
                os.unlink(p['output_file'])
            return {'status': 'TIME_LIMIT'}
        return result
    return guarded
