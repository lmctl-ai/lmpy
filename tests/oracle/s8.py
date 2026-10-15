"""Independent S8a references: a numpy-only table reader (no pandas, no pycli code), hand-built joins, rolling windows and
classification/regression metrics in pure Python, and hand-coded moments / Kolmogorov-Smirnov statistics for the sampler."""
import json, math
import numpy as np


def read_table(path):
    """{'columns': [...], 'meta': {...}, 'n': rows, 'data': {col: [python values, None where masked]}} straight from the NPZ members."""
    with np.load(path, allow_pickle=False) as z:
        metadata = json.loads(z['__pycli_table__'].tobytes().decode('utf-8'))
        names = metadata['columns']
        data = {}
        for i, name in enumerate(names):
            raw, mask = z['c%d' % i], z['m%d' % i]
            spec = metadata['column_meta'][name]
            dtype = spec['dtype']
            values = []
            for k in range(len(raw)):
                if mask[k]:
                    values.append(None)
                elif dtype == 'string':
                    values.append(bytes(raw[k]).decode('utf-8'))
                elif dtype == 'category':
                    values.append(spec['categories'][int(raw[k])])
                elif dtype == 'bool':
                    values.append(bool(raw[k]))
                elif dtype in ('int64', 'datetime64[ns]'):
                    values.append(int(raw[k]))
                else:
                    values.append(float(raw[k]))
            data[name] = values
    return {'columns': names, 'meta': metadata['column_meta'], 'n': metadata['n_rows'], 'data': data}


def same(a, b, tol=0.0):
    """Element-wise equality where None equals None and NaN equals NaN; floats within tol (relative+absolute)."""
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x is None or y is None:
            if x is not y:
                return False
        elif isinstance(x, float) or isinstance(y, float):
            if math.isnan(x) and math.isnan(y):
                continue
            if math.isinf(x) or math.isinf(y):
                if x != y:
                    return False
                continue
            if abs(x - y) > tol * max(1.0, abs(x), abs(y)):
                return False
        elif x != y:
            return False
    return True


def hand_merge(left, right, how, lkeys, rkeys):
    """Row-index pairs (i, j) in the documented order; None marks an absent side. Rows with an NA key never match (SQL semantics).
    left/right: {col: [values]}; keys: lists of column names. inner/left are left-driven, right is right-driven, outer is left-driven
    followed by the unmatched right rows in right order."""
    nl, nr = len(next(iter(left.values()))), len(next(iter(right.values())))

    def key(table, cols, i):
        k = tuple(table[c][i] for c in cols)
        return None if any(v is None for v in k) else k
    lk = [key(left, lkeys, i) for i in range(nl)]
    rk = [key(right, rkeys, j) for j in range(nr)]
    matches_of_left = [[j for j in range(nr) if lk[i] is not None and rk[j] == lk[i]] for i in range(nl)]
    matches_of_right = [[i for i in range(nl) if rk[j] is not None and lk[i] == rk[j]] for j in range(nr)]
    pairs = []
    if how in ('inner', 'left', 'outer'):
        for i in range(nl):
            if matches_of_left[i]:
                pairs += [(i, j) for j in matches_of_left[i]]
            elif how != 'inner':
                pairs.append((i, None))
        if how == 'outer':
            pairs += [(None, j) for j in range(nr) if not matches_of_right[j]]
    else:
        for j in range(nr):
            if matches_of_right[j]:
                pairs += [(i, j) for i in matches_of_right[j]]
            else:
                pairs.append((None, j))
    return pairs


def hand_rolling(values, fn, window, min_periods, center=False, times=None, span=None):
    """values: list of float|None (None = missing). Count windows are trailing (or centred: [i - w//2, i + (w-1)//2]); time windows are
    the rows j <= i with times[i] - span < times[j] <= times[i]. Missing observations are skipped; a result needs min_periods
    observations (std/var also need two); count also requires min_periods observations."""
    n = len(values)
    out = []
    for i in range(n):
        if times is not None:
            idx = [j for j in range(i + 1) if times[i] - span < times[j] <= times[i]]
        elif center:
            idx = list(range(max(0, i - window // 2), min(n - 1, i + (window - 1) // 2) + 1))
        else:
            idx = list(range(max(0, i - window + 1), i + 1))
        v = [values[j] for j in idx if values[j] is not None]
        c = len(v)
        if fn == 'count':
            out.append(c if c >= min_periods else None)
            continue
        need = max(min_periods, 2 if fn in ('std', 'var') else 1)
        if c < need or c == 0:
            out.append(None)
            continue
        mean = math.fsum(v) / c
        if fn == 'sum':
            out.append(math.fsum(v))
        elif fn == 'mean':
            out.append(mean)
        elif fn in ('std', 'var'):
            var = math.fsum((x - mean) ** 2 for x in v) / (c - 1)
            out.append(math.sqrt(var) if fn == 'std' else var)
        elif fn == 'min':
            out.append(min(v))
        elif fn == 'max':
            out.append(max(v))
        else:
            s = sorted(v)
            out.append(s[c // 2] if c % 2 else (s[c // 2 - 1] + s[c // 2]) / 2)
    return out


def labels_sorted(values):
    try:
        return sorted(set(values), key=lambda v: (0, int(v)) if str(v).lstrip('-').isdigit() else (1, str(v)))
    except Exception:
        return sorted(set(values), key=str)


def hand_confusion(y_true, y_pred, labels):
    index = {l: k for k, l in enumerate(labels)}
    m = [[0] * len(labels) for _ in labels]
    for t, p in zip(y_true, y_pred):
        m[index[t]][index[p]] += 1
    return m


def hand_prf(y_true, y_pred, average, pos=None, zero_division=None):
    """precision, recall, f1 by average; None where undefined and zero_division is None."""
    labels = labels_sorted(list(y_true) + list(y_pred))

    def one(label):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        return tp, fp, fn

    def ratio(a, b):
        return a / b if b else zero_division

    def metrics(tp, fp, fn):
        p, r = ratio(tp, tp + fp), ratio(tp, tp + fn)
        f = ratio(2 * tp, 2 * tp + fp + fn)
        return p, r, f
    if average == 'binary':
        return metrics(*one(pos))
    if average == 'micro':
        tp = sum(one(l)[0] for l in labels)
        fp = sum(one(l)[1] for l in labels)
        fn = sum(one(l)[2] for l in labels)
        return metrics(tp, fp, fn)
    per = [metrics(*one(l)) for l in labels]
    weights = [sum(1 for t in y_true if t == l) for l in labels] if average == 'weighted' else [1] * len(labels)
    out = []
    for k in range(3):
        column = [m[k] for m in per]
        if any(v is None for v in column):
            out.append(None)
        else:
            out.append(sum(w * v for w, v in zip(weights, column)) / sum(weights))
    return tuple(out)


def hand_auc(y_true, score, pos):
    """P(score_pos > score_neg) + 0.5 P(tie), by brute-force pair counting."""
    positives = [s for t, s in zip(y_true, score) if t == pos]
    negatives = [s for t, s in zip(y_true, score) if t != pos]
    total = 0.0
    for a in positives:
        for b in negatives:
            total += 1.0 if a > b else 0.5 if a == b else 0.0
    return total / (len(positives) * len(negatives))


def ks_statistic(sample, cdf):
    s = sorted(sample)
    n = len(s)
    return max(max(cdf(x) - k / n, (k + 1) / n - cdf(x)) for k, x in enumerate(s))


def norm_cdf(x, mu=0.0, sigma=1.0):
    return 0.5 * (1.0 + math.erf((x - mu) / (sigma * math.sqrt(2.0))))
