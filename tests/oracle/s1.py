"""Independent S1 references: exact rational algebra, a strict safetensors reader and
NIST StRD data. Nothing here imports or calls the pycli worker."""
import json, math, struct
from fractions import Fraction as F
import numpy as np

NUMPY_DTYPE = {'F64': '<f8', 'F32': '<f4', 'I64': '<i8', 'I32': '<i4', 'U8': 'u1', 'BOOL': '?'}


def read_safetensors(path):
    """Strict independent reader: returns {key: ndarray}; asserts a canonical layout."""
    raw = open(path, 'rb').read()
    n = struct.unpack('<Q', raw[:8])[0]
    assert 8 + n <= len(raw)
    header = json.loads(raw[8:8 + n])
    assert (8 + n) % 8 == 0, 'payload must stay 8-byte aligned'
    payload = raw[8 + n:]
    out, cursor = {}, 0
    for key, entry in sorted(((k, v) for k, v in header.items() if k != '__metadata__'), key=lambda kv: kv[1]['data_offsets'][0]):
        begin, end = entry['data_offsets']
        assert begin == cursor, 'offsets must be contiguous'
        cursor = end
        out[key] = np.frombuffer(payload[begin:end], dtype=NUMPY_DTYPE[entry['dtype']]).reshape(entry['shape']).copy()
    assert cursor == len(payload), 'data section fully covered'
    return out


def write_safetensors(path, tensors, header_pad=0, metadata=None):
    """tensors: {key: ndarray}. Builds a valid file (optionally with extra header padding)."""
    names = {np.dtype('<f8'): 'F64', np.dtype('<f4'): 'F32', np.dtype('<i8'): 'I64', np.dtype('<i4'): 'I32', np.dtype('u1'): 'U8', np.dtype('?'): 'BOOL'}
    header, payload = {}, b''
    for key, array in tensors.items():
        array = np.ascontiguousarray(array)
        header[key] = {'dtype': names[array.dtype], 'shape': list(array.shape), 'data_offsets': [len(payload), len(payload) + array.nbytes]}
        payload += array.tobytes()
    if metadata:
        header['__metadata__'] = metadata
    text = json.dumps(header, separators=(',', ':')).encode()
    text += b' ' * ((-len(text)) % 8 + header_pad * 8)
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(text)) + text + payload)


def fmat(a):
    return [[F(x) for x in row] for row in a]


def gauss_jordan(m, columns):
    """Reduce augmented matrix m in place over Q; returns (rank over the first `columns`, sign of row swaps)."""
    rows, rank, sign = len(m), 0, 1
    for c in range(columns):
        pivot = next((r for r in range(rank, rows) if m[r][c] != 0), None)
        if pivot is None:
            continue
        if pivot != rank:
            m[rank], m[pivot] = m[pivot], m[rank]
            sign = -sign
        lead = m[rank][c]
        m[rank] = [v / lead for v in m[rank]]
        for r in range(rows):
            if r != rank and m[r][c] != 0:
                factor = m[r][c]
                m[r] = [a - factor * b for a, b in zip(m[r], m[rank])]
        rank += 1
    return rank, sign


def det_exact(a):
    m, n = fmat(a), len(a)
    value = F(1)
    for c in range(n):
        pivot = next((r for r in range(c, n) if m[r][c] != 0), None)
        if pivot is None:
            return F(0)
        if pivot != c:
            m[c], m[pivot] = m[pivot], m[c]
            value = -value
        value *= m[c][c]
        for r in range(c + 1, n):
            factor = m[r][c] / m[c][c]
            m[r] = [x - factor * y for x, y in zip(m[r], m[c])]
    return value


def rank_exact(a):
    m = fmat(a)
    return gauss_jordan(m, len(a[0]))[0]


def inverse_exact(a):
    n = len(a)
    m = [row + [F(int(i == j)) for j in range(n)] for i, row in enumerate(fmat(a))]
    assert gauss_jordan(m, n)[0] == n, 'singular'
    return [row[n:] for row in m]


def lstsq_exact(a, b):
    """Exact least squares of the doubles actually posed (full column rank), via normal equations over Q."""
    A, B = fmat(a), [[F(v)] for v in b] if np.ndim(b) == 1 else fmat(b)
    n, k = len(A[0]), len(B[0])
    m = [[sum(A[r][i] * A[r][j] for r in range(len(A))) for j in range(n)] + [sum(A[r][i] * B[r][c] for r in range(len(A))) for c in range(k)] for i in range(n)]
    assert gauss_jordan(m, n)[0] == n, 'rank deficient'
    return [[m[i][n + c] for c in range(k)] for i in range(n)]


def lre(computed, certified):
    """NIST log relative error: -log10(|c - q| / |q|), capped at 15 for exact agreement."""
    error = abs(float(computed) - float(certified))
    return 15.0 if error == 0 else min(15.0, -math.log10(error / abs(float(certified))))


# NIST StRD Longley (certified to 15 digits). Columns: x1..x6 (GNP deflator, GNP, unemployed,
# armed forces, population, year); response y = employed.
LONGLEY_Y = "60323 61122 60171 61187 63221 63639 64989 63761 66019 67857 68169 66513 68655 69564 69331 70551".split()
LONGLEY_X = [
    "83.0 88.5 88.2 89.5 96.2 98.1 99.0 100.0 101.2 104.6 108.4 110.8 112.6 114.2 115.7 116.9".split(),
    "234289 259426 258054 284599 328975 346999 365385 363112 397469 419180 442769 444546 482704 502601 518173 554894".split(),
    "2356 2325 3682 3351 2099 1932 1870 3578 2904 2822 2936 4681 3813 3931 4806 4007".split(),
    "1590 1456 1616 1650 3099 3594 3547 3350 3048 2857 2798 2637 2552 2514 2572 2827".split(),
    "107608 108632 109773 110929 112075 113270 115094 116219 117388 118734 120445 121950 123366 125368 127852 130081".split(),
    [str(1947 + i) for i in range(16)],
]
LONGLEY_A = [[1.0] + [float(col[i]) for col in LONGLEY_X] for i in range(16)]
LONGLEY_B = [float(v) for v in LONGLEY_Y]
LONGLEY_CERTIFIED = [-3482258.63459582, 15.0618722713733, -0.358191792925910E-01, -2.02022980381683,
                     -1.03322686717359, -0.511041056535807E-01, 1829.15146461355]
LONGLEY_RESIDUAL_SS = 836424.055505915

# NIST Wampler 1: y = 1 + x + x^2 + x^3 + x^4 + x^5, x = 0..20; certified coefficients are all exactly 1.
WAMPLER1_A = [[float(x) ** p for p in range(6)] for x in range(21)]
WAMPLER1_B = [float(sum(x ** p for p in range(6))) for x in range(21)]
