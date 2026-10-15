"""Expression-semantics grid shared by pycli and R (no numpy: the host script imports this too)."""
import math

SEMANTIC_X = [-math.inf, -1000, -2.5, -2, -1, -0.5, 0, 1e-300, 0.5, 1, 2, 2.5, 3, 10, 30, 100, 1e5, 1e300, math.inf, math.nan]
SEMANTIC_Y = [-math.inf, -3, -2.5, -1, -0.5, 0, 0.5, 1, 2, 2.4, 2.5, 3, 5, 30, 31, 100, 1e10, math.inf, math.nan]


def token(value):
    """Canonical text of a result: the same strings the R side prints."""
    value = float(value)
    if math.isnan(value):
        return 'NaN'
    if math.isinf(value):
        return 'Inf' if value > 0 else '-Inf'
    return repr(value)


def semantic_expressions(spec):
    out = []
    for name, info in spec['functions'].items():
        if info.get('reducer') or name == 'ifelse':
            continue
        out.append(f'{name}(x)' if info['arity'] == 1 else f'{name}(x,y)')
    out += ['x+y', 'x-y', 'x*y', 'x/y', 'x^y', '-x', '--x', 'pi*x', 'e^x']
    out += [f'ifelse(x {op} y,1,2)' for op in ('<', '<=', '>', '>=', '==', '!=')]
    out += ['ifelse(x<1 & y<1,1,2)', 'ifelse(x<1 | y<1,1,2)', 'ifelse(x>0 & y,1,2)', 'ifelse(x>0 | y,1,2)', 'ifelse(x<y,x,y)',
            'ifelse(x<y,x/0,2)', 'ifelse(x<y & x>y,1,2)', 'ifelse(x<y | x>=y,1,2)']
    return out


def uses_y(expression):
    import re
    return re.search(r'\by\b', expression) is not None


def semantic_cell_matches(expression, y, expected, got):
    """Compare one grid cell. Tokens must match exactly for NaN/Inf/ERR; finite values to 1e-9 relative. choose/lchoose with a
    huge k (>= 1e9) are the one documented numerical difference between R and SciPy: only the finite/non-finite class must agree."""
    if expected == got:
        return True
    special = ('NaN', 'Inf', '-Inf', 'ERR')
    if expected in special or got in special:
        return False
    if expression in ('choose(x,y)', 'lchoose(x,y)') and y is not None and abs(y) >= 1e9:
        return True
    a, b = float(expected), float(got)
    return abs(a - b) <= 1e-9 * max(abs(a), abs(b), 1e-300)
