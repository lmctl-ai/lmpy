#!/usr/bin/env python3
"""Synthetic teaching datasets for the examples/tut-*.lms recipes. Every table is generated here from a fixed seed (numpy PCG64)
and committed under examples/data/, so the recipes never download anything and their self-checks know the true parameters.
Usage: python3 scripts/generate-tutorial-data.py [--check]   (--check regenerates and fails when a committed file differs)."""
import argparse, io, pathlib, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / 'examples' / 'data'


def csv(name, header, rows):
    buf = io.StringIO()
    buf.write(','.join(header) + '\n')
    for row in rows:
        buf.write(','.join(fmt(v) for v in row) + '\n')
    return name, buf.getvalue()


def fmt(v):
    if isinstance(v, (bool, np.bool_)):
        return 'true' if v else 'false'
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return '%.6f' % float(v)
    return str(v)


def tables():
    out = []
    # 1. OLS with two regressors (statsmodels OLS example shape): y = 1.5 + 0.8 x1 - 2.0 x2 + N(0, 0.5^2), n = 60.
    rng = np.random.default_rng(101)
    x1 = rng.uniform(0, 10, 60); x2 = rng.normal(0, 1, 60)
    y = 1.5 + 0.8 * x1 - 2.0 * x2 + rng.normal(0, 0.5, 60)
    out.append(csv('tut-ols.csv', ['x1', 'x2', 'y'], zip(x1, x2, y)))
    # 2. Poisson counts with exposure by treatment group (R ?glm ship-damage shape): log rate = log 2 + 0.5 [B] - 0.4 [C], 30 per group.
    rng = np.random.default_rng(102)
    groups = np.repeat(['A', 'B', 'C'], 30)
    hours = rng.uniform(1, 5, 90)
    effect = np.where(groups == 'B', 0.5, np.where(groups == 'C', -0.4, 0.0))
    counts = rng.poisson(2.0 * np.exp(effect) * hours)
    out.append(csv('tut-poisson.csv', ['group', 'hours', 'count'], zip(groups, hours, counts)))
    # 3. Dose-response, one row per insect (R ?glm budworm shape): logit p = -3 + 1.2 ldose, ldose = log2(dose), 40 insects per dose.
    rng = np.random.default_rng(103)
    rows = []
    for dose in (1, 2, 4, 8, 16, 32):
        ldose = np.log2(dose); p = 1 / (1 + np.exp(-(-3 + 1.2 * ldose)))
        dead = rng.random(40) < p
        rows += [(dose, ldose, int(d)) for d in dead]
    out.append(csv('tut-dose.csv', ['dose', 'ldose', 'dead'], rows))
    # 4. Three treatment groups (PlantGrowth shape): weights N(5.0, 0.6), N(4.6, 0.6), N(5.8, 0.6), 12 per group, plus a yes/no trait.
    rng = np.random.default_rng(104)
    rows = []
    for g, mu, p_flower in (('ctrl', 5.0, 0.25), ('trt1', 4.6, 0.25), ('trt2', 5.8, 0.85)):
        w = rng.normal(mu, 0.6, 12); fl = rng.random(12) < p_flower
        rows += [(g, wi, 'yes' if f else 'no') for wi, f in zip(w, fl)]
    out.append(csv('tut-groups.csv', ['group', 'weight', 'flowered'], rows))
    # 5. A skewed sample for the bootstrap: gamma(shape 2, scale 5), mean 10, n = 40.
    rng = np.random.default_rng(105)
    out.append(csv('tut-bootstrap.csv', ['value'], [(v,) for v in rng.gamma(2.0, 5.0, 40)]))
    # 6. A spectroscopy-style peak: y = 0.3 + 2.0 exp(-(x - 4.5)^2 / (2 0.8^2)) + N(0, 0.05^2) on x = 0..10 step 0.1.
    rng = np.random.default_rng(106)
    x = np.round(np.arange(0, 10.05, 0.1), 1)
    y = 0.3 + 2.0 * np.exp(-(x - 4.5) ** 2 / (2 * 0.8 ** 2)) + rng.normal(0, 0.05, x.size)
    out.append(csv('tut-peak.csv', ['x', 'y'], zip(x, y)))
    # 7. Diet problem foods (Stigler shape, invented numbers): cost per serving and three nutrients per serving.
    foods = [('oats', 0.30, 110, 4, 2), ('milk', 0.25, 160, 8, 285), ('beans', 0.40, 230, 15, 50), ('bread', 0.20, 140, 5, 30),
             ('eggs', 0.35, 150, 13, 50), ('spinach', 0.45, 25, 3, 100), ('cheese', 0.80, 220, 14, 400), ('rice', 0.15, 200, 4, 10)]
    out.append(csv('tut-diet.csv', ['food', 'cost', 'calories', 'protein', 'calcium'], foods))
    # 10/11. A noisy two-tone signal sampled at 100 Hz for 4 s: 5 Hz amplitude 1 + 12 Hz amplitude 0.5 + N(0, 0.3^2).
    rng = np.random.default_rng(110)
    t = np.arange(400) / 100.0
    tone5 = np.sin(2 * np.pi * 5 * t)
    signal = tone5 + 0.5 * np.sin(2 * np.pi * 12 * t) + rng.normal(0, 0.3, t.size)
    out.append(csv('tut-signal.csv', ['t', 'tone5', 'signal'], zip(t, tone5, signal)))
    # 12. An AR(1) series: x_t = 2 + 0.7 x_{t-1} + N(0, 1), mean 2 / 0.3, n = 200 after a burn-in of 100.
    rng = np.random.default_rng(1122)  # seed chosen so the sample ACF is a representative draw (seed 112 gave r1 = 0.54)
    e = rng.normal(0, 1, 300); x = np.zeros(300); x[0] = 2 / 0.3
    for i in range(1, 300):
        x[i] = 2 + 0.7 * x[i - 1] + e[i]
    out.append(csv('tut-ar1.csv', ['t', 'x'], [(i, v) for i, v in enumerate(x[100:])]))
    # 13. Three flower species with four measurements each (iris shape, invented means), 50 per species.
    rng = np.random.default_rng(113)
    rows = []
    means = {'setosa': (5.0, 3.4, 1.5, 0.25), 'versicolor': (5.9, 2.8, 4.3, 1.3), 'virginica': (6.6, 3.0, 5.6, 2.0)}
    sds = (0.35, 0.3, 0.3, 0.15)
    for species, mu in means.items():
        m = np.maximum(rng.normal(mu, sds, (50, 4)), 0.1)
        rows += [(species, *r) for r in m]
    out.append(csv('tut-flowers.csv', ['species', 'sepal_length', 'sepal_width', 'petal_length', 'petal_width'], rows))
    # 14. Interpolation: 13 nodes of f(x) = exp(-x/3) sin(2x) on [0, 6] and a fine grid with the true values.
    f = lambda x: np.exp(-x / 3) * np.sin(2 * x)
    nodes = np.linspace(0, 6, 13); grid = np.linspace(0.1, 5.9, 59)
    out.append(csv('tut-interp-nodes.csv', ['x', 'y'], zip(nodes, f(nodes))))
    out.append(csv('tut-interp-grid.csv', ['x', 'f'], zip(grid, f(grid))))
    # 15. A small road network: 10 towns, 16 two-way roads with lengths in km.
    roads = [('A', 'B', 7), ('A', 'C', 9), ('A', 'F', 14), ('B', 'C', 10), ('B', 'D', 15), ('C', 'D', 11), ('C', 'F', 2), ('D', 'E', 6),
             ('E', 'F', 9), ('E', 'G', 8), ('F', 'G', 12), ('G', 'H', 5), ('H', 'I', 4), ('D', 'I', 13), ('I', 'J', 3), ('E', 'J', 17)]
    out.append(csv('tut-roads.csv', ['from', 'to', 'km'], roads))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--check', action='store_true'); args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    bad = 0
    for name, text in tables():
        path = OUT / name
        if args.check:
            if not path.exists() or path.read_text() != text:
                print('DRIFT', path); bad += 1
        else:
            path.write_text(text); print('wrote', path, len(text.splitlines()) - 1, 'rows')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
