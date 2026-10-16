#!/usr/bin/env python3
"""README timing tables from the checked-in measurement files: per-operation-family p50 through the host shim, worker import cost per
import group, the runtime image size history and the hardened-container peaks. Every number cites the docs/measurement-*.json it comes
from; nothing is typed by hand. Usage: python3 scripts/timing-table.py  (prints Markdown)"""
import json, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
D = ROOT / 'docs'


def load(name):
    return json.loads((D / name).read_text())


def p50(entry):
    """p50 wall milliseconds of one case in any of the measurement layouts."""
    if 'wall' in entry:
        return entry['wall']['p50_ms']
    return entry['p50_ms']


def imp(entry):
    if 'worker_import_ms' in entry:
        return entry['worker_import_ms']['p50_ms']
    return entry.get('import_p50_ms')


# (family, file, case key, label) in the order of the README; one representative small call per family.
FAMILIES = [
    ('py_info (numpy only)', 'measurement-s8b2-shim.json', 'py_info', 'py_info'),
    ('array_import / array_inspect (pure C++, no Python)', 'measurement-s1-shim.json', 'array_import', 'array_import'),
    ('la_solve (3x3)', 'measurement-s1-shim.json', 'la_solve', 'la_solve'),
    ('la_lstsq (200 x 5; la_props, la_decomp similar)', 'measurement-s1-shim.json', 'la_lstsq_200x5', 'la_lstsq'),
    ('frames (frame_import ... frame_resample)', 'measurement-s2-review-shim.json', 'frame_import', 'frame_import'),
    ('opt_minimize (Rosenbrock, BFGS)', 'measurement-s3-shim.json', 'opt_minimize_rosenbrock_bfgs', 'opt_minimize'),
    ('opt_linprog (small LP)', 'measurement-s4-review-shim.json', 'LP', 'opt_linprog'),
    ('ode_solve (decay, RK45)', 'measurement-s4-review-shim.json', 'ODE_decay_RK45', 'ode_solve'),
    ('sm_ols (40 x 6)', 'measurement-s5-shim.json', 'sm_ols_40x6', 'sm_ols'),
    ('ml_fit (logistic, 40 x 1)', 'measurement-s6-review-shim.json', 'ml_fit_logistic_40x1', 'ml_fit'),
    ('interp_eval (pchip, 40 nodes)', 'measurement-s7-review-shim.json', 'interp_eval_pchip_40', 'interp_eval'),
    ('frame_to_array (100 x 2)', 'measurement-s8a-review-shim.json', 'frame_to_array_100x2', 'frame_to_array'),
    ('la_matrix_fn (expm 8x8)', 'measurement-s8b1-shim.json', 'la_matrix_fn_expm_8x8', 'la_matrix_fn'),
    ('special_fn (erf, 100 values)', 'measurement-s8b2-shim.json', 'special_fn_erf_100', 'special_fn'),
    ('graph_paths (dijkstra, 6 nodes)', 'measurement-s8b2-shim.json', 'graph_paths_dijkstra_6', 'graph_paths'),
    ('graph_flow (max_flow, 6 nodes)', 'measurement-s8b2-shim.json', 'graph_flow_max_flow_6', 'graph_flow'),
    ('stats_dist (genextreme cdf)', 'measurement-s8b2-shim.json', 'stats_dist_genextreme_cdf', 'stats_dist'),
    ('stats_dist rvs (norm, numpy only)', 'measurement-s8b2-shim.json', 'stats_dist_norm_rvs', 'stats_dist rvs'),
]
IMPORT_GROUPS = [
    ('numpy only (py_info, stats_dist rvs of the S8a families, frame bridge)', 'measurement-s8b2-shim.json', 'py_info'),
    ('scipy.special (special_fn)', 'measurement-s8b2-shim.json', 'special_fn_erf_100'),
    ('scipy.stats (stats_dist evaluations)', 'measurement-s8b2-shim.json', 'stats_dist_genextreme_cdf'),
    ('networkx + scipy.sparse.csgraph (graph_paths, graph_flow)', 'measurement-s8b2-shim.json', 'graph_paths_dijkstra_6'),
    ('scipy.optimize (opt_minimize, opt_least_squares, num_quad, opt_root)', 'measurement-s3-shim.json', 'opt_minimize_rosenbrock_bfgs'),
    ('scipy.optimize HiGHS / scipy.integrate (opt_linprog, ode_solve)', 'measurement-s4-review-shim.json', 'LP'),
    ('scipy.interpolate / signal / spatial (S7)', 'measurement-s7-review-shim.json', 'interp_eval_pchip_40'),
    ('pandas (frames)', 'measurement-s2-review-shim.json', 'frame_import'),
    ('statsmodels (sm_*)', 'measurement-s5-shim.json', 'sm_ols_40x6'),
    ('scikit-learn (ml_*)', 'measurement-s6-review-shim.json', 'ml_fit_logistic_40x1'),
    ('scipy.linalg (la_matrix_fn, la_sparse)', 'measurement-s8b1-shim.json', 'la_matrix_fn_expm_8x8'),
]
SIZES = [  # (stage, file, path to the byte count)
    ('S0 (first runtime)', 'measurement-runtime.json', ('pycli:runtime', 'image_bytes')),
    ('S0 review fix (pruned, fresh base)', 'measurement-review-packaging.json', ('runtime_content_bytes',)),
    ('S2 (pandas)', 'measurement-s2-packaging.json', ('pycli:runtime', 'content_bytes')),
    ('S3 (expressions)', 'measurement-s3-shim.json', ('image_bytes',)),
    ('S4 (HiGHS, ODE)', 'measurement-s4-review-packaging.json', ('pycli:runtime', 'content_bytes')),
    ('S5 (statsmodels)', 'measurement-s5-shim.json', ('image_bytes',)),
    ('S6 (scikit-learn)', 'measurement-s6-review-shim.json', ('image_inspect_bytes',)),
    ('S7 (interp / signal / spatial)', 'measurement-s7-review-shim.json', ('image_bytes',)),
    ('S8a (bridge, metrics)', 'measurement-s8a-review-shim.json', ('image_bytes',)),
    ('S8b-1 (matrix functions, sparse)', 'measurement-s8b1-shim.json', ('image_bytes',)),
    ('S8b-2 + review fixes (networkx; now)', 'measurement-s8b2-shim.json', ('image_bytes',)),
]
PEAKS = [  # hardened-container cgroup peaks per cap run (bytes), where the file records one
    ('S2 frame caps (200000 rows / 500000 cells / 16 MiB CSV)', 'measurement-s2-review-caps.json', ('cgroup_peak_bytes',)),
    ('S4 LP / ODE caps', 'measurement-s4-review-caps.json', ('cgroup_peak_bytes',)),
]


def dig(obj, path):
    for key in path:
        obj = obj[key]
    return obj


def main():
    print('| Operation family (representative small call) | p50 wall ms through the shim | worker import p50 ms | Source |')
    print('|---|---:|---:|---|')
    for family, file, key, _ in FAMILIES:
        data = load(file)
        entry = data['cases'][key] if 'cases' in data else data[key] if key in data else data['pycli:runtime'][key]
        importms = imp(entry)
        print('| %s | %.0f | %s | `docs/%s` (`%s`) |' % (family, p50(entry), ('%.0f' % importms) if importms is not None else 'n/a', file, key))
    print()
    print('| Import group (what the worker imports for the family) | worker import p50 ms | Source |')
    print('|---|---:|---|')
    for group, file, key in IMPORT_GROUPS:
        data = load(file)
        entry = data['cases'][key] if 'cases' in data else data[key]
        print('| %s | %.0f | `docs/%s` (`%s`) |' % (group, imp(entry), file, key))
    print()
    print('| Runtime image (content bytes, `docker image inspect .Size`) | Bytes | MB | Source |')
    print('|---|---:|---:|---|')
    for stage, file, path in SIZES:
        value = dig(load(file), path)
        print('| %s | %d | %.1f | `docs/%s` |' % (stage, value, value / 1e6, file))
    print()
    print('| Hardened-container peak (2 CPUs, 4 GiB, read-only root, `--network=none`) | cgroup peak MB | Source |')
    print('|---|---:|---|')
    for label, file, path in PEAKS:
        value = dig(load(file), path)
        print('| %s | %.0f | `docs/%s` |' % (label, value / 1e6, file))


if __name__ == '__main__':
    main()
