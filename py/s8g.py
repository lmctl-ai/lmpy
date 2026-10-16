"""S8b-2 graph operations: graph_paths (networkx for one source and k shortest simple paths, scipy.sparse.csgraph for the all-pairs matrix) and graph_flow
(networkx Dinitz / min cut / network simplex). Assembled after s8b.py. The C++ front end (src/s8g.hpp) has validated the table, the node class, the caps and
every node argument; the worker repeats the data-level checks and re-verifies every flow it returns against the edge list."""
S8G_OPS = ('graph_paths', 'graph_paths_to', 'graph_flow', 'graph_flow_to')
S8G_INLINE_FLOW_EDGES = 300
S8G_TOL = 1e-9
S8G_LIMIT = 1e15


def g_node_column(np, cmeta, raw, mask, name):
    if mask.any():
        raise BadData('endpoint column %s has a missing value' % name, 'NA_NOT_ALLOWED')
    dtype = cmeta['dtype']
    if dtype == 'int64':
        return [int(v) for v in raw.tolist()]
    if dtype == 'string':
        return [bytes(v).decode('utf-8') for v in raw]
    if dtype == 'category':
        levels = cmeta['categories']
        return [levels[int(v)] for v in raw]
    raise BadData('node ids are int64, string or category', 'BAD_DTYPE')


def g_number_column(np, cmeta, raw, mask, name):
    if mask.any():
        raise BadData('column %s has %d missing value(s)' % (name, int(mask.sum())), 'NA_NOT_ALLOWED')
    if cmeta['dtype'] == 'int64':
        return [int(v) for v in raw.tolist()]
    if cmeta['dtype'] != 'float64':
        raise BadData('column %s must be int64 or float64' % name, 'BAD_DTYPE')
    values = raw.astype(float)
    if not np.isfinite(values).all():
        raise BadData('column %s holds NaN or infinite values' % name, 'NON_FINITE')
    return values.tolist()


def g_read(np, p, extra):
    """(from_nodes, to_nodes, {extra column: values}) in row order."""
    if 'frame_file' in p:
        names, cmeta, n, raws, masks = s8_read(p['frame_file'])
        index = {c: i for i, c in enumerate(names)}
        src = g_node_column(np, cmeta[p['from_column']], raws[index[p['from_column']]], masks[index[p['from_column']]], p['from_column'])
        dst = g_node_column(np, cmeta[p['to_column']], raws[index[p['to_column']]], masks[index[p['to_column']]], p['to_column'])
        columns = {c: g_number_column(np, cmeta[c], raws[index[c]], masks[index[c]], c) for c in extra}
        return src, dst, columns
    records = p['records']
    src, dst = [r[p['from_column']] for r in records], [r[p['to_column']] for r in records]
    return src, dst, {c: [r[c] for r in records] for c in extra}


def g_order(nodes):
    return sorted(nodes)


def g_budget(p, body):
    return s7_guard(p, body)


def g_inline(ok, what):
    if not ok:
        raise BadData('%s does not fit the inline result: use the _to twin to write an artifact' % what, 'OUTPUT_TOO_LARGE')


def g_is_file(p):
    return 'output_file' in p or 'output_prefix' in p


# --------------------------------------------------------------------------------------------------------------------- paths
def g_tree(adj, dist, source):
    """Predecessor and hop count of every reached node from a breadth-first walk over the tight edges (dist[u] + w == dist[v]): an acyclic tree even with
    zero-weight cycles, with the fewest edges among shortest paths."""
    pred, hops = {source: None}, {source: 0}
    queue = [source]
    for u in queue:
        for v, w in adj[u].items():
            if v not in hops and v in dist and dist[u] + w == dist[v]:
                pred[v] = u
                hops[v] = hops[u] + 1
                queue.append(v)
    return pred, hops


def g_path(pred, target):
    out = [target]
    while pred[out[-1]] is not None:
        out.append(pred[out[-1]])
    out.reverse()
    return out


def g_adjacency(src, dst, weights, directed):
    adj = {}
    for k, (u, v) in enumerate(zip(src, dst)):
        adj.setdefault(u, {})
        adj.setdefault(v, {})
        if u == v:
            continue
        w = weights[k] if weights is not None else 1
        for a, b in (((u, v),) if directed else ((u, v), (v, u))):
            if b not in adj[a] or w < adj[a][b]:
                adj[a][b] = w
    return adj


def g_all_pairs(np, p, adj, weights_used):
    from scipy import sparse
    from scipy.sparse import csgraph
    nodes = g_order(adj)
    index = {u: i for i, u in enumerate(nodes)}
    rows, cols, data = [], [], []
    for u, nbrs in adj.items():
        for v, w in nbrs.items():
            rows.append(index[u])
            cols.append(index[v])
            data.append(float(w))
    n = len(nodes)
    s7_cap(n <= 5000 and n * n <= 1900000, 'all-pairs matrix')
    matrix = sparse.csr_matrix((np.asarray(data, dtype=float), (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64))), shape=(n, n))
    matrix.sum_duplicates()
    try:
        if p['method'] == 'bfs':
            d = csgraph.dijkstra(matrix, directed=True, unweighted=True)
        elif p['method'] == 'dijkstra':
            d = csgraph.dijkstra(matrix, directed=True)
        else:
            d = csgraph.johnson(matrix, directed=True)
    except csgraph.NegativeCycleError:
        return None, nodes
    return np.where(np.isinf(d), np.nan, d), nodes


def g_save(p, items):
    """Raw tensors (NaN allowed: it marks unreachable pairs); integer arrays are stored as int64."""
    import numpy as np
    outputs = []
    for i, (key, array) in enumerate(items):
        array = np.ascontiguousarray(array)
        dtype = '<i8' if np.issubdtype(array.dtype, np.integer) else '<f8'
        array = np.ascontiguousarray(array, dtype=dtype)
        s8_cap(array.size <= 1900000, 'artifact values')
        array.tofile(p['output_prefix'] + str(i) + '.bin')
        outputs.append({'key': key, 'dtype': 'I64' if dtype == '<i8' else 'F64', 'shape': list(array.shape)})
    return outputs


def g_node_array(np, nodes, kind):
    if kind == 'int':
        return np.asarray(nodes, dtype='<i8'), {'dtype': 'int64'}
    encoded = [n.encode('utf-8') for n in nodes]
    width = max(1, max((len(e) for e in encoded), default=1))
    return np.asarray(encoded, dtype='S%d' % width), {'dtype': 'string'}


def graph_paths(p):
    import numpy as np
    import networkx as nx
    method, directed, kind = p['method'], p['directed'], p['node_kind']
    src, dst, columns = g_read(np, p, [p['weight_column']] if p.get('weight_column') else [])
    weights = columns.get(p.get('weight_column')) if p.get('weight_column') else None
    if method == 'dijkstra' and weights is not None and min(weights) < 0:
        raise BadData('dijkstra needs non-negative weights', 'BAD_PARAM')
    adj = g_adjacency(src, dst, weights, directed)
    n_edges = sum(len(v) for v in adj.values()) // (1 if directed else 2)
    result = {'status': 'OK', 'method': method, 'directed': directed, 'n_nodes': len(adj), 'n_edges': n_edges}
    if p['all_pairs']:
        def body():
            return g_all_pairs(np, p, adj, weights is not None)
        got = g_budget(p, body)
        if got is None:
            return dict(result, status='TIME_LIMIT')
        matrix, nodes = got
        if matrix is None:
            return dict(result, status='NEGATIVE_CYCLE', message='the graph has a negative cycle: shortest paths are undefined')
        result['n_unreachable_pairs'] = int(np.isnan(matrix).sum())
        result['outputs'] = g_save(p, [('distance', matrix), ('nodes', np.asarray(nodes, dtype=np.int64))])
        return result
    source = p['source']
    G = nx.DiGraph() if directed else nx.Graph()
    for u, nbrs in adj.items():
        G.add_node(u)
    for u, nbrs in adj.items():
        for v, w in nbrs.items():
            if directed or u <= v:
                G.add_edge(u, v, weight=w)
    weight = None if method == 'bfs' else 'weight'
    state = {}

    def body():
        if method == 'bfs':
            dist = dict(nx.single_source_shortest_path_length(G, source))
        elif method == 'dijkstra':
            dist = dict(nx.single_source_dijkstra_path_length(G, source, weight='weight'))
        else:
            try:
                dist = dict(nx.single_source_bellman_ford_path_length(G, source, weight='weight'))
            except nx.NetworkXUnbounded:
                cycle = nx.find_negative_cycle(G, source, weight='weight')
                if len(cycle) > 1 and cycle[0] == cycle[-1]:
                    cycle = cycle[:-1]
                state['cycle'] = cycle
                return 'NEGATIVE_CYCLE'
        state['dist'] = dist
        if p.get('k_shortest') and p.get('target') is not None and p['target'] in dist:
            from itertools import islice
            found = []
            for path in islice(nx.shortest_simple_paths(G, source, p['target'], weight=weight), p['k_shortest']):
                length = sum(G[a][b]['weight'] if weight else 1 for a, b in zip(path, path[1:]))
                found.append({'nodes': path, 'length': length, 'hops': len(path) - 1})
            state['paths'] = found
        return 'OK'
    outcome = g_budget(p, body)
    if outcome is None:
        return dict(result, status='TIME_LIMIT')
    if outcome == 'NEGATIVE_CYCLE':
        return dict(result, status='NEGATIVE_CYCLE', cycle=state['cycle'], message='a negative cycle is reachable from the source: shortest paths are undefined')
    dist = state['dist']
    pred, hops = g_tree(adj if method != 'bfs' else {u: {v: 1 for v in nbrs} for u, nbrs in adj.items()}, dist, source)
    result['n_reachable'] = len(dist)
    result['n_unreachable'] = len(adj) - len(dist)
    target = p.get('target')
    if not g_is_file(p) and target is not None:
        if target not in dist:
            return dict(result, status='UNREACHABLE', target=target, message='the target cannot be reached from the source')
        if 'paths' in state:
            result['paths'] = state['paths']
            result['k_found'] = len(state['paths'])
            total = sum(len(q['nodes']) + 2 for q in state['paths'])
            g_inline(total <= 900, 'k_shortest result')
            return result
        path = g_path(pred, target)
        g_inline(len(path) <= 900, 'path length')
        result.update({'target': target, 'path': path, 'length': dist[target], 'hops': len(path) - 1})
        return result
    reached = g_order(dist)
    if not g_is_file(p):
        g_inline(2 * len(reached) <= 900, 'distance listing')
        result.update({'nodes': reached, 'distance': [dist[u] for u in reached]})
        return result
    nodes = g_order(adj)
    names = ['node', 'distance', 'hops', 'predecessor']
    node_raw, node_meta = g_node_array(np, nodes, kind)
    reachable = np.asarray([u in dist for u in nodes], dtype=bool)
    distance = np.asarray([float(dist[u]) if u in dist else np.nan for u in nodes], dtype='<f8')
    hop_raw = np.asarray([hops.get(u, 0) for u in nodes], dtype='<i8')
    pred_values = [pred[u] if u in pred and pred[u] is not None else None for u in nodes]
    pred_mask = np.asarray([v is None for v in pred_values], dtype=bool)
    if kind == 'int':
        pred_raw = np.asarray([0 if v is None else v for v in pred_values], dtype='<i8')
    else:
        pred_raw, _ = g_node_array(np, ['' if v is None else v for v in pred_values], kind)
    raws = [node_raw, distance, hop_raw, pred_raw]
    masks = [np.zeros(len(nodes), dtype=bool), ~reachable, ~reachable, pred_mask]
    cmeta = {'node': node_meta, 'distance': {'dtype': 'float64'}, 'hops': {'dtype': 'int64'}, 'predecessor': dict(node_meta)}
    s8_write(p, names, cmeta, len(nodes), raws, masks)
    return s8_summary(names, cmeta, len(nodes), raws, masks, result)


# ---------------------------------------------------------------------------------------------------------------------- flow
def g_edge_table(np, p):
    src, dst, columns = g_read(np, p, [p['capacity_column']] + ([p['cost_column']] if p.get('cost_column') else []))
    caps = columns[p['capacity_column']]
    if any(c < 0 for c in caps):
        raise BadData('capacities must be non-negative', 'BAD_DATA')
    if any(c > S8G_LIMIT for c in caps):
        raise BadData('capacities must not exceed 1e15', 'BAD_DATA')
    costs = columns[p['cost_column']] if p.get('cost_column') else None
    return src, dst, caps, costs


def g_write_table(p, names, columns, result):
    """Write a flow / cut table whose first two columns are node ids (a zero-row table is a valid answer)."""
    import numpy as np
    kind, n = p['node_kind'], len(columns[0])
    node_a, meta_a = g_node_array(np, columns[0], kind)
    node_b, _ = g_node_array(np, columns[1], kind)
    value = np.asarray(columns[2], dtype='<f8') if any(isinstance(v, float) for v in columns[2]) else np.asarray(columns[2], dtype='<i8')
    cmeta = {names[0]: meta_a, names[1]: dict(meta_a), names[2]: {'dtype': 'float64' if value.dtype.kind == 'f' else 'int64'}}
    masks = [np.zeros(n, dtype=bool) for _ in names]
    s8_write(p, names, cmeta, n, [node_a, node_b, value], masks)
    return s8_summary(names, cmeta, n, [node_a, node_b, value], masks, result)


def graph_flow(p):
    import numpy as np
    import networkx as nx
    from networkx.algorithms.flow import dinitz
    kind = p['kind']
    src, dst, caps, costs = g_edge_table(np, p)
    integral = all(isinstance(c, int) for c in caps) and (costs is None or all(isinstance(c, int) for c in costs))
    result = {'status': 'OK', 'kind': kind, 'integral': bool(integral), 'n_nodes': 0}
    if kind == 'min_cost_flow':
        seen = set()
        for u, v in zip(src, dst):
            if (u, v) in seen:
                raise BadData('min_cost_flow needs one edge per (from, to) pair; (%r, %r) is repeated' % (u, v), 'DUPLICATE_EDGE')
            seen.add((u, v))
        if not (all(float(c).is_integer() for c in caps) and all(float(c).is_integer() for c in costs)):
            raise BadData('min_cost_flow needs integer capacities and costs (network simplex is exact only for integers)', 'NOT_INTEGER')
        caps, costs = [int(c) for c in caps], [int(c) for c in costs]
        if max(abs(c) for c in costs + caps) > 10 ** 12:
            raise BadData('capacities and costs must be at most 1e12 in magnitude for min_cost_flow', 'BAD_DATA')
        result['integral'] = True
    merged = {}
    for u, v, c in zip(src, dst, caps):
        if u != v:
            merged[(u, v)] = merged.get((u, v), 0) + c
    if integral or kind == 'min_cost_flow':
        merged = {k: int(v) for k, v in merged.items()}
    scale = max([1.0] + [float(c) for c in merged.values()])
    tolerance = 0.0 if result['integral'] else S8G_TOL * scale
    result['tolerance'] = tolerance
    G = nx.DiGraph()
    for node in set(src) | set(dst):
        G.add_node(node)
    cost_of = {}
    if kind == 'min_cost_flow':
        for u, v, k in zip(src, dst, costs):
            cost_of[(u, v)] = k
        for (u, v), c in merged.items():
            G.add_edge(u, v, capacity=c, weight=cost_of[(u, v)])
    else:
        for (u, v), c in merged.items():
            G.add_edge(u, v, capacity=c)
    result['n_nodes'] = G.number_of_nodes()
    state = {}
    if kind == 'min_cost_flow':
        if p.get('supplies') is not None:
            supplies = {pair[0]: int(pair[1]) for pair in p['supplies']}
        else:
            supplies = {p['source']: int(p['flow_value']), p['sink']: -int(p['flow_value'])}
        if sum(supplies.values()) != 0:
            return dict(result, status='UNBALANCED', message='the supplies sum to %d: supply and demand must balance' % sum(supplies.values()))
        for node, amount in supplies.items():
            G.nodes[node]['demand'] = -amount

        def body():
            try:
                state['cost'], state['flow'] = nx.network_simplex(G)
            except nx.NetworkXUnfeasible:
                return 'INFEASIBLE'
            except nx.NetworkXUnbounded:
                return 'UNBOUNDED'
            return 'OK'
    else:
        source, sink = p['source'], p['sink']

        def body():
            if kind == 'max_flow':
                state['value'], state['flow'] = nx.maximum_flow(G, source, sink, flow_func=dinitz)
            else:
                state['value'], state['partition'] = nx.minimum_cut(G, source, sink, flow_func=dinitz)
                state['flow'] = None
            return 'OK'
    outcome = g_budget(p, body)
    if outcome is None:
        return dict(result, status='TIME_LIMIT')
    if outcome != 'OK':
        return dict(result, status=outcome, message='the supplies cannot be routed within the capacities' if outcome == 'INFEASIBLE' else 'a negative-cost cycle has unlimited capacity')
    nodes = g_order(G.nodes)
    tol = tolerance
    if kind == 'min_cut':
        reach = set(state['partition'][0])
        cut = [(u, v, c) for (u, v), c in merged.items() if u in reach and v not in reach and c > 0]
        cut.sort(key=lambda e: (e[0], e[1]))
        total = sum(c for _, _, c in cut)
        if abs(total - state['value']) > tol + 1e-12 * max(1.0, abs(total)):
            return dict(result, status='NUMERICAL_FAILURE', message='the cut capacity %r differs from the flow value %r' % (total, state['value']))
        result.update({'value': state['value'], 'n_cut_edges': len(cut), 'source_side_size': len(reach)})
        if not g_is_file(p):
            g_inline(len(cut) <= S8G_INLINE_FLOW_EDGES and len(reach) <= S8G_INLINE_FLOW_EDGES, 'cut size')
            result.update({'cut_from': [e[0] for e in cut], 'cut_to': [e[1] for e in cut], 'cut_capacity': [e[2] for e in cut], 'source_side': g_order(reach)})
            return result
        return g_write_table(p, ['from', 'to', 'capacity'], [[e[0] for e in cut], [e[1] for e in cut], [e[2] for e in cut]], result)
    flow = state['flow']
    edges = sorted(((u, v, f) for u, nbrs in flow.items() for v, f in nbrs.items() if f > (tol if tol else 0)), key=lambda e: (e[0], e[1]))
    # independent verification of the returned flow against the raw edge list
    net = {}
    for u, v, f in edges:
        cap = merged[(u, v)]
        if f > cap + tol + 1e-12 * max(1.0, float(cap)):
            return dict(result, status='NUMERICAL_FAILURE', message='the flow on (%r, %r) exceeds its capacity' % (u, v))
        net[u] = net.get(u, 0) - f
        net[v] = net.get(v, 0) + f
    if kind == 'max_flow':
        source, sink = p['source'], p['sink']
        value = state['value']
        for node, excess in net.items():
            if node not in (source, sink) and abs(excess) > tol + 1e-12:
                return dict(result, status='NUMERICAL_FAILURE', message='flow is not conserved at node %r (excess %r)' % (node, excess))
        if abs(-net.get(source, 0) - value) > tol + 1e-12 * max(1.0, abs(value)):
            return dict(result, status='NUMERICAL_FAILURE', message='the flow out of the source differs from the reported value')
        result['value'] = value
    else:
        for node in G.nodes:
            if abs(-net.get(node, 0) - supplies.get(node, 0)) > 0:
                return dict(result, status='NUMERICAL_FAILURE', message='the flow does not meet the supply at node %r' % (node,))
        recomputed = sum(f * cost_of[(u, v)] for u, v, f in edges)
        if recomputed != state['cost']:
            return dict(result, status='NUMERICAL_FAILURE', message='the reported cost %r differs from the cost of the returned flow %r' % (state['cost'], recomputed))
        result['cost'] = state['cost']
    result['n_flow_edges'] = len(edges)
    if not g_is_file(p):
        g_inline(len(edges) <= S8G_INLINE_FLOW_EDGES, 'flow edge list')
        result.update({'flow_from': [e[0] for e in edges], 'flow_to': [e[1] for e in edges], 'flow': [e[2] for e in edges]})
        return result
    return g_write_table(p, ['from', 'to', 'flow'], [[e[0] for e in edges], [e[1] for e in edges], [e[2] for e in edges]], result)


def s8g_operation(name):
    return graph_paths if name.startswith('graph_paths') else graph_flow
