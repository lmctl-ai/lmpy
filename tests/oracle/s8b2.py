"""Independent S8b-2 references: hand Dijkstra / Bellman-Ford / BFS / Floyd-Warshall, brute-force k shortest simple paths, Edmonds-Karp with
the residual-reachability min cut, and successive shortest paths for min-cost flow, all in pure Python (no networkx, no scipy). Random graph
generators are seeded. Closed forms for distributions (mpmath-free hand formulas) live in the test module."""
import heapq, itertools, math, random


def adjacency(edges, directed):
    """{u: {v: min weight}} with self loops dropped, duplicate edges reduced to the lightest, undirected edges mirrored."""
    adj = {}
    for u, v, w in edges:
        adj.setdefault(u, {})
        adj.setdefault(v, {})
        if u == v:
            continue
        for a, b in ((u, v),) if directed else ((u, v), (v, u)):
            if b not in adj[a] or w < adj[a][b]:
                adj[a][b] = w
    return adj


def dijkstra(adj, s):
    dist = {s: 0}
    heap = [(0, 0, s)]
    seen, tick = set(), 1
    while heap:
        d, _, u = heapq.heappop(heap)
        if u in seen:
            continue
        seen.add(u)
        for v, w in adj[u].items():
            if v not in dist or d + w < dist[v]:
                dist[v] = d + w
                heapq.heappush(heap, (d + w, tick, v))
                tick += 1
    return dist


def bellman_ford(adj, s):
    """(distances, has_negative_cycle_reachable_from_s)."""
    dist = {s: 0}
    for _ in range(len(adj)):
        changed = False
        for u in list(dist):
            for v, w in adj[u].items():
                if v not in dist or dist[u] + w < dist[v] - 0:
                    dist[v] = dist[u] + w
                    changed = True
        if not changed:
            return dist, False
    return dist, True


def bfs_hops(adj, s):
    dist, queue = {s: 0}, [s]
    for u in queue:
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                queue.append(v)
    return dist


def floyd_warshall(adj):
    nodes = sorted(adj)
    inf = math.inf
    d = {u: {v: (0 if u == v else adj[u].get(v, inf)) for v in nodes} for u in nodes}
    for k in nodes:
        for i in nodes:
            for j in nodes:
                if d[i][k] + d[k][j] < d[i][j]:
                    d[i][j] = d[i][k] + d[k][j]
    negative = any(d[u][u] < 0 for u in nodes)
    return nodes, d, negative


def simple_paths(adj, s, t):
    """All simple s-t paths with their lengths (tiny graphs only)."""
    out = []

    def walk(u, path, length):
        if u == t:
            out.append((length, list(path)))
            return
        for v, w in adj[u].items():
            if v not in path:
                path.append(v)
                walk(v, path, length + w)
                path.pop()
    walk(s, [s], 0)
    out.sort(key=lambda item: (item[0], item[1]))
    return out


def random_graph(rng, n, m, directed=True, weights=(1, 20), integer=True, negative=False, dag=False, nodes=None):
    nodes = nodes or list(range(n))
    edges = set()
    tries = 0
    while len(edges) < m and tries < 20 * m:
        tries += 1
        u, v = rng.sample(nodes, 2)
        if dag and nodes.index(u) > nodes.index(v):
            u, v = v, u
        edges.add((u, v))
    out = []
    for u, v in sorted(edges, key=lambda e: (str(e[0]), str(e[1]))):
        w = rng.randint(*weights) if integer else round(rng.uniform(*weights), 3)
        if negative and rng.random() < 0.3:
            w = -abs(w) // 3 if integer else -abs(w) / 3
        out.append((u, v, w))
    return out


def edmonds_karp(caps, s, t):
    """caps: {(u, v): capacity} (parallel edges already summed). Returns (value, flow {(u, v): f}, source-side set of the residual min cut)."""
    residual = {}
    for (u, v), c in caps.items():
        residual.setdefault(u, {})[v] = residual.get(u, {}).get(v, 0) + c
        residual.setdefault(v, {}).setdefault(u, 0)
    value = 0
    while True:
        parent = {s: None}
        queue = [s]
        for u in queue:
            for v, c in residual[u].items():
                if c > 1e-12 and v not in parent:
                    parent[v] = u
                    queue.append(v)
        if t not in parent:
            break
        bottleneck, v = math.inf, t
        while parent[v] is not None:
            bottleneck = min(bottleneck, residual[parent[v]][v])
            v = parent[v]
        v = t
        while parent[v] is not None:
            residual[parent[v]][v] -= bottleneck
            residual[v][parent[v]] += bottleneck
            v = parent[v]
        value += bottleneck
    flow = {(u, v): caps[(u, v)] - residual[u][v] for (u, v) in caps if caps[(u, v)] - residual[u][v] > 1e-12}
    return value, flow, set(parent)


def min_cost_flow(edges, supplies):
    """Successive shortest paths with Bellman-Ford on the residual graph. edges: (u, v, capacity, cost); supplies {node: +supply/-demand}.
    Returns (cost, flow {(u, v): f}) or None when infeasible."""
    arcs = []  # [u, v, residual capacity, cost, reverse index]
    graph = {}

    def add(u, v, cap, cost):
        graph.setdefault(u, []).append(len(arcs))
        arcs.append([u, v, cap, cost])
        graph.setdefault(v, []).append(len(arcs))
        arcs.append([v, u, 0, -cost])
    for u, v, cap, cost in edges:
        add(u, v, cap, cost)
    S, T = '__super_source__', '__super_sink__'
    need = 0
    for node, amount in supplies.items():
        if amount > 0:
            add(S, node, amount, 0)
            need += amount
        elif amount < 0:
            add(node, T, -amount, 0)
    graph.setdefault(S, [])
    graph.setdefault(T, [])
    sent = 0
    while sent < need:
        dist, via = {S: 0}, {}
        for _ in range(len(graph)):
            changed = False
            for u in list(dist):
                for a in graph[u]:
                    _, v, cap, cost = arcs[a]
                    if cap > 0 and (v not in dist or dist[u] + cost < dist[v]):
                        dist[v] = dist[u] + cost
                        via[v] = a
                        changed = True
            if not changed:
                break
        if T not in dist:
            return None
        push, v = need - sent, T
        while v != S:
            a = via[v]
            push = min(push, arcs[a][2])
            v = arcs[a][0]
        v = T
        while v != S:
            a = via[v]
            arcs[a][2] -= push
            arcs[a ^ 1][2] += push
            v = arcs[a][0]
        sent += push
    flow, total = {}, 0
    for i in range(0, len(arcs), 2):
        u, v, cap, cost = arcs[i]
        if S in (u, v) or T in (u, v):
            continue
        used = arcs[i + 1][2]
        if used:
            flow[(u, v)] = used
            total += used * cost
    return total, flow
