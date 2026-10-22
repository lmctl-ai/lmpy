# Source: networkx 3.x documentation examples.
#   Weighted graph gallery example ("Weighted Graph",
#     https://networkx.org/documentation/stable/auto_examples/drawing/plot_weighted_graph.html): the six-edge graph
#     a-b 0.6, a-c 0.2, c-d 0.1, c-e 0.7, c-f 0.9, a-d 0.3, here queried with dijkstra_path / shortest_path_length.
#   networkx.algorithms.flow.maximum_flow and minimum_cut docstring examples (the x..y capacity graph, flow value 3.0).
#   networkx.algorithms.flow.min_cost_flow / min_cost_flow_cost docstring example (demand -5 at a, 5 at d; cost 24).
# Prints the answers as one JSON object; examples/python/networkx-paths-flows.lms pins them.
import json
import networkx as nx

# weighted graph gallery example
G = nx.Graph()
G.add_edge("a", "b", weight=0.6)
G.add_edge("a", "c", weight=0.2)
G.add_edge("c", "d", weight=0.1)
G.add_edge("c", "e", weight=0.7)
G.add_edge("c", "f", weight=0.9)
G.add_edge("a", "d", weight=0.3)
path_af = nx.dijkstra_path(G, "a", "f")
len_af = nx.dijkstra_path_length(G, "a", "f")
path_bd = nx.dijkstra_path(G, "b", "d")
len_bd = nx.shortest_path_length(G, "b", "d", weight="weight")
hops_bf = nx.shortest_path_length(G, "b", "f")  # unweighted: number of edges
dist_from_a = nx.single_source_dijkstra_path_length(G, "a")

# maximum_flow / minimum_cut docstring example
F = nx.DiGraph()
F.add_edge("x", "a", capacity=3.0)
F.add_edge("x", "b", capacity=1.0)
F.add_edge("a", "c", capacity=3.0)
F.add_edge("b", "c", capacity=5.0)
F.add_edge("b", "d", capacity=4.0)
F.add_edge("d", "e", capacity=2.0)
F.add_edge("c", "y", capacity=2.0)
F.add_edge("e", "y", capacity=3.0)
flow_value, flow_dict = nx.maximum_flow(F, "x", "y")
cut_value, (reachable, non_reachable) = nx.minimum_cut(F, "x", "y")

# min_cost_flow docstring example
M = nx.DiGraph()
M.add_node("a", demand=-5)
M.add_node("d", demand=5)
M.add_edge("a", "b", weight=3, capacity=4)
M.add_edge("a", "c", weight=6, capacity=10)
M.add_edge("b", "d", weight=1, capacity=9)
M.add_edge("c", "d", weight=2, capacity=5)
flowDict = nx.min_cost_flow(M)
mcf_cost = nx.min_cost_flow_cost(M)

print(json.dumps({
    "path_a_f": path_af, "length_a_f": len_af, "path_b_d": path_bd, "length_b_d": len_bd, "hops_b_f": hops_bf,
    "dist_from_a": [dist_from_a[n] for n in sorted(dist_from_a)],
    "flow_value": flow_value, "cut_value": cut_value, "source_side": sorted(reachable),
    # edges carrying flow, sorted by (from, to)
    "max_flow_edges": [f"{u}->{v}" for u in sorted(flow_dict) for v in sorted(flow_dict[u]) if flow_dict[u][v] > 0],
    "max_flow_amounts": [flow_dict[u][v] for u in sorted(flow_dict) for v in sorted(flow_dict[u]) if flow_dict[u][v] > 0],
    "min_cost_flow_cost": mcf_cost,
    "min_cost_flow_edges": [f"{u}->{v}" for u in sorted(flowDict) for v in sorted(flowDict[u]) if flowDict[u][v] > 0],
    "min_cost_flow_amounts": [flowDict[u][v] for u in sorted(flowDict) for v in sorted(flowDict[u]) if flowDict[u][v] > 0],
}))
