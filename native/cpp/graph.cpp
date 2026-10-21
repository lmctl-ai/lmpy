// Graphs with string node names: shortest paths (Dijkstra, Bellman-Ford, BFS, Yen's k shortest simple paths, all pairs)
// and flows (maximum flow, minimum cut, minimum-cost flow). pycli ops covered: graph_paths(_to), graph_flow(_to).
//
// The semantics follow pycli's s8g.py, which drove networkx and scipy.sparse.csgraph:
// - Parallel edges keep the smallest weight for paths and add their capacities for flows; self loops are ignored except
//   that a negative self loop is a negative cycle (paths) and a negative-cost self loop is saturated (min-cost flow).
// - The reported path is taken from a breadth-first walk over the tight edges (dist[u] + w == dist[v]) in edge insertion
//   order: among equally short routes it has the fewest edges (pycli g_tree).
// - Node lists are ordered by name (numerically when every name is an integer, as pycli sorted integer ids).
// - k shortest simple paths are a port of networkx.shortest_simple_paths (Yen's algorithm with networkx's bidirectional
//   Dijkstra / BFS and the same tie order); maximum flow and minimum cut are a port of networkx's Dinitz implementation and
//   minimum_cut partition rule (networkx is BSD-3-Clause, (c) NetworkX Developers). Minimum-cost flow uses successive shortest
//   paths with potentials on integer data (networkx used network simplex: the optimal cost is the same; when several
//   flows are optimal the returned flow may differ).
#include "frame.hpp"
#include <algorithm>
#include <deque>
#include <queue>
#include <set>
#include <unordered_map>

namespace lmpy {

struct Graph : Obj {
  static constexpr const char* TYPE = "Graph";
  bool directed = true;
  std::vector<std::string> names;
  std::unordered_map<std::string, int> ids;
  struct E { int u, v; double w; double cap; bool has_cap; };
  std::vector<E> edges;
  int node(const std::string& s) {
    auto it = ids.find(s);
    if (it != ids.end()) return it->second;
    if (s.empty()) fail("Graph: a node name is empty");
    ids.emplace(s, int(names.size()));
    names.push_back(s);
    return int(names.size()) - 1;
  }
  int need(const std::string& s, const char* op) const {
    auto it = ids.find(s);
    if (it == ids.end()) fail(std::string("Graph.") + op + ": unknown node '" + s + "'");
    return it->second;
  }
  void add(const std::string& u, const std::string& v, double w, double cap, bool has_cap, const char* op) {
    if (!std::isfinite(w)) fail(std::string(op) + ": edge " + u + " -> " + v + ": the weight is not finite");
    if (std::fabs(w) > 1e15) fail(std::string(op) + ": edge " + u + " -> " + v + ": weights must be at most 1e15 in magnitude");
    if (has_cap && !(cap >= 0 && cap <= 1e15)) fail(std::string(op) + ": edge " + u + " -> " + v + ": capacities must be finite, >= 0 and <= 1e15; got " + jnum(cap));
    int a = node(u), b = node(v);
    edges.push_back({a, b, w, cap, has_cap});
  }
};

namespace {

using std::string;
using std::vector;
const string Gt = "Graph", Rt = "Res", St = "StrVec", Vt = "Vec", Ot = "Opts", Ft = "Frame";
Graph& gg(void* s) { return *static_cast<Graph*>(s); }

// ---- node order ----------------------------------------------------------------------------------------------------
bool is_int_name(const string& s) {
  size_t i = s[0] == '-' ? 1 : 0;
  if (i >= s.size() || s.size() - i > 18) return false;
  for (; i < s.size(); ++i) if (s[i] < '0' || s[i] > '9') return false;
  return true;
}
struct Order {
  const Graph& g; bool numeric = true; vector<long long> key;
  explicit Order(const Graph& gr) : g(gr) {
    for (auto& n : g.names) if (!is_int_name(n)) { numeric = false; break; }
    if (numeric) for (auto& n : g.names) key.push_back(std::stoll(n));
  }
  bool less(int a, int b) const { return numeric ? key[size_t(a)] < key[size_t(b)] : g.names[size_t(a)] < g.names[size_t(b)]; }
  vector<int> sorted(vector<int> v) const { std::sort(v.begin(), v.end(), [this](int a, int b) { return less(a, b); }); return v; }
  vector<int> all() const { vector<int> v(g.names.size()); for (size_t i = 0; i < v.size(); ++i) v[i] = int(i); return sorted(v); }
};
vector<string> names_of(const Graph& g, const vector<int>& v) { vector<string> s; for (int i : v) s.push_back(g.names[size_t(i)]); return s; }

// ---- adjacency for paths (pycli g_adjacency) -------------------------------------------------------------------------
struct Adj {
  vector<vector<std::pair<int, double>>> out;  // insertion order, smallest parallel weight
  int64_t n_edges = 0;
  bool negative = false;          // any raw weight < 0 (self loops included, as pycli checked)
  std::map<int, double> loops;    // node -> most negative self loop
};
Adj adjacency(const Graph& g, bool unit) {
  Adj a;
  a.out.resize(g.names.size());
  vector<std::unordered_map<int, size_t>> pos(g.names.size());
  for (auto& e : g.edges) {
    double w = unit ? 1.0 : e.w;
    if (!unit && e.w < 0) a.negative = true;
    if (e.u == e.v) {
      if (!unit && e.w < 0 && (!a.loops.count(e.u) || e.w < a.loops[e.u])) a.loops[e.u] = e.w;
      continue;
    }
    for (int k = 0; k < (g.directed ? 1 : 2); ++k) {
      int x = k ? e.v : e.u, y = k ? e.u : e.v;
      auto it = pos[size_t(x)].find(y);
      if (it == pos[size_t(x)].end()) { pos[size_t(x)][y] = a.out[size_t(x)].size(); a.out[size_t(x)].push_back({y, w}); }
      else if (w < a.out[size_t(x)][it->second].second) a.out[size_t(x)][it->second].second = w;
    }
  }
  for (auto& o : a.out) a.n_edges += int64_t(o.size());
  if (!g.directed) a.n_edges /= 2;
  return a;
}

const double UNSEEN = std::numeric_limits<double>::infinity();

vector<double> dijkstra(const Adj& a, int s) {
  vector<double> d(a.out.size(), UNSEEN);
  vector<char> done(a.out.size(), 0);
  using QE = std::pair<double, int>;
  std::priority_queue<QE, vector<QE>, std::greater<QE>> q;
  d[size_t(s)] = 0; q.push({0, s});
  while (!q.empty()) {
    auto [dv, v] = q.top(); q.pop();
    if (done[size_t(v)]) continue;
    done[size_t(v)] = 1;
    for (auto& [w, len] : a.out[size_t(v)]) {
      double nd = dv + len;
      if (nd < d[size_t(w)]) { d[size_t(w)] = nd; q.push({nd, w}); }
    }
  }
  return d;
}

vector<double> bfs(const Adj& a, int s) {
  vector<double> d(a.out.size(), UNSEEN);
  std::deque<int> q{s}; d[size_t(s)] = 0;
  while (!q.empty()) { int v = q.front(); q.pop_front(); for (auto& e : a.out[size_t(v)]) if (std::isinf(d[size_t(e.first)])) { d[size_t(e.first)] = d[size_t(v)] + 1; q.push_back(e.first); } }
  return d;
}

// Bellman-Ford over the nodes reachable from s (s < 0: a virtual source joined to every node by a zero edge). Returns false
// and a negative cycle (forward order, rotated to start at its smallest node) when one is reachable.
bool bellman_ford(const Adj& a, int s, const Order& ord, vector<double>& d, vector<int>& cycle) {
  size_t n = a.out.size();
  d.assign(n, UNSEEN);
  vector<int> pred(n, -1);
  if (s >= 0) d[size_t(s)] = 0; else std::fill(d.begin(), d.end(), 0.0);
  int last = -1;
  for (size_t it = 0; it <= n; ++it) {
    last = -1;
    for (size_t u = 0; u < n; ++u) {
      if (std::isinf(d[u])) continue;
      for (auto& [v, w] : a.out[u]) if (d[u] + w < d[size_t(v)]) { d[size_t(v)] = d[u] + w; pred[size_t(v)] = int(u); last = v; }
    }
    if (last < 0) return true;
  }
  int x = last;
  for (size_t i = 0; i < n; ++i) x = pred[size_t(x)];
  cycle.clear();
  for (int y = x;; y = pred[size_t(y)]) { cycle.push_back(y); if (pred[size_t(y)] == x) break; }
  std::reverse(cycle.begin(), cycle.end());
  auto m = std::min_element(cycle.begin(), cycle.end(), [&](int p, int q) { return ord.less(p, q); });
  std::rotate(cycle.begin(), m, cycle.end());
  return false;
}

// pycli g_tree: predecessor and hop count of every reached node from a BFS over the tight edges.
void tight_tree(const Adj& a, const vector<double>& dist, int s, bool unit, vector<int>& pred, vector<int>& hops) {
  size_t n = a.out.size();
  pred.assign(n, -1); hops.assign(n, -1);
  hops[size_t(s)] = 0;
  vector<int> q{s};
  for (size_t i = 0; i < q.size(); ++i) {
    int u = q[i];
    for (auto& [v, w0] : a.out[size_t(u)]) {
      double w = unit ? 1.0 : w0;
      if (hops[size_t(v)] < 0 && !std::isinf(dist[size_t(v)]) && std::fabs(dist[size_t(u)] + w - dist[size_t(v)]) <= 1e-12 * std::max(1.0, std::fabs(dist[size_t(v)]))) {
        pred[size_t(v)] = u; hops[size_t(v)] = hops[size_t(u)] + 1; q.push_back(v);
      }
    }
  }
}

string read_method(const Opts& o, const char* op) {
  string m = o.str("method", "dijkstra");
  if (m != "dijkstra" && m != "bellman_ford" && m != "bfs") fail(string("Graph.") + op + ": method must be dijkstra, bellman_ford or bfs; got '" + m + "'");
  return m;
}

Res* base_res(const Graph& g, const string& method, const Adj& a) {
  auto* r = make<Res>();
  r->str("status", "OK").str("method", method).flag("directed", g.directed).integer("n_nodes", int64_t(g.names.size())).integer("n_edges", a.n_edges);
  return r;
}

// Single-source search shared by shortest_path and distances. Returns null on success (dist filled) or a status result.
Res* single_source(const Graph& g, int s, const string& method, const Adj& a, const Order& ord, vector<double>& dist, const char* op) {
  if (method == "dijkstra" && a.negative) fail(string("Graph.") + op + ": dijkstra needs non-negative weights; use method bellman_ford");
  if (method == "bfs") { dist = bfs(a, s); return nullptr; }
  if (method == "dijkstra") { dist = dijkstra(a, s); return nullptr; }
  if (!a.loops.empty()) {  // a reachable negative self loop is a one-node negative cycle
    vector<double> reach = bfs(a, s);
    vector<int> hit;
    for (auto& l : a.loops) if (!std::isinf(reach[size_t(l.first)])) hit.push_back(l.first);
    if (!hit.empty()) {
      auto* r = base_res(g, method, a);
      r->str("status", "NEGATIVE_CYCLE").strs("cycle", {g.names[size_t(ord.sorted(hit)[0])]}).str("message", "a negative self loop is reachable from the source: shortest paths are undefined");
      return r;
    }
  }
  vector<int> cycle;
  if (!bellman_ford(a, s, ord, dist, cycle)) {
    auto* r = base_res(g, method, a);
    r->str("status", "NEGATIVE_CYCLE").strs("cycle", names_of(g, cycle)).str("message", "a negative cycle is reachable from the source: shortest paths are undefined");
    return r;
  }
  return nullptr;
}

Res* shortest_path(const Graph& g, const string& src, const string& dst, const Opts& o) {
  string method = read_method(o, "shortest_path");
  o.finish("Graph.shortest_path");
  int s = g.need(src, "shortest_path"), t = g.need(dst, "shortest_path");
  Order ord(g);
  bool unit = method == "bfs";
  Adj a = adjacency(g, unit);
  vector<double> dist;
  if (Res* bad = single_source(g, s, method, a, ord, dist, "shortest_path")) return bad;
  auto* r = base_res(g, method, a);
  int64_t reached = 0;
  for (double d : dist) reached += !std::isinf(d);
  r->integer("n_reachable", reached).integer("n_unreachable", int64_t(dist.size()) - reached);
  if (std::isinf(dist[size_t(t)])) {
    r->str("status", "UNREACHABLE").str("target", dst).str("message", "the target cannot be reached from the source");
    return r;
  }
  vector<int> pred, hops;
  tight_tree(a, dist, s, unit, pred, hops);
  vector<int> path{t};
  while (pred[size_t(path.back())] >= 0) path.push_back(pred[size_t(path.back())]);
  std::reverse(path.begin(), path.end());
  r->str("target", dst).strs("path", names_of(g, path)).num("length", dist[size_t(t)]).integer("hops", int64_t(path.size()) - 1);
  return r;
}

Res* distances(const Graph& g, const string& src, const Opts& o) {
  string method = read_method(o, "distances");
  o.finish("Graph.distances");
  int s = g.need(src, "distances");
  Order ord(g);
  bool unit = method == "bfs";
  Adj a = adjacency(g, unit);
  vector<double> dist;
  if (Res* bad = single_source(g, s, method, a, ord, dist, "distances")) return bad;
  vector<int> pred, hops;
  tight_tree(a, dist, s, unit, pred, hops);
  vector<int> reached;
  for (size_t i = 0; i < dist.size(); ++i) if (!std::isinf(dist[i])) reached.push_back(int(i));
  reached = ord.sorted(reached);
  vector<double> d, h; vector<string> p;
  for (int v : reached) { d.push_back(dist[size_t(v)]); h.push_back(double(hops[size_t(v)])); p.push_back(pred[size_t(v)] >= 0 ? g.names[size_t(pred[size_t(v)])] : ""); }
  auto* r = base_res(g, method, a);
  r->str("source", src).integer("n_reachable", int64_t(reached.size())).integer("n_unreachable", int64_t(dist.size() - reached.size()))
      .strs("nodes", names_of(g, reached)).vec("distance", d).vec("hops", h).strs("predecessor", p);
  return r;
}

Res* all_distances(const Graph& g, const Opts& o) {
  string method = read_method(o, "all_distances");
  o.finish("Graph.all_distances");
  Order ord(g);
  bool unit = method == "bfs";
  Adj a = adjacency(g, unit);
  if (method == "dijkstra" && a.negative) fail("Graph.all_distances: dijkstra needs non-negative weights; use method bellman_ford");
  size_t n = g.names.size();
  if (double(n) * double(n) > 4e7) fail("Graph.all_distances: " + std::to_string(n) + " nodes make a matrix above 4e7 entries");
  auto* r = base_res(g, method, a);
  vector<double> h(n, 0.0);
  if (method == "bellman_ford") {
    if (!a.loops.empty()) {
      vector<int> ls; for (auto& l : a.loops) ls.push_back(l.first);
      r->str("status", "NEGATIVE_CYCLE").strs("cycle", {g.names[size_t(ord.sorted(ls)[0])]}).str("message", "the graph has a negative cycle: shortest paths are undefined");
      return r;
    }
    vector<int> cycle;
    if (!bellman_ford(a, -1, ord, h, cycle)) {
      r->str("status", "NEGATIVE_CYCLE").strs("cycle", names_of(g, cycle)).str("message", "the graph has a negative cycle: shortest paths are undefined");
      return r;
    }
  }
  // Johnson reweighting (h = 0 unless bellman_ford): w'(u, v) = w + h[u] - h[v] >= 0
  Adj rw = a;
  for (size_t u = 0; u < n; ++u) for (auto& e : rw.out[u]) e.second = std::max(0.0, e.second + h[u] - h[size_t(e.first)]);
  vector<int> order = ord.all();
  Eigen::MatrixXd m(static_cast<long>(n), static_cast<long>(n));
  int64_t unreachable = 0;
  for (size_t i = 0; i < n; ++i) {
    int s = order[i];
    vector<double> d = unit ? bfs(rw, s) : dijkstra(rw, s);
    for (size_t j = 0; j < n; ++j) {
      int t = order[j];
      double v = d[size_t(t)];
      if (std::isinf(v)) { m(long(i), long(j)) = NAN; ++unreachable; }
      else m(long(i), long(j)) = method == "bellman_ford" ? v - h[size_t(s)] + h[size_t(t)] : v;
    }
  }
  r->strs("nodes", names_of(g, order)).mat("distance", m).integer("n_unreachable_pairs", unreachable);
  return r;
}

// ---- Yen's k shortest simple paths (port of networkx.shortest_simple_paths) ------------------------------------------
struct NxGraph {  // the networkx (Di)Graph pycli built: neighbour order matters for ties
  bool directed; vector<vector<int>> succ, pred; std::map<std::pair<int, int>, double> w;
};
NxGraph nx_graph(const Graph& g, const Adj& a, const Order& ord) {
  NxGraph G; G.directed = g.directed;
  size_t n = a.out.size();
  G.succ.resize(n); G.pred.resize(n);
  for (size_t u = 0; u < n; ++u)
    for (auto& [v, w] : a.out[u]) {
      if (!g.directed && ord.less(v, int(u))) continue;  // undirected: pycli added (u, v) only when u <= v
      auto has = [](const vector<int>& l, int x) { return std::find(l.begin(), l.end(), x) != l.end(); };
      if (g.directed) { G.succ[u].push_back(v); G.pred[size_t(v)].push_back(int(u)); G.w[{int(u), v}] = w; }
      else {
        if (!has(G.succ[u], v)) G.succ[u].push_back(v);
        if (!has(G.succ[size_t(v)], int(u))) G.succ[size_t(v)].push_back(int(u));
        G.w[{int(u), v}] = w; G.w[{v, int(u)}] = w;
      }
    }
  if (!g.directed) G.pred = G.succ;
  return G;
}
struct Ignore { std::set<int> nodes; std::set<std::pair<int, int>> edges; };
// neighbours of v in direction dir (0 forward / successors, 1 backward / predecessors) after the ignore filters
vector<int> nbrs(const NxGraph& G, int v, int dir, const Ignore& ig) {
  vector<int> out;
  for (int w : dir == 0 ? G.succ[size_t(v)] : G.pred[size_t(v)]) {
    if (ig.nodes.count(w)) continue;
    if (G.directed) { if (dir == 0 ? ig.edges.count({v, w}) : ig.edges.count({w, v})) continue; }
    else if (ig.edges.count({v, w}) || ig.edges.count({w, v})) continue;
    out.push_back(w);
  }
  return out;
}
// _bidirectional_dijkstra; returns false when there is no path
bool bidir_dijkstra(const NxGraph& G, int s, int t, const Ignore& ig, double& length, vector<int>& path) {
  if (ig.nodes.count(s) || ig.nodes.count(t)) return false;
  if (s == t) { length = 0; path = {s}; return true; }
  size_t n = G.succ.size();
  vector<double> dists[2] = {vector<double>(n, UNSEEN), vector<double>(n, UNSEEN)};
  vector<char> fin[2] = {vector<char>(n, 0), vector<char>(n, 0)};
  vector<double> seen[2] = {vector<double>(n, UNSEEN), vector<double>(n, UNSEEN)};
  vector<char> sn[2] = {vector<char>(n, 0), vector<char>(n, 0)};
  vector<int> par[2] = {vector<int>(n, -1), vector<int>(n, -1)};
  using QE = std::tuple<double, long, int>;
  std::priority_queue<QE, vector<QE>, std::greater<QE>> fringe[2];
  long c = 0;
  seen[0][size_t(s)] = 0; sn[0][size_t(s)] = 1; seen[1][size_t(t)] = 0; sn[1][size_t(t)] = 1;
  fringe[0].push({0.0, c++, s}); fringe[1].push({0.0, c++, t});
  bool found = false; double finaldist = 0; vector<int> finalpath;
  int dir = 1;
  auto chain = [&](int d, int v) { vector<int> p; for (int x = v; x >= 0; x = par[d][size_t(x)]) p.push_back(x); std::reverse(p.begin(), p.end()); return p; };
  while (!fringe[0].empty() && !fringe[1].empty()) {
    dir = 1 - dir;
    auto [dist, cc, v] = fringe[dir].top(); fringe[dir].pop(); (void)cc;
    if (fin[dir][size_t(v)]) continue;
    fin[dir][size_t(v)] = 1; dists[dir][size_t(v)] = dist;
    if (fin[1 - dir][size_t(v)]) { length = finaldist; path = finalpath; return found; }
    for (int w : nbrs(G, v, dir, ig)) {
      double mw = dir == 0 ? G.w.at({v, w}) : G.w.at({w, v});
      double vw = dists[dir][size_t(v)] + mw;
      if (fin[dir][size_t(w)]) continue;  // (a shorter path would mean negative weights, refused earlier)
      if (!sn[dir][size_t(w)] || vw < seen[dir][size_t(w)]) {
        seen[dir][size_t(w)] = vw; sn[dir][size_t(w)] = 1;
        fringe[dir].push({vw, c++, w});
        par[dir][size_t(w)] = v;
        if (sn[0][size_t(w)] && sn[1][size_t(w)]) {
          double total = seen[0][size_t(w)] + seen[1][size_t(w)];
          if (!found || finaldist > total) {
            found = true; finaldist = total;
            vector<int> f = chain(0, w), b = chain(1, w);  // b runs t ... w
            std::reverse(b.begin(), b.end());
            finalpath = f; finalpath.insert(finalpath.end(), b.begin() + 1, b.end());
          }
        }
      }
    }
  }
  return false;
}
// _bidirectional_shortest_path (unweighted); length is the node count, as networkx returns it
bool bidir_bfs(const NxGraph& G, int s, int t, const Ignore& ig, double& length, vector<int>& path) {
  if (ig.nodes.count(s) || ig.nodes.count(t)) return false;
  size_t n = G.succ.size();
  vector<int> pred(n, -2), succ(n, -2);  // -2 absent, -1 None
  int meet = -1;
  if (s == t) meet = s;
  else {
    pred[size_t(s)] = -1; succ[size_t(t)] = -1;
    vector<int> ff{s}, rf{t};
    while (meet < 0 && !ff.empty() && !rf.empty()) {
      if (ff.size() <= rf.size()) {
        vector<int> level; level.swap(ff);
        for (int v : level) { for (int w : nbrs(G, v, 0, ig)) { if (pred[size_t(w)] == -2) { ff.push_back(w); pred[size_t(w)] = v; } if (succ[size_t(w)] != -2) { meet = w; break; } } if (meet >= 0) break; }
      } else {
        vector<int> level; level.swap(rf);
        for (int v : level) { for (int w : nbrs(G, v, 1, ig)) { if (succ[size_t(w)] == -2) { succ[size_t(w)] = v; rf.push_back(w); } if (pred[size_t(w)] != -2) { meet = w; break; } } if (meet >= 0) break; }
      }
    }
    if (meet < 0) return false;
  }
  path.clear();
  if (s == t) { path = {s}; length = 1; return true; }
  for (int w = meet; w >= 0; w = succ[size_t(w)]) path.push_back(w);
  for (int w = pred[size_t(path[0])]; w >= 0; w = pred[size_t(w)]) path.insert(path.begin(), w);
  length = double(path.size());
  return true;
}

Res* k_shortest(const Graph& g, const string& src, const string& dst, int64_t k, const Opts& o) {
  string method = read_method(o, "k_shortest");
  o.finish("Graph.k_shortest");
  int s = g.need(src, "k_shortest"), t = g.need(dst, "k_shortest");
  if (k < 1 || k > 1000) fail("Graph.k_shortest: k must be in [1, 1000]; got " + std::to_string(k));
  Order ord(g);
  bool unit = method == "bfs";
  Adj a = adjacency(g, unit);
  if (!unit && a.negative) fail("Graph.k_shortest: k shortest simple paths need non-negative weights (Yen's algorithm runs Dijkstra)");
  auto* r = base_res(g, method, a);
  vector<double> reach = bfs(a, s);
  if (std::isinf(reach[size_t(t)])) {
    r->str("status", "UNREACHABLE").str("target", dst).str("message", "the target cannot be reached from the source");
    return r;
  }
  NxGraph G = nx_graph(g, a, ord);
  auto length_of = [&](const vector<int>& p) { double sum = 0; for (size_t i = 1; i < p.size(); ++i) sum += unit ? 1.0 : G.w.at({p[i - 1], p[i]}); return sum; };
  auto spf = [&](int x, const Ignore& ig, double& len, vector<int>& p) { return unit ? bidir_bfs(G, x, t, ig, len, p) : bidir_dijkstra(G, x, t, ig, len, p); };
  // PathBuffer: a heap of (cost, counter, path) plus the set of queued paths
  using BE = std::tuple<double, long, vector<int>>;
  std::priority_queue<BE, vector<BE>, std::greater<BE>> heap;
  std::set<vector<int>> queued;
  long counter = 0;
  auto push = [&](double cost, const vector<int>& p) { if (queued.insert(p).second) heap.push({cost, counter++, p}); };
  vector<vector<int>> listA;
  vector<int> prev;
  while (int64_t(listA.size()) < k) {
    if (prev.empty()) {
      double len; vector<int> p;
      if (spf(s, Ignore{}, len, p)) push(len, p);
    } else {
      Ignore ig;
      for (size_t i = 1; i < prev.size(); ++i) {
        vector<int> root(prev.begin(), prev.begin() + long(i));
        double root_len = unit ? double(root.size()) : length_of(root);
        for (auto& p : listA) if (p.size() > i && std::equal(root.begin(), root.end(), p.begin())) ig.edges.insert({p[i - 1], p[i]});
        double len; vector<int> spur;
        if (spf(root.back(), ig, len, spur)) {
          vector<int> p(root.begin(), root.end() - 1);
          p.insert(p.end(), spur.begin(), spur.end());
          push(root_len + len, p);
        }
        ig.nodes.insert(root.back());
      }
    }
    if (heap.empty()) break;
    vector<int> p = std::get<2>(heap.top()); heap.pop();
    queued.erase(p);
    listA.push_back(p);
    prev = p;
  }
  vector<string> paths; vector<double> lens, hops;
  for (auto& p : listA) {
    string text;
    for (size_t i = 0; i < p.size(); ++i) text += (i ? "," : "") + g.names[size_t(p[i])];
    paths.push_back(text); lens.push_back(length_of(p)); hops.push_back(double(p.size()) - 1);
  }
  r->str("target", dst).integer("k_found", int64_t(listA.size())).strs("paths", paths).vec("lengths", lens).vec("hops", hops);
  return r;
}

// ---- flows ---------------------------------------------------------------------------------------------------------
struct Arcs { vector<std::pair<int, int>> key; vector<double> cap; std::map<std::pair<int, int>, size_t> at; bool integral = true; };
Arcs merged_arcs(const Graph& g, const char* op) {
  Arcs m;
  for (auto& e : g.edges) {
    if (!e.has_cap) fail(string("Graph.") + op + ": edge " + g.names[size_t(e.u)] + " -> " + g.names[size_t(e.v)] + " has no capacity (add edges with edge(u, v, weight, capacity) or graph(frame, ..., opts().set(\"capacity\", column)))");
    if (e.cap != std::floor(e.cap)) m.integral = false;
    if (e.u == e.v) continue;
    for (int k = 0; k < (g.directed ? 1 : 2); ++k) {
      std::pair<int, int> key = k ? std::make_pair(e.v, e.u) : std::make_pair(e.u, e.v);
      auto it = m.at.find(key);
      if (it == m.at.end()) { m.at[key] = m.key.size(); m.key.push_back(key); m.cap.push_back(e.cap); }
      else m.cap[it->second] += e.cap;
    }
  }
  return m;
}

// networkx build_residual_network + dinitz_impl (directed form: an undirected graph arrives here as two arcs per edge)
struct Residual {
  struct RE { int u, v; double cap, flow; };
  vector<RE> e;
  vector<vector<std::pair<int, int>>> succ, pred;  // (neighbour, edge index) in insertion order
  vector<std::unordered_map<int, int>> at;           // at[u][v] = edge index of u -> v
  double inf = 1;
  int edge(int u, int v) const { auto it = at[size_t(u)].find(v); return it == at[size_t(u)].end() ? -1 : it->second; }
  int add(int u, int v, double cap) {
    int k = edge(u, v);
    if (k >= 0) { e[size_t(k)].cap = cap; return k; }
    e.push_back({u, v, cap, 0}); k = int(e.size()) - 1;
    at[size_t(u)][v] = k; succ[size_t(u)].push_back({v, k}); pred[size_t(v)].push_back({u, k});
    return k;
  }
};
Residual dinitz(size_t n, const vector<int>& node_order, const Arcs& arcs, int s, int t, double& value) {
  Residual R;
  R.succ.resize(n); R.pred.resize(n); R.at.resize(n);
  // G.edges order: nodes in G order (sorted), then each node's arcs in insertion order
  vector<vector<size_t>> by_u(n);
  for (size_t i = 0; i < arcs.key.size(); ++i) by_u[size_t(arcs.key[i].first)].push_back(i);
  vector<size_t> list;
  double sum = 0;
  for (int u : node_order) for (size_t i : by_u[size_t(u)]) if (arcs.cap[i] > 0) { list.push_back(i); sum += arcs.cap[i]; }
  R.inf = 3 * sum; if (R.inf == 0) R.inf = 1;
  for (size_t i : list) {
    int u = arcs.key[i].first, v = arcs.key[i].second;
    double c = std::min(arcs.cap[i], R.inf);
    if (R.edge(u, v) < 0) { R.add(u, v, c); if (R.edge(v, u) < 0) R.add(v, u, 0); }
    else R.e[size_t(R.edge(u, v))].cap = c;
  }
  value = 0;
  for (;;) {
    // breadth-first layering: parents[v] lists every u with dist[u] + 1 == dist[v]
    vector<std::deque<int>> parents(n);
    vector<char> has(n, 0);
    vector<int> vd(n, -1);
    vd[size_t(s)] = 0;
    std::deque<std::pair<int, int>> q{{s, 0}};
    while (!q.empty()) {
      if (has[size_t(t)]) break;
      auto [u, d] = q.front(); q.pop_front();
      for (auto& [v, k] : R.succ[size_t(u)]) {
        const auto& ed = R.e[size_t(k)];
        if (ed.cap - ed.flow > 0) {
          if (has[size_t(v)]) { if (vd[size_t(v)] == d + 1) parents[size_t(v)].push_back(u); }
          else { parents[size_t(v)] = std::deque<int>{u}; has[size_t(v)] = 1; vd[size_t(v)] = d + 1; q.push_back({v, d + 1}); }
        }
      }
    }
    if (!has[size_t(t)]) break;
    // depth-first search from the sink over the parents lists
    double total = 0;
    int u = t;
    vector<int> path{u};
    for (;;) {
      int v;
      if (has[size_t(u)] && !parents[size_t(u)].empty()) { v = parents[size_t(u)].front(); path.push_back(v); }
      else {
        path.pop_back();
        if (path.empty()) break;
        v = path.back();
        parents[size_t(v)].pop_front();
      }
      if (v == s) {
        double flow = R.inf;
        for (size_t i = 0; i + 1 < path.size(); ++i) { const auto& ed = R.e[size_t(R.edge(path[i + 1], path[i]))]; flow = std::min(flow, ed.cap - ed.flow); }
        for (size_t i = path.size() - 1; i >= 1; --i) {
          int a = path[i], b = path[i - 1];  // pairwise(reversed(path)): (u, v) = (path[i], path[i-1]); R_pred[v][u] is u -> v
          auto& fwd = R.e[size_t(R.edge(a, b))];
          fwd.flow += flow;
          R.e[size_t(R.edge(b, a))].flow -= flow;
          if (fwd.cap - fwd.flow == 0) {
            parents[size_t(b)].pop_front();
            while (path.back() != b) path.pop_back();
          }
        }
        total += flow;
        v = path.back();
      }
      u = v;
    }
    if (total * 2 > R.inf) fail("max_flow: infinite capacity path");
    value += total;
  }
  return R;
}

double flow_tol(bool integral, double value) { return integral ? 0.0 : 1e-9 * std::fabs(value); }

Res* flow_base(const Graph& g, const string& kind, bool integral) {
  auto* r = make<Res>();
  r->str("status", "OK").str("kind", kind).flag("integral", integral).integer("n_nodes", int64_t(g.names.size()));
  return r;
}

Res* max_flow(const Graph& g, const string& src, const string& dst, bool cut) {
  const char* op = cut ? "min_cut" : "max_flow";
  int s = g.need(src, op), t = g.need(dst, op);
  if (s == t) fail(string("Graph.") + op + ": source and sink are the same node");
  Order ord(g);
  Arcs arcs = merged_arcs(g, op);
  size_t n = g.names.size();
  double value = 0;
  Residual R = dinitz(n, ord.all(), arcs, s, t, value);
  double tol = flow_tol(arcs.integral, value);
  auto* r = flow_base(g, op, arcs.integral);
  r->num("tolerance", tol);
  auto order_pair = [&](const std::tuple<int, int, double>& x, const std::tuple<int, int, double>& y) {
    if (std::get<0>(x) != std::get<0>(y)) return ord.less(std::get<0>(x), std::get<0>(y));
    return ord.less(std::get<1>(x), std::get<1>(y));
  };
  auto put_value = [&](Res* res, const char* key, double v) { if (arcs.integral && std::fabs(v) < 9e15) res->integer(key, int64_t(v)); else res->num(key, v); };
  if (cut) {
    // networkx minimum_cut: drop saturated residual edges; the sink side is every node that can still reach t
    vector<char> sink(n, 0);
    vector<int> q{t}; sink[size_t(t)] = 1;
    for (size_t i = 0; i < q.size(); ++i)
      for (auto& [u, k] : R.pred[size_t(q[i])]) { const auto& ed = R.e[size_t(k)]; if (ed.flow != ed.cap && !sink[size_t(u)]) { sink[size_t(u)] = 1; q.push_back(u); } }
    vector<std::tuple<int, int, double>> ce;
    for (size_t i = 0; i < arcs.key.size(); ++i) {
      int u = arcs.key[i].first, v = arcs.key[i].second;
      if (!sink[size_t(u)] && sink[size_t(v)] && arcs.cap[i] > 0) ce.push_back({u, v, arcs.cap[i]});
    }
    std::sort(ce.begin(), ce.end(), order_pair);
    double total = 0;
    for (auto& c : ce) total += std::get<2>(c);
    if (std::fabs(total - value) > tol) {
      r->str("status", "NUMERICAL_FAILURE").str("message", "the cut capacity " + jnum(total) + " differs from the flow value " + jnum(value));
      return r;
    }
    vector<int> side;
    for (size_t i = 0; i < n; ++i) if (!sink[i]) side.push_back(int(i));
    vector<string> cf, ct; vector<double> cc;
    for (auto& c : ce) { cf.push_back(g.names[size_t(std::get<0>(c))]); ct.push_back(g.names[size_t(std::get<1>(c))]); cc.push_back(std::get<2>(c)); }
    put_value(r, "value", value);
    r->integer("n_cut_edges", int64_t(ce.size())).integer("source_side_size", int64_t(side.size()))
        .strs("cut_from", cf).strs("cut_to", ct).vec("cut_capacity", cc).strs("source_side", names_of(g, ord.sorted(side)));
    return r;
  }
  double noise = arcs.integral ? 0.0 : 1e-12 * std::fabs(value);
  vector<std::tuple<int, int, double>> fe;
  for (auto& ed : R.e) if (ed.flow > noise && ed.u != ed.v) fe.push_back({ed.u, ed.v, ed.flow});
  std::sort(fe.begin(), fe.end(), order_pair);
  // independent verification against the merged capacities
  vector<double> net(n, 0.0);
  for (auto& [u, v, f] : fe) {
    auto it = arcs.at.find({u, v});
    double cap = it == arcs.at.end() ? 0.0 : arcs.cap[it->second];
    if (f > cap + tol) { r->str("status", "NUMERICAL_FAILURE").str("message", "the flow on (" + g.names[size_t(u)] + ", " + g.names[size_t(v)] + ") exceeds its capacity"); return r; }
    net[size_t(u)] -= f; net[size_t(v)] += f;
  }
  for (size_t i = 0; i < n; ++i)
    if (int(i) != s && int(i) != t && std::fabs(net[i]) > tol) { r->str("status", "NUMERICAL_FAILURE").str("message", "flow is not conserved at node " + g.names[i]); return r; }
  if (std::fabs(-net[size_t(s)] - value) > tol) { r->str("status", "NUMERICAL_FAILURE").str("message", "the flow out of the source differs from the reported value"); return r; }
  vector<string> ff, ft; vector<double> fv;
  for (auto& [u, v, f] : fe) { ff.push_back(g.names[size_t(u)]); ft.push_back(g.names[size_t(v)]); fv.push_back(f); }
  put_value(r, "value", value);
  r->integer("n_flow_edges", int64_t(fe.size())).strs("flow_from", ff).strs("flow_to", ft).vec("flow", fv);
  return r;
}

// Minimum-cost flow on integer data: negative-cost arcs are saturated first, then successive shortest paths with
// Dijkstra on reduced costs move every remaining supply to the demands.
Res* min_cost_flow(const Graph& g, const vector<std::pair<int, int64_t>>& supply_list) {
  const char* op = "min_cost_flow";
  if (!g.directed) fail("Graph.min_cost_flow: needs a directed graph");
  Order ord(g);
  size_t n = g.names.size();
  std::set<std::pair<int, int>> seen;
  vector<int64_t> loop_flow(n, 0);
  __int128 loop_cost = 0;
  struct A { int u, v; int64_t cap, cost, flow; };
  vector<A> arcs;
  for (auto& e : g.edges) {
    if (!e.has_cap) fail(string("Graph.") + op + ": edge " + g.names[size_t(e.u)] + " -> " + g.names[size_t(e.v)] + " has no capacity");
    if (seen.count({e.u, e.v}) && e.u != e.v) fail(string("Graph.") + op + ": needs one edge per (from, to) pair; (" + g.names[size_t(e.u)] + ", " + g.names[size_t(e.v)] + ") is repeated");
    seen.insert({e.u, e.v});
    if (e.cap != std::floor(e.cap) || e.w != std::floor(e.w)) fail(string("Graph.") + op + ": needs integer capacities and costs (the edge weight is the unit cost); edge " + g.names[size_t(e.u)] + " -> " + g.names[size_t(e.v)] + " has capacity " + jnum(e.cap) + ", cost " + jnum(e.w));
    if (std::fabs(e.w) > 1e12) fail(string("Graph.") + op + ": costs must be at most 1e12 in magnitude");
    int64_t c = int64_t(e.cap), k = int64_t(e.w);
    if (e.u == e.v) { if (k < 0 && c > 0) { loop_flow[size_t(e.u)] += c; loop_cost += __int128(c) * k; } continue; }
    arcs.push_back({e.u, e.v, c, k, 0});
  }
  auto* r = flow_base(g, op, true);
  vector<int64_t> supply(n, 0);
  __int128 balance = 0;
  for (auto& [v, amt] : supply_list) { supply[size_t(v)] += amt; balance += amt; }
  if (balance != 0) {
    r->str("status", "UNBALANCED").str("message", "the supplies sum to " + std::to_string(int64_t(balance)) + ": supply and demand must balance");
    return r;
  }
  // residual graph: arc 2i forward, 2i+1 backward
  vector<int64_t> excess = supply;  // > 0 still to send, < 0 still to receive
  struct RA { int to; int64_t res, cost; };
  vector<RA> ra; vector<vector<int>> out(n + 2);
  for (auto& a : arcs) {
    int64_t f = a.cost < 0 ? a.cap : 0;  // saturate negative-cost arcs
    a.flow = f; excess[size_t(a.u)] -= f; excess[size_t(a.v)] += f;
    out[size_t(a.u)].push_back(int(ra.size())); ra.push_back({a.v, a.cap - f, a.cost});
    out[size_t(a.v)].push_back(int(ra.size())); ra.push_back({a.u, f, -a.cost});
  }
  int S = int(n), T = int(n) + 1;
  int64_t need = 0;
  vector<int> term(n, -1);
  for (size_t v = 0; v < n; ++v) {
    if (excess[v] > 0) { term[v] = int(ra.size()); out[size_t(S)].push_back(int(ra.size())); ra.push_back({int(v), excess[v], 0}); out[v].push_back(int(ra.size())); ra.push_back({S, 0, 0}); need += excess[v]; }
    else if (excess[v] < 0) { term[v] = int(ra.size()); out[v].push_back(int(ra.size())); ra.push_back({T, -excess[v], 0}); out[size_t(T)].push_back(int(ra.size())); ra.push_back({int(v), 0, 0}); }
  }
  size_t N = n + 2;
  vector<int64_t> pot(N, 0);
  int64_t sent = 0;
  const int64_t BIG = std::numeric_limits<int64_t>::max() / 4;
  while (sent < need) {
    vector<int64_t> d(N, BIG); vector<int> via(N, -1);
    using QE = std::pair<int64_t, int>;
    std::priority_queue<QE, vector<QE>, std::greater<QE>> q;
    d[size_t(S)] = 0; q.push({0, S});
    while (!q.empty()) {
      auto [dv, v] = q.top(); q.pop();
      if (dv != d[size_t(v)]) continue;
      for (int k : out[size_t(v)]) {
        const RA& e = ra[size_t(k)];
        if (e.res <= 0) continue;
        int64_t nd = dv + e.cost + pot[size_t(v)] - pot[size_t(e.to)];
        if (nd < d[size_t(e.to)]) { d[size_t(e.to)] = nd; via[size_t(e.to)] = k; q.push({nd, e.to}); }
      }
    }
    if (d[size_t(T)] >= BIG) break;
    for (size_t v = 0; v < N; ++v) if (d[v] < BIG) pot[v] += d[v];
    int64_t push = need - sent;
    for (int v = T; v != S; v = ra[size_t(via[size_t(v)] ^ 1)].to) push = std::min(push, ra[size_t(via[size_t(v)])].res);
    for (int v = T; v != S; v = ra[size_t(via[size_t(v)] ^ 1)].to) { ra[size_t(via[size_t(v)])].res -= push; ra[size_t(via[size_t(v)] ^ 1)].res += push; }
    sent += push;
  }
  if (sent < need) {
    r->str("status", "INFEASIBLE").str("message", "the supplies cannot be routed within the capacities");
    return r;
  }
  // read the flows back and verify them independently
  vector<std::tuple<int, int, int64_t>> fe;
  __int128 cost = 0;
  vector<__int128> net(n, 0);
  for (size_t i = 0; i < arcs.size(); ++i) {
    int64_t f = ra[2 * i + 1].res;  // the backward residual equals the flow
    if (f < 0 || f > arcs[i].cap) { r->str("status", "NUMERICAL_FAILURE").str("message", "a flow exceeds its capacity"); return r; }
    if (f > 0) { fe.push_back({arcs[i].u, arcs[i].v, f}); cost += __int128(f) * arcs[i].cost; net[size_t(arcs[i].u)] -= f; net[size_t(arcs[i].v)] += f; }
  }
  for (size_t v = 0; v < n; ++v)
    if (-net[v] != supply[v]) { r->str("status", "NUMERICAL_FAILURE").str("message", "the flow does not meet the supply at node " + g.names[v]); return r; }
  for (size_t v = 0; v < n; ++v) if (loop_flow[v] > 0) fe.push_back({int(v), int(v), loop_flow[v]});
  std::sort(fe.begin(), fe.end(), [&](const auto& x, const auto& y) {
    if (std::get<0>(x) != std::get<0>(y)) return ord.less(std::get<0>(x), std::get<0>(y));
    return ord.less(std::get<1>(x), std::get<1>(y)); });
  cost += loop_cost;
  if (cost >= -(__int128(1) << 63) && cost < (__int128(1) << 63)) r->integer("cost", int64_t(cost));
  else {
    r->num("cost", double(cost));
    __int128 c = cost < 0 ? -cost : cost; string digits;
    while (c > 0) { digits.insert(digits.begin(), char('0' + int(c % 10))); c /= 10; }
    r->str("cost_exact", (cost < 0 ? "-" : "") + digits);
  }
  vector<string> ff, ft; vector<double> fv;
  for (auto& [u, v, f] : fe) { ff.push_back(g.names[size_t(u)]); ft.push_back(g.names[size_t(v)]); fv.push_back(double(f)); }
  r->integer("n_flow_edges", int64_t(fe.size())).strs("flow_from", ff).strs("flow_to", ft).vec("flow", fv);
  return r;
}

int64_t integral_amount(double x, const char* what) {
  if (!std::isfinite(x) || x != std::floor(x) || std::fabs(x) > 9e15) fail(string("Graph.min_cost_flow: ") + what + " must be an integer; got " + jnum(x));
  return int64_t(x);
}

// ---- builders ------------------------------------------------------------------------------------------------------
vector<string> node_column(const Frame& f, const string& name) {
  const Column& c = f.get(name);
  vector<string> out;
  for (size_t i = 0; i < c.size(); ++i) {
    if (c.numeric && std::isnan(c.num[i])) fail("graph: endpoint column '" + name + "' has a missing value in row " + std::to_string(i));
    out.push_back(c.cell_text(i));
    if (out.back().empty()) fail("graph: endpoint column '" + name + "' has an empty name in row " + std::to_string(i));
  }
  return out;
}
const vector<double>& number_column(const Frame& f, const string& name) {
  const Column& c = f.num_col(name);
  for (size_t i = 0; i < c.num.size(); ++i) if (!std::isfinite(c.num[i])) fail("graph: column '" + name + "' has a missing or non-finite value in row " + std::to_string(i));
  return c.num;
}
Graph* from_frame(const Frame& f, const string& from, const string& to, bool directed, const string& weight, const string& capacity) {
  auto* g = make<Graph>();
  g->directed = directed;
  vector<string> a = node_column(f, from), b = node_column(f, to);
  const vector<double>* w = weight.empty() ? nullptr : &number_column(f, weight);
  const vector<double>* c = capacity.empty() ? nullptr : &number_column(f, capacity);
  for (size_t i = 0; i < a.size(); ++i) g->add(a[i], b[i], w ? (*w)[i] : 1.0, c ? (*c)[i] : 0.0, c != nullptr, "graph");
  return g;
}

}  // namespace

void register_graph(Api& api) {
  api.section("Graphs", "Directed or undirected graphs with string node names, built edge by edge or from a frame's edge list; shortest paths "
                        "(Dijkstra, Bellman-Ford, BFS hops, Yen's k shortest simple paths, all pairs) and flows (maximum flow, minimum cut, "
                        "minimum-cost flow) with pycli's `graph_paths` / `graph_flow` result fields and statuses (OK, UNREACHABLE, "
                        "NEGATIVE_CYCLE, INFEASIBLE, UNBALANCED, NUMERICAL_FAILURE). Node lists are sorted by name (numerically when every "
                        "name is an integer). Parallel edges keep the smallest weight for paths and add capacities for flows.");
  api.type(Gt, "Graph: nodes are strings; each edge has a weight (path length, or the unit cost for min_cost_flow) and an optional capacity.");
  api.fn("graph", {"bool"}, Gt, "empty graph (true: directed); add edges with `edge`", [](Args& a) { auto* g = make<Graph>(); g->directed = a[0].b; return O(g); });
  api.fn("graph", {Ft, "string", "string", "bool"}, Gt, "graph from an edge list frame (from column, to column, directed) with unit weights",
         [](Args& a) { return O(from_frame(obj<Frame>(a[0]), a[1].s, a[2].s, a[3].b, "", "")); });
  api.fn("graph", {Ft, "string", "string", "string", "bool"}, Gt, "graph from an edge list frame with a weight column (from, to, weight, directed)",
         [](Args& a) { return O(from_frame(obj<Frame>(a[0]), a[1].s, a[2].s, a[4].b, a[3].s, "")); });
  api.fn("graph", {Ft, "string", "string", "bool", Ot}, Gt, "graph from an edge list frame with options `weight` (column; default unit weights) and `capacity` (column, for flows)",
         [](Args& a) {
           const Opts& o = obj<Opts>(a[4]);
           string w = o.str("weight", ""), c = o.str("capacity", "");
           o.finish("graph");
           return O(from_frame(obj<Frame>(a[0]), a[1].s, a[2].s, a[3].b, w, c)); });
  api.method(Gt, "edge", {"string", "string", "float"}, Gt, "*in place*: add an edge u -> v (both ways when undirected) with a weight; returns the graph",
             [](void* s, Args& a) { gg(s).add(a[0].s, a[1].s, a[2].f, 0, false, "Graph.edge"); return O(static_cast<Graph*>(s)); });
  api.method(Gt, "edge", {"string", "string", "float", "float"}, Gt, "*in place*: add an edge with a weight and a capacity (for flows); returns the graph",
             [](void* s, Args& a) { gg(s).add(a[0].s, a[1].s, a[2].f, a[3].f, true, "Graph.edge"); return O(static_cast<Graph*>(s)); });
  api.method(Gt, "node", {"string"}, Gt, "*in place*: add a node (no-op when present); returns the graph",
             [](void* s, Args& a) { gg(s).node(a[0].s); return O(static_cast<Graph*>(s)); });
  api.method(Gt, "nodes", {}, St, "node names in sorted order", [](void* s, Args&) { Order o(gg(s)); return strsv(names_of(gg(s), o.all())); });
  api.method(Gt, "has_node", {"string"}, "bool", "whether a node exists", [](void* s, Args& a) { return B(gg(s).ids.count(a[0].s) > 0); });
  api.method(Gt, "n_nodes", {}, "int", "node count", [](void* s, Args&) { return I(int64_t(gg(s).names.size())); });
  api.method(Gt, "n_edges", {}, "int", "edges as added (parallel edges and self loops included)", [](void* s, Args&) { return I(int64_t(gg(s).edges.size())); });
  api.method(Gt, "directed", {}, "bool", "whether the graph is directed", [](void* s, Args&) { return B(gg(s).directed); });

  api.method(Gt, "shortest_path", {"string", "string"}, Rt, "Dijkstra shortest path (pycli graph_paths with a target): status OK/UNREACHABLE, path (StrVec), length, hops, n_reachable, n_unreachable, n_nodes, n_edges",
             [](void* s, Args& a) { return O(shortest_path(gg(s), a[0].s, a[1].s, no_opts())); });
  api.method(Gt, "shortest_path", {"string", "string", Ot}, Rt, "shortest path with `method` dijkstra (non-negative weights), bellman_ford (negative weights; status NEGATIVE_CYCLE with `cycle`) or bfs (fewest edges; length = hops)",
             [](void* s, Args& a) { return O(shortest_path(gg(s), a[0].s, a[1].s, obj<Opts>(a[2]))); });
  api.method(Gt, "distances", {"string"}, Rt, "Dijkstra distances from a source to every reachable node (pycli graph_paths without a target): nodes, distance, hops, predecessor (\"\" for the source), n_reachable, n_unreachable",
             [](void* s, Args& a) { return O(distances(gg(s), a[0].s, no_opts())); });
  api.method(Gt, "distances", {"string", Ot}, Rt, "distances with `method` dijkstra, bellman_ford or bfs",
             [](void* s, Args& a) { return O(distances(gg(s), a[0].s, obj<Opts>(a[1]))); });
  api.method(Gt, "k_shortest", {"string", "string", "int"}, Rt, "the k shortest simple paths, shortest first (Yen; networkx shortest_simple_paths): k_found, paths (StrVec of comma-joined node names), lengths, hops; status UNREACHABLE when there is none",
             [](void* s, Args& a) { return O(k_shortest(gg(s), a[0].s, a[1].s, a[2].i, no_opts())); });
  api.method(Gt, "k_shortest", {"string", "string", "int", Ot}, Rt, "k_shortest with `method` dijkstra (weights) or bfs (edge counts)",
             [](void* s, Args& a) { return O(k_shortest(gg(s), a[0].s, a[1].s, a[2].i, obj<Opts>(a[3]))); });
  api.method(Gt, "all_distances", {}, Rt, "all-pairs Dijkstra distances (scipy csgraph): nodes (sorted), distance (Mat, NaN = unreachable), n_unreachable_pairs",
             [](void* s, Args&) { return O(all_distances(gg(s), no_opts())); });
  api.method(Gt, "all_distances", {Ot}, Rt, "all pairs with `method` dijkstra, bellman_ford (Johnson; status NEGATIVE_CYCLE with `cycle`) or bfs",
             [](void* s, Args& a) { return O(all_distances(gg(s), obj<Opts>(a[0]))); });

  api.method(Gt, "max_flow", {"string", "string"}, Rt, "maximum flow (Dinitz; pycli graph_flow max_flow): value, flow_from, flow_to, flow (edges with positive flow, sorted), n_flow_edges, integral, tolerance",
             [](void* s, Args& a) { return O(max_flow(gg(s), a[0].s, a[1].s, false)); });
  api.method(Gt, "min_cut", {"string", "string"}, Rt, "minimum s-t cut (networkx minimum_cut partition): value, cut_from, cut_to, cut_capacity, n_cut_edges, source_side, source_side_size",
             [](void* s, Args& a) { return O(max_flow(gg(s), a[0].s, a[1].s, true)); });
  api.method(Gt, "min_cost_flow", {"string", "string", "float"}, Rt, "cheapest way to send an integer amount from source to sink (edge weight = unit cost; integer capacities and costs): status OK/INFEASIBLE, cost, flow_from, flow_to, flow",
             [](void* s, Args& a) {
               const Graph& g = gg(s);
               int src = g.need(a[0].s, "min_cost_flow"), dst = g.need(a[1].s, "min_cost_flow");
               int64_t v = integral_amount(a[2].f, "flow_value");
               return O(min_cost_flow(g, {{src, v}, {dst, -v}})); });
  api.method(Gt, "min_cost_flow", {St, Vt}, Rt, "minimum-cost flow for node supplies (positive) and demands (negative); status UNBALANCED when they do not sum to zero",
             [](void* s, Args& a) {
               const Graph& g = gg(s);
               const auto& names = obj<StrVec>(a[0]).v; const auto& amt = obj<Vec>(a[1]).v;
               if (names.size() != amt.size()) fail("Graph.min_cost_flow: " + std::to_string(names.size()) + " nodes but " + std::to_string(amt.size()) + " supplies");
               vector<std::pair<int, int64_t>> sup;
               for (size_t i = 0; i < names.size(); ++i) sup.push_back({g.need(names[i], "min_cost_flow"), integral_amount(amt[i], "a supply")});
               return O(min_cost_flow(g, sup)); });
}

}  // namespace lmpy
