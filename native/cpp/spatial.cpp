// Spatial algorithms on point matrices (rows are points): exact k-nearest neighbours and radius neighbours, pairwise
// distances (scipy.spatial.distance.cdist / pdist metrics) and convex hulls in 2-D and 3-D.
// pycli ops covered: spatial_knn(_to), spatial_dist(_to), spatial_hull(_to) (kind convex_hull).
//
// - knn: Euclidean queries use a nanoflann kd-tree for candidates; every reported distance is recomputed directly and rows
//   are ordered by (distance, index), so equal distances list the lower index first (pycli's lexsort). Other Minkowski p
//   use an exact brute-force scan with the same ordering.
// - hull: an own quickhull (2-D: Andrew's monotone chain; 3-D: quickhull with conflict lists). Like Qhull with option Qt,
//   points on an edge or inside a facet are not vertices, coplanar facets are merged and re-triangulated, and the 2-D
//   vertices run counter-clockwise. Volume is the enclosed area (2-D) or volume (3-D); area is the perimeter (2-D) or the
//   surface area (3-D), as in scipy.spatial.ConvexHull.
#include "host.hpp"
#include <nanoflann.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>

namespace lmpy {
namespace {

using Eigen::MatrixXd;
const std::string Mt = "Mat", Vt = "Vec", Rt = "Res", Ot = "Opts";

void need_points(const MatrixXd& m, const std::string& op, const char* what) {
  if (m.rows() == 0 || m.cols() == 0) fail(op + ": " + what + " is empty (" + std::to_string(m.rows()) + "x" + std::to_string(m.cols()) + ")");
  if (!m.allFinite()) fail(op + ": " + what + " contains NaN or infinity");
}

// ---- Minkowski distances -------------------------------------------------------------------------------------------
// p = 2: sqrt of the plain sum of squares (scipy cdist); rescaled by the largest component when that sum over- or
// underflows (pycli's stable form). p = inf: largest component. Other p: largest component times (sum (d/max)^p)^(1/p).
double minkowski(const double* a, const double* b, Eigen::Index d, double p) {
  double mx = 0;
  if (p == 2) {
    double s = 0;
    for (Eigen::Index j = 0; j < d; ++j) { double t = a[j] - b[j]; s += t * t; mx = std::max(mx, std::fabs(t)); }
    if (std::isfinite(s) && (s >= 1e-280 || mx == 0)) return std::sqrt(s);
    double r = 0;
    for (Eigen::Index j = 0; j < d; ++j) { double t = (a[j] - b[j]) / mx; r += t * t; }
    return mx * std::sqrt(r);
  }
  if (p == 1) { double s = 0; for (Eigen::Index j = 0; j < d; ++j) s += std::fabs(a[j] - b[j]); return s; }
  for (Eigen::Index j = 0; j < d; ++j) mx = std::max(mx, std::fabs(a[j] - b[j]));
  if (std::isinf(p) || mx == 0) return mx;
  double s = 0;
  for (Eigen::Index j = 0; j < d; ++j) s += std::pow(std::fabs(a[j] - b[j]) / mx, p);
  return mx * std::pow(s, 1.0 / p);
}

double read_p(const Opts& o, const std::string& op) {
  double p = o.num("p", 2.0);
  if (!(std::isinf(p) && p > 0) && !(p >= 1 && p <= 64)) fail(op + ": p must be in [1, 64] or inf(); got " + jnum(p));
  return p;
}

// Row-major point storage for nanoflann.
struct Cloud {
  std::vector<double> x; size_t n = 0, d = 0;
  size_t kdtree_get_point_count() const { return n; }
  double kdtree_get_pt(size_t i, size_t j) const { return x[i * d + j]; }
  template <class B> bool kdtree_get_bbox(B&) const { return false; }
};
using Tree = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double, Cloud>, Cloud, -1, uint32_t>;

struct Searcher {
  const MatrixXd& pts; double p; Cloud cloud; std::unique_ptr<Tree> tree;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> rows;
  Searcher(const MatrixXd& points, double pp) : pts(points), p(pp), rows(points) {
    if (p == 2) {
      cloud.n = size_t(points.rows()); cloud.d = size_t(points.cols());
      cloud.x.assign(rows.data(), rows.data() + rows.size());
      tree = std::make_unique<Tree>(int(cloud.d), cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    }
  }
  double dist(const double* q, size_t i) const { return minkowski(rows.data() + i * size_t(pts.cols()), q, pts.cols(), p); }
  // All points within distance r of q (r inclusive), as (distance, index) sorted ascending; `self` (or -1) is skipped.
  std::vector<std::pair<double, size_t>> within(const double* q, double r, long self) const {
    std::vector<std::pair<double, size_t>> out;
    size_t n = size_t(pts.rows());
    if (tree) {
      std::vector<nanoflann::ResultItem<uint32_t, double>> hits;
      double r2 = r * r; r2 = r2 * (1 + 1e-9) + 1e-300;
      if (std::isinf(r)) { for (size_t i = 0; i < n; ++i) hits.push_back({uint32_t(i), 0.0}); }
      else tree->radiusSearch(q, r2, hits, nanoflann::SearchParameters(0, false));
      for (auto& h : hits) { if (long(h.first) == self) continue; double dd = dist(q, h.first); if (dd <= r) out.push_back({dd, h.first}); }
    } else {
      for (size_t i = 0; i < n; ++i) { if (long(i) == self) continue; double dd = dist(q, i); if (dd <= r) out.push_back({dd, i}); }
    }
    std::sort(out.begin(), out.end());
    return out;
  }
  // The k nearest, ordered by (distance, index).
  std::vector<std::pair<double, size_t>> nearest(const double* q, size_t k, long self) const {
    size_t n = size_t(pts.rows());
    if (tree) {
      size_t want = std::min(n, k + (self >= 0 ? 1 : 0));
      std::vector<uint32_t> idx(want); std::vector<double> d2(want);
      size_t got = tree->knnSearch(q, want, idx.data(), d2.data());
      double worst = 0;
      for (size_t i = 0; i < got; ++i) if (long(idx[i]) != self) worst = std::max(worst, dist(q, idx[i]));
      auto all = within(q, worst * (1 + 1e-12) + 1e-300, self);  // every tie at the k-th distance, then exact ordering
      if (all.size() >= k) { all.resize(k); return all; }
      // fall through to the exact scan (cannot happen unless the tree missed points)
    }
    std::vector<std::pair<double, size_t>> all;
    all.reserve(n);
    for (size_t i = 0; i < n; ++i) if (long(i) != self) all.push_back({dist(q, i), i});
    size_t kk = std::min(k, all.size());
    std::partial_sort(all.begin(), all.begin() + long(kk), all.end());
    all.resize(kk);
    return all;
  }
};

Res* knn(const MatrixXd& pts, const MatrixXd* queries, int64_t k, const Opts& o) {
  const std::string op = "knn";
  double p = read_p(o, op);
  bool exclude = o.flag("exclude_self", false);
  o.finish(op);
  need_points(pts, op, "points");
  if (queries) { need_points(*queries, op, "queries"); if (queries->cols() != pts.cols()) fail(op + ": queries have " + std::to_string(queries->cols()) + " columns; the points have " + std::to_string(pts.cols())); }
  if (exclude && queries) fail(op + ": exclude_self is only used when the points query themselves (no queries argument)");
  int64_t n = pts.rows();
  if (k < 1) fail(op + ": k must be at least 1; got " + std::to_string(k));
  if (k > n - (exclude ? 1 : 0)) fail(op + ": k = " + std::to_string(k) + " exceeds the " + std::to_string(n - (exclude ? 1 : 0)) + " available points");
  const MatrixXd& q = queries ? *queries : pts;
  Searcher s(pts, p);
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> qr(q);
  MatrixXd ind(q.rows(), k), dst(q.rows(), k);
  for (Eigen::Index i = 0; i < q.rows(); ++i) {
    auto nb = s.nearest(qr.data() + i * q.cols(), size_t(k), exclude ? long(i) : -1);
    for (int64_t j = 0; j < k; ++j) { ind(i, j) = double(nb[size_t(j)].second); dst(i, j) = nb[size_t(j)].first; }
  }
  auto* r = make<Res>();
  r->str("status", "OK").integer("n_points", n).integer("n_queries", q.rows()).integer("k", k).num("p", p)
      .str("search", p == 2 ? "tree" : "brute").mat("indices", ind).mat("distances", dst);
  return r;
}

Res* knn_radius(const MatrixXd& pts, const MatrixXd* queries, double radius, const Opts& o) {
  const std::string op = "knn_radius";
  double p = read_p(o, op);
  bool exclude = o.flag("exclude_self", false);
  int64_t max_pairs = o.integer("max_pairs", 1000000);
  o.finish(op);
  need_points(pts, op, "points");
  if (queries) { need_points(*queries, op, "queries"); if (queries->cols() != pts.cols()) fail(op + ": queries have " + std::to_string(queries->cols()) + " columns; the points have " + std::to_string(pts.cols())); }
  if (exclude && queries) fail(op + ": exclude_self is only used when the points query themselves (no queries argument)");
  if (!(radius >= 0) || std::isinf(radius)) fail(op + ": radius must be finite and >= 0; got " + jnum(radius));
  const MatrixXd& q = queries ? *queries : pts;
  Searcher s(pts, p);
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> qr(q);
  std::vector<double> counts, ind, dst;
  for (Eigen::Index i = 0; i < q.rows(); ++i) {
    auto nb = s.within(qr.data() + i * q.cols(), radius, exclude ? long(i) : -1);
    if (int64_t(ind.size() + nb.size()) > max_pairs) fail(op + ": more than max_pairs = " + std::to_string(max_pairs) + " neighbour pairs (stopped at query " + std::to_string(i) + "); use a smaller radius or raise max_pairs");
    counts.push_back(double(nb.size()));
    for (auto& e : nb) { ind.push_back(double(e.second)); dst.push_back(e.first); }
  }
  auto* r = make<Res>();
  r->str("status", "OK").integer("n_points", pts.rows()).integer("n_queries", q.rows()).num("radius", radius).num("p", p)
      .str("search", p == 2 ? "tree" : "brute").integer("n_pairs", int64_t(ind.size())).vec("counts", counts).vec("indices", ind).vec("distances", dst);
  return r;
}

// ---- cdist / pdist -------------------------------------------------------------------------------------------------
struct Metric {
  std::string name; double p = 2; MatrixXd vi; bool allow_nan = false;
  // Pre-centred rows for correlation are computed by the caller.
  double operator()(const double* a, const double* b, Eigen::Index d) const {
    if (name == "euclidean") return minkowski(a, b, d, 2);
    if (name == "sqeuclidean") { double s = 0; for (Eigen::Index j = 0; j < d; ++j) { double t = a[j] - b[j]; s += t * t; } return s; }
    if (name == "cityblock") return minkowski(a, b, d, 1);
    if (name == "chebyshev") return minkowski(a, b, d, INFINITY);
    if (name == "minkowski") return minkowski(a, b, d, p);
    if (name == "cosine" || name == "correlation") {
      // scipy: 1 - u.v / sqrt(u.u * v.v), the cosine similarity clipped to [-1, 1]
      double uv = 0, uu = 0, vv = 0;
      for (Eigen::Index j = 0; j < d; ++j) { uv += a[j] * b[j]; uu += a[j] * a[j]; vv += b[j] * b[j]; }
      double c = uv / std::sqrt(uu * vv);
      if (std::isnan(c)) return NAN;
      c = std::max(-1.0, std::min(1.0, c));
      return 1.0 - c;
    }
    if (name == "hamming") { double s = 0; for (Eigen::Index j = 0; j < d; ++j) s += a[j] != b[j]; return s / double(d); }
    if (name == "jaccard") {  // non-zero is true (scipy 1.16)
      double num = 0, den = 0;
      for (Eigen::Index j = 0; j < d; ++j) { bool x = a[j] != 0, y = b[j] != 0; num += x != y; den += x || y; }
      return den == 0 ? 0.0 : num / den;
    }
    if (name == "canberra") {
      double s = 0;
      for (Eigen::Index j = 0; j < d; ++j) { double den = std::fabs(a[j]) + std::fabs(b[j]); if (den != 0) s += std::fabs(a[j] - b[j]) / den; }
      return s;
    }
    if (name == "braycurtis") {
      double num = 0, den = 0;
      for (Eigen::Index j = 0; j < d; ++j) { num += std::fabs(a[j] - b[j]); den += std::fabs(a[j] + b[j]); }
      return num / den;
    }
    if (name == "mahalanobis") {
      double s = 0;
      for (Eigen::Index i = 0; i < d; ++i) { double t = 0; for (Eigen::Index j = 0; j < d; ++j) t += vi(i, j) * (a[j] - b[j]); s += (a[i] - b[i]) * t; }
      return std::sqrt(s);
    }
    fail("cdist: unknown metric '" + name + "'");
  }
};

const char* METRICS = "euclidean, sqeuclidean, cityblock, chebyshev, minkowski, cosine, correlation, hamming, jaccard, canberra, braycurtis, mahalanobis";

Metric read_metric(const std::string& name, Eigen::Index d, const Opts& o, const std::string& op) {
  Metric m; m.name = name;
  static const std::vector<std::string> known = {"euclidean", "sqeuclidean", "cityblock", "chebyshev", "minkowski", "cosine", "correlation",
                                                 "hamming", "jaccard", "canberra", "braycurtis", "mahalanobis"};
  if (std::find(known.begin(), known.end(), name) == known.end()) fail(op + ": unknown metric '" + name + "' (metrics: " + METRICS + ")");
  if (name == "minkowski") m.p = read_p(o, op);
  else if (o.has("p")) fail(op + ": option p is only used with metric minkowski");
  const MatrixXd* vi = o.mat("vi");
  if (name == "mahalanobis") {
    if (!vi) fail(op + ": metric mahalanobis needs option vi (the d x d inverse covariance matrix)");
    if (vi->rows() != d || vi->cols() != d) fail(op + ": vi must be " + std::to_string(d) + "x" + std::to_string(d) + "; got " + std::to_string(vi->rows()) + "x" + std::to_string(vi->cols()));
    if (!vi->allFinite()) fail(op + ": vi contains NaN or infinity");
    double scale = vi->cwiseAbs().maxCoeff();
    m.vi = *vi;
    for (Eigen::Index i = 0; i < d; ++i) for (Eigen::Index j = 0; j < i; ++j) {
      double x = m.vi(i, j), y = m.vi(j, i);
      if (std::fabs(x - y) > 1e-12 * scale) fail(op + ": vi must be symmetric to relative tolerance 1e-12");
      m.vi(i, j) = m.vi(j, i) = x / 2 + y / 2;  // pycli symmetrises accepted round-off
    }
  } else if (vi) fail(op + ": option vi is only used with metric mahalanobis");
  m.allow_nan = o.flag("allow_nan", false);
  o.finish(op);
  return m;
}

MatrixXd centred(const MatrixXd& x) { return x.colwise() - x.rowwise().mean(); }

// Evaluates f(i, j) for the requested pairs and enforces the undefined-pair rule.
MatrixXd cdist_m(const MatrixXd& a0, const MatrixXd& b0, const std::string& metric, const Opts& o) {
  const std::string op = "cdist";
  need_points(a0, op, "a"); need_points(b0, op, "b");
  if (a0.cols() != b0.cols()) fail(op + ": a has " + std::to_string(a0.cols()) + " columns and b has " + std::to_string(b0.cols()));
  Metric m = read_metric(metric, a0.cols(), o, op);
  if (double(a0.rows()) * double(b0.rows()) > 1e8) fail(op + ": " + std::to_string(a0.rows()) + " x " + std::to_string(b0.rows()) + " pairs exceed the 1e8 limit");
  using RM = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  RM a = metric == "correlation" ? RM(centred(a0)) : RM(a0), b = metric == "correlation" ? RM(centred(b0)) : RM(b0);
  MatrixXd d(a.rows(), b.rows());
  int64_t bad = 0; Eigen::Index bi = -1, bj = -1;
  for (Eigen::Index i = 0; i < a.rows(); ++i)
    for (Eigen::Index j = 0; j < b.rows(); ++j) {
      double v = m(a.data() + i * a.cols(), b.data() + j * b.cols(), a.cols());
      if (!std::isfinite(v)) { if (!bad) { bi = i; bj = j; } ++bad; }
      d(i, j) = v;
    }
  if (bad && !m.allow_nan)
    fail(op + ": metric " + metric + " is undefined for " + std::to_string(bad) + " pair(s), first (" + std::to_string(bi) + ", " + std::to_string(bj) +
         ") (a zero vector for cosine, a constant row for correlation, an indefinite vi, or overflow); set allow_nan to get NaN entries");
  for (Eigen::Index i = 0; i < d.rows(); ++i) for (Eigen::Index j = 0; j < d.cols(); ++j) if (!std::isfinite(d(i, j))) d(i, j) = NAN;
  return d;
}

std::vector<double> pdist_v(const MatrixXd& a, const std::string& metric, const Opts& o) {
  MatrixXd d = cdist_m(a, a, metric, o);
  std::vector<double> out;
  for (Eigen::Index i = 0; i < d.rows(); ++i) for (Eigen::Index j = i + 1; j < d.cols(); ++j) out.push_back(d(i, j));
  return out;
}

// ---- convex hull ---------------------------------------------------------------------------------------------------
Res* degenerate(int64_t n, int64_t dim, const std::string& msg) {
  auto* r = make<Res>();
  r->str("status", "DEGENERATE").str("kind", "convex_hull").integer("n_points", n).integer("dim", dim).str("message", msg);
  return r;
}

// Distance tolerance for "on the boundary": relative to the largest absolute coordinate, like Qhull's round-off bound.
double hull_tol(const MatrixXd& p) { return 1e-13 * std::max(p.cwiseAbs().maxCoeff(), 1e-300) * double(p.cols()); }

// 2-D hull vertices (counter-clockwise, strictly convex) by the monotone chain over the points sorted by (x, y, index).
std::vector<size_t> chain2(const std::vector<std::array<double, 2>>& pt, const std::vector<size_t>& ids, double tol) {
  std::vector<size_t> s = ids;
  std::sort(s.begin(), s.end(), [&](size_t a, size_t b) { return pt[a][0] != pt[b][0] ? pt[a][0] < pt[b][0] : pt[a][1] != pt[b][1] ? pt[a][1] < pt[b][1] : a < b; });
  // b is kept between a and c when it lies right of a -> c by more than tol (a strict left turn a -> b -> c)
  auto left = [&](size_t a, size_t b, size_t c) {
    double ux = pt[c][0] - pt[a][0], uy = pt[c][1] - pt[a][1], len = std::hypot(ux, uy);
    double cr = ux * (pt[b][1] - pt[a][1]) - uy * (pt[b][0] - pt[a][0]);
    return len > 0 && cr < -tol * len;
  };
  std::vector<size_t> h;
  for (int pass = 0; pass < 2; ++pass) {
    size_t base = h.size();
    for (size_t k = 0; k < s.size(); ++k) {
      size_t i = pass == 0 ? s[k] : s[s.size() - 1 - k];
      while (h.size() >= base + 2 && !left(h[h.size() - 2], h[h.size() - 1], i)) h.pop_back();
      // collapse exact duplicates of the last point
      if (h.size() > base && pt[h.back()][0] == pt[i][0] && pt[h.back()][1] == pt[i][1]) continue;
      h.push_back(i);
    }
    h.pop_back();
  }
  return h;
}

Res* hull2(const MatrixXd& p) {
  int64_t n = p.rows();
  std::vector<std::array<double, 2>> pt(size_t(n), std::array<double, 2>{});
  std::vector<size_t> ids(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) { pt[size_t(i)] = {p(i, 0), p(i, 1)}; ids[size_t(i)] = size_t(i); }
  double tol = hull_tol(p);
  std::vector<size_t> h = chain2(pt, ids, tol);
  if (h.size() < 3) return degenerate(n, 2, "the points are collinear (or coincide): no 2-D hull");
  std::rotate(h.begin(), std::min_element(h.begin(), h.end()), h.end());
  double area = 0, per = 0;
  const auto& o = pt[h[0]];
  for (size_t i = 0; i < h.size(); ++i) {
    const auto& a = pt[h[i]]; const auto& b = pt[h[(i + 1) % h.size()]];
    area += (a[0] - o[0]) * (b[1] - o[1]) - (b[0] - o[0]) * (a[1] - o[1]);
    per += std::hypot(b[0] - a[0], b[1] - a[1]);
  }
  MatrixXd simp(h.size(), 2);
  std::vector<double> verts;
  for (size_t i = 0; i < h.size(); ++i) { simp(long(i), 0) = double(h[i]); simp(long(i), 1) = double(h[(i + 1) % h.size()]); verts.push_back(double(h[i])); }
  auto* r = make<Res>();
  r->str("status", "OK").str("kind", "convex_hull").integer("n_points", n).integer("dim", 2).num("volume", area / 2).num("area", per)
      .integer("n_vertices", int64_t(h.size())).integer("n_simplices", int64_t(h.size())).vec("vertices", verts).mat("simplices", simp);
  return r;
}

// 3-D quickhull. Faces are counter-clockwise seen from outside; nb[e] is the face across edge (v[e], v[e+1]).
struct Hull3 {
  using P = Eigen::Vector3d;
  struct Face { std::array<int, 3> v; std::array<int, 3> nb; P n; bool alive = true; std::vector<int> out; };
  std::vector<P> x; std::vector<Face> f; double tol;

  P normal(int a, int b, int c) const { P n = (x[size_t(b)] - x[size_t(a)]).cross(x[size_t(c)] - x[size_t(a)]); double l = n.norm(); return l > 0 ? P(n / l) : P(0, 0, 0); }
  double dist(const Face& fc, int i) const { return fc.n.dot(x[size_t(i)] - x[size_t(fc.v[0])]); }
  int add(int a, int b, int c) { Face fc; fc.v = {a, b, c}; fc.nb = {-1, -1, -1}; fc.n = normal(a, b, c); f.push_back(fc); return int(f.size()) - 1; }
  int edge_of(int face, int a, int b) const { for (int e = 0; e < 3; ++e) if (f[size_t(face)].v[size_t(e)] == a && f[size_t(face)].v[size_t((e + 1) % 3)] == b) return e; return -1; }

  // Returns "" on success or the degeneracy message.
  std::string build(const std::vector<int>& ids) {
    // initial simplex: the widest axis-extreme pair, then the farthest point from that line, then from that plane
    int i0 = ids[0], i1 = ids[0]; double best = -1;
    for (int ax = 0; ax < 3; ++ax) {
      int lo = ids[0], hi = ids[0];
      for (int i : ids) { if (x[size_t(i)][ax] < x[size_t(lo)][ax]) lo = i; if (x[size_t(i)][ax] > x[size_t(hi)][ax]) hi = i; }
      double w = (x[size_t(hi)] - x[size_t(lo)]).norm();
      if (w > best) { best = w; i0 = lo; i1 = hi; }
    }
    if (best <= tol) return "all points coincide";
    P dir = (x[size_t(i1)] - x[size_t(i0)]).normalized();
    int i2 = -1; best = -1;
    for (int i : ids) { P v = x[size_t(i)] - x[size_t(i0)]; double dd = (v - dir * v.dot(dir)).norm(); if (dd > best) { best = dd; i2 = i; } }
    if (best <= tol) return "the points are collinear: no 3-D hull";
    P n = normal(i0, i1, i2);
    int i3 = -1; best = -1;
    for (int i : ids) { double dd = std::fabs(n.dot(x[size_t(i)] - x[size_t(i0)])); if (dd > best) { best = dd; i3 = i; } }
    if (best <= tol) return "the points are coplanar: no 3-D hull (initial simplex is flat)";
    if (n.dot(x[size_t(i3)] - x[size_t(i0)]) > 0) std::swap(i1, i2);  // i3 below the base (i0, i1, i2)
    int f0 = add(i0, i1, i2), f1 = add(i0, i3, i1), f2 = add(i1, i3, i2), f3 = add(i2, i3, i0);
    link_all({f0, f1, f2, f3});
    std::vector<int> cand;
    for (int i : ids) if (i != i0 && i != i1 && i != i2 && i != i3) cand.push_back(i);
    assign(cand, {f0, f1, f2, f3});
    for (size_t cur = 0; cur < f.size(); ++cur) {
      while (f[cur].alive && !f[cur].out.empty()) {
        int far = f[cur].out[0]; double fd = dist(f[cur], far);
        for (int i : f[cur].out) { double dd = dist(f[cur], i); if (dd > fd) { fd = dd; far = i; } }
        add_point(int(cur), far);
        if (!f[cur].alive) break;
      }
    }
    return "";
  }
  void link_all(const std::vector<int>& fs) {
    for (int a : fs) for (int e = 0; e < 3; ++e) {
      int u = f[size_t(a)].v[size_t(e)], w = f[size_t(a)].v[size_t((e + 1) % 3)];
      for (int b : fs) if (b != a) { int k = edge_of(b, w, u); if (k >= 0) f[size_t(a)].nb[size_t(e)] = b; }
    }
  }
  void assign(const std::vector<int>& pts, const std::vector<int>& fs) {
    for (int i : pts) {
      int bestf = -1; double bd = tol;
      for (int k : fs) { double dd = dist(f[size_t(k)], i); if (dd > bd) { bd = dd; bestf = k; } }
      if (bestf >= 0) f[size_t(bestf)].out.push_back(i);
    }
  }
  void add_point(int start, int p) {
    // visible faces by depth-first search; horizon edges collected in counter-clockwise order
    std::vector<char> vis(f.size(), 0);
    std::vector<std::pair<int, int>> horizon;  // (face across, edge index in that visible face) -> edge vertices
    std::vector<std::array<int, 3>> hz;         // a, b, outside face
    std::vector<int> visible;
    std::function<void(int, int)> walk = [&](int fc, int from_edge) {
      vis[size_t(fc)] = 1; visible.push_back(fc);
      for (int k = 0; k < 3; ++k) {
        int e = (from_edge + k) % 3;
        int nb = f[size_t(fc)].nb[size_t(e)];
        if (vis[size_t(nb)]) continue;
        if (dist(f[size_t(nb)], p) > tol) {
          int back = edge_of(nb, f[size_t(fc)].v[size_t((e + 1) % 3)], f[size_t(fc)].v[size_t(e)]);
          walk(nb, back);
        } else hz.push_back({f[size_t(fc)].v[size_t(e)], f[size_t(fc)].v[size_t((e + 1) % 3)], nb});
      }
    };
    walk(start, 0);
    std::vector<int> orphans;
    for (int fc : visible) { f[size_t(fc)].alive = false; for (int i : f[size_t(fc)].out) if (i != p) orphans.push_back(i); f[size_t(fc)].out.clear(); }
    std::vector<int> nf;
    for (auto& h : hz) {
      int id = add(h[0], h[1], p);
      nf.push_back(id);
      f[size_t(id)].nb[0] = h[2];
      int k = edge_of(h[2], h[1], h[0]);
      f[size_t(h[2])].nb[size_t(k)] = id;
    }
    for (size_t i = 0; i < nf.size(); ++i) {
      size_t j = (i + 1) % nf.size();
      if (f[size_t(nf[i])].v[1] != f[size_t(nf[j])].v[0]) fail("hull: internal error (horizon is not a cycle)");
      f[size_t(nf[i])].nb[1] = nf[j];
      f[size_t(nf[j])].nb[2] = nf[i];
    }
    std::sort(orphans.begin(), orphans.end());
    assign(orphans, nf);
  }
};

Res* hull3(const MatrixXd& p) {
  int64_t n = p.rows();
  Hull3 h;
  h.tol = hull_tol(p);
  for (int64_t i = 0; i < n; ++i) h.x.push_back(Hull3::P(p(i, 0), p(i, 1), p(i, 2)));
  std::vector<int> ids(static_cast<size_t>(n));
  std::iota(ids.begin(), ids.end(), 0);
  std::string msg = h.build(ids);
  if (!msg.empty()) return degenerate(n, 3, msg);
  // merge coplanar neighbouring faces (union-find), then keep the strictly convex corners of every merged facet
  std::vector<int> alive, par(h.f.size());
  std::iota(par.begin(), par.end(), 0);
  std::function<int(int)> root = [&](int a) { while (par[size_t(a)] != a) a = par[size_t(a)] = par[size_t(par[size_t(a)])]; return a; };
  for (size_t i = 0; i < h.f.size(); ++i) if (h.f[i].alive) alive.push_back(int(i));
  for (int a : alive) for (int e = 0; e < 3; ++e) {
    int b = h.f[size_t(a)].nb[size_t(e)];
    if (b < a) continue;
    const auto& fa = h.f[size_t(a)]; const auto& fb = h.f[size_t(b)];
    bool flat = true;
    for (int k = 0; k < 3; ++k) flat = flat && std::fabs(h.dist(fa, fb.v[size_t(k)])) <= h.tol && std::fabs(h.dist(fb, fa.v[size_t(k)])) <= h.tol;
    if (flat && fa.n.dot(fb.n) > 0) par[size_t(root(a))] = root(b);
  }
  std::map<int, std::vector<int>> groups;
  for (int a : alive) groups[root(a)].push_back(a);
  Hull3::P c(0, 0, 0); int cnt = 0;
  std::vector<std::array<int, 3>> tris;
  std::vector<char> is_vertex(size_t(n), 0);
  double area = 0;
  for (auto& g : groups) {
    // facet plane: area-weighted normal of its triangles
    Hull3::P nrm(0, 0, 0);
    std::vector<int> vs;
    for (int a : g.second) {
      const auto& fa = h.f[size_t(a)];
      nrm += (h.x[size_t(fa.v[1])] - h.x[size_t(fa.v[0])]).cross(h.x[size_t(fa.v[2])] - h.x[size_t(fa.v[0])]);
      for (int v : fa.v) vs.push_back(v);
    }
    std::sort(vs.begin(), vs.end()); vs.erase(std::unique(vs.begin(), vs.end()), vs.end());
    nrm.normalize();
    Hull3::P u = (std::fabs(nrm[0]) < 0.9 ? Hull3::P(1, 0, 0) : Hull3::P(0, 1, 0)).cross(nrm).normalized(), w = nrm.cross(u);
    std::vector<std::array<double, 2>> pt(vs.size());
    std::vector<size_t> local(vs.size());
    for (size_t k = 0; k < vs.size(); ++k) { Hull3::P d = h.x[size_t(vs[k])] - h.x[size_t(vs[0])]; pt[k] = {d.dot(u), d.dot(w)}; local[k] = k; }
    std::vector<size_t> ring = chain2(pt, local, h.tol);
    for (auto& k : ring) k = size_t(vs[k]);  // (u, w, nrm) is right-handed: counter-clockwise seen from outside
    if (ring.size() < 3) continue;
    std::rotate(ring.begin(), std::min_element(ring.begin(), ring.end()), ring.end());
    for (size_t k = 0; k < ring.size(); ++k) is_vertex[ring[k]] = 1;
    for (size_t k = 1; k + 1 < ring.size(); ++k) {
      std::array<int, 3> t = {int(ring[0]), int(ring[k]), int(ring[k + 1])};
      tris.push_back(t);
      area += 0.5 * (h.x[size_t(t[1])] - h.x[size_t(t[0])]).cross(h.x[size_t(t[2])] - h.x[size_t(t[0])]).norm();
    }
  }
  std::vector<double> verts;
  for (int64_t i = 0; i < n; ++i) if (is_vertex[size_t(i)]) { verts.push_back(double(i)); c += h.x[size_t(i)]; ++cnt; }
  c /= double(cnt);
  double vol = 0;
  for (auto& t : tris) vol += std::fabs((h.x[size_t(t[0])] - c).dot((h.x[size_t(t[1])] - c).cross(h.x[size_t(t[2])] - c))) / 6.0;
  MatrixXd simp(long(tris.size()), 3);
  for (size_t i = 0; i < tris.size(); ++i) for (int k = 0; k < 3; ++k) simp(long(i), k) = double(tris[i][size_t(k)]);
  auto* r = make<Res>();
  r->str("status", "OK").str("kind", "convex_hull").integer("n_points", n).integer("dim", 3).num("volume", vol).num("area", area)
      .integer("n_vertices", int64_t(verts.size())).integer("n_simplices", int64_t(tris.size())).vec("vertices", verts).mat("simplices", simp);
  return r;
}

Res* hull(const MatrixXd& p) {
  if (p.cols() != 2 && p.cols() != 3) fail("hull: points must have 2 or 3 columns (2-D or 3-D hulls); got " + std::to_string(p.cols()));
  need_points(p, "hull", "points");
  if (p.rows() > 1000000) fail("hull: more than 1,000,000 points");
  if (p.rows() < p.cols() + 1) return degenerate(p.rows(), p.cols(), "need at least dim + 1 points, got " + std::to_string(p.rows()));
  return p.cols() == 2 ? hull2(p) : hull3(p);
}

}  // namespace

void register_spatial(Api& api) {
  api.section("Spatial", "Nearest neighbours, distance matrices and convex hulls on point matrices whose rows are points "
                         "(pycli `spatial_knn`, `spatial_dist`, `spatial_hull`; scipy.spatial semantics). Neighbour rows are ordered by "
                         "distance and then by point index; indices are 0-based and returned as float matrices.");
  auto M = [](const Value& v) -> const MatrixXd& { return obj<Mat>(v).m; };
  api.fn("knn", {Mt, Mt, "int"}, Rt, "k nearest points of every query row, Euclidean (pycli spatial_knn): indices, distances (queries x k), status, n_points, n_queries, k, p, search",
         [M](Args& a) { return O(knn(M(a[0]), &M(a[1]), a[2].i, no_opts())); });
  api.fn("knn", {Mt, Mt, "int", Ot}, Rt, "knn with options: `p` (Minkowski norm in [1, 64] or inf(); default 2)",
         [M](Args& a) { return O(knn(M(a[0]), &M(a[1]), a[2].i, obj<Opts>(a[3]))); });
  api.fn("knn", {Mt, "int"}, Rt, "k nearest points of every point among the points themselves (a point is its own nearest neighbour)",
         [M](Args& a) { return O(knn(M(a[0]), nullptr, a[1].i, no_opts())); });
  api.fn("knn", {Mt, "int", Ot}, Rt, "self-query knn with options: `p`, `exclude_self` (drop each point from its own list)",
         [M](Args& a) { return O(knn(M(a[0]), nullptr, a[1].i, obj<Opts>(a[2]))); });
  api.fn("knn_radius", {Mt, Mt, "float"}, Rt, "all points within a distance (inclusive) of every query (pycli spatial_knn radius mode): counts per query, flat indices and distances (CSR layout), n_pairs",
         [M](Args& a) { return O(knn_radius(M(a[0]), &M(a[1]), a[2].f, no_opts())); });
  api.fn("knn_radius", {Mt, Mt, "float", Ot}, Rt, "knn_radius with options: `p`, `max_pairs` (default 1000000; more is an error)",
         [M](Args& a) { return O(knn_radius(M(a[0]), &M(a[1]), a[2].f, obj<Opts>(a[3]))); });
  api.fn("knn_radius", {Mt, "float", Ot}, Rt, "radius neighbours of every point among the points themselves; options `p`, `max_pairs`, `exclude_self`",
         [M](Args& a) { return O(knn_radius(M(a[0]), nullptr, a[1].f, obj<Opts>(a[2]))); });
  api.fn("cdist", {Mt, Mt, "string"}, Mt, "distance matrix between the rows of a and b (scipy cdist; pycli spatial_dist): euclidean, sqeuclidean, cityblock, chebyshev, minkowski, cosine, correlation, hamming, jaccard (non-zero is true), canberra, braycurtis, mahalanobis",
         [M](Args& a) { return matv(cdist_m(M(a[0]), M(a[1]), a[2].s, no_opts())); });
  api.fn("cdist", {Mt, Mt, "string", Ot}, Mt, "cdist with options: `p` (minkowski), `vi` (Mat, mahalanobis inverse covariance, symmetric within 1e-12), `allow_nan` (undefined pairs become NaN instead of an error)",
         [M](Args& a) { return matv(cdist_m(M(a[0]), M(a[1]), a[2].s, obj<Opts>(a[3]))); });
  api.fn("cdist", {Mt, "string"}, Mt, "distance matrix of the rows of a with themselves (symmetric, zero diagonal for a true metric)",
         [M](Args& a) { return matv(cdist_m(M(a[0]), M(a[0]), a[1].s, no_opts())); });
  api.fn("pdist", {Mt, "string"}, Vt, "condensed pairwise distances (scipy pdist): d(0,1), d(0,2), ..., d(n-2,n-1)",
         [M](Args& a) { return vecv(pdist_v(M(a[0]), a[1].s, no_opts())); });
  api.fn("pdist", {Mt, "string", Ot}, Vt, "pdist with the cdist options",
         [M](Args& a) { return vecv(pdist_v(M(a[0]), a[1].s, obj<Opts>(a[2]))); });
  api.fn("hull", {Mt}, Rt, "convex hull of 2-D or 3-D points (scipy ConvexHull, Qhull Qt; pycli spatial_hull convex_hull): status OK/DEGENERATE, volume, area, "
                           "vertices (Vec of indices; 2-D counter-clockwise from the lowest index), simplices (Mat: edges in 2-D, outward counter-clockwise triangles in 3-D), n_vertices, n_simplices, n_points, dim",
         [M](Args& a) { return O(hull(M(a[0]))); });
}

}  // namespace lmpy
