// Sparse matrices (Eigen SparseMatrix, CSC) with direct and iterative solvers and symmetric eigenvalues (Spectra's
// implicitly restarted Lanczos, the ARPACK method). pycli op covered: la_sparse (spsolve, cg, gmres, bicgstab, eigsh).
#include "host.hpp"
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <Spectra/MatOp/SparseSymMatProd.h>
#include <Spectra/SymEigsSolver.h>
#include <unsupported/Eigen/IterativeSolvers>

namespace lmpy {
namespace {

using SpMat = Eigen::SparseMatrix<double>;
using Eigen::VectorXd;
const std::string Spt = "Sparse", Vt = "Vec", Rt = "Res", Mt = "Mat";

struct Sparse : Obj {
  static constexpr const char* TYPE = "Sparse";
  int64_t rows = 0, cols = 0;
  std::vector<Eigen::Triplet<double>> trips;  // pending entries (duplicates are summed, like scipy COO -> CSR)
  SpMat m;
  bool built = true;
  const SpMat& mat() {
    if (!built) { m.resize(Eigen::Index(rows), Eigen::Index(cols)); m.setFromTriplets(trips.begin(), trips.end()); m.makeCompressed(); built = true; }
    return m;
  }
};
Sparse& sp(void* s) { return *static_cast<Sparse*>(s); }

VectorXd rhs(Sparse& a, const Value& v, const char* op) {
  const auto& b = obj<Vec>(v).v;
  if (int64_t(b.size()) != a.rows) fail(std::string("Sparse.") + op + ": right-hand side has " + std::to_string(b.size()) + " values; the matrix has " + std::to_string(a.rows) + " rows");
  return to_eigen(b);
}
void need_square(const Sparse& a, const char* op) {
  if (a.rows != a.cols || a.rows == 0) fail(std::string("Sparse.") + op + ": needs a non-empty square matrix");
}

template <class Solver> Res* iterative(Sparse& a, const VectorXd& b, const Opts& o, const char* name, Solver& solver) {
  double rtol = o.num("rtol", 1e-5);
  double atol = o.num("atol", 0.0);
  int64_t maxiter = o.integer("maxiter", 10 * a.rows);
  const std::vector<double>* x0 = o.vec("x0");
  if (rtol < 0 || atol < 0) fail(std::string("Sparse.") + name + ": rtol and atol must be >= 0");
  double bn = b.norm();
  // scipy's criterion ||r|| <= max(rtol * ||b||, atol); Eigen's tolerance is relative to ||b||
  double tol = bn > 0 ? std::max(rtol, atol / bn) : rtol;
  solver.setTolerance(tol > 0 ? tol : std::numeric_limits<double>::epsilon());
  solver.setMaxIterations(Eigen::Index(maxiter));
  solver.compute(a.mat());
  VectorXd x = x0 ? to_eigen(*x0) : VectorXd::Zero(b.size());
  if (x0 && int64_t(x0->size()) != a.cols) fail(std::string("Sparse.") + name + ": x0 has the wrong length");
  x = solver.solveWithGuess(b, x);
  double rr = (b - a.mat() * x).norm();
  auto* r = make<Res>();
  bool ok = rr <= std::max(rtol * bn, atol) * (1 + 1e-9) || bn == 0;
  r->vec("x", x).str("status", ok ? "OK" : "NO_CONVERGENCE").integer("iterations", int64_t(solver.iterations()))
      .num("residual_norm", rr).num("relative_residual", bn > 0 ? rr / bn : rr);
  return r;
}

}  // namespace

void register_sparse(Api& api) {
  api.section("Sparse matrices", "`Sparse` is a compressed sparse matrix (pycli `la_sparse`). Build it from COO triplets (duplicates add up, "
                                 "like scipy) or entry by entry, then solve directly (`solve`, SuperLU-style sparse LU), iteratively "
                                 "(`cg`, `gmres`, `bicgstab`; options `rtol`, `atol`, `maxiter`, `x0`, gmres `restart`) or find "
                                 "eigenvalues of a symmetric matrix (`eigsh`, implicitly restarted Lanczos as in ARPACK).");
  api.type(Spt, "Sparse float64 matrix. Operator `*` multiplies by a Vec.");
  api.fn("sparse", {"int", "int"}, Spt, "empty rows x cols sparse matrix (fill with `add`)", [](Args& a) {
    if (a[0].i < 0 || a[1].i < 0) fail("sparse: negative size"); auto* s = make<Sparse>(); s->rows = a[0].i; s->cols = a[1].i; s->built = false; return O(s); });
  api.fn("sparse", {Vt, Vt, Vt, "int", "int"}, Spt, "from COO triplets (row indexes, column indexes, values, rows, cols); duplicates are summed", [](Args& a) {
    const auto& r = obj<Vec>(a[0]).v; const auto& c = obj<Vec>(a[1]).v; const auto& v = obj<Vec>(a[2]).v;
    if (r.size() != c.size() || r.size() != v.size()) fail("sparse: row, col and val have different lengths");
    auto* s = make<Sparse>(); s->rows = a[3].i; s->cols = a[4].i; s->built = false;
    for (size_t k = 0; k < r.size(); ++k) {
      if (r[k] != std::floor(r[k]) || c[k] != std::floor(c[k]) || r[k] < 0 || c[k] < 0 || r[k] >= double(s->rows) || c[k] >= double(s->cols))
        fail("sparse: entry " + std::to_string(k) + " (" + jnum(r[k]) + ", " + jnum(c[k]) + ") is outside " + std::to_string(s->rows) + "x" + std::to_string(s->cols));
      s->trips.emplace_back(int(r[k]), int(c[k]), v[k]);
    }
    return O(s); });
  api.fn("sparse", {Mt}, Spt, "from a dense matrix (zeros dropped)", [](Args& a) {
    const auto& m = obj<Mat>(a[0]).m; auto* s = make<Sparse>(); s->rows = m.rows(); s->cols = m.cols(); s->m = m.sparseView(); s->m.makeCompressed(); return O(s); });
  api.fn("speye", {"int"}, Spt, "n x n sparse identity", [](Args& a) {
    auto* s = make<Sparse>(); s->rows = s->cols = a[0].i; s->built = false; for (int64_t i = 0; i < a[0].i; ++i) s->trips.emplace_back(int(i), int(i), 1.0); return O(s); });
  api.fn("tridiag", {"int", "float", "float", "float"}, Spt, "n x n tridiagonal matrix with constant (lower, diagonal, upper) bands", [](Args& a) {
    auto* s = make<Sparse>(); s->rows = s->cols = a[0].i; s->built = false;
    for (int64_t i = 0; i < a[0].i; ++i) { s->trips.emplace_back(int(i), int(i), a[2].f); if (i + 1 < a[0].i) { s->trips.emplace_back(int(i + 1), int(i), a[1].f); s->trips.emplace_back(int(i), int(i + 1), a[3].f); } }
    return O(s); });
  api.method(Spt, "add", {"int", "int", "float"}, Spt, "*in place*: A[i, j] += x; returns the matrix", [](void* s, Args& a) {
    Sparse& m = sp(s);
    if (a[0].i < 0 || a[1].i < 0 || a[0].i >= m.rows || a[1].i >= m.cols) fail("Sparse.add: (" + std::to_string(a[0].i) + ", " + std::to_string(a[1].i) + ") is outside the matrix");
    if (m.built) { for (int k = 0; k < m.m.outerSize(); ++k) for (SpMat::InnerIterator it(m.m, k); it; ++it) m.trips.emplace_back(int(it.row()), int(it.col()), it.value()); m.built = false; }
    m.trips.emplace_back(int(a[0].i), int(a[1].i), a[2].f); return O(&m); });
  api.method(Spt, "rows", {}, "int", "row count", [](void* s, Args&) { return I(sp(s).rows); });
  api.method(Spt, "cols", {}, "int", "column count", [](void* s, Args&) { return I(sp(s).cols); });
  api.method(Spt, "nnz", {}, "int", "stored nonzeros (after summing duplicates)", [](void* s, Args&) { return I(int64_t(sp(s).mat().nonZeros())); });
  api.method(Spt, "mul", {Vt}, Vt, "matrix-vector product", [](void* s, Args& a) {
    Sparse& m = sp(s); const auto& x = obj<Vec>(a[0]).v; if (int64_t(x.size()) != m.cols) fail("Sparse.mul: vector length differs from the column count");
    return vecv(VectorXd(m.mat() * to_eigen(x))); });
  api.op(Spt, "*", "mul");
  api.method(Spt, "to_dense", {}, Mt, "dense copy", [](void* s, Args&) { return matv(Eigen::MatrixXd(sp(s).mat())); });
  api.method(Spt, "t", {}, Spt, "transpose", [](void* s, Args&) { auto* r = make<Sparse>(); r->rows = sp(s).cols; r->cols = sp(s).rows; r->m = sp(s).mat().transpose(); r->m.makeCompressed(); return O(r); });
  api.method(Spt, "solve", {Vt}, Vt, "direct sparse LU solve (scipy spsolve); singular matrices are refused", [](void* s, Args& a) {
    Sparse& m = sp(s); need_square(m, "solve"); VectorXd b = rhs(m, a[0], "solve");
    Eigen::SparseLU<SpMat, Eigen::COLAMDOrdering<int>> lu; lu.analyzePattern(m.mat()); lu.factorize(m.mat());
    if (lu.info() != Eigen::Success) fail("Sparse.solve: the matrix is singular (" + lu.lastErrorMessage() + ")");
    return vecv(VectorXd(lu.solve(b))); });
  api.method(Spt, "cg", {Vt}, Rt, "conjugate gradient for symmetric positive definite A: x, status, iterations, residual_norm, relative_residual", [](void* s, Args& a) {
    Sparse& m = sp(s); need_square(m, "cg"); Eigen::ConjugateGradient<SpMat, Eigen::Lower | Eigen::Upper> solver;
    return O(iterative(m, rhs(m, a[0], "cg"), no_opts(), "cg", solver)); });
  api.method(Spt, "cg", {Vt, "Opts"}, Rt, "cg with `rtol` (default 1e-5), `atol`, `maxiter` (default 10 n), `x0`", [](void* s, Args& a) {
    Sparse& m = sp(s); need_square(m, "cg"); Eigen::ConjugateGradient<SpMat, Eigen::Lower | Eigen::Upper> solver; const Opts& o = obj<Opts>(a[1]);
    Res* r = iterative(m, rhs(m, a[0], "cg"), o, "cg", solver); o.finish("cg"); return O(r); });
  api.method(Spt, "gmres", {Vt, "Opts"}, Rt, "restarted GMRES: `rtol`, `atol`, `restart` (default 20), `maxiter`, `x0`", [](void* s, Args& a) {
    Sparse& m = sp(s); need_square(m, "gmres"); const Opts& o = obj<Opts>(a[1]);
    Eigen::GMRES<SpMat, Eigen::DiagonalPreconditioner<double>> solver; solver.set_restart(int(o.integer("restart", 20)));
    int64_t restart = o.integer("restart", 20); (void)restart;
    // pycli/scipy count maxiter in restart cycles; Eigen counts inner iterations
    Opts adjusted = o; if (o.has("maxiter")) { Item it; it.kind = Item::INT; it.i = o.integer("maxiter", 0) * o.integer("restart", 20); adjusted.put("maxiter", it); }
    Res* r = iterative(m, rhs(m, a[0], "gmres"), adjusted, "gmres", solver);
    for (size_t i = 0; i < adjusted.items.size(); ++i) if (adjusted.used[i]) o.find(adjusted.items[i].first);
    o.finish("gmres"); return O(r); });
  api.method(Spt, "bicgstab", {Vt, "Opts"}, Rt, "BiCGSTAB for general A: `rtol`, `atol`, `maxiter`, `x0`", [](void* s, Args& a) {
    Sparse& m = sp(s); need_square(m, "bicgstab"); const Opts& o = obj<Opts>(a[1]); Eigen::BiCGSTAB<SpMat> solver;
    Res* r = iterative(m, rhs(m, a[0], "bicgstab"), o, "bicgstab", solver); o.finish("bicgstab"); return O(r); });
  api.method(Spt, "eigsh", {"int", "Opts"}, Rt, "k eigenvalues of a symmetric matrix (ARPACK-style Lanczos): `which` LA, SA, LM or SM (default LM), `seed` (start vector), `ncv`, `tol`, `maxiter`; fields w (ascending), vectors (Mat, columns), max_residual, nconv, status", [](void* s, Args& a) {
    Sparse& m = sp(s); need_square(m, "eigsh"); const Opts& o = obj<Opts>(a[1]);
    int64_t k = a[0].i, n = m.rows;
    std::string which = o.str("which", "LM");
    int64_t seed = o.integer("seed", 0);
    int64_t ncv = o.integer("ncv", std::min<int64_t>(n, std::max<int64_t>(2 * k + 1, 20)));
    double tol = o.num("tol", 0.0);
    int64_t maxiter = o.integer("maxiter", n * 10);
    o.finish("eigsh");
    if (k < 1 || k >= n) fail("Sparse.eigsh: k must satisfy 1 <= k < n");
    if (ncv <= k || ncv > n) fail("Sparse.eigsh: ncv must satisfy k < ncv <= n");
    if ((m.mat() - SpMat(m.mat().transpose())).norm() != 0.0) fail("Sparse.eigsh: the matrix is not exactly symmetric");
    Spectra::SortRule rule;
    if (which == "LA") rule = Spectra::SortRule::LargestAlge;
    else if (which == "SA") rule = Spectra::SortRule::SmallestAlge;
    else if (which == "LM") rule = Spectra::SortRule::LargestMagn;
    else if (which == "SM") rule = Spectra::SortRule::SmallestMagn;
    else fail("Sparse.eigsh: which must be LA, SA, LM or SM");
    Spectra::SparseSymMatProd<double> op(m.mat());
    Spectra::SymEigsSolver<Spectra::SparseSymMatProd<double>> solver(op, Eigen::Index(k), Eigen::Index(ncv));
    VectorXd v0(n);
    uint64_t state = uint64_t(seed) * 0x9E3779B97F4A7C15ULL + 0x632BE59BD9B4E019ULL;  // splitmix64
    for (int64_t i = 0; i < n; ++i) {
      uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
      z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL; z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL; z ^= z >> 31;
      v0(i) = double(z >> 11) * 0x1.0p-53 - 0.5;
    }
    solver.init(v0.data());
    Eigen::Index nconv = solver.compute(rule, Eigen::Index(maxiter), tol > 0 ? tol : 1e-12, Spectra::SortRule::SmallestAlge);
    auto* r = make<Res>();
    if (solver.info() != Spectra::CompInfo::Successful) {
      r->vec("w", std::vector<double>{}).integer("nconv", int64_t(nconv)).str("status", "NO_CONVERGENCE").num("max_residual", NAN);
      return O(r);
    }
    VectorXd w = solver.eigenvalues();
    Eigen::MatrixXd vecs = solver.eigenvectors();
    for (Eigen::Index j = 0; j < vecs.cols(); ++j) { Eigen::Index i; vecs.col(j).cwiseAbs().maxCoeff(&i); if (vecs(i, j) < 0) vecs.col(j) *= -1; }
    double maxres = 0;
    for (Eigen::Index j = 0; j < vecs.cols(); ++j) maxres = std::max(maxres, (m.mat() * vecs.col(j) - w(j) * vecs.col(j)).norm());
    r->vec("w", w).mat("vectors", vecs).num("max_residual", maxres).integer("nconv", int64_t(nconv)).str("status", "OK");
    return O(r); });
}

}  // namespace lmpy
