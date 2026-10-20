// Dense linear algebra on Mat (Eigen): solves, least squares, properties, decompositions, matrix functions.
// pycli ops covered: la_solve(_to), la_lstsq(_to), la_props, la_decomp(_to), la_matrix_fn(_to).
#include "host.hpp"
#include <unsupported/Eigen/MatrixFunctions>
#include <Eigen/Eigenvalues>

namespace lmnum {
namespace {

using Eigen::MatrixXd;
using Eigen::VectorXd;
const std::string Mt = "Mat", Vt = "Vec", Rt = "Res";

const MatrixXd& self_m(void* s) { return static_cast<Mat*>(s)->m; }
void need_square(const MatrixXd& m, const char* op) {
  if (m.rows() != m.cols() || m.rows() == 0) fail(std::string("Mat.") + op + ": needs a non-empty square matrix; got " + std::to_string(m.rows()) + "x" + std::to_string(m.cols()));
}
void need_finite(const MatrixXd& m, const char* op) {
  if (!m.allFinite()) fail(std::string("Mat.") + op + ": the matrix contains NaN or infinity");
}
bool exactly_symmetric(const MatrixXd& m) { return m.rows() == m.cols() && (m - m.transpose()).cwiseAbs().maxCoeff() == 0.0; }

// LAPACK-style reciprocal condition number estimate in the 1-norm from an LU factorisation (exact inverse norm; fine
// for the matrix sizes the DSL targets).
double rcond_1(const MatrixXd& a, const Eigen::PartialPivLU<MatrixXd>& lu) {
  double an = a.cwiseAbs().colwise().sum().maxCoeff();
  MatrixXd inv = lu.inverse();
  double in = inv.cwiseAbs().colwise().sum().maxCoeff();
  if (an == 0 || !std::isfinite(in)) return 0.0;
  return 1.0 / (an * in);
}

MatrixXd solve_any(const MatrixXd& a, const MatrixXd& b, const std::string& assume) {
  need_square(a, "solve");
  need_finite(a, "solve");
  need_finite(b, "solve");
  if (b.rows() != a.rows()) fail("Mat.solve: right-hand side has " + std::to_string(b.rows()) + " rows; the matrix has " + std::to_string(a.rows()));
  if (assume == "pos") {
    if (!exactly_symmetric(a)) fail("Mat.solve(pos): the matrix is not exactly symmetric");
    Eigen::LLT<MatrixXd> llt(a);
    if (llt.info() != Eigen::Success) fail("Mat.solve(pos): the matrix is not positive definite");
    return llt.solve(b);
  }
  if (assume == "sym" && !exactly_symmetric(a)) fail("Mat.solve(sym): the matrix is not exactly symmetric");
  if (assume != "gen" && assume != "sym") fail("Mat.solve: assume_a must be gen, sym or pos; got '" + assume + "'");
  Eigen::FullPivLU<MatrixXd> full(a);
  if (!full.isInvertible()) fail("Mat.solve: the matrix is singular");
  Eigen::PartialPivLU<MatrixXd> lu(a);
  double rc = rcond_1(a, lu);
  if (rc < std::numeric_limits<double>::epsilon()) fail("Mat.solve: the matrix is singular to working precision (rcond " + jnum(rc) + ")");
  return lu.solve(b);
}

Res* lstsq(const MatrixXd& a, const VectorXd& b, const Opts& o) {
  double rcond = o.num("rcond", -1);
  o.finish("lstsq");
  need_finite(a, "lstsq");
  if (!b.allFinite()) fail("Mat.lstsq: the right-hand side contains NaN or infinity");
  if (b.size() != a.rows()) fail("Mat.lstsq: right-hand side has " + std::to_string(b.size()) + " values; the matrix has " + std::to_string(a.rows()) + " rows");
  Eigen::JacobiSVD<MatrixXd> svd(a, Eigen::ComputeThinU | Eigen::ComputeThinV);
  VectorXd s = svd.singularValues();
  double eps = std::numeric_limits<double>::epsilon();
  double cut = (rcond < 0 ? eps * double(std::max(a.rows(), a.cols())) : rcond) * (s.size() ? s(0) : 0.0);
  Eigen::Index rank = 0;
  for (Eigen::Index i = 0; i < s.size(); ++i) if (s(i) > cut) ++rank;
  VectorXd utb = svd.matrixU().transpose() * b;
  VectorXd y = VectorXd::Zero(s.size());
  for (Eigen::Index i = 0; i < rank; ++i) y(i) = utb(i) / s(i);
  VectorXd x = svd.matrixV() * y;
  VectorXd r = b - a * x;
  auto* res = make<Res>();
  res->vec("x", x).integer("rank", rank).vec("singular_values", s);
  // numpy returns residues (sum of squared residuals) only for full-rank, overdetermined systems
  if (rank == a.cols() && a.rows() > a.cols()) res->vec("residues", std::vector<double>{r.squaredNorm()});
  else res->vec("residues", std::vector<double>{});
  res->num("residual_norm", r.norm());
  res->str("status", rank < std::min(a.rows(), a.cols()) ? "RANK_DEFICIENT" : "OK");
  return res;
}

double norm_of(const MatrixXd& m, const std::string& ord) {
  if (ord == "fro") return m.norm();
  if (ord == "1") return m.cwiseAbs().colwise().sum().maxCoeff();
  if (ord == "inf") return m.cwiseAbs().rowwise().sum().maxCoeff();
  if (ord == "-1") return m.cwiseAbs().colwise().sum().minCoeff();
  if (ord == "-inf") return m.cwiseAbs().rowwise().sum().minCoeff();
  Eigen::JacobiSVD<MatrixXd> svd(m);
  if (ord == "2") return svd.singularValues().size() ? svd.singularValues()(0) : 0.0;
  if (ord == "-2") return svd.singularValues().size() ? svd.singularValues().tail(1)(0) : 0.0;
  if (ord == "nuc") return svd.singularValues().sum();
  fail("Mat.norm: ord must be fro, nuc, 1, -1, 2, -2, inf or -inf; got '" + ord + "'");
}

}  // namespace

void register_linalg(Api& api) {
  api.section("Linear algebra", "Dense solves, least squares, matrix properties, decompositions and matrix functions on `Mat` "
                                "(pycli `la_solve`, `la_lstsq`, `la_props`, `la_decomp`, `la_matrix_fn`; Eigen 3.4 underneath). "
                                "A singular or ill-posed input is a run-time error, except `lstsq`, which reports `status` RANK_DEFICIENT.");
  api.method(Mt, "solve", {Vt}, Vt, "x with A x = b (LU with partial pivoting; singular matrices are refused)", [](void* s, Args& a) {
    MatrixXd b = to_eigen(obj<Vec>(a[0]).v);
    return vecv(VectorXd(solve_any(self_m(s), b, "gen").col(0))); });
  api.method(Mt, "solve", {Vt, "string"}, Vt, "solve with assume_a = gen, sym (exactly symmetric) or pos (Cholesky; must be positive definite)", [](void* s, Args& a) {
    MatrixXd b = to_eigen(obj<Vec>(a[0]).v);
    return vecv(VectorXd(solve_any(self_m(s), b, a[1].s).col(0))); });
  api.method(Mt, "solve", {Mt}, Mt, "X with A X = B (several right-hand sides)", [](void* s, Args& a) { return matv(solve_any(self_m(s), obj<Mat>(a[0]).m, "gen")); });
  api.method(Mt, "lstsq", {Vt}, Rt, "least squares (SVD, numpy.linalg.lstsq): x, rank, singular_values, residues, residual_norm, status OK/RANK_DEFICIENT",
             [](void* s, Args& a) { return O(lstsq(self_m(s), to_eigen(obj<Vec>(a[0]).v), no_opts())); });
  api.method(Mt, "lstsq", {Vt, "Opts"}, Rt, "least squares with `rcond` (relative singular-value cutoff; default eps * max(m, n))",
             [](void* s, Args& a) { return O(lstsq(self_m(s), to_eigen(obj<Vec>(a[0]).v), obj<Opts>(a[1]))); });
  api.method(Mt, "inv", {}, Mt, "inverse (singular matrices are refused)", [](void* s, Args&) {
    return matv(solve_any(self_m(s), MatrixXd::Identity(self_m(s).rows(), self_m(s).cols()), "gen")); });
  api.method(Mt, "pinv", {}, Mt, "Moore-Penrose pseudo-inverse (cutoff eps * max(m, n) * s_max)", [](void* s, Args&) {
    const MatrixXd& m = self_m(s); need_finite(m, "pinv");
    Eigen::JacobiSVD<MatrixXd> svd(m, Eigen::ComputeThinU | Eigen::ComputeThinV);
    VectorXd sv = svd.singularValues(); double cut = std::numeric_limits<double>::epsilon() * double(std::max(m.rows(), m.cols())) * (sv.size() ? sv(0) : 0);
    VectorXd inv = sv.unaryExpr([cut](double x) { return x > cut ? 1.0 / x : 0.0; });
    return matv(svd.matrixV() * inv.asDiagonal() * svd.matrixU().transpose()); });
  api.method(Mt, "det", {}, "float", "determinant", [](void* s, Args&) { need_square(self_m(s), "det"); need_finite(self_m(s), "det"); return F(self_m(s).partialPivLu().determinant()); });
  api.method(Mt, "slogdet", {}, Rt, "sign and log|det| (numpy.linalg.slogdet): fields sign, logabsdet", [](void* s, Args&) {
    const MatrixXd& m = self_m(s); need_square(m, "slogdet"); need_finite(m, "slogdet");
    Eigen::PartialPivLU<MatrixXd> lu(m); const MatrixXd& f = lu.matrixLU(); double sign = lu.permutationP().determinant(), logabs = 0;
    for (Eigen::Index i = 0; i < f.rows(); ++i) { double d = f(i, i); if (d == 0) { sign = 0; logabs = -INFINITY; break; } if (d < 0) sign = -sign; logabs += std::log(std::fabs(d)); }
    auto* r = make<Res>(); r->num("sign", sign).num("logabsdet", logabs); return O(r); });
  api.method(Mt, "rank", {}, "int", "numerical rank (SVD, tolerance s_max * max(m, n) * eps)", [](void* s, Args&) {
    const MatrixXd& m = self_m(s); need_finite(m, "rank"); if (m.size() == 0) return I(0);
    Eigen::JacobiSVD<MatrixXd> svd(m); VectorXd sv = svd.singularValues(); double tol = sv(0) * double(std::max(m.rows(), m.cols())) * std::numeric_limits<double>::epsilon();
    int64_t r = 0; for (Eigen::Index i = 0; i < sv.size(); ++i) r += sv(i) > tol; return I(r); });
  api.method(Mt, "cond", {}, "float", "2-norm condition number s_max / s_min", [](void* s, Args&) {
    const MatrixXd& m = self_m(s); need_finite(m, "cond"); Eigen::JacobiSVD<MatrixXd> svd(m); VectorXd sv = svd.singularValues();
    if (sv.size() == 0) fail("Mat.cond: empty matrix"); return F(sv.tail(1)(0) == 0 ? INFINITY : sv(0) / sv.tail(1)(0)); });
  api.method(Mt, "norm", {}, "float", "Frobenius norm", [](void* s, Args&) { return F(self_m(s).norm()); });
  api.method(Mt, "norm", {"string"}, "float", "matrix norm: fro, nuc, 1, -1, 2, -2, inf, -inf (numpy ord)", [](void* s, Args& a) { return F(norm_of(self_m(s), a[0].s)); });
  api.method(Mt, "trace", {}, "float", "sum of the diagonal", [](void* s, Args&) { return F(self_m(s).trace()); });
  api.method(Mt, "is_symmetric", {"float"}, "bool", "max |A - A^T| <= tol", [](void* s, Args& a) {
    const MatrixXd& m = self_m(s); return B(m.rows() == m.cols() && (m.size() == 0 || (m - m.transpose()).cwiseAbs().maxCoeff() <= a[0].f)); });

  api.method(Mt, "svd", {}, Rt, "thin SVD A = U diag(s) Vt: fields u (Mat), s (Vec, descending), vt (Mat), reconstruction_error", [](void* s, Args&) {
    const MatrixXd& m = self_m(s); need_finite(m, "svd");
    Eigen::JacobiSVD<MatrixXd> svd(m, Eigen::ComputeThinU | Eigen::ComputeThinV);
    MatrixXd u = svd.matrixU(), vt = svd.matrixV().transpose(); VectorXd sv = svd.singularValues();
    auto* r = make<Res>(); r->mat("u", u).vec("s", sv).mat("vt", vt).num("reconstruction_error", (u * sv.asDiagonal() * vt - m).cwiseAbs().maxCoeff()); return O(r); });
  api.method(Mt, "qr", {}, Rt, "reduced QR A = Q R (Householder): fields q, r, reconstruction_error", [](void* s, Args&) {
    const MatrixXd& m = self_m(s); need_finite(m, "qr");
    Eigen::HouseholderQR<MatrixXd> qr(m); Eigen::Index k = std::min(m.rows(), m.cols());
    MatrixXd q = qr.householderQ() * MatrixXd::Identity(m.rows(), k);
    MatrixXd rr = qr.matrixQR().topRows(k).triangularView<Eigen::Upper>();
    // numpy/LAPACK sign convention is not fixed; make diag(R) non-negative for a canonical answer
    for (Eigen::Index i = 0; i < k; ++i) if (rr(i, i) < 0) { rr.row(i) *= -1; q.col(i) *= -1; }
    auto* r = make<Res>(); r->mat("q", q).mat("r", rr).num("reconstruction_error", (q * rr - m).cwiseAbs().maxCoeff()); return O(r); });
  api.method(Mt, "eigh", {}, Rt, "symmetric eigen-decomposition: fields values (ascending), vectors (columns), reconstruction_error", [](void* s, Args&) {
    const MatrixXd& m = self_m(s); need_square(m, "eigh"); need_finite(m, "eigh");
    Eigen::SelfAdjointEigenSolver<MatrixXd> es(m);
    if (es.info() != Eigen::Success) fail("Mat.eigh: no convergence");
    MatrixXd v = es.eigenvectors();
    for (Eigen::Index j = 0; j < v.cols(); ++j) { Eigen::Index k; v.col(j).cwiseAbs().maxCoeff(&k); if (v(k, j) < 0) v.col(j) *= -1; }  // canonical sign
    auto* r = make<Res>(); r->vec("values", es.eigenvalues()).mat("vectors", v).num("reconstruction_error", (v * es.eigenvalues().asDiagonal() * v.transpose() - m).cwiseAbs().maxCoeff()); return O(r); });
  api.method(Mt, "eig", {}, Rt, "general eigenvalues: fields values_re, values_im (sorted by real part, then imaginary), real (bool: all imaginary parts 0)", [](void* s, Args&) {
    const MatrixXd& m = self_m(s); need_square(m, "eig"); need_finite(m, "eig");
    Eigen::EigenSolver<MatrixXd> es(m, false);
    if (es.info() != Eigen::Success) fail("Mat.eig: no convergence");
    std::vector<std::pair<double, double>> ev; for (Eigen::Index i = 0; i < es.eigenvalues().size(); ++i) ev.push_back({es.eigenvalues()(i).real(), es.eigenvalues()(i).imag()});
    std::sort(ev.begin(), ev.end());
    std::vector<double> re, im; bool real = true; for (auto& e : ev) { re.push_back(e.first); im.push_back(e.second); real &= e.second == 0; }
    auto* r = make<Res>(); r->vec("values_re", re).vec("values_im", im).flag("real", real); return O(r); });
  api.method(Mt, "cholesky", {}, Mt, "lower-triangular L with A = L L^T (must be symmetric positive definite)", [](void* s, Args&) {
    const MatrixXd& m = self_m(s); need_square(m, "cholesky"); need_finite(m, "cholesky");
    if (!exactly_symmetric(m)) fail("Mat.cholesky: the matrix is not exactly symmetric");
    Eigen::LLT<MatrixXd> llt(m); if (llt.info() != Eigen::Success) fail("Mat.cholesky: the matrix is not positive definite");
    return matv(MatrixXd(llt.matrixL())); });
  api.method(Mt, "lu", {}, Rt, "LU with partial pivoting A = P L U (scipy.linalg.lu): fields p, l, u", [](void* s, Args&) {
    const MatrixXd& m = self_m(s); need_square(m, "lu"); need_finite(m, "lu");
    Eigen::PartialPivLU<MatrixXd> lu(m); MatrixXd l = MatrixXd(lu.matrixLU().triangularView<Eigen::UnitLower>()), u = MatrixXd(lu.matrixLU().triangularView<Eigen::Upper>());
    MatrixXd p = lu.permutationP().transpose().toDenseMatrix().cast<double>();
    auto* r = make<Res>(); r->mat("p", p).mat("l", l).mat("u", u); return O(r); });
  api.method(Mt, "expm", {}, Mt, "matrix exponential (Padé scaling and squaring)", [](void* s, Args&) { need_square(self_m(s), "expm"); need_finite(self_m(s), "expm"); return matv(MatrixXd(self_m(s).exp())); });
  api.method(Mt, "logm", {}, Mt, "principal matrix logarithm (real result required)", [](void* s, Args&) {
    const MatrixXd& m = self_m(s); need_square(m, "logm"); need_finite(m, "logm");
    Eigen::EigenSolver<MatrixXd> es(m, false);
    for (Eigen::Index i = 0; i < es.eigenvalues().size(); ++i) if (es.eigenvalues()(i).imag() == 0 && es.eigenvalues()(i).real() <= 0) fail("Mat.logm: a real eigenvalue <= 0 has no real logarithm");
    return matv(MatrixXd(m.log())); });
  api.method(Mt, "sqrtm", {}, Mt, "principal matrix square root (real result required)", [](void* s, Args&) {
    const MatrixXd& m = self_m(s); need_square(m, "sqrtm"); need_finite(m, "sqrtm");
    Eigen::EigenSolver<MatrixXd> es(m, false);
    for (Eigen::Index i = 0; i < es.eigenvalues().size(); ++i) if (es.eigenvalues()(i).imag() == 0 && es.eigenvalues()(i).real() < 0) fail("Mat.sqrtm: a negative real eigenvalue has no real square root");
    return matv(MatrixXd(m.sqrt())); });
}

}  // namespace lmnum
