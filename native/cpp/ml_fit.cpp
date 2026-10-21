// Machine learning estimators (scikit-learn 1.7.2 subset, pycli ml_fit / ml_predict).
// Algorithms follow scikit-learn's BSD-3-Clause sources (sklearn/linear_model/_cd_fast.pyx enet_coordinate_descent,
// _ridge.py _solve_svd, discriminant_analysis.py _solve_svd, naive_bayes.py GaussianNB, decomposition/_pca.py _fit_full
// with utils.extmath.svd_flip, cluster/_kmeans.py _kmeans_plusplus and _kmeans_single_lloyd, svm liblinear's
// L2R_L2LOSS_SVC objective); credit to the scikit-learn developers. Logistic regression and LinearSVC minimise the same
// convex objectives as sklearn's lbfgs / liblinear, by Newton's method to a tighter tolerance (same optimum, different
// iterates and n_iter).
#include "ml.hpp"
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>

namespace lmpy {
namespace ml {

// ------------------------------------------------------------------------------------------------ RandomState
RandomState::RandomState(uint32_t seed) {
  mt[0] = seed;
  for (int i = 1; i < 624; ++i) mt[i] = 1812433253U * (mt[i - 1] ^ (mt[i - 1] >> 30)) + uint32_t(i);
  pos = 624;
}
uint32_t RandomState::next32() {
  static const uint32_t mag01[2] = {0x0U, 0x9908b0dfU};
  if (pos >= 624) {
    int kk = 0;
    for (; kk < 624 - 397; ++kk) {
      uint32_t y = (mt[kk] & 0x80000000U) | (mt[kk + 1] & 0x7fffffffU);
      mt[kk] = mt[kk + 397] ^ (y >> 1) ^ mag01[y & 1U];
    }
    for (; kk < 623; ++kk) {
      uint32_t y = (mt[kk] & 0x80000000U) | (mt[kk + 1] & 0x7fffffffU);
      mt[kk] = mt[kk + (397 - 624)] ^ (y >> 1) ^ mag01[y & 1U];
    }
    uint32_t y = (mt[623] & 0x80000000U) | (mt[0] & 0x7fffffffU);
    mt[623] = mt[396] ^ (y >> 1) ^ mag01[y & 1U];
    pos = 0;
  }
  uint32_t y = mt[pos++];
  y ^= (y >> 11);
  y ^= (y << 7) & 0x9d2c5680U;
  y ^= (y << 15) & 0xefc60000U;
  y ^= (y >> 18);
  return y;
}
uint64_t RandomState::interval(uint64_t max) {
  if (max == 0) return 0;
  uint64_t mask = max, value;
  mask |= mask >> 1; mask |= mask >> 2; mask |= mask >> 4; mask |= mask >> 8; mask |= mask >> 16; mask |= mask >> 32;
  if (max <= 0xffffffffULL) { while ((value = (next32() & mask)) > max) {} }
  else { while ((value = (next64() & mask)) > max) {} }
  return value;
}

// ------------------------------------------------------------------------------------------------ Estimator state
const VectorXd& Estimator::v(const std::string& k) const {
  auto it = vecs.find(k); if (it == vecs.end()) fail("Estimator(" + kind + "): internal: no vector '" + k + "'"); return it->second;
}
const MatrixXd& Estimator::m(const std::string& k) const {
  auto it = mats.find(k); if (it == mats.end()) fail("Estimator(" + kind + "): internal: no matrix '" + k + "'"); return it->second;
}
double Estimator::n(const std::string& k) const {
  auto it = nums.find(k); if (it == nums.end()) fail("Estimator(" + kind + "): internal: no number '" + k + "'"); return it->second;
}

namespace {
const std::vector<std::pair<std::string, std::string>>& families() {
  static const std::vector<std::pair<std::string, std::string>> f = {
      {"linear", "regression"}, {"ridge", "regression"}, {"lasso", "regression"}, {"elastic_net", "regression"},
      {"logistic", "classification"}, {"linear_svc", "classification"}, {"gaussian_nb", "classification"}, {"lda", "classification"},
      {"knn_classifier", "classification"}, {"knn_regressor", "regression"}, {"kmeans", "clustering"}, {"pca", "decomposition"},
      {"standard_scaler", "scaling"}};
  return f;
}
}  // namespace

std::string task_of(const std::string& kind) {
  for (auto& f : families()) if (f.first == kind) return f.second;
  if (kind.rfind("random_forest_", 0) == 0 || kind.rfind("hist_gradient_boosting_", 0) == 0)
    fail("ml: kind '" + kind + "' (tree ensembles) is not supported by lmpy: its result is defined only by scikit-learn's random streams and tree builder");
  std::string known;
  for (auto& f : families()) known += (known.empty() ? "" : ", ") + f.first;
  fail("ml: unknown kind '" + kind + "' (known: " + known + ")");
}

Hyper read_hyper(const std::string& kind, const Opts& o, const std::string& op) {
  Hyper h;
  std::string task = task_of(kind);
  auto positive = [&](const char* key, double v) { if (!(v > 0) || !std::isfinite(v)) fail(op + "(" + kind + "): " + key + " must be a finite number > 0; got " + jnum(v)); return v; };
  auto at_least = [&](const char* key, int64_t v, int64_t lo) { if (v < lo) fail(op + "(" + kind + "): " + key + " must be >= " + std::to_string(lo) + "; got " + std::to_string(v)); return v; };
  if (task == "regression" || task == "classification") h.scale = o.flag("scale", false);
  if (kind == "linear" || kind == "ridge" || kind == "lasso" || kind == "elastic_net" || kind == "logistic" || kind == "linear_svc")
    h.fit_intercept = o.flag("fit_intercept", true);
  if (kind == "ridge" || kind == "lasso" || kind == "elastic_net") h.alpha = positive("alpha", o.num("alpha", 1.0));
  if (kind == "lasso" || kind == "elastic_net") {
    h.max_iter = at_least("max_iter", o.integer("max_iter", 1000), 1);
    h.tol = positive("tol", o.num("tol", 1e-8));
  }
  if (kind == "elastic_net") {
    h.l1_ratio = o.num("l1_ratio", 0.5);
    if (!(h.l1_ratio >= 0 && h.l1_ratio <= 1)) fail(op + "(elastic_net): l1_ratio must be in [0, 1]; got " + jnum(h.l1_ratio));
  }
  if (kind == "logistic" || kind == "linear_svc") {
    h.C = positive("C", o.num("C", 1.0));
    h.max_iter = at_least("max_iter", o.integer("max_iter", 1000), 1);
    h.tol = positive("tol", o.num("tol", 1e-8));
    h.class_weight = o.str("class_weight", "none");
    if (h.class_weight != "none" && h.class_weight != "balanced") fail(op + "(" + kind + "): class_weight must be none or balanced; got '" + h.class_weight + "'");
  }
  if (kind == "gaussian_nb") h.var_smoothing = positive("var_smoothing", o.num("var_smoothing", 1e-9));
  if (kind == "lda") h.n_components = at_least("n_components", o.integer("n_components", 0), 0);
  if (kind == "knn_classifier" || kind == "knn_regressor") {
    h.n_neighbors = at_least("n_neighbors", o.integer("n_neighbors", 5), 1);
    h.weights = o.str("weights", "uniform");
    if (h.weights != "uniform" && h.weights != "distance") fail(op + "(" + kind + "): weights must be uniform or distance; got '" + h.weights + "'");
  }
  if (kind == "kmeans") {
    h.n_clusters = at_least("n_clusters", o.integer("n_clusters", 2), 1);
    h.n_init = at_least("n_init", o.integer("n_init", 10), 1);
    h.max_iter = at_least("max_iter", o.integer("max_iter", 300), 1);
    h.tol = o.num("tol", 1e-8);
    if (!(h.tol >= 0) || !std::isfinite(h.tol)) fail(op + "(kmeans): tol must be a finite number >= 0");
    h.seed = o.integer("seed", 0);
    if (h.seed < 0 || h.seed > int64_t(0xffffffffLL)) fail(op + "(kmeans): seed must be in [0, 2^32 - 1] (numpy RandomState); got " + std::to_string(h.seed));
    h.init = o.str("init", "k-means++");
    if (h.init != "k-means++" && h.init != "random") fail(op + "(kmeans): init must be k-means++ or random; got '" + h.init + "'");
  }
  if (kind == "pca") {
    h.n_components = at_least("n_components", o.integer("n_components", 2), 1);
    h.whiten = o.flag("whiten", false);
  }
  return h;
}

namespace {

std::string sh(const MatrixXd& m) { return std::to_string(m.rows()) + "x" + std::to_string(m.cols()); }

// sklearn StandardScaler: mean, population variance, scale = sqrt(var) with near-constant columns scaled by 1
// (sklearn.preprocessing._data._is_constant_feature).
void scaler_stats(const MatrixXd& x, VectorXd& mean, VectorXd& var, VectorXd& scale) {
  const double n = double(x.rows()), eps = std::numeric_limits<double>::epsilon();
  mean = x.colwise().mean().transpose();
  var.resize(x.cols()); scale.resize(x.cols());
  for (Eigen::Index j = 0; j < x.cols(); ++j) {
    double s = 0; for (Eigen::Index i = 0; i < x.rows(); ++i) { double d = x(i, j) - mean(j); s += d * d; }
    var(j) = s / n;
    double upper = n * eps * var(j) + (n * mean(j) * eps) * (n * mean(j) * eps);
    scale(j) = var(j) <= upper ? 1.0 : std::sqrt(var(j));
  }
}

MatrixXd apply_scale(const MatrixXd& x, const VectorXd& mean, const VectorXd& scale) {
  MatrixXd r = x;
  for (Eigen::Index j = 0; j < x.cols(); ++j) r.col(j) = (r.col(j).array() - mean(j)) / scale(j);
  return r;
}

// Classes are the sorted distinct target values; codes index them.
std::vector<int> encode(const VectorXd& y, const std::vector<double>& classes) {
  std::vector<int> c(y.size());
  for (Eigen::Index i = 0; i < y.size(); ++i) c[i] = int(std::lower_bound(classes.begin(), classes.end(), y(i)) - classes.begin());
  return c;
}

VectorXd class_weights(const Hyper& h, const std::vector<int>& code, int k) {  // sklearn compute_class_weight
  VectorXd w = VectorXd::Ones(k);
  if (h.class_weight == "balanced") {
    VectorXd cnt = VectorXd::Zero(k); for (int c : code) cnt(c) += 1;
    for (int j = 0; j < k; ++j) w(j) = double(code.size()) / (double(k) * cnt(j));
  }
  return w;
}

// ---- linear / ridge (centred SVD solves, as sklearn's LinearRegression via scipy lstsq and Ridge(solver="svd"))
void fit_linear(Estimator& e, const MatrixXd& x, const VectorXd& y) {
  VectorXd xm = VectorXd::Zero(x.cols()); double ym = 0;
  if (e.hp.fit_intercept) { xm = x.colwise().mean().transpose(); ym = y.mean(); }
  MatrixXd xc = x.rowwise() - xm.transpose(); VectorXd yc = y.array() - ym;
  Eigen::BDCSVD<MatrixXd> svd(xc, Eigen::ComputeThinU | Eigen::ComputeThinV);
  VectorXd s = svd.singularValues(); VectorXd uty = svd.matrixU().transpose() * yc;
  VectorXd d = VectorXd::Zero(s.size());
  int64_t rank = 0;
  if (e.kind == "linear") {
    double cut = std::numeric_limits<double>::epsilon() * (s.size() ? s(0) : 0.0);  // scipy.linalg.lstsq cond default eps
    for (Eigen::Index i = 0; i < s.size(); ++i) if (s(i) > cut) { d(i) = 1.0 / s(i); ++rank; }
    e.rank = rank;
    if (rank < x.cols()) e.status = "RANK_DEFICIENT";
  } else {
    for (Eigen::Index i = 0; i < s.size(); ++i) if (s(i) > 1e-15) d(i) = s(i) / (s(i) * s(i) + e.hp.alpha);
  }
  VectorXd coef = svd.matrixV() * d.cwiseProduct(uty);
  e.vecs["coef"] = coef;
  e.nums["intercept"] = e.hp.fit_intercept ? ym - xm.dot(coef) : 0.0;
}

// ---- lasso / elastic net: sklearn enet_coordinate_descent (cyclic), on centred data, alpha scaled by n
void fit_enet(Estimator& e, const MatrixXd& x, const VectorXd& y) {
  const Eigen::Index n = x.rows(), p = x.cols();
  VectorXd xm = VectorXd::Zero(p); double ym = 0;
  if (e.hp.fit_intercept) { xm = x.colwise().mean().transpose(); ym = y.mean(); }
  MatrixXd X = x.rowwise() - xm.transpose(); VectorXd Y = y.array() - ym;
  double l1 = e.kind == "lasso" ? 1.0 : e.hp.l1_ratio;
  double alpha = e.hp.alpha * l1 * double(n), beta = e.hp.alpha * (1.0 - l1) * double(n);
  VectorXd w = VectorXd::Zero(p), norm_cols = X.colwise().squaredNorm().transpose();
  VectorXd R = Y;  // y - X w with w = 0
  double tol = e.hp.tol * Y.dot(Y), d_w_tol = e.hp.tol, gap = tol + 1.0;
  int64_t it = 0; bool converged = false;
  for (it = 0; it < e.hp.max_iter; ++it) {
    double w_max = 0, d_w_max = 0;
    for (Eigen::Index j = 0; j < p; ++j) {
      if (norm_cols(j) == 0.0) continue;
      double w_old = w(j);
      if (w_old != 0.0) R += w_old * X.col(j);
      double tmp = X.col(j).dot(R);
      w(j) = (tmp > 0 ? 1.0 : tmp < 0 ? -1.0 : 0.0) * std::max(std::fabs(tmp) - alpha, 0.0) / (norm_cols(j) + beta);
      if (w(j) != 0.0) R -= w(j) * X.col(j);
      d_w_max = std::max(d_w_max, std::fabs(w(j) - w_old));
      w_max = std::max(w_max, std::fabs(w(j)));
    }
    if (w_max == 0.0 || d_w_max / w_max < d_w_tol || it == e.hp.max_iter - 1) {
      VectorXd XtA = X.transpose() * R - beta * w;
      double dual_norm = XtA.cwiseAbs().maxCoeff(), R2 = R.dot(R), w2 = w.dot(w), cst;
      if (dual_norm > alpha) { cst = alpha / dual_norm; gap = 0.5 * (R2 + R2 * cst * cst); }
      else { cst = 1.0; gap = R2; }
      gap += alpha * w.cwiseAbs().sum() - cst * R.dot(Y) + 0.5 * beta * (1 + cst * cst) * w2;
      if (gap < tol) { converged = true; break; }
    }
  }
  e.n_iter = converged ? it + 1 : e.hp.max_iter;
  if (!converged) e.status = "NO_CONVERGENCE";
  e.vecs["coef"] = w;
  e.nums["intercept"] = e.hp.fit_intercept ? ym - xm.dot(w) : 0.0;
  e.nums["dual_gap"] = gap / double(n);
}

// ---- logistic regression: minimise sum_i sw_i * loss_i + ||W||^2 / (2C) (intercept unpenalised); binary uses one
// decision row (classes[1] positive), K > 2 classes the multinomial (softmax) loss with K rows, as sklearn's lbfgs.
// Solved by damped Newton (exact Hessian) to a gradient tolerance well below sklearn's.
void fit_logistic(Estimator& e, const MatrixXd& x, const std::vector<int>& code, int K) {
  const Eigen::Index n = x.rows(), p = x.cols();
  const int rows = K == 2 ? 1 : K;
  const Eigen::Index q = p + (e.hp.fit_intercept ? 1 : 0), dim = rows * q;
  VectorXd cw = class_weights(e.hp, code, K), sw(n);
  for (Eigen::Index i = 0; i < n; ++i) sw(i) = cw(code[i]);
  MatrixXd xa(n, q); xa.leftCols(p) = x; if (e.hp.fit_intercept) xa.col(p).setOnes();
  const double lam = 1.0 / e.hp.C;
  auto unpack = [&](const VectorXd& th) { return Eigen::Map<const MatrixXd>(th.data(), q, rows); };  // column r = row r of coef
  auto objective = [&](const VectorXd& th, VectorXd* grad, MatrixXd* hess) {
    MatrixXd W = unpack(th);
    MatrixXd z = xa * W;  // n x rows
    double f = 0;
    MatrixXd G = MatrixXd::Zero(q, rows);
    if (hess) hess->setZero(dim, dim);
    if (rows == 1) {
      VectorXd d(n);
      for (Eigen::Index i = 0; i < n; ++i) {
        double zi = z(i, 0), yi = code[i] == 1 ? 1.0 : 0.0;
        double lse = zi > 0 ? zi + std::log1p(std::exp(-zi)) : std::log1p(std::exp(zi));
        f += sw(i) * (lse - yi * zi);
        double pr = zi >= 0 ? 1.0 / (1.0 + std::exp(-zi)) : std::exp(zi) / (1.0 + std::exp(zi));
        G.col(0) += sw(i) * (pr - yi) * xa.row(i).transpose();
        d(i) = sw(i) * pr * (1 - pr);
      }
      if (hess) *hess = xa.transpose() * d.asDiagonal() * xa;
    } else {
      MatrixXd P(n, rows);
      for (Eigen::Index i = 0; i < n; ++i) {
        double mx = z.row(i).maxCoeff(), s = 0;
        for (int r = 0; r < rows; ++r) s += std::exp(z(i, r) - mx);
        double lse = mx + std::log(s);
        f += sw(i) * (lse - z(i, code[i]));
        for (int r = 0; r < rows; ++r) P(i, r) = std::exp(z(i, r) - lse);
      }
      MatrixXd Rr = P; for (Eigen::Index i = 0; i < n; ++i) Rr(i, code[i]) -= 1.0;
      G = xa.transpose() * (sw.asDiagonal() * Rr);
      if (hess) {
        for (int a = 0; a < rows; ++a)
          for (int b = a; b < rows; ++b) {
            VectorXd d(n);
            for (Eigen::Index i = 0; i < n; ++i) d(i) = sw(i) * P(i, a) * ((a == b ? 1.0 : 0.0) - P(i, b));
            MatrixXd blk = xa.transpose() * d.asDiagonal() * xa;
            hess->block(a * q, b * q, q, q) = blk;
            if (a != b) hess->block(b * q, a * q, q, q) = blk.transpose();
          }
      }
    }
    // penalty on the coefficients (not the intercept)
    double pen = 0;
    for (int r = 0; r < rows; ++r) {
      pen += W.col(r).head(p).squaredNorm();
      G.col(r).head(p) += lam * W.col(r).head(p);
      if (hess) for (Eigen::Index j = 0; j < p; ++j) (*hess)(r * q + j, r * q + j) += lam;
    }
    f += 0.5 * lam * pen;
    if (grad) *grad = Eigen::Map<VectorXd>(G.data(), dim);
    // multinomial with an intercept: the loss is invariant to a common shift of the intercepts. The gradient is orthogonal
    // to that direction, so adding 1 1^T over the intercept entries makes the Newton system definite without changing the
    // step (the iterates keep sklearn's zero-sum intercepts).
    if (hess && rows > 1 && e.hp.fit_intercept)
      for (int a = 0; a < rows; ++a) for (int b = 0; b < rows; ++b) (*hess)(a * q + p, b * q + p) += 1.0;
    return f;
  };
  VectorXd th = VectorXd::Zero(dim), g; MatrixXd H;
  double f = objective(th, &g, &H);
  const double g0 = std::max(g.cwiseAbs().maxCoeff(), 1e-300);
  bool converged = false; int64_t it = 0;
  for (it = 0; it < e.hp.max_iter; ++it) {
    double gmax = g.cwiseAbs().maxCoeff();
    if (gmax <= 1e-12 * std::max(1.0, g0) || gmax == 0) { converged = true; break; }
    Eigen::LDLT<MatrixXd> ldlt(H);
    VectorXd step = -ldlt.solve(g);
    if (!step.allFinite() || step.dot(g) >= 0) step = -g;
    double t = 1.0, fn = f; VectorXd thn;
    for (int ls = 0; ls < 60; ++ls) {
      thn = th + t * step; fn = objective(thn, nullptr, nullptr);
      if (std::isfinite(fn) && fn <= f + 1e-4 * t * step.dot(g)) break;
      t *= 0.5;
    }
    if (!(fn <= f)) { converged = gmax <= 1e-8 * std::max(1.0, g0); break; }  // no further decrease is representable
    double df = f - fn;
    th = thn; f = objective(th, &g, &H);
    if (df <= 1e-16 * std::max(1.0, std::fabs(f)) && g.cwiseAbs().maxCoeff() <= 1e-8 * std::max(1.0, g0)) { converged = true; ++it; break; }
  }
  if (!converged) e.status = "NO_CONVERGENCE";
  e.n_iter = it;
  MatrixXd W = unpack(th);
  e.mats["coef"] = W.topRows(p).transpose();
  e.vecs["intercept"] = e.hp.fit_intercept ? VectorXd(W.row(p).transpose()) : VectorXd::Zero(rows);
}

// ---- LinearSVC: liblinear L2R_L2LOSS_SVC, primal  0.5 ||w~||^2 + sum_i C_i max(0, 1 - y_i w~.x~_i)^2  with the intercept
// as a penalised synthetic feature (intercept_scaling 1); one-vs-rest for K > 2 (positives weighted C*cw_k, negatives
// C, as liblinear's train()); binary: one row, classes[1] positive. Generalised Newton with backtracking.
void fit_svc(Estimator& e, const MatrixXd& x, const std::vector<int>& code, int K) {
  const Eigen::Index n = x.rows(), p = x.cols(), q = p + (e.hp.fit_intercept ? 1 : 0);
  const int rows = K == 2 ? 1 : K;
  MatrixXd xa(n, q); xa.leftCols(p) = x; if (e.hp.fit_intercept) xa.col(p).setOnes();
  VectorXd cw = class_weights(e.hp, code, K);
  MatrixXd coef(rows, p); VectorXd icpt = VectorXd::Zero(rows);
  int64_t max_it = 0; bool all_conv = true;
  for (int r = 0; r < rows; ++r) {
    int pos_class = rows == 1 ? 1 : r;
    VectorXd yy(n), C(n);
    for (Eigen::Index i = 0; i < n; ++i) {
      bool pos = code[i] == pos_class; yy(i) = pos ? 1.0 : -1.0;
      C(i) = rows == 1 ? e.hp.C * cw(code[i]) : (pos ? e.hp.C * cw(r) : e.hp.C);
    }
    auto obj = [&](const VectorXd& w, VectorXd* g, MatrixXd* h) {
      VectorXd z = xa * w; double f = 0.5 * w.squaredNorm();
      if (g) *g = w;
      if (h) *h = MatrixXd::Identity(q, q);
      for (Eigen::Index i = 0; i < n; ++i) {
        double m = 1 - yy(i) * z(i);
        if (m > 0) {
          f += C(i) * m * m;
          if (g) *g -= 2 * C(i) * yy(i) * m * xa.row(i).transpose();
          if (h) h->noalias() += 2 * C(i) * xa.row(i).transpose() * xa.row(i);
        }
      }
      return f;
    };
    VectorXd w = VectorXd::Zero(q), g; MatrixXd H;
    double f = obj(w, &g, &H), g0 = std::max(g.norm(), 1e-300);
    bool conv = false; int64_t it = 0;
    for (it = 0; it < e.hp.max_iter; ++it) {
      if (g.norm() <= 1e-13 * g0) { conv = true; break; }
      VectorXd step = -H.ldlt().solve(g);
      double t = 1.0, fn = f; VectorXd wn;
      for (int ls = 0; ls < 60; ++ls) {
        wn = w + t * step; fn = obj(wn, nullptr, nullptr);
        if (fn <= f + 1e-4 * t * step.dot(g)) break;
        t *= 0.5;
      }
      if (!(fn <= f)) { conv = g.norm() <= 1e-9 * g0; break; }
      double df = f - fn;
      w = wn; f = obj(w, &g, &H);
      if (df <= 1e-16 * std::max(1.0, f) && g.norm() <= 1e-9 * g0) { conv = true; ++it; break; }
    }
    all_conv = all_conv && conv; max_it = std::max(max_it, it);
    coef.row(r) = w.head(p).transpose();
    if (e.hp.fit_intercept) icpt(r) = w(p);
  }
  if (!all_conv) e.status = "NO_CONVERGENCE";
  e.n_iter = max_it;
  e.mats["coef"] = coef; e.vecs["intercept"] = icpt;
}

// ---- Gaussian naive Bayes (sklearn GaussianNB.fit without priors or sample weights)
void fit_nb(Estimator& e, const MatrixXd& x, const std::vector<int>& code, int K) {
  const Eigen::Index n = x.rows(), p = x.cols();
  VectorXd gm = x.colwise().mean().transpose(); double vmax = 0;
  for (Eigen::Index j = 0; j < p; ++j) vmax = std::max(vmax, (x.col(j).array() - gm(j)).square().mean());
  double eps = e.hp.var_smoothing * vmax;
  MatrixXd theta = MatrixXd::Zero(K, p), var = MatrixXd::Zero(K, p); VectorXd cnt = VectorXd::Zero(K);
  for (Eigen::Index i = 0; i < n; ++i) { theta.row(code[i]) += x.row(i); cnt(code[i]) += 1; }
  for (int k = 0; k < K; ++k) theta.row(k) /= cnt(k);
  for (Eigen::Index i = 0; i < n; ++i) var.row(code[i]) += (x.row(i) - theta.row(code[i])).array().square().matrix();
  for (int k = 0; k < K; ++k) var.row(k) = (var.row(k) / cnt(k)).array() + eps;
  e.mats["theta"] = theta; e.mats["var"] = var; e.vecs["class_count"] = cnt; e.vecs["class_prior"] = cnt / double(n);
  e.nums["epsilon"] = eps;
}

// ---- LDA, solver="svd" (sklearn LinearDiscriminantAnalysis._solve_svd, tol 1e-4). Singular-vector signs are fixed so
// that each scaling column's largest-magnitude entry is positive (sklearn leaves them to LAPACK; coef/intercept/predict
// do not depend on them, transform does up to the sign of each axis).
void fit_lda(Estimator& e, const MatrixXd& x, const std::vector<int>& code, int K) {
  const Eigen::Index n = x.rows(), p = x.cols();
  const double tol = 1e-4;
  MatrixXd means = MatrixXd::Zero(K, p); VectorXd cnt = VectorXd::Zero(K);
  for (Eigen::Index i = 0; i < n; ++i) { means.row(code[i]) += x.row(i); cnt(code[i]) += 1; }
  for (int k = 0; k < K; ++k) means.row(k) /= cnt(k);
  VectorXd priors = cnt / double(n);
  int64_t maxc = std::min<int64_t>(K - 1, p);
  if (e.hp.n_components > 0) {
    if (e.hp.n_components > maxc) fail("ml_fit(lda): n_components " + std::to_string(e.hp.n_components) + " exceeds min(n_features, n_classes - 1) = " + std::to_string(maxc));
    maxc = e.hp.n_components;
  }
  Eigen::RowVectorXd xbar = priors.transpose() * means;
  MatrixXd xc(n, p);
  for (Eigen::Index i = 0; i < n; ++i) xc.row(i) = x.row(i) - means.row(code[i]);
  if (xc.cwiseAbs().maxCoeff() == 0) fail("ml_fit(lda): DEGENERATE_DATA: every row equals its class mean (zero within-class scatter)");
  VectorXd sd(p);
  for (Eigen::Index j = 0; j < p; ++j) { double m = xc.col(j).mean(); sd(j) = std::sqrt((xc.col(j).array() - m).square().mean()); if (sd(j) == 0) sd(j) = 1.0; }
  MatrixXd xs = std::sqrt(1.0 / double(n - K)) * (xc.array().rowwise() / sd.transpose().array()).matrix();
  Eigen::BDCSVD<MatrixXd> s1(xs, Eigen::ComputeThinV);
  VectorXd S = s1.singularValues(); Eigen::Index rank = 0;
  for (Eigen::Index i = 0; i < S.size(); ++i) rank += S(i) > tol;
  if (rank == 0) fail("ml_fit(lda): DEGENERATE_DATA: within-class scatter has rank 0");
  MatrixXd scal = s1.matrixV().leftCols(rank);  // p x rank (Vt[:rank].T)
  for (Eigen::Index j = 0; j < rank; ++j) scal.col(j) = scal.col(j).array() / sd.array() / S(j);
  double fac = K == 1 ? 1.0 : 1.0 / double(K - 1);
  MatrixXd cen = means.rowwise() - xbar;  // K x p
  MatrixXd xb(K, p);
  for (int k = 0; k < K; ++k) xb.row(k) = std::sqrt(double(n) * priors(k) * fac) * cen.row(k);
  MatrixXd xbs = xb * scal;  // K x rank
  Eigen::BDCSVD<MatrixXd> s2(xbs, Eigen::ComputeThinV);
  VectorXd S2 = s2.singularValues();
  VectorXd evr = S2.array().square() / S2.array().square().sum();
  Eigen::Index rank2 = 0;
  for (Eigen::Index i = 0; i < S2.size(); ++i) rank2 += S2(i) > tol * S2(0);
  MatrixXd scalings = scal * s2.matrixV().leftCols(rank2);
  for (Eigen::Index j = 0; j < scalings.cols(); ++j) { Eigen::Index k; scalings.col(j).cwiseAbs().maxCoeff(&k); if (scalings(k, j) < 0) scalings.col(j) *= -1; }
  MatrixXd coef = cen * scalings;  // K x rank2
  VectorXd icpt(K);
  for (int k = 0; k < K; ++k) icpt(k) = -0.5 * coef.row(k).squaredNorm() + std::log(priors(k));
  MatrixXd coef_full = coef * scalings.transpose();  // K x p
  icpt -= coef_full * xbar.transpose();
  if (K == 2) {
    MatrixXd c1 = coef_full.row(1) - coef_full.row(0);
    VectorXd i1(1); i1(0) = icpt(1) - icpt(0);
    coef_full = c1; icpt = i1;
  }
  int64_t r = std::min<int64_t>(maxc, scalings.cols());
  e.mats["coef"] = coef_full; e.vecs["intercept"] = icpt; e.mats["means"] = means; e.vecs["priors"] = priors;
  e.mats["scalings"] = scalings.leftCols(r); e.vecs["xbar"] = xbar.transpose();
  e.vecs["explained_variance_ratio"] = evr.head(std::min<Eigen::Index>(evr.size(), maxc));
  e.nums["components"] = double(r);
}

// ---- k-means (sklearn KMeans(algorithm="lloyd")): data centred by its mean, k-means++ (greedy, 2 + floor(ln k) local
// trials) or random init from numpy RandomState(seed), Lloyd iterations with sklearn's stopping rule (labels unchanged,
// or squared centre shift <= tol * mean feature variance), empty clusters relocated to the farthest points, best of
// n_init by inertia.
struct KmRun { std::vector<int> labels; MatrixXd centers; double inertia; int64_t n_iter; };

// squared Euclidean distances from row c of `cand` to every row of X via ||c||^2 - 2 c.x + ||x||^2, clipped at 0
VectorXd sqdist_rows(const Eigen::RowVectorXd& c, const MatrixXd& X, const VectorXd& xsq) {
  double cc = c.squaredNorm();
  VectorXd d = (-2.0 * (X * c.transpose())).array() + cc;
  d += xsq;
  return d.cwiseMax(0.0);
}

MatrixXd kmeanspp(const MatrixXd& X, int k, const VectorXd& xsq, RandomState& rng) {
  const Eigen::Index n = X.rows();
  MatrixXd centers(k, X.cols());
  int trials = 2 + int(std::log(double(k)));
  // first centre: random_state.choice(n, p=uniform weights): cdf = cumsum(1/n), cdf /= cdf[-1], searchsorted(u, "right")
  std::vector<double> cdf(n); double acc = 0; for (Eigen::Index i = 0; i < n; ++i) { acc += 1.0 / double(n); cdf[i] = acc; }
  for (auto& c : cdf) c /= acc;
  double u = rng.random_sample();
  Eigen::Index first = std::upper_bound(cdf.begin(), cdf.end(), u) - cdf.begin();
  if (first >= n) first = n - 1;
  centers.row(0) = X.row(first);
  VectorXd closest = sqdist_rows(X.row(first), X, xsq);
  double pot = closest.sum();
  for (int c = 1; c < k; ++c) {
    std::vector<double> cum(n); double s = 0;
    for (Eigen::Index i = 0; i < n; ++i) { s += closest(i); cum[i] = s; }
    std::vector<Eigen::Index> cand(trials);
    for (int t = 0; t < trials; ++t) {
      double v = rng.random_sample() * pot;
      Eigen::Index id = std::lower_bound(cum.begin(), cum.end(), v) - cum.begin();
      cand[t] = std::min<Eigen::Index>(id, n - 1);
    }
    int best = 0; double best_pot = 0; VectorXd best_d;
    for (int t = 0; t < trials; ++t) {
      VectorXd d = sqdist_rows(X.row(cand[t]), X, xsq).cwiseMin(closest);
      double pt = d.sum();
      if (t == 0 || pt < best_pot) { best = t; best_pot = pt; best_d = d; }
    }
    pot = best_pot; closest = best_d;
    centers.row(c) = X.row(cand[best]);
  }
  return centers;
}

MatrixXd kmeans_random_init(const MatrixXd& X, int k, RandomState& rng) {
  // random_state.choice(n, size=k, replace=False, p=uniform) (numpy legacy algorithm)
  const Eigen::Index n = X.rows();
  std::vector<double> p(n, 1.0 / double(n));
  std::vector<Eigen::Index> found;
  while (int(found.size()) < k) {
    int need = k - int(found.size());
    std::vector<double> u(need); for (auto& x : u) x = rng.random_sample();
    for (auto f : found) p[f] = 0;
    std::vector<double> cdf(n); double s = 0; for (Eigen::Index i = 0; i < n; ++i) { s += p[i]; cdf[i] = s; }
    for (auto& c : cdf) c /= s;
    std::vector<Eigen::Index> nw; for (double x : u) nw.push_back(std::min<Eigen::Index>(std::upper_bound(cdf.begin(), cdf.end(), x) - cdf.begin(), n - 1));
    // np.unique(new, return_index=True) then the first occurrences in original order
    std::vector<std::pair<Eigen::Index, size_t>> firsts;
    for (size_t i = 0; i < nw.size(); ++i) { bool seen = false; for (size_t j = 0; j < i; ++j) seen |= nw[j] == nw[i]; if (!seen) firsts.push_back({nw[i], i}); }
    std::sort(firsts.begin(), firsts.end(), [](auto& a, auto& b) { return a.second < b.second; });
    for (auto& f : firsts) found.push_back(f.first);
  }
  MatrixXd c(k, X.cols()); for (int j = 0; j < k; ++j) c.row(j) = X.row(found[j]);
  return c;
}

void assign(const MatrixXd& X, const MatrixXd& C, std::vector<int>& labels) {
  VectorXd cn = C.rowwise().squaredNorm();
  MatrixXd d = -2.0 * X * C.transpose();
  d.rowwise() += cn.transpose();
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    int best = 0; double m = d(i, 0);
    for (Eigen::Index j = 1; j < C.rows(); ++j) if (d(i, j) < m) { m = d(i, j); best = int(j); }
    labels[i] = best;
  }
}

double inertia_of(const MatrixXd& X, const MatrixXd& C, const std::vector<int>& labels) {
  double s = 0;
  for (Eigen::Index i = 0; i < X.rows(); ++i) s += (X.row(i) - C.row(labels[i])).squaredNorm();
  return s;
}

KmRun lloyd(const MatrixXd& X, MatrixXd centers, int64_t max_iter, double tol) {
  const Eigen::Index n = X.rows(), k = centers.rows();
  std::vector<int> labels(n, -1), old(n, -1);
  bool strict = false; int64_t i = 0;
  for (i = 0; i < max_iter; ++i) {
    assign(X, centers, labels);
    MatrixXd nc = MatrixXd::Zero(k, X.cols()); VectorXd w = VectorXd::Zero(k);
    for (Eigen::Index r = 0; r < n; ++r) { nc.row(labels[r]) += X.row(r); w(labels[r]) += 1; }
    // relocate empty clusters to the points farthest from their current centre (sklearn _relocate_empty_clusters_dense)
    std::vector<int> empty; for (Eigen::Index j = 0; j < k; ++j) if (w(j) == 0) empty.push_back(int(j));
    if (!empty.empty()) {
      std::vector<std::pair<double, Eigen::Index>> far;
      for (Eigen::Index r = 0; r < n; ++r) far.push_back({(X.row(r) - centers.row(labels[r])).squaredNorm(), r});
      std::stable_sort(far.begin(), far.end(), [](auto& a, auto& b) { return a.first > b.first; });
      for (size_t t = 0; t < empty.size() && t < far.size(); ++t) {
        Eigen::Index r = far[t].second; int oldc = labels[r];
        nc.row(oldc) -= X.row(r); nc.row(empty[t]) = X.row(r);
        w(empty[t]) = 1; w(oldc) -= 1;
      }
    }
    for (Eigen::Index j = 0; j < k; ++j) if (w(j) > 0) nc.row(j) /= w(j);
    double shift = (nc - centers).rowwise().squaredNorm().sum();
    centers = nc;
    if (labels == old) { strict = true; break; }
    if (shift <= tol) break;
    old = labels;
  }
  if (!strict) assign(X, centers, labels);
  KmRun r; r.labels = labels; r.centers = centers; r.inertia = inertia_of(X, centers, labels);
  r.n_iter = std::min<int64_t>(i + 1, max_iter);
  return r;
}

bool same_clustering(const std::vector<int>& a, const std::vector<int>& b, int k) {
  std::vector<int> map(k, -1);
  for (size_t i = 0; i < a.size(); ++i) {
    if (map[b[i]] == -1) map[b[i]] = a[i];
    else if (map[b[i]] != a[i]) return false;
  }
  return true;
}

void fit_kmeans(Estimator& e, const MatrixXd& x) {
  const Eigen::Index n = x.rows();
  const int k = int(e.hp.n_clusters);
  if (k > n) fail("ml_fit(kmeans): n_clusters " + std::to_string(k) + " exceeds the number of rows " + std::to_string(n));
  {
    std::set<std::vector<double>> distinct;
    for (Eigen::Index i = 0; i < n && int(distinct.size()) < k; ++i) {
      std::vector<double> row(x.cols());
      for (Eigen::Index j = 0; j < x.cols(); ++j) row[j] = x(i, j);
      distinct.insert(row);
    }
    if (int(distinct.size()) < k) fail("ml_fit(kmeans): DEGENERATE_CLUSTERS: fewer distinct rows than n_clusters " + std::to_string(k));
  }
  Eigen::RowVectorXd mean = x.colwise().mean();
  MatrixXd X = x.rowwise() - mean;
  double var_mean = 0;
  for (Eigen::Index j = 0; j < X.cols(); ++j) { double m = X.col(j).mean(); var_mean += (X.col(j).array() - m).square().mean(); }
  var_mean /= double(X.cols());
  double tol = e.hp.tol * var_mean;
  VectorXd xsq = X.rowwise().squaredNorm();
  RandomState rng{uint32_t(e.hp.seed)};
  KmRun best; bool have = false;
  for (int64_t run = 0; run < e.hp.n_init; ++run) {
    MatrixXd init = e.hp.init == "random" ? kmeans_random_init(X, k, rng) : kmeanspp(X, k, xsq, rng);
    KmRun r = lloyd(X, init, e.hp.max_iter, tol);
    if (!have || (r.inertia < best.inertia && !same_clustering(r.labels, best.labels, k))) { best = r; have = true; }
  }
  best.centers.rowwise() += mean;
  std::set<int> used(best.labels.begin(), best.labels.end());
  if (int(used.size()) < k) fail("ml_fit(kmeans): DEGENERATE_CLUSTERS: only " + std::to_string(used.size()) + " distinct clusters found for n_clusters " + std::to_string(k));
  VectorXd lab(n); for (Eigen::Index i = 0; i < n; ++i) lab(i) = best.labels[i];
  e.mats["cluster_centers"] = best.centers; e.vecs["labels"] = lab; e.nums["inertia"] = best.inertia;
  e.n_iter = best.n_iter;
  if (best.n_iter >= e.hp.max_iter) e.status = "NO_CONVERGENCE";  // pycli: reaching the cap is conservatively NO_CONVERGENCE
}

// ---- PCA (svd_solver="full"; svd_flip with u_based_decision=False: each component's largest |loading| is positive)
void fit_pca(Estimator& e, const MatrixXd& x) {
  const Eigen::Index n = x.rows(), p = x.cols(), mn = std::min(n, p);
  if (e.hp.n_components > mn) fail("ml_fit(pca): n_components " + std::to_string(e.hp.n_components) + " exceeds min(n_samples, n_features) = " + std::to_string(mn));
  if (n < 2) fail("ml_fit(pca): needs at least 2 rows");
  VectorXd mean = x.colwise().mean().transpose();
  MatrixXd xc = x.rowwise() - mean.transpose();
  Eigen::BDCSVD<MatrixXd> svd(xc, Eigen::ComputeThinV);
  VectorXd S = svd.singularValues(); MatrixXd Vt = svd.matrixV().transpose();
  for (Eigen::Index r = 0; r < Vt.rows(); ++r) { Eigen::Index j; Vt.row(r).cwiseAbs().maxCoeff(&j); if (Vt(r, j) < 0) Vt.row(r) *= -1; }
  VectorXd ev = S.array().square() / double(n - 1);
  double total = ev.sum();
  int64_t k = e.hp.n_components;
  e.mats["components"] = Vt.topRows(k); e.vecs["mean"] = mean;
  e.vecs["explained_variance"] = ev.head(k);
  e.vecs["explained_variance_ratio"] = total > 0 ? VectorXd(ev.head(k) / total) : VectorXd(VectorXd::Constant(k, NAN));
  e.vecs["singular_values"] = S.head(k);
  e.nums["noise_variance"] = k < mn ? ev.segment(k, mn - k).mean() : 0.0;
}

// ---- k nearest neighbours (brute force Euclidean). Neighbours are ordered by distance, ties by training row order.
std::vector<std::vector<std::pair<double, Eigen::Index>>> neighbours(const Estimator& e, const MatrixXd& q) {
  const MatrixXd& tx = e.m("train_x");
  const Eigen::Index k = e.hp.n_neighbors;
  std::vector<std::vector<std::pair<double, Eigen::Index>>> out(q.rows());
  std::vector<std::pair<double, Eigen::Index>> d(tx.rows());
  for (Eigen::Index i = 0; i < q.rows(); ++i) {
    for (Eigen::Index r = 0; r < tx.rows(); ++r) d[r] = {(tx.row(r) - q.row(i)).norm(), r};
    std::partial_sort(d.begin(), d.begin() + k, d.end());
    out[i].assign(d.begin(), d.begin() + k);
  }
  return out;
}

VectorXd knn_weights(const Estimator& e, const std::vector<std::pair<double, Eigen::Index>>& nb) {
  VectorXd w = VectorXd::Ones(Eigen::Index(nb.size()));
  if (e.hp.weights == "distance") {
    bool zero = false; for (auto& p : nb) zero |= p.first == 0.0;
    for (size_t j = 0; j < nb.size(); ++j) w(j) = zero ? (nb[j].first == 0.0 ? 1.0 : 0.0) : 1.0 / nb[j].first;
  }
  return w;
}

}  // namespace

// ------------------------------------------------------------------------------------------------ fit / predict
Estimator* fit(const std::string& kind, const MatrixXd& x0, const VectorXd* y, const Hyper& hp) {
  auto* e = make<Estimator>();
  e->kind = kind; e->task = task_of(kind); e->hp = hp;
  const std::string op = "ml_fit(" + kind + ")";
  if (x0.rows() == 0 || x0.cols() == 0) fail(op + ": x is empty (" + sh(x0) + ")");
  if (!x0.allFinite()) fail(op + ": x contains NaN or infinity (no missing values are allowed)");
  bool supervised = e->task == "regression" || e->task == "classification";
  if (supervised) {
    if (!y) fail(op + ": a supervised kind needs a target y: ml_fit(kind, x, y)");
    if (y->size() != x0.rows()) fail(op + ": y has " + std::to_string(y->size()) + " values; x has " + std::to_string(x0.rows()) + " rows");
    if (!y->allFinite()) fail(op + ": y contains NaN or infinity");
  } else if (y) fail(op + ": '" + kind + "' is unsupervised: call ml_fit(kind, x) or ml_fit(kind, x, opts) without y");
  e->n_features = x0.cols(); e->n_train = x0.rows();
  MatrixXd x = x0;
  if (hp.scale) {
    VectorXd mean, var, scale; scaler_stats(x0, mean, var, scale);
    e->vecs["pp_mean"] = mean; e->vecs["pp_scale"] = scale;
    x = apply_scale(x0, mean, scale);
  }
  if (e->task == "classification") {
    std::vector<double> classes = sorted_unique(*y);
    if (classes.size() < 2) fail(op + ": classification needs at least 2 classes in y; found " + std::to_string(classes.size()));
    std::vector<int> code = encode(*y, classes);
    e->vecs["classes"] = to_eigen(classes);
    int K = int(classes.size());
    if (kind == "logistic") fit_logistic(*e, x, code, K);
    else if (kind == "linear_svc") fit_svc(*e, x, code, K);
    else if (kind == "gaussian_nb") fit_nb(*e, x, code, K);
    else if (kind == "lda") {
      if (x.rows() <= K) fail(op + ": needs more rows than classes");
      fit_lda(*e, x, code, K);
    } else if (kind == "knn_classifier") {
      if (hp.n_neighbors > x.rows()) fail(op + ": n_neighbors " + std::to_string(hp.n_neighbors) + " exceeds the number of training rows " + std::to_string(x.rows()));
      VectorXd yc(x.rows()); for (Eigen::Index i = 0; i < x.rows(); ++i) yc(i) = code[i];
      e->mats["train_x"] = x; e->vecs["train_y"] = yc;
    }
  } else if (e->task == "regression") {
    if (kind == "linear" || kind == "ridge") fit_linear(*e, x, *y);
    else if (kind == "lasso" || kind == "elastic_net") fit_enet(*e, x, *y);
    else if (kind == "knn_regressor") {
      if (hp.n_neighbors > x.rows()) fail(op + ": n_neighbors " + std::to_string(hp.n_neighbors) + " exceeds the number of training rows " + std::to_string(x.rows()));
      e->mats["train_x"] = x; e->vecs["train_y"] = *y;
    }
  } else if (kind == "kmeans") fit_kmeans(*e, x);
  else if (kind == "pca") fit_pca(*e, x);
  else if (kind == "standard_scaler") {
    VectorXd mean, var, scale; scaler_stats(x, mean, var, scale);
    e->vecs["mean"] = mean; e->vecs["var"] = var; e->vecs["scale"] = scale;
  }
  return e;
}

MatrixXd prepare(const Estimator& e, const MatrixXd& x, const char* op) {
  if (x.cols() != e.n_features) fail(std::string("Estimator.") + op + "(" + e.kind + "): x has " + std::to_string(x.cols()) + " columns; the model was fitted on " + std::to_string(e.n_features));
  if (!x.allFinite()) fail(std::string("Estimator.") + op + "(" + e.kind + "): x contains NaN or infinity");
  if (e.hp.scale) return apply_scale(x, e.v("pp_mean"), e.v("pp_scale"));
  return x;
}

namespace {
MatrixXd linear_decision(const Estimator& e, const MatrixXd& x) {
  MatrixXd d = x * e.m("coef").transpose();
  d.rowwise() += e.v("intercept").transpose();
  return d;
}
MatrixXd nb_jll(const Estimator& e, const MatrixXd& x) {
  const MatrixXd &th = e.m("theta"), &var = e.m("var"); const VectorXd& prior = e.v("class_prior");
  MatrixXd j(x.rows(), th.rows());
  for (Eigen::Index k = 0; k < th.rows(); ++k) {
    double n_ij = -0.5 * (2.0 * M_PI * var.row(k).array()).log().sum();
    for (Eigen::Index i = 0; i < x.rows(); ++i)
      j(i, k) = std::log(prior(k)) + n_ij - 0.5 * ((x.row(i) - th.row(k)).array().square() / var.row(k).array()).sum();
  }
  return j;
}
MatrixXd softmax_rows(MatrixXd z) {
  for (Eigen::Index i = 0; i < z.rows(); ++i) { double mx = z.row(i).maxCoeff(); z.row(i) = (z.row(i).array() - mx).exp(); z.row(i) /= z.row(i).sum(); }
  return z;
}
double expit(double t) { return t >= 0 ? 1.0 / (1.0 + std::exp(-t)) : std::exp(t) / (1.0 + std::exp(t)); }
}  // namespace

MatrixXd decision_function(const Estimator& e, const MatrixXd& x0) {
  if (e.kind != "logistic" && e.kind != "linear_svc" && e.kind != "lda") fail("Estimator.decision_function: available for logistic, linear_svc and lda; this model is " + e.kind);
  return linear_decision(e, prepare(e, x0, "decision_function"));
}

MatrixXd predict_proba(const Estimator& e, const MatrixXd& x0) {
  MatrixXd x = prepare(e, x0, "predict_proba");
  const Eigen::Index K = e.classifier() ? e.v("classes").size() : 0;
  if (e.kind == "logistic" || e.kind == "lda") {
    MatrixXd d = linear_decision(e, x);
    if (d.cols() == 1) { MatrixXd p(x.rows(), 2); for (Eigen::Index i = 0; i < x.rows(); ++i) { p(i, 1) = expit(d(i, 0)); p(i, 0) = 1 - p(i, 1); } return p; }
    return softmax_rows(d);
  }
  if (e.kind == "gaussian_nb") return softmax_rows(nb_jll(e, x));
  if (e.kind == "knn_classifier") {
    auto nb = neighbours(e, x); MatrixXd p = MatrixXd::Zero(x.rows(), K); const VectorXd& ty = e.v("train_y");
    for (Eigen::Index i = 0; i < x.rows(); ++i) {
      VectorXd w = knn_weights(e, nb[i]);
      for (size_t j = 0; j < nb[i].size(); ++j) p(i, Eigen::Index(ty(nb[i][j].second))) += w(j);
      p.row(i) /= p.row(i).sum();
    }
    return p;
  }
  fail("Estimator.predict_proba: available for logistic, gaussian_nb, lda and knn_classifier; this model is " + e.kind);
}

VectorXd predict(const Estimator& e, const MatrixXd& x0) {
  if (e.kind == "pca" || e.kind == "standard_scaler") fail("Estimator.predict: use transform for " + e.kind);
  if (e.kind == "kmeans") {
    MatrixXd x = prepare(e, x0, "predict"); std::vector<int> lab(x.rows());
    assign(x, e.m("cluster_centers"), lab);
    VectorXd r(x.rows()); for (Eigen::Index i = 0; i < x.rows(); ++i) r(i) = lab[i]; return r;
  }
  if (e.task == "regression") {
    MatrixXd x = prepare(e, x0, "predict");
    if (e.kind == "knn_regressor") {
      auto nb = neighbours(e, x); VectorXd r(x.rows()); const VectorXd& ty = e.v("train_y");
      for (Eigen::Index i = 0; i < x.rows(); ++i) {
        VectorXd w = knn_weights(e, nb[i]); double s = 0;
        for (size_t j = 0; j < nb[i].size(); ++j) s += w(j) * ty(nb[i][j].second);
        r(i) = s / w.sum();
      }
      return r;
    }
    return (x * e.v("coef")).array() + e.n("intercept");
  }
  const VectorXd& classes = e.v("classes");
  MatrixXd score;
  if (e.kind == "logistic" || e.kind == "linear_svc" || e.kind == "lda") {
    score = decision_function(e, x0);
    if (!score.allFinite()) fail("Estimator.predict(" + e.kind + "): NUMERICAL_FAILURE: nonfinite decision score");
    if (score.cols() == 1) { VectorXd r(score.rows()); for (Eigen::Index i = 0; i < score.rows(); ++i) r(i) = classes(score(i, 0) > 0 ? 1 : 0); return r; }
  } else if (e.kind == "gaussian_nb") score = nb_jll(e, prepare(e, x0, "predict"));
  else score = predict_proba(e, x0);
  if (!score.allFinite()) fail("Estimator.predict(" + e.kind + "): NUMERICAL_FAILURE: nonfinite class score");
  VectorXd r(score.rows());
  for (Eigen::Index i = 0; i < score.rows(); ++i) { Eigen::Index k = 0; double m = score(i, 0); for (Eigen::Index j = 1; j < score.cols(); ++j) if (score(i, j) > m) { m = score(i, j); k = j; } r(i) = classes(k); }
  return r;
}

MatrixXd transform(const Estimator& e, const MatrixXd& x0) {
  MatrixXd x = prepare(e, x0, "transform");
  if (e.kind == "standard_scaler") return apply_scale(x, e.v("mean"), e.v("scale"));
  if (e.kind == "pca") {
    const MatrixXd& c = e.m("components");
    MatrixXd t = x * c.transpose();
    t.rowwise() -= (e.v("mean").transpose() * c.transpose());
    if (e.hp.whiten) t = t.array().rowwise() / e.v("explained_variance").transpose().array().sqrt();
    return t;
  }
  if (e.kind == "kmeans") {  // distances to every centre
    const MatrixXd& c = e.m("cluster_centers"); MatrixXd d(x.rows(), c.rows());
    for (Eigen::Index i = 0; i < x.rows(); ++i) for (Eigen::Index j = 0; j < c.rows(); ++j) d(i, j) = (x.row(i) - c.row(j)).norm();
    return d;
  }
  if (e.kind == "lda") {
    MatrixXd t = x.rowwise() - e.v("xbar").transpose();
    return t * e.m("scalings");
  }
  fail("Estimator.transform: available for standard_scaler, pca, kmeans and lda; this model is " + e.kind);
}

Res* params(const Estimator& e) {
  auto* r = make<Res>();
  r->str("kind", e.kind).str("task", e.task).str("status", e.status);
  r->integer("n_features", e.n_features).integer("n_train", e.n_train);
  if (e.classifier()) r->vec("classes", e.v("classes"));
  auto mat_or_vec = [&](const char* k) { if (e.mats.count(k)) r->mat(k, e.m(k)); else if (e.vecs.count(k)) r->vec(k, e.v(k)); };
  if (e.task == "regression" && e.kind.rfind("knn", 0) != 0) { r->vec("coef", e.v("coef")); r->num("intercept", e.n("intercept")); }
  if (e.kind == "linear") r->integer("rank", e.rank);
  if (e.kind == "lasso" || e.kind == "elastic_net") r->num("dual_gap", e.n("dual_gap"));
  if (e.kind == "logistic" || e.kind == "linear_svc" || e.kind == "lda") { mat_or_vec("coef"); mat_or_vec("intercept"); }
  if (e.kind == "lda") { r->mat("means", e.m("means")).vec("priors", e.v("priors")).mat("scalings", e.m("scalings")).vec("xbar", e.v("xbar")).vec("explained_variance_ratio", e.v("explained_variance_ratio")).integer("components", int64_t(e.n("components"))); }
  if (e.kind == "gaussian_nb") r->mat("theta", e.m("theta")).mat("var", e.m("var")).vec("class_prior", e.v("class_prior")).vec("class_count", e.v("class_count")).num("epsilon", e.n("epsilon"));
  if (e.kind == "kmeans") r->mat("cluster_centers", e.m("cluster_centers")).num("inertia", e.n("inertia")).vec("labels", e.v("labels"));
  if (e.kind == "pca") r->mat("components", e.m("components")).vec("mean", e.v("mean")).vec("explained_variance", e.v("explained_variance")).vec("explained_variance_ratio", e.v("explained_variance_ratio")).vec("singular_values", e.v("singular_values")).num("noise_variance", e.n("noise_variance"));
  if (e.kind == "standard_scaler") r->vec("mean", e.v("mean")).vec("var", e.v("var")).vec("scale", e.v("scale"));
  if (e.kind == "logistic" || e.kind == "linear_svc" || e.kind == "lasso" || e.kind == "elastic_net" || e.kind == "kmeans") r->integer("n_iter", e.n_iter);
  if (e.hp.scale) r->vec("pp_mean", e.v("pp_mean")).vec("pp_scale", e.v("pp_scale"));
  return r;
}

}  // namespace ml
}  // namespace lmpy
