// Nonlinear least squares behind `least_squares`: scipy.optimize.least_squares' trf (bounded and unbounded) and dogbox
// methods with the exact (SVD / lstsq) trust-region solvers and the robust losses are C++ ports of scipy 1.16
// (_lsq/trf.py, _lsq/dogbox.py, _lsq/common.py, _lsq/least_squares.py; BSD-3-Clause, Copyright (c) 2001-2002
// Enthought, Inc. and 2003-2024 SciPy Developers). Method lm is MINPACK lmder (Eigen's unsupported
// NonLinearOptimization translation, MPL-2.0; MINPACK by Garbow, Hillstrom and More, Argonne) with scipy's
// forward-difference Jacobian, as scipy's call_minpack does.
#include "optimize_impl.hpp"
#include <unsupported/Eigen/NonLinearOptimization>
#include <algorithm>
#include <cmath>

namespace lmpy::opt {
namespace {

using Eigen::VectorXi;

// ---- robust losses: rho(z) rows 0..2 for z = (f/f_scale)^2 ------------------------------------------------------
struct Loss {
  std::string name;
  double C;  // f_scale
  bool linear() const { return name == "linear"; }
  void rho_z(double z, double& r0, double& r1, double& r2) const {
    if (name == "huber") {
      if (z <= 1) { r0 = z; r1 = 1; r2 = 0; }
      else { r0 = 2 * std::sqrt(z) - 1; r1 = 1 / std::sqrt(z); r2 = -0.5 * std::pow(z, -1.5); }
    } else if (name == "soft_l1") {
      double t = 1 + z;
      r0 = 2 * (std::sqrt(t) - 1); r1 = 1 / std::sqrt(t); r2 = -0.5 * std::pow(t, -1.5);
    } else if (name == "cauchy") {
      double t = 1 + z;
      r0 = std::log1p(z); r1 = 1 / t; r2 = -1 / (t * t);
    } else {  // arctan
      double t = 1 + z * z;
      r0 = std::atan(z); r1 = 1 / t; r2 = -2 * z / (t * t);
    }
  }
  double cost(const VectorXd& f) const {
    if (linear()) return 0.5 * f.dot(f);
    double s = 0;
    for (Eigen::Index i = 0; i < f.size(); ++i) {
      double r0, r1, r2;
      rho_z((f(i) / C) * (f(i) / C), r0, r1, r2);
      s += r0;
    }
    return 0.5 * C * C * s;
  }
  // scale_for_robust_loss_function: J and f modified in place
  void scale(MatrixXd& J, VectorXd& f) const {
    for (Eigen::Index i = 0; i < f.size(); ++i) {
      double r0, r1, r2;
      rho_z((f(i) / C) * (f(i) / C), r0, r1, r2);
      r2 /= C * C;
      double js = r1 + 2 * r2 * f(i) * f(i);
      if (js < EPS) js = EPS;
      js = std::sqrt(js);
      f(i) *= r1 / js;
      J.row(i) *= js;
    }
  }
};

void jac_scale(const MatrixXd& J, VectorXd& scale, VectorXd& scale_inv, bool first) {
  VectorXd si = J.colwise().norm().transpose();
  if (first) { for (Eigen::Index i = 0; i < si.size(); ++i) if (si(i) == 0) si(i) = 1; }
  else si = si.cwiseMax(scale_inv);
  scale_inv = si;
  scale = si.cwiseInverse();
}

int check_termination(double dF, double F, double dx_norm, double x_norm, double ratio, double ftol, double xtol) {
  bool f_ok = dF < ftol * F && ratio > 0.25;
  bool x_ok = dx_norm < xtol * (xtol + x_norm);
  if (f_ok && x_ok) return 4;
  if (f_ok) return 2;
  if (x_ok) return 3;
  return -99;
}

void update_tr_radius(double& Delta, double actual, double predicted, double step_norm, bool bound_hit, double& ratio) {
  if (predicted > 0) ratio = actual / predicted;
  else if (predicted == actual && actual == 0) ratio = 1;
  else ratio = 0;
  if (ratio < 0.25) Delta = 0.25 * step_norm;
  else if (ratio > 0.75 && bound_hit) Delta *= 2.0;
}

double evaluate_quadratic(const MatrixXd& J, const VectorXd& g, const VectorXd& s, const VectorXd* diag = nullptr) {
  VectorXd Js = J * s;
  double q = Js.dot(Js);
  if (diag) q += s.cwiseProduct(*diag).dot(s);
  return 0.5 * q + s.dot(g);
}

void build_quadratic_1d(const MatrixXd& J, const VectorXd& g, const VectorXd& s, const VectorXd* diag, const VectorXd* s0, double& a, double& b, double& c) {
  VectorXd v = J * s;
  a = v.dot(v);
  if (diag) a += s.cwiseProduct(*diag).dot(s);
  a *= 0.5;
  b = g.dot(s);
  c = 0;
  if (s0) {
    VectorXd u = J * *s0;
    b += u.dot(v);
    c = 0.5 * u.dot(u) + g.dot(*s0);
    if (diag) {
      b += s0->cwiseProduct(*diag).dot(s);
      c += 0.5 * s0->cwiseProduct(*diag).dot(*s0);
    }
  }
}

std::pair<double, double> minimize_quadratic_1d(double a, double b, double lb, double ub, double c = 0) {
  double t[3] = {lb, ub, 0};
  int nt = 2;
  if (a != 0) {
    double ext = -0.5 * b / a;
    if (lb < ext && ext < ub) t[nt++] = ext;
  }
  int best = 0;
  double by = INF;
  for (int i = 0; i < nt; ++i) {
    double y = t[i] * (a * t[i] + b) + c;
    if (i == 0 || y < by) { by = y; best = i; }
  }
  return {t[best], by};
}

bool in_bounds(const VectorXd& x, const VectorXd& lb, const VectorXd& ub) {
  return ((x.array() >= lb.array()) && (x.array() <= ub.array())).all();
}

std::pair<double, VectorXd> step_size_to_bound(const VectorXd& x, const VectorXd& s, const VectorXd& lb, const VectorXd& ub) {
  VectorXd steps = VectorXd::Constant(x.size(), INF);
  for (Eigen::Index i = 0; i < x.size(); ++i)
    if (s(i) != 0) steps(i) = std::max((lb(i) - x(i)) / s(i), (ub(i) - x(i)) / s(i));
  double mn = steps.minCoeff();
  VectorXd hits(x.size());
  for (Eigen::Index i = 0; i < x.size(); ++i) hits(i) = steps(i) == mn ? (s(i) > 0 ? 1.0 : s(i) < 0 ? -1.0 : 0.0) : 0.0;
  return {mn, hits};
}

std::pair<double, double> intersect_trust_region(const VectorXd& x, const VectorXd& s, double Delta) {
  double a = s.dot(s);
  if (a == 0) fail("least_squares: internal error (zero trust-region direction)");
  double b = x.dot(s), c = x.dot(x) - Delta * Delta;
  double d = std::sqrt(std::max(0.0, b * b - a * c));
  double q = -(b + std::copysign(d, b));
  double t1 = q / a, t2 = c / q;
  return t1 < t2 ? std::make_pair(t1, t2) : std::make_pair(t2, t1);
}

// solve_lsq_trust_region: (p, alpha) from the SVD J_h = U diag(s) V'
VectorXd solve_lsq_trust_region(Eigen::Index n, Eigen::Index m, const VectorXd& uf, const VectorXd& s, const MatrixXd& V, double Delta, double& alpha) {
  const double rtol = 0.01;
  const int max_iter = 10;
  VectorXd suf = s.cwiseProduct(uf);
  auto phi_and_derivative = [&](double al, double& phi, double& dphi) {
    VectorXd denom = s.array().square() + al;
    double p_norm = suf.cwiseQuotient(denom).norm();
    phi = p_norm - Delta;
    dphi = -(suf.array().square() / denom.array().cube()).sum() / p_norm;
  };
  bool full_rank = false;
  if (m >= n && s.size() > 0) full_rank = s(s.size() - 1) > EPS * double(m) * s(0);
  if (full_rank) {
    VectorXd p = -V * uf.cwiseQuotient(s);
    if (p.norm() <= Delta) { alpha = 0.0; return p; }
  }
  double alpha_upper = suf.norm() / Delta, alpha_lower = 0.0;
  if (full_rank) {
    double phi, dphi;
    phi_and_derivative(0.0, phi, dphi);
    alpha_lower = -phi / dphi;
  }
  if (!full_rank && alpha == 0) alpha = std::max(0.001 * alpha_upper, std::sqrt(alpha_lower * alpha_upper));
  for (int it = 0; it < max_iter; ++it) {
    if (alpha < alpha_lower || alpha > alpha_upper) alpha = std::max(0.001 * alpha_upper, std::sqrt(alpha_lower * alpha_upper));
    double phi, dphi;
    phi_and_derivative(alpha, phi, dphi);
    if (phi < 0) alpha_upper = alpha;
    double ratio = phi / dphi;
    alpha_lower = std::max(alpha_lower, alpha - ratio);
    alpha -= (phi + Delta) * ratio / Delta;
    if (std::fabs(phi) < rtol * Delta) break;
  }
  VectorXd p = -V * suf.cwiseQuotient((s.array().square() + alpha).matrix());
  p *= Delta / p.norm();
  return p;
}

void thin_svd(const MatrixXd& A, MatrixXd& U, VectorXd& s, MatrixXd& V) {
  Eigen::JacobiSVD<MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
  U = svd.matrixU();
  s = svd.singularValues();
  V = svd.matrixV();
}

VectorXi find_active(const VectorXd& x, const VectorXd& lb, const VectorXd& ub, double rtol) {
  VectorXi active = VectorXi::Zero(x.size());
  for (Eigen::Index i = 0; i < x.size(); ++i) {
    if (rtol == 0) {
      if (x(i) <= lb(i)) active(i) = -1;
      if (x(i) >= ub(i)) active(i) = 1;
      continue;
    }
    double ld = x(i) - lb(i), ud = ub(i) - x(i);
    double lt = rtol * std::max(1.0, std::fabs(lb(i))), ut = rtol * std::max(1.0, std::fabs(ub(i)));
    if (std::isfinite(lb(i)) && ld <= std::min(ud, lt)) active(i) = -1;
    if (std::isfinite(ub(i)) && ud <= std::min(ld, ut)) active(i) = 1;
  }
  return active;
}

VectorXd make_strictly_feasible(const VectorXd& x, const VectorXd& lb, const VectorXd& ub, double rstep) {
  VectorXd xn = x;
  VectorXi active = find_active(x, lb, ub, rstep);
  for (Eigen::Index i = 0; i < x.size(); ++i) {
    if (active(i) == -1) xn(i) = rstep == 0 ? std::nextafter(lb(i), ub(i)) : lb(i) + rstep * std::max(1.0, std::fabs(lb(i)));
    if (active(i) == 1) xn(i) = rstep == 0 ? std::nextafter(ub(i), lb(i)) : ub(i) - rstep * std::max(1.0, std::fabs(ub(i)));
    if (xn(i) < lb(i) || xn(i) > ub(i)) xn(i) = 0.5 * (lb(i) + ub(i));
  }
  return xn;
}

void CL_scaling(const VectorXd& x, const VectorXd& g, const VectorXd& lb, const VectorXd& ub, VectorXd& v, VectorXd& dv) {
  v = VectorXd::Ones(x.size());
  dv = VectorXd::Zero(x.size());
  for (Eigen::Index i = 0; i < x.size(); ++i) {
    if (g(i) < 0 && std::isfinite(ub(i))) { v(i) = ub(i) - x(i); dv(i) = -1; }
    if (g(i) > 0 && std::isfinite(lb(i))) { v(i) = x(i) - lb(i); dv(i) = 1; }
  }
}

// trf select_step: returns (step, step_h, predicted reduction)
void select_step(const VectorXd& x, const MatrixXd& J_h, const VectorXd& diag_h, const VectorXd& g_h, VectorXd p, VectorXd p_h, const VectorXd& d,
                 double Delta, const VectorXd& lb, const VectorXd& ub, double theta, VectorXd& step, VectorXd& step_h, double& pred) {
  if (in_bounds(x + p, lb, ub)) {
    step = p; step_h = p_h;
    pred = -evaluate_quadratic(J_h, g_h, p_h, &diag_h);
    return;
  }
  auto [p_stride, hits] = step_size_to_bound(x, p, lb, ub);
  VectorXd r_h = p_h;
  for (Eigen::Index i = 0; i < r_h.size(); ++i) if (hits(i) != 0) r_h(i) = -r_h(i);
  VectorXd r = d.cwiseProduct(r_h);
  p *= p_stride;
  p_h *= p_stride;
  VectorXd x_on_bound = x + p;
  double to_tr = intersect_trust_region(p_h, r_h, Delta).second;
  double to_bound = step_size_to_bound(x_on_bound, r, lb, ub).first;
  double r_stride = std::min(to_bound, to_tr), r_stride_l, r_stride_u;
  if (r_stride > 0) {
    r_stride_l = (1 - theta) * p_stride / r_stride;
    r_stride_u = r_stride == to_bound ? theta * to_bound : to_tr;
  } else {
    r_stride_l = 0;
    r_stride_u = -1;
  }
  double r_value = INF;
  if (r_stride_l <= r_stride_u) {
    double a, b, c;
    build_quadratic_1d(J_h, g_h, r_h, &diag_h, &p_h, a, b, c);
    auto mq = minimize_quadratic_1d(a, b, r_stride_l, r_stride_u, c);
    r_h *= mq.first;
    r_h += p_h;
    r = r_h.cwiseProduct(d);
    r_value = mq.second;
  }
  p *= theta;
  p_h *= theta;
  double p_value = evaluate_quadratic(J_h, g_h, p_h, &diag_h);
  VectorXd ag_h = -g_h, ag = d.cwiseProduct(ag_h);
  to_tr = Delta / ag_h.norm();
  to_bound = step_size_to_bound(x, ag, lb, ub).first;
  double ag_stride = to_bound < to_tr ? theta * to_bound : to_tr;
  double a, b, c;
  build_quadratic_1d(J_h, g_h, ag_h, &diag_h, nullptr, a, b, c);
  auto mq = minimize_quadratic_1d(a, b, 0, ag_stride);
  ag_h *= mq.first;
  ag *= mq.first;
  double ag_value = mq.second;
  if (p_value < r_value && p_value < ag_value) { step = p; step_h = p_h; pred = -p_value; }
  else if (r_value < p_value && r_value < ag_value) { step = r; step_h = r_h; pred = -r_value; }
  else { step = ag; step_h = ag_h; pred = -ag_value; }
}

struct Ctx {
  const LsqProblem& p;
  Loss loss;
  VectorXd lb, ub;
  int64_t nfev = 0, njev = 0;
  VectorXd fun(const VectorXd& x) { return p.residual(x); }
  MatrixXd jac(const VectorXd& x, const VectorXd& f) { ++njev; return fd_jacobian(p.residual, x, f, 0.0, lb, ub); }
};

LsqResult trf(Ctx& C, const VectorXd& x0, VectorXd f, MatrixXd J) {
  const LsqProblem& P = C.p;
  bool bounded = false;
  for (Eigen::Index i = 0; i < x0.size(); ++i) bounded = bounded || std::isfinite(C.lb(i)) || std::isfinite(C.ub(i));
  Eigen::Index m = J.rows(), n = J.cols();
  VectorXd x = x0, f_true = f;
  int64_t nfev = 1;
  double cost = C.loss.cost(f);
  if (!C.loss.linear()) C.loss.scale(J, f);
  VectorXd g = J.transpose() * f, scale, scale_inv;
  if (P.x_scale_jac) jac_scale(J, scale, scale_inv, true);
  else { scale = VectorXd::Ones(n); scale_inv = scale; }
  double Delta;
  VectorXd v, dv;
  if (bounded) {
    CL_scaling(x, g, C.lb, C.ub, v, dv);
    for (Eigen::Index i = 0; i < n; ++i) if (dv(i) != 0) v(i) *= scale_inv(i);
    Delta = (x0.cwiseProduct(scale_inv).array() / v.array().sqrt()).matrix().norm();
  } else {
    Delta = x0.cwiseProduct(scale_inv).norm();
  }
  if (Delta == 0) Delta = 1.0;
  double alpha = 0.0, g_norm = 0;
  int status = -99;
  for (;;) {
    if (bounded) {
      CL_scaling(x, g, C.lb, C.ub, v, dv);
      g_norm = g.cwiseProduct(v).cwiseAbs().maxCoeff();
    } else {
      g_norm = g.cwiseAbs().maxCoeff();
    }
    if (g_norm < P.gtol) status = 1;
    if (status != -99 || nfev == P.max_nfev) break;
    VectorXd d, diag_h, g_h, uf, s;
    MatrixXd J_h, U, V;
    if (bounded) {
      for (Eigen::Index i = 0; i < n; ++i) if (dv(i) != 0) v(i) *= scale_inv(i);
      d = v.array().sqrt().matrix().cwiseProduct(scale);
      diag_h = g.cwiseProduct(dv).cwiseProduct(scale);
      g_h = d.cwiseProduct(g);
      J_h = J * d.asDiagonal();
      MatrixXd Ja(m + n, n);
      Ja.topRows(m) = J_h;
      Ja.bottomRows(n) = diag_h.array().sqrt().matrix().asDiagonal();
      VectorXd fa = VectorXd::Zero(m + n);
      fa.head(m) = f;
      thin_svd(Ja, U, s, V);
      uf = U.transpose() * fa;
    } else {
      d = scale;
      g_h = d.cwiseProduct(g);
      J_h = J * d.asDiagonal();
      thin_svd(J_h, U, s, V);
      uf = U.transpose() * f;
    }
    double theta = std::max(0.995, 1 - g_norm);
    double actual_reduction = -1, cost_new = cost;
    VectorXd x_new, f_new, step, step_h;
    while (actual_reduction <= 0 && nfev < P.max_nfev) {
      double pred;
      if (bounded) {
        VectorXd p_h = solve_lsq_trust_region(n, m + n, uf, s, V, Delta, alpha);
        VectorXd p = d.cwiseProduct(p_h);
        select_step(x, J_h, diag_h, g_h, p, p_h, d, Delta, C.lb, C.ub, theta, step, step_h, pred);
        x_new = make_strictly_feasible(x + step, C.lb, C.ub, 0);
      } else {
        step_h = solve_lsq_trust_region(n, m, uf, s, V, Delta, alpha);
        pred = -evaluate_quadratic(J_h, g_h, step_h);
        step = d.cwiseProduct(step_h);
        x_new = x + step;
      }
      f_new = C.fun(x_new);
      ++nfev;
      double step_h_norm = step_h.norm();
      if (!f_new.allFinite()) { Delta = 0.25 * step_h_norm; continue; }
      cost_new = C.loss.cost(f_new);
      actual_reduction = cost - cost_new;
      double Delta_new = Delta, ratio;
      update_tr_radius(Delta_new, actual_reduction, pred, step_h_norm, step_h_norm > 0.95 * Delta, ratio);
      double step_norm = step.norm();
      status = check_termination(actual_reduction, cost, step_norm, x.norm(), ratio, P.ftol, P.xtol);
      if (status != -99) break;
      alpha *= Delta / Delta_new;
      Delta = Delta_new;
    }
    if (actual_reduction > 0) {
      x = x_new;
      f = f_new;
      f_true = f;
      cost = cost_new;
      J = C.jac(x, f);
      if (!C.loss.linear()) C.loss.scale(J, f);
      g = J.transpose() * f;
      if (P.x_scale_jac) jac_scale(J, scale, scale_inv, false);
    }
  }
  LsqResult r;
  r.x = x; r.cost = cost; r.fun = f_true; r.jac = J; r.optimality = g_norm; r.nfev = nfev; r.njev = C.njev + 1;
  r.status = status == -99 ? 0 : status;
  return r;
}

void dogleg_step(const VectorXd& x, const VectorXd& newton, const VectorXd& g, double a, double b, const VectorXd& tr_bounds, const VectorXd& lb,
                 const VectorXd& ub, VectorXd& step, VectorXi& bound_hits, bool& tr_hit) {
  Eigen::Index n = x.size();
  VectorXd lbc = lb - x, ubc = ub - x;
  VectorXd lbt = lbc.cwiseMax(-tr_bounds), ubt = ubc.cwiseMin(tr_bounds);
  bound_hits = VectorXi::Zero(n);
  tr_hit = false;
  if (in_bounds(newton, lbt, ubt)) { step = newton; return; }
  double to_bounds = step_size_to_bound(VectorXd::Zero(n), -g, lbt, ubt).first;
  VectorXd cauchy = -minimize_quadratic_1d(a, b, 0, to_bounds).first * g;
  VectorXd diff = newton - cauchy;
  auto [ss, hits] = step_size_to_bound(cauchy, diff, lbt, ubt);
  for (Eigen::Index i = 0; i < n; ++i) {
    bool orig_l = lbt(i) == lbc(i), orig_u = ubt(i) == ubc(i), tr_l = lbt(i) == -tr_bounds(i), tr_u = ubt(i) == tr_bounds(i);
    if (hits(i) < 0 && orig_l) bound_hits(i) = -1;
    if (hits(i) > 0 && orig_u) bound_hits(i) = 1;
    if ((hits(i) < 0 && tr_l) || (hits(i) > 0 && tr_u)) tr_hit = true;
  }
  step = cauchy + ss * diff;
}

LsqResult dogbox(Ctx& C, const VectorXd& x0, VectorXd f, MatrixXd J) {
  const LsqProblem& P = C.p;
  Eigen::Index n = J.cols();
  VectorXd f_true = f;
  int64_t nfev = 1;
  double cost = C.loss.cost(f);
  if (!C.loss.linear()) C.loss.scale(J, f);
  VectorXd g = J.transpose() * f, scale, scale_inv;
  if (P.x_scale_jac) jac_scale(J, scale, scale_inv, true);
  else { scale = VectorXd::Ones(n); scale_inv = scale; }
  double Delta = x0.cwiseProduct(scale_inv).cwiseAbs().maxCoeff();
  if (Delta == 0) Delta = 1.0;
  VectorXi on_bound = VectorXi::Zero(n);
  for (Eigen::Index i = 0; i < n; ++i) {
    if (x0(i) == C.lb(i)) on_bound(i) = -1;
    if (x0(i) == C.ub(i)) on_bound(i) = 1;
  }
  VectorXd x = x0, g_full = g;
  double g_norm = 0;
  int status = -99;
  for (;;) {
    std::vector<Eigen::Index> fr;
    g_full = g;
    for (Eigen::Index i = 0; i < n; ++i) {
      if (double(on_bound(i)) * g(i) < 0) g(i) = 0;
      else fr.push_back(i);
    }
    g_norm = g.cwiseAbs().maxCoeff();
    if (g_norm < P.gtol) status = 1;
    if (status != -99 || nfev == P.max_nfev) break;
    Eigen::Index nf = Eigen::Index(fr.size());
    VectorXd x_free(nf), lb_free(nf), ub_free(nf), scale_free(nf), g_free(nf);
    MatrixXd J_free(J.rows(), nf);
    for (Eigen::Index k = 0; k < nf; ++k) {
      Eigen::Index i = fr[size_t(k)];
      x_free(k) = x(i); lb_free(k) = C.lb(i); ub_free(k) = C.ub(i); scale_free(k) = scale(i); g_free(k) = g_full(i);
      J_free.col(k) = J.col(i);
    }
    // numpy.linalg.lstsq(J_free, -f, rcond=-1): minimum-norm least squares
    Eigen::CompleteOrthogonalDecomposition<MatrixXd> cod(J_free);
    VectorXd newton = cod.solve(-f);
    double a, b, c;
    build_quadratic_1d(J_free, g_free, -g_free, nullptr, nullptr, a, b, c);
    double actual_reduction = -1.0, cost_new = cost;
    VectorXd x_new, f_new, step = VectorXd::Zero(n), step_free;
    VectorXi on_bound_free;
    while (actual_reduction <= 0 && nfev < P.max_nfev) {
      VectorXd tr_bounds = Delta * scale_free;
      bool tr_hit;
      dogleg_step(x_free, newton, g_free, a, b, tr_bounds, lb_free, ub_free, step_free, on_bound_free, tr_hit);
      step.setZero();
      for (Eigen::Index k = 0; k < nf; ++k) step(fr[size_t(k)]) = step_free(k);
      double pred = -evaluate_quadratic(J_free, g_free, step_free);
      x_new = (x + step).cwiseMax(C.lb).cwiseMin(C.ub);
      f_new = C.fun(x_new);
      ++nfev;
      double step_h_norm = step.cwiseProduct(scale_inv).cwiseAbs().maxCoeff();
      if (!f_new.allFinite()) { Delta = 0.25 * step_h_norm; continue; }
      cost_new = C.loss.cost(f_new);
      actual_reduction = cost - cost_new;
      double ratio;
      update_tr_radius(Delta, actual_reduction, pred, step_h_norm, tr_hit, ratio);
      status = check_termination(actual_reduction, cost, step.norm(), x.norm(), ratio, P.ftol, P.xtol);
      if (status != -99) break;
    }
    if (actual_reduction > 0) {
      for (Eigen::Index k = 0; k < nf; ++k) on_bound(fr[size_t(k)]) = on_bound_free(k);
      x = x_new;
      for (Eigen::Index i = 0; i < n; ++i) {
        if (on_bound(i) == -1) x(i) = C.lb(i);
        if (on_bound(i) == 1) x(i) = C.ub(i);
      }
      f = f_new;
      f_true = f;
      cost = cost_new;
      J = C.jac(x, f);
      if (!C.loss.linear()) C.loss.scale(J, f);
      g = J.transpose() * f;
      if (P.x_scale_jac) jac_scale(J, scale, scale_inv, false);
    }
  }
  LsqResult r;
  r.x = x; r.cost = cost; r.fun = f_true; r.jac = J; r.optimality = g_norm; r.nfev = nfev; r.njev = C.njev + 1;
  r.status = status == -99 ? 0 : status;
  return r;
}

struct LmFunctor {
  typedef double Scalar;
  enum { InputsAtCompileTime = Eigen::Dynamic, ValuesAtCompileTime = Eigen::Dynamic };
  typedef VectorXd InputType;
  typedef VectorXd ValueType;
  typedef MatrixXd JacobianType;
  Ctx& C;
  Eigen::Index n, m;
  int inputs() const { return int(n); }
  int values() const { return int(m); }
  int operator()(const VectorXd& x, VectorXd& fv) {
    fv = C.fun(x);
    if (fv.size() != m) fail("least_squares: the number of residuals changed between evaluations");
    return 0;
  }
  int df(const VectorXd& x, MatrixXd& J) {
    VectorXd f0 = C.fun(x);
    J = C.jac(x, f0);
    return 0;
  }
};

LsqResult lm(Ctx& C, const VectorXd& x0, const VectorXd& f0) {
  const LsqProblem& P = C.p;
  LmFunctor fn{C, x0.size(), f0.size()};
  Eigen::LevenbergMarquardt<LmFunctor> solver(fn);
  solver.parameters.factor = 100.0;
  solver.parameters.maxfev = P.max_nfev;
  solver.parameters.ftol = P.ftol;
  solver.parameters.xtol = P.xtol;
  solver.parameters.gtol = P.gtol;
  VectorXd x = x0;
  int info = int(solver.minimize(x));
  LsqResult r;
  r.x = x;
  r.fun = C.fun(x);
  r.jac = C.jac(x, r.fun);
  r.cost = 0.5 * r.fun.dot(r.fun);
  r.optimality = (r.jac.transpose() * r.fun).cwiseAbs().maxCoeff();
  r.nfev = solver.nfev;
  r.njev = solver.njev;
  static const int map[] = {-1, 2, 3, 4, 1, 0, 2, 3, 1, -1};
  r.status = info >= 0 && info <= 9 ? map[info] : -1;
  return r;
}

}  // namespace

LsqResult least_squares_solve(const LsqProblem& p, const VectorXd& x0in) {
  Eigen::Index n = x0in.size();
  Ctx C{p, Loss{p.loss, p.f_scale}, p.lb.size() ? p.lb : VectorXd::Constant(n, -INF), p.ub.size() ? p.ub : VectorXd::Constant(n, INF)};
  VectorXd x0 = p.method == "trf" ? make_strictly_feasible(x0in, C.lb, C.ub, 1e-10) : x0in;
  VectorXd f0 = C.fun(x0);
  LsqResult r;
  if (p.method == "lm") {
    r = lm(C, x0, f0);
  } else {
    MatrixXd J0 = C.jac(x0, f0);
    C.njev = 0;
    r = p.method == "trf" ? trf(C, x0, f0, J0) : dogbox(C, x0, f0, J0);
  }
  static const char* msgs[] = {"Improper input parameters status returned from `leastsq`", "The maximum number of function evaluations is exceeded.",
                               "`gtol` termination condition is satisfied.", "`ftol` termination condition is satisfied.",
                               "`xtol` termination condition is satisfied.", "Both `ftol` and `xtol` termination conditions are satisfied."};
  r.message = msgs[r.status + 1];
  return r;
}

}  // namespace lmpy::opt
