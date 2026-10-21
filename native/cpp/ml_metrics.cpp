// Metrics for the machine-learning family: pycli ml_metrics (hand formulas from the confusion counts, rank AUC with
// ties counting 1/2, clipped log loss, Brier; r2/mse/mae/mape) and the sklearn scorers ml_cv uses (accuracy, f1_macro,
// roc_auc, log_loss, r2, mse, mae), plus adjusted_rand_score (sklearn.metrics, pair-confusion form).
#include "ml.hpp"
#include <algorithm>
#include <cmath>
#include <set>

namespace lmpy {
namespace ml {

std::vector<double> sorted_unique(const VectorXd& v) {
  std::vector<double> s(v.data(), v.data() + v.size());
  std::sort(s.begin(), s.end());
  s.erase(std::unique(s.begin(), s.end()), s.end());
  return s;
}

double accuracy(const VectorXd& t, const VectorXd& p) {
  double c = 0; for (Eigen::Index i = 0; i < t.size(); ++i) c += t(i) == p(i);
  return c / double(t.size());
}

double r2_score(const VectorXd& t, const VectorXd& p) {
  double m = t.mean(), ss_res = (t - p).squaredNorm(), ss_tot = (t.array() - m).square().sum();
  if (ss_tot == 0) return NAN;
  return 1.0 - ss_res / ss_tot;
}

double f1_macro(const VectorXd& t, const VectorXd& p) {
  std::vector<double> labels = sorted_unique(t);
  for (double v : sorted_unique(p)) labels.push_back(v);
  std::sort(labels.begin(), labels.end()); labels.erase(std::unique(labels.begin(), labels.end()), labels.end());
  double s = 0;
  for (double l : labels) {
    double tp = 0, fp = 0, fn = 0;
    for (Eigen::Index i = 0; i < t.size(); ++i) { bool a = t(i) == l, b = p(i) == l; tp += a && b; fp += !a && b; fn += a && !b; }
    s += 2 * tp + fp + fn > 0 ? 2 * tp / (2 * tp + fp + fn) : 0.0;
  }
  return s / double(labels.size());
}

double roc_auc(const VectorXd& target, const VectorXd& score) {
  const Eigen::Index n = score.size();
  double npos = target.sum(), nneg = double(n) - npos;
  if (npos == 0 || nneg == 0) return NAN;
  std::vector<Eigen::Index> idx(n); for (Eigen::Index i = 0; i < n; ++i) idx[i] = i;
  std::stable_sort(idx.begin(), idx.end(), [&](Eigen::Index a, Eigen::Index b) { return score(a) < score(b); });
  double rank_pos = 0;
  for (Eigen::Index i = 0; i < n;) {
    Eigen::Index j = i; while (j < n && score(idx[j]) == score(idx[i])) ++j;
    double avg = (double(i + 1) + double(j)) / 2.0;
    for (Eigen::Index k = i; k < j; ++k) if (target(idx[k]) == 1) rank_pos += avg;
    i = j;
  }
  return (rank_pos - npos * (npos + 1) / 2.0) / (npos * nneg);
}

double log_loss(const VectorXd& t, const MatrixXd& proba, const VectorXd& classes) {
  // sklearn.metrics.log_loss(labels=classes): clip to [eps, 1 - eps] with eps = float64 machine epsilon
  const double eps = std::numeric_limits<double>::epsilon();
  double s = 0;
  for (Eigen::Index i = 0; i < t.size(); ++i) {
    Eigen::Index k = 0; while (k < classes.size() && classes(k) != t(i)) ++k;
    if (k == classes.size()) return NAN;
    double pr = std::min(std::max(proba(i, k), eps), 1 - eps);
    s -= std::log(pr);
  }
  return s / double(t.size());
}

double adjusted_rand(const VectorXd& a, const VectorXd& b) {
  if (a.size() != b.size()) fail("adjusted_rand_score: lengths differ (" + std::to_string(a.size()) + " and " + std::to_string(b.size()) + ")");
  const double n = double(a.size());
  std::vector<double> ua = sorted_unique(a), ub = sorted_unique(b);
  MatrixXd c = MatrixXd::Zero(Eigen::Index(ua.size()), Eigen::Index(ub.size()));
  for (Eigen::Index i = 0; i < a.size(); ++i)
    c(std::lower_bound(ua.begin(), ua.end(), a(i)) - ua.begin(), std::lower_bound(ub.begin(), ub.end(), b(i)) - ub.begin()) += 1;
  double ss = c.array().square().sum();
  double rows2 = c.rowwise().sum().array().square().sum(), cols2 = c.colwise().sum().array().square().sum();
  double c11 = ss - n, c01 = cols2 - ss, c10 = rows2 - ss, c00 = n * n - c01 - c10 - ss;
  if (c10 == 0 && c01 == 0) return 1.0;
  return 2.0 * (c11 * c00 - c10 * c01) / ((c11 + c10) * (c10 + c00) + (c11 + c01) * (c01 + c00));
}

Res* metrics(const VectorXd& yt, const VectorXd* yp_arg, const Opts& o) {
  const std::string op = "ml_metrics";
  const std::vector<double>* yp_opt = o.vec("y_pred");
  const std::vector<double>* ys = o.vec("y_score");
  const std::vector<std::string>* wanted_opt = o.strs("metrics");
  std::string kind = o.str("kind", "auto");
  bool has_average = o.has("average");
  std::string average = o.str("average", "binary");
  bool has_pos = o.has("pos_label");
  double pos = o.num("pos_label", 1.0);
  std::string zd = o.str("zero_division", "undefined");
  o.finish(op);
  if (yp_arg && yp_opt) fail(op + ": y_pred is given twice (argument and option)");
  VectorXd yp; bool have_pred = yp_arg || yp_opt;
  if (yp_arg) yp = *yp_arg; else if (yp_opt) yp = to_eigen(*yp_opt);
  const Eigen::Index n = yt.size();
  if (n == 0) fail(op + ": y_true is empty");
  if (have_pred && yp.size() != n) fail(op + ": y_true, y_pred and y_score must have the same length (" + std::to_string(n) + " and " + std::to_string(yp.size()) + ")");
  if (ys && Eigen::Index(ys->size()) != n) fail(op + ": y_true, y_pred and y_score must have the same length (" + std::to_string(n) + " and " + std::to_string(ys->size()) + ")");
  if (!have_pred && !ys) fail(op + ": needs y_pred (argument or option) and/or the option y_score");
  if (!yt.allFinite() || (have_pred && !yp.allFinite())) fail(op + ": y_true and y_pred must be finite (NA_NOT_ALLOWED)");
  if (average != "binary" && average != "macro" && average != "micro" && average != "weighted") fail(op + ": average must be binary, macro, micro or weighted; got '" + average + "'");
  if (zd != "undefined" && zd != "0" && zd != "1") fail(op + ": zero_division must be undefined, 0 or 1; got '" + zd + "'");
  if (kind != "auto" && kind != "classification" && kind != "regression") fail(op + ": kind must be auto, classification or regression; got '" + kind + "'");

  static const std::vector<std::string> known = {"accuracy", "balanced_accuracy", "precision", "recall", "f1", "confusion_matrix", "roc_auc", "log_loss", "brier", "r2", "mse", "rmse", "mae", "mape"};
  std::vector<std::string> wanted;
  if (wanted_opt) {
    if (wanted_opt->empty()) fail(op + ": metrics is empty");
    for (auto& m : *wanted_opt) {
      if (std::find(known.begin(), known.end(), m) == known.end()) fail(op + ": unknown metric '" + m + "' (known: accuracy, balanced_accuracy, precision, recall, f1, confusion_matrix, roc_auc, log_loss, brier, r2, mse, rmse, mae, mape)");
      if (std::find(wanted.begin(), wanted.end(), m) == wanted.end()) wanted.push_back(m);
    }
  } else {
    if (kind == "auto") {
      bool integral = true; std::set<double> distinct;
      for (Eigen::Index i = 0; i < n; ++i) { integral &= yt(i) == std::floor(yt(i)); distinct.insert(yt(i)); if (have_pred) { integral &= yp(i) == std::floor(yp(i)); distinct.insert(yp(i)); } }
      kind = ys || (integral && distinct.size() <= 30) ? "classification" : "regression";
    }
    if (kind == "classification") {
      if (have_pred) wanted = {"accuracy", "balanced_accuracy", "precision", "recall", "f1", "confusion_matrix"};
      if (ys) for (const char* m : {"roc_auc", "log_loss", "brier"}) wanted.push_back(m);
    } else {
      if (!have_pred) fail(op + ": regression metrics need y_pred");
      wanted = {"r2", "mse", "rmse", "mae"};
    }
  }
  auto want = [&](const char* m) { return std::find(wanted.begin(), wanted.end(), m) != wanted.end(); };
  auto* r = make<Res>();
  r->integer("n", n);
  std::vector<std::pair<std::string, double>> vals;
  std::vector<std::pair<std::string, std::string>> undefined;
  bool label_metrics = want("accuracy") || want("balanced_accuracy") || want("precision") || want("recall") || want("f1") || want("confusion_matrix");
  bool scored = want("roc_auc") || want("log_loss") || want("brier");
  bool regression = want("r2") || want("mse") || want("rmse") || want("mae") || want("mape");
  if ((label_metrics || regression) && !have_pred) fail(op + ": the requested label/regression metrics need y_pred");
  if (scored && !ys) fail(op + ": roc_auc, log_loss and brier need the option y_score");
  if (label_metrics) {
    std::vector<double> labels = sorted_unique(yt);
    for (double v : sorted_unique(yp)) labels.push_back(v);
    std::sort(labels.begin(), labels.end()); labels.erase(std::unique(labels.begin(), labels.end()), labels.end());
    if (labels.size() > 30) fail(op + ": classification metrics allow at most 30 distinct labels (found " + std::to_string(labels.size()) + "); for regression pass opts().set(\"kind\", \"regression\")");
    const Eigen::Index L = Eigen::Index(labels.size());
    auto at = [&](double v) { return Eigen::Index(std::lower_bound(labels.begin(), labels.end(), v) - labels.begin()); };
    MatrixXd conf = MatrixXd::Zero(L, L);
    for (Eigen::Index i = 0; i < n; ++i) conf(at(yt(i)), at(yp(i))) += 1;
    VectorXd tp = conf.diagonal(), fp = conf.colwise().sum().transpose() - tp, fn = conf.rowwise().sum() - tp, support = conf.rowwise().sum();
    bool zero_undef = zd == "undefined"; double zval = zd == "1" ? 1.0 : 0.0;
    if (!has_average && !wanted_opt) {
      bool zero_one = true; for (double l : labels) zero_one &= l == 0 || l == 1;
      if (!has_pos && !zero_one) average = "macro";  // lmpy default for multi-class data (pycli: binary needs pos_label)
    }
    if (want("accuracy")) vals.push_back({"accuracy", tp.sum() / double(n)});
    if (want("balanced_accuracy")) {
      double s = 0; int c = 0; for (Eigen::Index k = 0; k < L; ++k) if (support(k) > 0) { s += tp(k) / support(k); ++c; }
      vals.push_back({"balanced_accuracy", s / c});
    }
    if (want("precision") || want("recall") || want("f1")) {
      struct T { double v; bool def; };
      auto ratio = [&](double a, double b) { return b != 0 ? T{a / b, true} : T{zval, !zero_undef}; };
      auto prf = [&](double t, double f_p, double f_n) { return std::vector<T>{ratio(t, t + f_p), ratio(t, t + f_n), ratio(2 * t, 2 * t + f_p + f_n)}; };
      std::vector<T> triple;
      if (average == "binary") {
        if (!has_pos) {
          for (double l : labels) if (l != 0 && l != 1) fail(op + ": average=binary needs pos_label for labels other than 0/1 (BAD_PARAM)");
          pos = 1;
        }
        auto it = std::find(labels.begin(), labels.end(), pos);
        if (it == labels.end()) fail(op + ": pos_label " + jnum(pos) + " does not occur among the labels (BAD_PARAM)");
        Eigen::Index k = it - labels.begin();
        triple = prf(tp(k), fp(k), fn(k));
        r->num("pos_label", pos);
      } else if (average == "micro") triple = prf(tp.sum(), fp.sum(), fn.sum());
      else {
        double wsum = 0; std::vector<double> acc(3, 0.0); std::vector<bool> def(3, true);
        for (Eigen::Index k = 0; k < L; ++k) {
          double w = average == "weighted" ? support(k) : 1.0; wsum += w;
          auto rr = prf(tp(k), fp(k), fn(k));
          for (int c = 0; c < 3; ++c) { def[c] = def[c] && rr[c].def; acc[c] += w * rr[c].v; }
        }
        for (int c = 0; c < 3; ++c) triple.push_back(T{acc[c] / wsum, def[c]});
      }
      const char* names[3] = {"precision", "recall", "f1"};
      for (int c = 0; c < 3; ++c) {
        if (!want(names[c])) continue;
        if (triple[c].def) vals.push_back({names[c], triple[c].v});
        else undefined.push_back({names[c], "zero denominator (no predicted or no actual positives); set zero_division to 0 or 1 to define it"});
      }
    }
    if (want("confusion_matrix")) r->mat("confusion_matrix", conf);
    r->vec("labels", labels);
    r->str("average", average);
  }
  if (scored) {
    VectorXd score = to_eigen(*ys);
    if (!score.allFinite()) fail(op + ": y_score must contain only finite values (BAD_DATA)");
    if ((want("log_loss") || want("brier")) && (score.minCoeff() < 0 || score.maxCoeff() > 1)) fail(op + ": y_score must be a probability in [0, 1] for log_loss and brier (BAD_DATA)");
    std::vector<double> classes = sorted_unique(yt);
    if (has_pos && std::find(classes.begin(), classes.end(), pos) == classes.end()) fail(op + ": explicit pos_label " + jnum(pos) + " does not occur among the true labels (BAD_PARAM)");
    if (!has_pos) {
      if (classes.size() <= 2) for (double l : classes) if (l != 0 && l != 1) fail(op + ": scored metrics need pos_label for labels other than 0/1 (BAD_PARAM)");
      pos = 1;
    }
    if (!r->find("pos_label")) r->num("pos_label", pos);
    VectorXd target(n); for (Eigen::Index i = 0; i < n; ++i) target(i) = yt(i) == pos ? 1.0 : 0.0;
    if (classes.size() > 2) {
      for (const char* m : {"roc_auc", "log_loss", "brier"}) if (want(m)) undefined.push_back({m, "binary scores need y_true with at most two classes (found " + std::to_string(classes.size()) + ")"});
    } else {
      if (want("roc_auc")) {
        double a = roc_auc(target, score);
        if (std::isnan(a)) undefined.push_back({"roc_auc", "y_true holds only one class"}); else vals.push_back({"roc_auc", a});
      }
      if (want("log_loss")) {
        const double eps = 1e-15; double s = 0;
        for (Eigen::Index i = 0; i < n; ++i) { double q = std::min(std::max(score(i), eps), 1 - eps); s += target(i) * std::log(q) + (1 - target(i)) * std::log1p(-q); }
        vals.push_back({"log_loss", -s / double(n)});
        r->num("log_loss_eps", eps);
      }
      if (want("brier")) vals.push_back({"brier", (score - target).squaredNorm() / double(n)});
    }
  }
  if (regression) {
    // pycli accumulates in long double; so does this
    long double sse = 0, sae = 0, mean = 0, sst = 0, sape = 0; bool constant = true, zero = false; int zeros = 0;
    for (Eigen::Index i = 0; i < n; ++i) mean += (long double)yt(i);
    mean /= (long double)n;
    for (Eigen::Index i = 0; i < n; ++i) {
      long double e = (long double)yt(i) - (long double)yp(i);
      sse += e * e; sae += std::fabs(e); sst += ((long double)yt(i) - mean) * ((long double)yt(i) - mean);
      constant &= yt(i) == yt(0);
      if (yt(i) == 0) { zero = true; ++zeros; } else sape += std::fabs(e / (long double)yt(i));
    }
    if (want("r2")) { if (constant) undefined.push_back({"r2", "y_true is constant (zero total variance)"}); else vals.push_back({"r2", double(1.0L - sse / sst)}); }
    if (want("mse")) vals.push_back({"mse", double(sse / (long double)n)});
    if (want("rmse")) vals.push_back({"rmse", double(std::sqrt(sse / (long double)n))});
    if (want("mae")) vals.push_back({"mae", double(sae / (long double)n)});
    if (want("mape")) { if (zero) undefined.push_back({"mape", "y_true contains " + std::to_string(zeros) + " zero(s)"}); else vals.push_back({"mape", double(sape / (long double)n)}); }
  }
  for (auto& m : wanted)
    for (auto& v : vals) if (v.first == m) {
      if (!std::isfinite(v.second)) undefined.push_back({m, "result is outside representable float64 range"});
      else r->num(m, v.second);
    }
  auto und = std::make_shared<Res>();
  for (auto& m : wanted) for (auto& u : undefined) if (u.first == m) und->str(m, u.second);
  r->sub("undefined", und);
  r->strs("metrics", wanted);
  r->str("status", undefined.empty() ? "OK" : "UNDEFINED");
  return r;
}

}  // namespace ml
}  // namespace lmpy
