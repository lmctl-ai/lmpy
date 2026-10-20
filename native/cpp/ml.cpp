// Machine learning (scikit-learn 1.7.2 subset): pycli ml_fit, ml_predict, ml_cv and ml_metrics.
// Estimators live in ml_fit.cpp, metrics in ml_metrics.cpp; this file holds cross-validation, the model file format and
// the DSL registration.
#include "frame.hpp"
#include "ml.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <set>

namespace lmnum {
namespace ml {
namespace {

const std::string Et = "Estimator", Mt = "Mat", Vt = "Vec", Rt = "Res", Ot = "Opts", St = "StrVec";

Estimator& est(void* s) { return *static_cast<Estimator*>(s); }
VectorXd ev(const Value& v) { return to_eigen(obj<Vec>(v).v); }
const MatrixXd& em(const Value& v) { return obj<Mat>(v).m; }

Estimator* fit_call(const std::string& kind, const MatrixXd& x, const VectorXd* y, const Opts& o) {
  Hyper h = read_hyper(kind, o, "ml_fit");
  o.finish("ml_fit(" + kind + ")");
  return fit(kind, x, y, h);
}

// ------------------------------------------------------------------------------------------------ cross-validation
// Fold assignment exactly as sklearn.model_selection: KFold (first n % k folds one larger), StratifiedKFold (classes
// encoded by first appearance, round-robin allocation over the sorted codes, per-class blocks, per-class shuffle),
// GroupKFold (largest groups first into the lightest fold), TimeSeriesSplit; shuffles use numpy RandomState(seed).
struct Split { std::vector<Eigen::Index> train, test; };

std::vector<Split> make_splits(const std::string& cv, int64_t k, bool shuffle, int64_t seed, const VectorXd& y, const std::vector<double>* groups, bool classification) {
  const Eigen::Index n = y.size();
  std::vector<Split> out;
  auto from_fold = [&](const std::vector<int>& fold, int64_t nf) {
    for (int64_t f = 0; f < nf; ++f) {
      Split s;
      for (Eigen::Index i = 0; i < n; ++i) (fold[i] == f ? s.test : s.train).push_back(i);
      out.push_back(std::move(s));
    }
  };
  if (cv == "kfold") {
    if (k > n) fail("ml_cv: n_splits " + std::to_string(k) + " exceeds the number of rows " + std::to_string(n));
    std::vector<Eigen::Index> idx(n); for (Eigen::Index i = 0; i < n; ++i) idx[i] = i;
    if (shuffle) { RandomState rng{uint32_t(seed)}; rng.shuffle(idx); }
    std::vector<int> fold(n); Eigen::Index cur = 0;
    for (int64_t f = 0; f < k; ++f) {
      Eigen::Index size = n / k + (f < n % k ? 1 : 0);
      for (Eigen::Index j = cur; j < cur + size; ++j) fold[idx[j]] = int(f);
      cur += size;
    }
    from_fold(fold, k);
  } else if (cv == "stratified") {
    if (!classification) fail("ml_cv: cv=stratified needs a classification kind");
    // encode classes by order of first appearance
    std::vector<double> order; std::vector<int> enc(n);
    for (Eigen::Index i = 0; i < n; ++i) {
      auto it = std::find(order.begin(), order.end(), y(i));
      if (it == order.end()) { order.push_back(y(i)); enc[i] = int(order.size() - 1); } else enc[i] = int(it - order.begin());
    }
    const int K = int(order.size());
    std::vector<int64_t> counts(K, 0); for (int c : enc) ++counts[c];
    if (*std::max_element(counts.begin(), counts.end()) < k) fail("ml_cv: n_splits " + std::to_string(k) + " exceeds every class count");
    std::vector<int> sorted_enc = enc; std::sort(sorted_enc.begin(), sorted_enc.end());
    std::vector<std::vector<int64_t>> alloc(k, std::vector<int64_t>(K, 0));
    for (int64_t f = 0; f < k; ++f) for (size_t i = size_t(f); i < sorted_enc.size(); i += size_t(k)) ++alloc[f][sorted_enc[i]];
    RandomState rng{uint32_t(seed)};
    std::vector<int> fold(n);
    for (int c = 0; c < K; ++c) {
      std::vector<int> ff;
      for (int64_t f = 0; f < k; ++f) for (int64_t t = 0; t < alloc[f][c]; ++t) ff.push_back(int(f));
      if (shuffle) rng.shuffle(ff);
      size_t j = 0;
      for (Eigen::Index i = 0; i < n; ++i) if (enc[i] == c) fold[i] = ff[j++];
    }
    from_fold(fold, k);
  } else if (cv == "group") {
    if (!groups) fail("ml_cv: cv=group needs the option groups (a Vec with one group id per row)");
    if (Eigen::Index(groups->size()) != n) fail("ml_cv: groups has " + std::to_string(groups->size()) + " values; x has " + std::to_string(n) + " rows");
    if (shuffle) fail("ml_cv: cv=group does not support shuffle in lmnum");
    std::vector<double> ug(groups->begin(), groups->end()); std::sort(ug.begin(), ug.end()); ug.erase(std::unique(ug.begin(), ug.end()), ug.end());
    if (int64_t(ug.size()) < k) fail("ml_cv: group CV needs at least n_splits distinct groups (found " + std::to_string(ug.size()) + ")");
    std::vector<int> gi(n); std::vector<int64_t> cnt(ug.size(), 0);
    for (Eigen::Index i = 0; i < n; ++i) { gi[i] = int(std::lower_bound(ug.begin(), ug.end(), (*groups)[i]) - ug.begin()); ++cnt[gi[i]]; }
    std::vector<int> ord(ug.size()); for (size_t i = 0; i < ord.size(); ++i) ord[i] = int(i);
    std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return cnt[a] < cnt[b]; });
    std::reverse(ord.begin(), ord.end());
    std::vector<double> load(k, 0.0); std::vector<int> g2f(ug.size());
    for (int g : ord) { int64_t light = std::min_element(load.begin(), load.end()) - load.begin(); load[light] += double(cnt[g]); g2f[g] = int(light); }
    std::vector<int> fold(n); for (Eigen::Index i = 0; i < n; ++i) fold[i] = g2f[gi[i]];
    from_fold(fold, k);
  } else if (cv == "timeseries") {
    if (shuffle) fail("ml_cv: cv=timeseries does not support shuffle");
    Eigen::Index test_size = n / (k + 1);
    if (test_size < 1 || n - k * test_size <= 0) fail("ml_cv: too many splits (" + std::to_string(k) + ") for " + std::to_string(n) + " rows in a time-series split");
    for (Eigen::Index start = n - k * test_size; start < n; start += test_size) {
      Split s; for (Eigen::Index i = 0; i < start; ++i) s.train.push_back(i); for (Eigen::Index i = start; i < start + test_size; ++i) s.test.push_back(i);
      out.push_back(std::move(s));
    }
  } else fail("ml_cv: cv must be kfold, stratified, group or timeseries; got '" + cv + "'");
  return out;
}

MatrixXd rows_of(const MatrixXd& x, const std::vector<Eigen::Index>& idx) {
  MatrixXd r(Eigen::Index(idx.size()), x.cols()); for (size_t i = 0; i < idx.size(); ++i) r.row(Eigen::Index(i)) = x.row(idx[i]); return r;
}
VectorXd rows_of(const VectorXd& y, const std::vector<Eigen::Index>& idx) {
  VectorXd r(Eigen::Index(idx.size())); for (size_t i = 0; i < idx.size(); ++i) r(Eigen::Index(i)) = y(idx[i]); return r;
}

Res* cross_validate(const std::string& kind, const MatrixXd& x, const VectorXd& y, int64_t k, const Opts& o) {
  const std::string task = task_of(kind);
  if (task != "regression" && task != "classification") fail("ml_cv: '" + kind + "' is unsupervised; cross-validation needs a classifier or regressor (pycli defers unsupervised CV)");
  const bool cls = task == "classification";
  std::string cv = o.str("cv", cls ? "stratified" : "kfold");
  bool shuffle = o.flag("shuffle", false);
  int64_t seed = o.integer("seed", 0);
  const std::vector<double>* groups = o.vec("groups");
  const std::vector<std::string>* mopt = o.strs("metrics");
  std::string scoring = o.str("scoring", "");
  Hyper h = read_hyper(kind, o, "ml_cv");
  o.finish("ml_cv(" + kind + ")");
  if (k < 2 || k > 20) fail("ml_cv: n_splits must be in [2, 20]; got " + std::to_string(k));
  if (seed < 0 || seed > int64_t(0xffffffffLL)) fail("ml_cv: seed must be in [0, 2^32 - 1] (numpy RandomState); got " + std::to_string(seed));
  if (x.rows() != y.size()) fail("ml_cv: y has " + std::to_string(y.size()) + " values; x has " + std::to_string(x.rows()) + " rows");
  if (!x.allFinite() || !y.allFinite()) fail("ml_cv: x and y must be finite");
  std::vector<std::string> metrics;
  if (mopt) metrics = *mopt;
  if (!scoring.empty() && std::find(metrics.begin(), metrics.end(), scoring) == metrics.end()) metrics.insert(metrics.begin(), scoring);
  if (metrics.empty()) metrics.push_back(cls ? "accuracy" : "r2");
  if (scoring.empty()) scoring = metrics[0];
  const std::set<std::string> cm = {"accuracy", "f1_macro", "roc_auc", "log_loss"}, rm = {"mse", "mae", "r2"};
  for (auto& m : metrics) {
    if (cls && !cm.count(m)) fail("ml_cv: metric '" + m + "' is not a classifier metric (accuracy, f1_macro, roc_auc, log_loss)");
    if (!cls && !rm.count(m)) fail("ml_cv: metric '" + m + "' is not a regressor metric (mse, mae, r2)");
  }
  std::vector<double> labels = cls ? sorted_unique(y) : std::vector<double>{};
  if (cls && (labels.size() < 2)) fail("ml_cv: classification needs at least 2 classes in y");
  bool has_proba = kind == "logistic" || kind == "gaussian_nb" || kind == "lda" || kind == "knn_classifier";
  if (std::count(metrics.begin(), metrics.end(), "roc_auc") && (labels.size() != 2 || !has_proba)) fail("ml_cv: roc_auc requires a binary target and a predict_proba family (logistic, gaussian_nb, lda, knn_classifier)");
  if (std::count(metrics.begin(), metrics.end(), "log_loss") && !has_proba) fail("ml_cv: log_loss requires a predict_proba family (logistic, gaussian_nb, lda, knn_classifier)");

  std::vector<Split> splits = make_splits(cv, k, shuffle, seed, y, groups, cls);
  std::map<std::string, std::vector<double>> scores;
  std::vector<double> train_rows, test_rows, test_fold(size_t(y.size()), -1.0);
  auto* r = make<Res>();
  std::string status = "OK";
  auto stop = [&](const std::string& st, size_t f, const std::string& msg, const std::string& metric) {
    status = st;
    r->integer("fold_index", int64_t(f)).str("message", msg);
    if (!metric.empty()) r->str("metric", metric);
  };
  for (size_t f = 0; f < splits.size() && status == "OK"; ++f) {
    const Split& s = splits[f];
    MatrixXd xtr = rows_of(x, s.train), xte = rows_of(x, s.test);
    VectorXd ytr = rows_of(y, s.train), yte = rows_of(y, s.test);
    if (cls) {
      std::vector<double> present = sorted_unique(ytr);
      if (present.size() < 2) { stop("INSUFFICIENT_CLASSES", f, "training fold has fewer than two classes; consider stratified CV or shuffle", ""); break; }
      if (present.size() != labels.size() && std::count(metrics.begin(), metrics.end(), "log_loss")) { stop("UNDEFINED_METRIC", f, "training fold lacks classes; consider stratified CV or shuffle", "log_loss"); break; }
    }
    Estimator* e = fit(kind, xtr, &ytr, h);
    if (e->status == "NO_CONVERGENCE") { stop("NO_CONVERGENCE", f, "the estimator did not converge on this training fold", ""); break; }
    VectorXd pred = predict(*e, xte);
    std::map<std::string, double> fs;
    for (auto& m : metrics) {
      double sc = NAN;
      if (m == "accuracy") sc = accuracy(yte, pred);
      else if (m == "f1_macro") sc = f1_macro(yte, pred);
      else if (m == "roc_auc") {
        VectorXd t(yte.size()); for (Eigen::Index i = 0; i < yte.size(); ++i) t(i) = yte(i) == labels[1] ? 1.0 : 0.0;
        if (t.sum() == 0 || t.sum() == double(t.size())) { stop("UNDEFINED_METRIC", f, "held-out fold has only one class; consider stratified CV or shuffle", m); break; }
        sc = roc_auc(t, predict_proba(*e, xte).col(1));
      } else if (m == "log_loss") sc = log_loss(yte, predict_proba(*e, xte), e->v("classes"));
      else if (m == "mse") sc = (yte - pred).squaredNorm() / double(yte.size());
      else if (m == "mae") sc = (yte - pred).cwiseAbs().sum() / double(yte.size());
      else if (m == "r2") sc = r2_score(yte, pred);
      if (!std::isfinite(sc)) { stop("UNDEFINED_METRIC", f, "metric is not finite; consider stratified CV or shuffle", m); break; }
      fs[m] = sc;
    }
    if (status != "OK") break;
    for (auto& kv : fs) scores[kv.first].push_back(kv.second);
    train_rows.push_back(double(s.train.size())); test_rows.push_back(double(s.test.size()));
    for (auto i : s.test) test_fold[size_t(i)] = double(f);
    if (e->status == "RANK_DEFICIENT") status = "OK_RANK_DEFICIENT";
  }
  bool rank_def = status == "OK_RANK_DEFICIENT";
  if (rank_def) status = "RANK_DEFICIENT";
  auto mean_std = [](const std::vector<double>& v, double& m, double& sd) {
    m = NAN; sd = NAN; if (v.empty()) return;
    m = 0; for (double x : v) m += x; m /= double(v.size());
    sd = 0; for (double x : v) sd += (x - m) * (x - m); sd = std::sqrt(sd / double(v.size()));
  };
  auto ms = std::make_shared<Res>(), mm = std::make_shared<Res>(), msd = std::make_shared<Res>();
  for (auto& m : metrics) {
    double a, b; mean_std(scores[m], a, b);
    ms->vec(m, scores[m]); mm->num(m, a); msd->num(m, b);
  }
  double pm, psd; mean_std(scores[scoring], pm, psd);
  r->str("status", status).integer("completed_folds", int64_t(train_rows.size())).integer("n_splits", int64_t(splits.size()))
      .str("cv", cv).flag("shuffle", shuffle).integer("seed", seed).str("scoring", scoring)
      .vec("scores", scores[scoring]).num("mean", pm).num("std", psd)
      .sub("metric_scores", ms).sub("metric_mean", mm).sub("metric_std", msd)
      .vec("train_rows", train_rows).vec("test_rows", test_rows).vec("test_fold", test_fold)
      .flag("preprocessing_fitted_per_fold", true);
  if (shuffle) r->str("generator", "numpy RandomState(MT19937)");
  return r;
}

// ------------------------------------------------------------------------------------------------ model file (JSON)
struct J {
  enum T { NUL, BOOL, NUM, STR, ARR, OBJ } t = NUL;
  bool b = false; double num = 0; std::string s; std::vector<J> arr; std::vector<std::pair<std::string, J>> obj;
  const J* get(const std::string& k) const { for (auto& kv : obj) if (kv.first == k) return &kv.second; return nullptr; }
};
struct JParser {
  const std::string& s; size_t i = 0; std::string path;
  [[noreturn]] void err(const std::string& m) { fail("load_model: " + path + ": invalid model file: " + m + " at byte " + std::to_string(i)); }
  void ws() { while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\r' || s[i] == '\t')) ++i; }
  J value() {
    ws(); if (i >= s.size()) err("unexpected end");
    J j; char c = s[i];
    if (c == '{') {
      j.t = J::OBJ; ++i; ws();
      if (i < s.size() && s[i] == '}') { ++i; return j; }
      for (;;) { ws(); J k = value(); if (k.t != J::STR) err("object key must be a string"); ws(); if (i >= s.size() || s[i] != ':') err("expected ':'"); ++i; j.obj.push_back({k.s, value()}); ws(); if (i < s.size() && s[i] == ',') { ++i; continue; } if (i < s.size() && s[i] == '}') { ++i; return j; } err("expected ',' or '}'"); }
    }
    if (c == '[') {
      j.t = J::ARR; ++i; ws();
      if (i < s.size() && s[i] == ']') { ++i; return j; }
      for (;;) { j.arr.push_back(value()); ws(); if (i < s.size() && s[i] == ',') { ++i; continue; } if (i < s.size() && s[i] == ']') { ++i; return j; } err("expected ',' or ']'"); }
    }
    if (c == '"') {
      j.t = J::STR; ++i;
      while (i < s.size() && s[i] != '"') {
        if (s[i] == '\\') { ++i; if (i >= s.size()) err("bad escape"); char e = s[i]; j.s += e == 'n' ? '\n' : e == 't' ? '\t' : e == 'r' ? '\r' : e; ++i; }
        else j.s += s[i++];
      }
      if (i >= s.size()) err("unterminated string");
      ++i; return j;
    }
    if (s.compare(i, 4, "null") == 0) { i += 4; return j; }
    if (s.compare(i, 4, "true") == 0) { i += 4; j.t = J::BOOL; j.b = true; return j; }
    if (s.compare(i, 5, "false") == 0) { i += 5; j.t = J::BOOL; return j; }
    char* end = nullptr; j.num = std::strtod(s.c_str() + i, &end);
    if (end == s.c_str() + i) err("unexpected character");
    i = size_t(end - s.c_str()); j.t = J::NUM; return j;
  }
};

std::string hyper_json(const Hyper& h) {
  return std::string("{") + "\"fit_intercept\":" + (h.fit_intercept ? "true" : "false") + ",\"alpha\":" + jnum(h.alpha) + ",\"l1_ratio\":" + jnum(h.l1_ratio) +
         ",\"C\":" + jnum(h.C) + ",\"tol\":" + jnum(h.tol) + ",\"var_smoothing\":" + jnum(h.var_smoothing) + ",\"max_iter\":" + std::to_string(h.max_iter) +
         ",\"class_weight\":" + jstr(h.class_weight) + ",\"n_components\":" + std::to_string(h.n_components) + ",\"n_neighbors\":" + std::to_string(h.n_neighbors) +
         ",\"weights\":" + jstr(h.weights) + ",\"n_clusters\":" + std::to_string(h.n_clusters) + ",\"n_init\":" + std::to_string(h.n_init) +
         ",\"seed\":" + std::to_string(h.seed) + ",\"init\":" + jstr(h.init) + ",\"whiten\":" + (h.whiten ? "true" : "false") + ",\"scale\":" + (h.scale ? "true" : "false") + "}";
}

}  // namespace

std::string save_json(const Estimator& e) {
  std::string o = "{\"lmnum_model\":1,\"kind\":" + jstr(e.kind) + ",\"status\":" + jstr(e.status) + ",\"n_features\":" + std::to_string(e.n_features) +
                  ",\"n_train\":" + std::to_string(e.n_train) + ",\"n_iter\":" + std::to_string(e.n_iter) + ",\"rank\":" + std::to_string(e.rank) +
                  ",\"hyper\":" + hyper_json(e.hp) + ",\"nums\":{";
  bool first = true;
  for (auto& kv : e.nums) { o += (first ? "" : ",") + jstr(kv.first) + ":" + jnum(kv.second); first = false; }
  o += "},\"vecs\":{"; first = true;
  for (auto& kv : e.vecs) { o += (first ? "" : ",") + jstr(kv.first) + ":" + jvec(from_eigen(kv.second)); first = false; }
  o += "},\"mats\":{"; first = true;
  for (auto& kv : e.mats) {
    o += (first ? "" : ",") + jstr(kv.first) + ":{\"rows\":" + std::to_string(kv.second.rows()) + ",\"cols\":" + std::to_string(kv.second.cols()) + ",\"data\":" + jmat(kv.second) + "}";
    first = false;
  }
  return o + "}}\n";
}

Estimator* load_json(const std::string& text, const std::string& path) {
  JParser p{text, 0, path};
  J root = p.value();
  auto bad = [&](const std::string& m) -> void { fail("load_model: " + path + ": invalid model file: " + m); };
  if (root.t != J::OBJ || !root.get("lmnum_model") || root.get("lmnum_model")->num != 1) bad("not an lmnum model (version 1)");
  auto num = [&](const J* j, const char* what) { if (!j || (j->t != J::NUM && j->t != J::NUL)) bad(std::string("missing number ") + what); return j->t == J::NUL ? NAN : j->num; };
  auto str = [&](const J* j, const char* what) { if (!j || j->t != J::STR) bad(std::string("missing string ") + what); return j->s; };
  auto flag = [&](const J* j, const char* what) { if (!j || j->t != J::BOOL) bad(std::string("missing bool ") + what); return j->b; };
  auto* e = make<Estimator>();
  e->kind = str(root.get("kind"), "kind"); e->task = task_of(e->kind); e->status = str(root.get("status"), "status");
  e->n_features = int64_t(num(root.get("n_features"), "n_features")); e->n_train = int64_t(num(root.get("n_train"), "n_train"));
  e->n_iter = int64_t(num(root.get("n_iter"), "n_iter")); e->rank = int64_t(num(root.get("rank"), "rank"));
  const J* h = root.get("hyper"); if (!h || h->t != J::OBJ) bad("missing hyper");
  Hyper& hp = e->hp;
  hp.fit_intercept = flag(h->get("fit_intercept"), "fit_intercept"); hp.alpha = num(h->get("alpha"), "alpha"); hp.l1_ratio = num(h->get("l1_ratio"), "l1_ratio");
  hp.C = num(h->get("C"), "C"); hp.tol = num(h->get("tol"), "tol"); hp.var_smoothing = num(h->get("var_smoothing"), "var_smoothing");
  hp.max_iter = int64_t(num(h->get("max_iter"), "max_iter")); hp.class_weight = str(h->get("class_weight"), "class_weight");
  hp.n_components = int64_t(num(h->get("n_components"), "n_components")); hp.n_neighbors = int64_t(num(h->get("n_neighbors"), "n_neighbors"));
  hp.weights = str(h->get("weights"), "weights"); hp.n_clusters = int64_t(num(h->get("n_clusters"), "n_clusters")); hp.n_init = int64_t(num(h->get("n_init"), "n_init"));
  hp.seed = int64_t(num(h->get("seed"), "seed")); hp.init = str(h->get("init"), "init"); hp.whiten = flag(h->get("whiten"), "whiten"); hp.scale = flag(h->get("scale"), "scale");
  const J *nums = root.get("nums"), *vecs = root.get("vecs"), *mats = root.get("mats");
  if (!nums || nums->t != J::OBJ || !vecs || vecs->t != J::OBJ || !mats || mats->t != J::OBJ) bad("missing nums/vecs/mats");
  for (auto& kv : nums->obj) e->nums[kv.first] = num(&kv.second, kv.first.c_str());
  for (auto& kv : vecs->obj) {
    if (kv.second.t != J::ARR) bad("vector " + kv.first + " is not an array");
    VectorXd v(Eigen::Index(kv.second.arr.size())); for (size_t i = 0; i < kv.second.arr.size(); ++i) v(Eigen::Index(i)) = num(&kv.second.arr[i], kv.first.c_str());
    e->vecs[kv.first] = v;
  }
  for (auto& kv : mats->obj) {
    const J& m = kv.second; if (m.t != J::OBJ) bad("matrix " + kv.first + " is not an object");
    Eigen::Index rows = Eigen::Index(num(m.get("rows"), "rows")), cols = Eigen::Index(num(m.get("cols"), "cols"));
    const J* d = m.get("data"); if (!d || d->t != J::ARR || Eigen::Index(d->arr.size()) != rows) bad("matrix " + kv.first + " has the wrong row count");
    MatrixXd x(rows, cols);
    for (Eigen::Index i = 0; i < rows; ++i) {
      const J& row = d->arr[size_t(i)]; if (row.t != J::ARR || Eigen::Index(row.arr.size()) != cols) bad("matrix " + kv.first + " row " + std::to_string(i) + " has the wrong length");
      for (Eigen::Index j = 0; j < cols; ++j) x(i, j) = num(&row.arr[size_t(j)], kv.first.c_str());
    }
    e->mats[kv.first] = x;
  }
  // smoke the reconstructed state on one zero row, as pycli's ml_load does
  MatrixXd z = MatrixXd::Zero(1, e->n_features);
  if (e->kind == "pca" || e->kind == "standard_scaler") transform(*e, z); else predict(*e, z);
  return e;
}

}  // namespace ml

using namespace ml;

void register_ml(Api& api) {
  api.section("Machine learning", "scikit-learn 1.7.2 subset (pycli `ml_fit`, `ml_predict`, `ml_cv`, `ml_metrics`). `ml_fit(kind, x, y)` fits a "
              "supervised `Estimator` (kinds linear, ridge, lasso, elastic_net, logistic, linear_svc, gaussian_nb, lda, knn_classifier, "
              "knn_regressor; classes are the sorted distinct y values) and `ml_fit(kind, x)` an unsupervised one (kmeans, pca, standard_scaler). "
              "Hyperparameters use pycli's ml-spec names and defaults (alpha, l1_ratio, C, tol, max_iter, class_weight, var_smoothing, "
              "n_components, n_neighbors, weights, n_clusters, n_init, init, seed, whiten, fit_intercept) plus `scale` (pycli preprocess.scale). "
              "Estimator status is OK, RANK_DEFICIENT (linear) or NO_CONVERGENCE. Random forests and gradient boosting are not supported. "
              "k-means and shuffled CV draw from numpy's RandomState(seed) (MT19937), as sklearn does.");
  api.type(Et, "A fitted scikit-learn-style model: predict, predict_proba, decision_function, transform, score, params, save.");
  api.fn("ml_fit", {"string", Mt, Vt}, Et, "fit a supervised model (pycli ml_fit): kind, features x (rows x p), target y (class labels or values)", [](Args& a) {
    VectorXd y = ev(a[2]); return O(fit_call(a[0].s, em(a[1]), &y, no_opts())); });
  api.fn("ml_fit", {"string", Mt, Vt, Ot}, Et, "fit a supervised model with hyperparameters and `scale` (standardise features first; learned from the training rows)", [](Args& a) {
    VectorXd y = ev(a[2]); return O(fit_call(a[0].s, em(a[1]), &y, obj<Opts>(a[3]))); });
  api.fn("ml_fit", {"string", Mt}, Et, "fit an unsupervised model: kmeans (2 clusters), pca (2 components) or standard_scaler", [](Args& a) {
    return O(fit_call(a[0].s, em(a[1]), nullptr, no_opts())); });
  api.fn("ml_fit", {"string", Mt, Ot}, Et, "fit an unsupervised model with options: kmeans n_clusters, n_init (10), max_iter (300), tol (1e-8), init (k-means++|random), seed (0); pca n_components, whiten", [](Args& a) {
    return O(fit_call(a[0].s, em(a[1]), nullptr, obj<Opts>(a[2]))); });
  api.fn("kmeans", {Mt, "int"}, Et, "k-means with k clusters (sklearn KMeans, Lloyd, k-means++, n_init 10, seed 0)", [](Args& a) {
    Hyper h = read_hyper("kmeans", no_opts(), "kmeans"); h.n_clusters = a[1].i;
    if (h.n_clusters < 1) fail("kmeans: k must be >= 1");
    return O(fit("kmeans", em(a[0]), nullptr, h)); });
  api.fn("kmeans", {Mt, "int", Ot}, Et, "k-means with options seed, n_init, max_iter, tol, init", [](Args& a) {
    const Opts& o = obj<Opts>(a[2]);
    if (o.has("n_clusters")) fail("kmeans: give k as the second argument, not as n_clusters");
    Hyper h = read_hyper("kmeans", o, "kmeans"); o.finish("kmeans"); h.n_clusters = a[1].i;
    if (h.n_clusters < 1) fail("kmeans: k must be >= 1");
    return O(fit("kmeans", em(a[0]), nullptr, h)); });
  api.fn("load_model", {"string"}, Et, "read a model written by Estimator.save (JSON, format lmnum_model 1)", [](Args& a) {
    return O(load_json(read_file_text(a[0].s), a[0].s)); });

  api.method(Et, "kind", {}, "string", "the family (linear, logistic, kmeans, ...)", [](void* s, Args&) { return S(est(s).kind); });
  api.method(Et, "task", {}, "string", "regression, classification, clustering, decomposition or scaling", [](void* s, Args&) { return S(est(s).task); });
  api.method(Et, "status", {}, "string", "OK, RANK_DEFICIENT (linear: minimum-norm solution kept) or NO_CONVERGENCE", [](void* s, Args&) { return S(est(s).status); });
  api.method(Et, "ok", {}, "bool", "status is OK", [](void* s, Args&) { return B(est(s).status == "OK"); });
  api.method(Et, "predict", {Mt}, Vt, "predictions: class labels, regression values or cluster indexes (pycli ml_predict kind predict)", [](void* s, Args& a) {
    return vecv(predict(est(s), em(a[0]))); });
  api.method(Et, "predict_proba", {Mt}, Mt, "class probabilities, one column per class in classes() order (logistic, gaussian_nb, lda, knn_classifier)", [](void* s, Args& a) {
    return matv(predict_proba(est(s), em(a[0]))); });
  api.method(Et, "decision_function", {Mt}, Mt, "decision scores (logistic, linear_svc, lda); binary models give one column, positive for classes()[1]", [](void* s, Args& a) {
    return matv(decision_function(est(s), em(a[0]))); });
  api.method(Et, "transform", {Mt}, Mt, "standard_scaler: standardised x; pca: component scores; lda: discriminant axes; kmeans: distances to the centres", [](void* s, Args& a) {
    return matv(transform(est(s), em(a[0]))); });
  api.method(Et, "score", {Mt, Vt}, "float", "accuracy (classifiers) or R^2 (regressors) on x, y", [](void* s, Args& a) {
    Estimator& e = est(s); VectorXd y = ev(a[1]);
    if (e.task != "classification" && e.task != "regression") fail("Estimator.score: needs a classifier or regressor; this model is " + e.kind);
    VectorXd p = predict(e, em(a[0]));
    if (p.size() != y.size()) fail("Estimator.score: y has " + std::to_string(y.size()) + " values; x has " + std::to_string(p.size()) + " rows");
    return F(e.classifier() ? accuracy(y, p) : r2_score(y, p)); });
  api.method(Et, "params", {}, Rt, "fitted attributes (sklearn names without the trailing underscore): coef, intercept, classes, rank, theta, var, "
             "means, priors, scalings, cluster_centers, inertia, labels, components, mean, explained_variance(_ratio), singular_values, scale, n_iter ...",
             [](void* s, Args&) { return O(params(est(s))); });
  api.method(Et, "classes", {}, Vt, "sorted class labels (classifiers)", [](void* s, Args&) {
    if (!est(s).classifier()) fail("Estimator.classes: " + est(s).kind + " is not a classifier");
    return vecv(est(s).v("classes")); });
  api.method(Et, "coef", {}, Mt, "coefficients as a matrix: one row per decision function (regressors and binary classifiers: 1 row)", [](void* s, Args&) {
    Estimator& e = est(s);
    if (e.mats.count("coef")) return matv(e.m("coef"));
    if (e.vecs.count("coef")) return matv(MatrixXd(e.v("coef").transpose()));
    fail("Estimator.coef: " + e.kind + " has no coefficients"); });
  api.method(Et, "intercept", {}, Vt, "intercepts, one per decision function (regressors: one value)", [](void* s, Args&) {
    Estimator& e = est(s);
    if (e.vecs.count("intercept")) return vecv(e.v("intercept"));
    if (e.nums.count("intercept")) return vecv(std::vector<double>{e.n("intercept")});
    fail("Estimator.intercept: " + e.kind + " has no intercept"); });
  api.method(Et, "labels", {}, Vt, "k-means cluster index of every training row", [](void* s, Args&) {
    if (est(s).kind != "kmeans") fail("Estimator.labels: only for kmeans; this model is " + est(s).kind);
    return vecv(est(s).v("labels")); });
  api.method(Et, "centers", {}, Mt, "k-means cluster centres (one row per cluster)", [](void* s, Args&) {
    if (est(s).kind != "kmeans") fail("Estimator.centers: only for kmeans; this model is " + est(s).kind);
    return matv(est(s).m("cluster_centers")); });
  api.method(Et, "inertia", {}, "float", "k-means within-cluster sum of squared distances", [](void* s, Args&) {
    if (est(s).kind != "kmeans") fail("Estimator.inertia: only for kmeans; this model is " + est(s).kind);
    return F(est(s).n("inertia")); });
  api.method(Et, "n_iter", {}, "int", "iterations used (lasso/elastic_net epochs, logistic/linear_svc Newton steps, k-means Lloyd iterations of the best run)", [](void* s, Args&) { return I(est(s).n_iter); });
  api.method(Et, "save", {"string"}, "void", "write the model as JSON (atomic replace); load_model reads it back with identical predictions", [](void* s, Args& a) {
    write_file_atomic(a[0].s, save_json(est(s))); return Value::none(); });
  api.method(Et, "json", {}, "string", "the model file text (JSON)", [](void* s, Args&) { return S(save_json(est(s))); });

  api.fn("ml_cv", {"string", Mt, Vt, "int"}, Rt, "k-fold cross-validation (pycli ml_cv): stratified for classifiers (accuracy), kfold for regressors (r2), no shuffle", [](Args& a) {
    return O(cross_validate(a[0].s, em(a[1]), ev(a[2]), a[3].i, no_opts())); });
  api.fn("ml_cv", {"string", Mt, Vt, "int", Ot}, Rt, "cross-validation with options cv (kfold|stratified|group|timeseries), shuffle, seed, groups (Vec), "
         "metrics (StrVec: accuracy, f1_macro, roc_auc, log_loss | mse, mae, r2), scoring, and the kind's hyperparameters and scale. Result: status, scores, "
         "mean, std (scoring metric), metric_scores/metric_mean/metric_std (per metric), train_rows, test_rows, test_fold, completed_folds", [](Args& a) {
    return O(cross_validate(a[0].s, em(a[1]), ev(a[2]), a[3].i, obj<Opts>(a[4]))); });

  api.fn("ml_metrics", {Vt, Vt}, Rt, "metrics of predictions (pycli ml_metrics): classification (integral labels, <= 30) gives accuracy, balanced_accuracy, "
         "precision, recall, f1, confusion_matrix, labels; otherwise r2, mse, rmse, mae. status OK or UNDEFINED (see undefined)", [](Args& a) {
    VectorXd p = ev(a[1]); return O(metrics(ev(a[0]), &p, no_opts())); });
  api.fn("ml_metrics", {Vt, Vt, Ot}, Rt, "metrics with options metrics (StrVec), kind (auto|classification|regression), average (binary|macro|micro|weighted), "
         "pos_label, zero_division (undefined|0|1), y_score (Vec: roc_auc, log_loss, brier)", [](Args& a) {
    VectorXd p = ev(a[1]); return O(metrics(ev(a[0]), &p, obj<Opts>(a[2]))); });
  api.fn("ml_metrics", {Vt, Ot}, Rt, "metrics from y_true and options only (y_pred and/or y_score given as options)", [](Args& a) {
    return O(metrics(ev(a[0]), nullptr, obj<Opts>(a[1]))); });
  api.fn("adjusted_rand_score", {Vt, Vt}, "float", "adjusted Rand index of two labelings (sklearn.metrics.adjusted_rand_score; 1 = same partition)", [](Args& a) {
    return F(adjusted_rand(ev(a[0]), ev(a[1]))); });
  api.fn("label_encode", {St}, Rt, "text labels to codes (sklearn LabelEncoder): fields codes (Vec, 0-based index into classes), classes (sorted StrVec)", [](Args& a) {
    const auto& v = obj<StrVec>(a[0]).v;
    std::vector<std::string> cl(v); std::sort(cl.begin(), cl.end()); cl.erase(std::unique(cl.begin(), cl.end()), cl.end());
    std::vector<double> codes; for (auto& s : v) codes.push_back(double(std::lower_bound(cl.begin(), cl.end(), s) - cl.begin()));
    auto* r = make<Res>(); r->vec("codes", codes).strs("classes", cl); return O(r); });
}

}  // namespace lmnum
