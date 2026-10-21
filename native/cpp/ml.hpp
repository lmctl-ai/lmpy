// Machine learning family (scikit-learn 1.7.2 subset): shared declarations for ml.cpp, ml_fit.cpp and ml_metrics.cpp.
#pragma once
#include "host.hpp"
#include <map>

namespace lmpy {
namespace ml {

using Eigen::MatrixXd;
using Eigen::VectorXd;

// numpy's legacy RandomState(seed) bit generator: MT19937 seeded with init_genrand (numpy `mt19937_seed`), plus the
// helpers sklearn reaches through it (random_sample, legacy random_interval used by shuffle). Reimplemented from the
// MT19937 paper (Matsumoto & Nishimura 1998) and numpy's BSD-licensed distributions code, so that KFold/StratifiedKFold
// shuffles and k-means++ seeding draw the same numbers as sklearn with the same integer seed.
struct RandomState {
  uint32_t mt[624];
  int pos = 624;
  explicit RandomState(uint32_t seed);
  uint32_t next32();
  uint64_t next64() { uint64_t hi = next32(); return (hi << 32) | next32(); }
  double random_sample() { int32_t a = int32_t(next32() >> 5), b = int32_t(next32() >> 6); return (a * 67108864.0 + b) / 9007199254740992.0; }
  uint64_t interval(uint64_t max);  // uniform integer in [0, max] (numpy random_interval, masked rejection)
  template <class T> void shuffle(std::vector<T>& v) {  // RandomState.shuffle on a 1-D array
    for (size_t i = v.size(); i-- > 1;) { size_t j = size_t(interval(i)); std::swap(v[i], v[j]); }
  }
};

// Hyperparameters of every family (pycli ml-spec.json names and defaults).
struct Hyper {
  bool fit_intercept = true;
  double alpha = 1.0, l1_ratio = 0.5, C = 1.0, tol = 1e-8, var_smoothing = 1e-9;
  int64_t max_iter = -1;  // family default when -1
  std::string class_weight = "none";
  int64_t n_components = -1;  // pca default 2, lda 0 (= auto)
  int64_t n_neighbors = 5;
  std::string weights = "uniform";
  int64_t n_clusters = 2, n_init = 10, seed = 0;
  std::string init = "k-means++";
  bool whiten = false;
  bool scale = false;  // pycli preprocess.scale: standardise features with training mean / population sd first
};

// A fitted estimator. Numeric state lives in named vectors/matrices/numbers so that params(), save and load are generic.
struct Estimator : Obj {
  static constexpr const char* TYPE = "Estimator";
  std::string kind, task, status = "OK";
  Hyper hp;
  int64_t n_features = 0, n_train = 0, n_iter = 0, rank = -1;
  std::map<std::string, VectorXd> vecs;
  std::map<std::string, MatrixXd> mats;
  std::map<std::string, double> nums;
  const VectorXd& v(const std::string& k) const;
  const MatrixXd& m(const std::string& k) const;
  double n(const std::string& k) const;
  bool classifier() const { return task == "classification"; }
};

// family table: kind -> task ("regression", "classification", "clustering", "decomposition", "scaling")
std::string task_of(const std::string& kind);  // fails for an unknown kind, listing the known ones
Hyper read_hyper(const std::string& kind, const Opts& o, const std::string& op);

Estimator* fit(const std::string& kind, const MatrixXd& x, const VectorXd* y, const Hyper& hp);
MatrixXd prepare(const Estimator& e, const MatrixXd& x, const char* op);  // shape check + optional scaling
VectorXd predict(const Estimator& e, const MatrixXd& x);
MatrixXd predict_proba(const Estimator& e, const MatrixXd& x);
MatrixXd decision_function(const Estimator& e, const MatrixXd& x);
MatrixXd transform(const Estimator& e, const MatrixXd& x);
Res* params(const Estimator& e);
std::string save_json(const Estimator& e);
Estimator* load_json(const std::string& text, const std::string& path);

// metrics (ml_metrics.cpp)
double accuracy(const VectorXd& t, const VectorXd& p);
double r2_score(const VectorXd& t, const VectorXd& p);  // NaN for a constant target
double f1_macro(const VectorXd& t, const VectorXd& p);  // sklearn f1_score(average="macro", zero_division=0)
double roc_auc(const VectorXd& target01, const VectorXd& score);  // rank statistic, ties count 1/2; NaN with one class
double log_loss(const VectorXd& t, const MatrixXd& proba, const VectorXd& classes);  // sklearn log_loss (eps clip)
double adjusted_rand(const VectorXd& a, const VectorXd& b);
Res* metrics(const VectorXd& yt, const VectorXd* yp, const Opts& o);
std::vector<double> sorted_unique(const VectorXd& v);

}  // namespace ml
}  // namespace lmpy
