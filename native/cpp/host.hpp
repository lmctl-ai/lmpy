// lmpy host infrastructure: one definition per DSL member feeds the dslvm registry, the published schema
// (`lmpy --dsl-schema --json`) and the API reference (`lmpy --api-md`), so the three cannot drift.
#pragma once
#include "dslvm.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace lmpy {
using dslvm::Args;
using dslvm::Value;

// A host failure inside a DSL call: becomes {"ok": false, "phase": "run", "code": "RUN_HOST", ...} with the DSL line.
[[noreturn]] inline void fail(const std::string& message) { throw dslvm::Error("RUN_HOST", message); }

// ---- Host objects -----------------------------------------------------------------------------------------------
// Every object a DSL call returns lives in one arena for the rest of the process (the DSL has no destructor and a
// run is one short process). Objects are immutable unless a method is documented as "in place".
struct Obj { virtual ~Obj() = default; };
std::vector<std::unique_ptr<Obj>>& arena();
template <class T, class... A> T* make(A&&... a) {
  auto p = std::make_unique<T>(std::forward<A>(a)...);
  T* raw = p.get();
  arena().push_back(std::move(p));
  return raw;
}

struct Vec : Obj { static constexpr const char* TYPE = "Vec"; std::vector<double> v; Vec() = default; explicit Vec(std::vector<double> x) : v(std::move(x)) {} };
struct StrVec : Obj { static constexpr const char* TYPE = "StrVec"; std::vector<std::string> v; StrVec() = default; explicit StrVec(std::vector<std::string> x) : v(std::move(x)) {} };
struct Mat : Obj { static constexpr const char* TYPE = "Mat"; Eigen::MatrixXd m; Mat() = default; explicit Mat(Eigen::MatrixXd x) : m(std::move(x)) {} };

// A typed value used by Opts and Res.
struct Item {
  enum Kind { NUM, INT, STR, BOOL, VEC, STRS, MAT, RES } kind = NUM;
  double f = 0; int64_t i = 0; std::string s; bool b = false;
  std::vector<double> vec; std::vector<std::string> strs; Eigen::MatrixXd mat; std::shared_ptr<struct Res> res;
};
const char* item_kind_name(Item::Kind k);

// Options builder: opts().str("method", "BFGS").num("tol", 1e-8). An operation reads the keys it knows and then calls
// finish(), which refuses any key it did not read, so a misspelt option is an error rather than a silent default.
struct Opts : Obj {
  static constexpr const char* TYPE = "Opts";
  std::vector<std::pair<std::string, Item>> items;
  mutable std::vector<bool> used;
  const Item* find(const std::string& key) const;
  bool has(const std::string& key) const { return find(key) != nullptr; }
  double num(const std::string& key, double def) const;      // accepts num or int
  int64_t integer(const std::string& key, int64_t def) const;
  std::string str(const std::string& key, const std::string& def) const;
  bool flag(const std::string& key, bool def) const;
  const std::vector<double>* vec(const std::string& key) const;  // null when absent
  const std::vector<std::string>* strs(const std::string& key) const;
  const Eigen::MatrixXd* mat(const std::string& key) const;
  void finish(const std::string& op) const;
  void put(const std::string& key, Item item);
};
// The empty options object used by overloads without an Opts argument.
const Opts& no_opts();

// Generic operation result: typed, ordered fields read with r.num("key"), r.vec("key") ... (a missing key or a wrong kind
// is a run-time error naming the available keys).
struct Res : Obj {
  static constexpr const char* TYPE = "Res";
  std::vector<std::pair<std::string, Item>> items;
  Res& num(const std::string& k, double v) { Item it; it.kind = Item::NUM; it.f = v; return put(k, std::move(it)); }
  Res& integer(const std::string& k, int64_t v) { Item it; it.kind = Item::INT; it.i = v; return put(k, std::move(it)); }
  Res& str(const std::string& k, std::string v) { Item it; it.kind = Item::STR; it.s = std::move(v); return put(k, std::move(it)); }
  Res& flag(const std::string& k, bool v) { Item it; it.kind = Item::BOOL; it.b = v; return put(k, std::move(it)); }
  Res& vec(const std::string& k, std::vector<double> v) { Item it; it.kind = Item::VEC; it.vec = std::move(v); return put(k, std::move(it)); }
  Res& vec(const std::string& k, const Eigen::VectorXd& v) { return vec(k, std::vector<double>(v.data(), v.data() + v.size())); }
  Res& strs(const std::string& k, std::vector<std::string> v) { Item it; it.kind = Item::STRS; it.strs = std::move(v); return put(k, std::move(it)); }
  Res& mat(const std::string& k, Eigen::MatrixXd m) { Item it; it.kind = Item::MAT; it.mat = std::move(m); return put(k, std::move(it)); }
  Res& sub(const std::string& k, std::shared_ptr<Res> r) { Item it; it.kind = Item::RES; it.res = std::move(r); return put(k, std::move(it)); }
  Res& put(const std::string& k, Item it);
  const Item& get(const std::string& k, Item::Kind want) const;
  const Item* find(const std::string& k) const;
  std::string keys_text() const;
};

// ---- JSON text ----------------------------------------------------------------------------------------------------
std::string jnum(double v);  // shortest round-trip text; NaN and infinities become null (JSON has no such numbers)
std::string jstr(const std::string& s);
std::string jvec(const std::vector<double>& v);
std::string jstrs(const std::vector<std::string>& v);
std::string jmat(const Eigen::MatrixXd& m);  // row-major nested arrays
std::string jitem(const Item& it);
std::string jres(const Res& r);

// ---- Value helpers --------------------------------------------------------------------------------------------------
inline Value F(double v) { return Value::real(v); }
inline Value I(int64_t v) { return Value::integer(v); }
inline Value S(std::string v) { return Value::string(std::move(v)); }
inline Value B(bool v) { return Value::boolean(v); }
template <class T> Value O(T* p) { return Value::object(T::TYPE, p); }
template <class T> T& obj(const Value& v) { return *static_cast<T*>(v.obj); }
inline Value vecv(std::vector<double> v) { return O(make<Vec>(std::move(v))); }
inline Value vecv(const Eigen::VectorXd& v) { return vecv(std::vector<double>(v.data(), v.data() + v.size())); }
inline Value matv(Eigen::MatrixXd m) { return O(make<Mat>(std::move(m))); }
inline Value strsv(std::vector<std::string> v) { return O(make<StrVec>(std::move(v))); }
inline Value resv(Res* r) { return O(r); }
Eigen::VectorXd to_eigen(const std::vector<double>& v);
std::vector<double> from_eigen(const Eigen::VectorXd& v);
size_t index_arg(int64_t i, size_t n, const char* what);  // negative counts from the end; out of range is an error

// ---- API builder --------------------------------------------------------------------------------------------------
using Fn = std::function<Value(Args&)>;
using Meth = std::function<Value(void*, Args&)>;
struct Api {
  struct Overload { std::vector<std::string> params; std::string returns; std::string doc; std::string section; };
  struct TypeDef { std::string doc; std::string section; std::map<std::string, std::vector<Overload>> methods; std::map<std::string, std::string> operators; };
  dslvm::Registry reg;
  std::map<std::string, TypeDef> types;
  std::map<std::string, std::vector<Overload>> functions;
  std::map<std::string, std::string> context;
  std::map<std::string, std::vector<std::string>> enums;
  struct Conversion { std::string from, to, via; };
  std::vector<Conversion> conversions;
  std::vector<std::string> sections;  // in registration order
  std::map<std::string, std::string> section_docs;
  std::string current;

  void section(const std::string& name, const std::string& doc);
  void type(const std::string& name, const std::string& doc);
  void fn(const std::string& name, std::vector<std::string> params, const std::string& returns, const std::string& doc, Fn f);
  void method(const std::string& type, const std::string& name, std::vector<std::string> params, const std::string& returns, const std::string& doc, Meth m);
  void op(const std::string& type, const std::string& op, const std::string& method);
  void conversion(const std::string& from, const std::string& to, const std::string& via);
  void enumeration(const std::string& name, std::vector<std::string> values);
  void ctx(const std::string& name, const std::string& type, const std::string& doc);
  std::string schema_json() const;
  std::string summary() const;
  std::string markdown() const;
};

// Family registration entry points: one function per source file, called in this order by main.cpp
// (families.cpp lists the families beyond core/frame/linalg).
void register_core(Api&);
void register_frame(Api&);
void register_frame_ops(Api&);  // frame_ops.cpp: groupby, merge, rolling, resample, array files
void register_linalg(Api&);
void register_families(Api&);
void register_sparse(Api&);     // sparse.cpp
void register_stats(Api&);      // stats.cpp: distributions, special functions, random numbers, tests
void register_optimize(Api&);   // optimize.cpp: minimize, least squares, roots, quadrature, linear programming
void register_ode(Api&);        // ode.cpp: initial value problems
void register_interp(Api&);     // interp.cpp: splines and interpolators
void register_signal(Api&);     // signal.cpp: filters, spectra, peaks, FFT
void register_spatial(Api&);    // spatial.cpp: k-nearest neighbours, distances, convex hulls
void register_graph(Api&);      // graph.cpp: shortest paths and flows
void register_models(Api&);     // models.cpp: OLS, GLM, ARIMA, time-series tests
void register_ml(Api&);         // ml.cpp: supervised and unsupervised models, cross-validation, metrics
}  // namespace lmpy
