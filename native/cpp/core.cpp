// Core DSL types: Vec, StrVec, Mat, Opts, Res and the `out` context object.
#include "expr.hpp"
#include "host.hpp"
#include "out.hpp"
#include <algorithm>
#include <cstdio>
#include <iostream>
#include <numeric>
#include <sstream>

namespace lmnum {

Out& the_out() { static Out o; return o; }

namespace {

using V = std::vector<double>;
const std::string Vt = "Vec", Mt = "Mat", St = "StrVec", Ot = "Opts", Rt = "Res";

V& vv(void* self) { return static_cast<Vec*>(self)->v; }
const V& av(const Value& x) { return obj<Vec>(x).v; }

void same_len(const V& a, const V& b, const char* op) {
  if (a.size() != b.size()) fail(std::string("Vec.") + op + ": lengths differ (" + std::to_string(a.size()) + " and " + std::to_string(b.size()) + ")");
}
template <class F> Value zip(const V& a, const V& b, const char* op, F f) {
  same_len(a, b, op);
  V r(a.size());
  for (size_t i = 0; i < a.size(); ++i) r[i] = f(a[i], b[i]);
  return vecv(std::move(r));
}
template <class F> Value each(const V& a, F f) {
  V r(a.size());
  for (size_t i = 0; i < a.size(); ++i) r[i] = f(a[i]);
  return vecv(std::move(r));
}
void need_nonempty(const V& a, const char* op) {
  if (a.empty()) fail(std::string("Vec.") + op + ": the vector is empty");
}
double vsum(const V& a) {  // Neumaier compensated summation
  double s = 0, c = 0;
  for (double x : a) {
    double t = s + x;
    c += std::fabs(s) >= std::fabs(x) ? (s - t) + x : (x - t) + s;
    s = t;
  }
  return s + c;
}
double vmean(const V& a) { return vsum(a) / double(a.size()); }
double vvar(const V& a, int64_t ddof) {
  if (int64_t(a.size()) <= ddof) fail("Vec.var: need more than ddof values");
  double m = vmean(a), s = 0;
  for (double x : a) s += (x - m) * (x - m);
  return s / double(int64_t(a.size()) - ddof);
}
double quantile_sorted(const V& s, double q) {  // numpy "linear" (Hyndman-Fan type 7)
  if (s.empty()) fail("quantile of an empty vector");
  if (!(q >= 0 && q <= 1)) fail("quantile: q must be in [0, 1]");
  double h = (double(s.size()) - 1) * q;
  size_t lo = size_t(std::floor(h));
  size_t hi = std::min(lo + 1, s.size() - 1);
  return s[lo] + (h - double(lo)) * (s[hi] - s[lo]);
}
V parse_numbers(const std::string& text) {
  V out;
  const char* p = text.c_str();
  while (*p) {
    while (*p && (*p == ' ' || *p == ',' || *p == '[' || *p == ']' || *p == '\n' || *p == '\t' || *p == ';')) ++p;
    if (!*p) break;
    char* e = nullptr;
    if (!std::strncmp(p, "nan", 3) || !std::strncmp(p, "NaN", 3)) { out.push_back(NAN); p += 3; continue; }
    double v = std::strtod(p, &e);
    if (e == p) fail("cannot parse a number at \"" + std::string(p).substr(0, 20) + "\"");
    out.push_back(v);
    p = e;
  }
  return out;
}
std::vector<std::string> split_text(const std::string& s, const std::string& sep) {
  std::vector<std::string> out;
  if (s.empty()) return out;
  if (sep.empty()) fail("split: empty separator");
  size_t start = 0;
  for (;;) {
    size_t at = s.find(sep, start);
    if (at == std::string::npos) { out.push_back(s.substr(start)); break; }
    out.push_back(s.substr(start, at - start));
    start = at + sep.size();
  }
  return out;
}
std::string trim(const std::string& s) {
  size_t a = s.find_first_not_of(" \t\n\r"), b = s.find_last_not_of(" \t\n\r");
  return a == std::string::npos ? "" : s.substr(a, b - a + 1);
}
Eigen::MatrixXd& mm(void* self) { return static_cast<Mat*>(self)->m; }
const Eigen::MatrixXd& am(const Value& x) { return obj<Mat>(x).m; }
void need_shape(bool ok, const std::string& what) { if (!ok) fail(what); }
std::string shape(const Eigen::MatrixXd& m) { return std::to_string(m.rows()) + "x" + std::to_string(m.cols()); }

Value item_value(const Item& it, Item::Kind want) {
  switch (want) {
    case Item::NUM: return F(it.kind == Item::INT ? double(it.i) : it.f);
    case Item::INT: return I(it.i);
    case Item::STR: return S(it.s);
    case Item::BOOL: return B(it.b);
    case Item::VEC: return vecv(it.vec);
    case Item::STRS: return strsv(it.strs);
    case Item::MAT: return matv(it.mat);
    case Item::RES: { auto* r = make<Res>(*it.res); return O(r); }
  }
  return Value::none();
}

}  // namespace

std::string fmt_fixed(double x, int64_t digits) {
  if (digits < 0 || digits > 40) fail("fmt: digits must be in [0, 40]");
  char buf[512];
  std::snprintf(buf, sizeof buf, "%.*f", int(digits), x);
  return buf;
}

void register_core(Api& api) {
  // ---------------------------------------------------------------------------------------------------- Vec
  api.section("Vectors", "`Vec` is a 1-D float64 array (numpy `ndarray` of rank 1). Methods return new vectors unless marked *in place*. "
                         "Arithmetic operators are elementwise; mixing a Vec with a float broadcasts the float.");
  api.type("Vec", "1-D float64 array. Operators: `+`, `-`, `*` elementwise (Vec or float on either side), unary `-`.");
  api.fn("vec", {}, Vt, "empty vector", [](Args&) { return vecv(V{}); });
  api.fn("vec", {"float"}, Vt, "one-element vector", [](Args& a) { return vecv(V{a[0].f}); });
  api.fn("vec", {"float", "float"}, Vt, "two-element vector", [](Args& a) { return vecv(V{a[0].f, a[1].f}); });
  api.fn("vec", {"float", "float", "float"}, Vt, "three-element vector", [](Args& a) { return vecv(V{a[0].f, a[1].f, a[2].f}); });
  api.fn("vec", {"float", "float", "float", "float"}, Vt, "four-element vector", [](Args& a) { return vecv(V{a[0].f, a[1].f, a[2].f, a[3].f}); });
  api.fn("vec", {"string"}, Vt, "parse numbers separated by spaces/commas (brackets allowed, `nan` accepted): `vec(\"1 2.5 -3\")`",
         [](Args& a) { return vecv(parse_numbers(a[0].s)); });
  api.fn("zeros", {"int"}, Vt, "n zeros", [](Args& a) { if (a[0].i < 0) fail("zeros: negative length"); return vecv(V(size_t(a[0].i), 0.0)); });
  api.fn("ones", {"int"}, Vt, "n ones", [](Args& a) { if (a[0].i < 0) fail("ones: negative length"); return vecv(V(size_t(a[0].i), 1.0)); });
  api.fn("full", {"int", "float"}, Vt, "n copies of a value", [](Args& a) { if (a[0].i < 0) fail("full: negative length"); return vecv(V(size_t(a[0].i), a[1].f)); });
  api.fn("linspace", {"float", "float", "int"}, Vt, "n evenly spaced values from start to stop inclusive (numpy.linspace)", [](Args& a) {
    int64_t n = a[2].i;
    if (n < 0) fail("linspace: negative count");
    V r(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) r[size_t(i)] = n == 1 ? a[0].f : a[0].f + (a[1].f - a[0].f) * double(i) / double(n - 1);
    if (n > 1) r.back() = a[1].f;
    return vecv(std::move(r));
  });
  api.fn("arange", {"float", "float", "float"}, Vt, "start, start+step, ... below stop (numpy.arange)", [](Args& a) {
    double start = a[0].f, stop = a[1].f, step = a[2].f;
    if (step == 0 || !std::isfinite(step)) fail("arange: step must be finite and nonzero");
    double count = std::ceil((stop - start) / step);
    if (count > 1e8) fail("arange: more than 1e8 values");
    V r;
    for (int64_t i = 0; i < int64_t(count); ++i) r.push_back(start + double(i) * step);
    return vecv(std::move(r));
  });
  api.fn("arange", {"int"}, Vt, "0, 1, ..., n-1 as floats", [](Args& a) {
    if (a[0].i < 0) fail("arange: negative length");
    V r(size_t(a[0].i));
    std::iota(r.begin(), r.end(), 0.0);
    return vecv(std::move(r));
  });
  api.fn("concat", {Vt, Vt}, Vt, "concatenation", [](Args& a) { V r = av(a[0]); r.insert(r.end(), av(a[1]).begin(), av(a[1]).end()); return vecv(std::move(r)); });

  api.method(Vt, "len", {}, "int", "length", [](void* s, Args&) { return I(int64_t(vv(s).size())); });
  api.method(Vt, "at", {"int"}, "float", "element i (negative counts from the end)", [](void* s, Args& a) { return F(vv(s)[index_arg(a[0].i, vv(s).size(), "Vec.at")]); });
  api.method(Vt, "set", {"int", "float"}, Vt, "*in place*: element i = x; returns the vector", [](void* s, Args& a) {
    vv(s)[index_arg(a[0].i, vv(s).size(), "Vec.set")] = a[1].f; return O(static_cast<Vec*>(s)); });
  api.method(Vt, "push", {"float"}, Vt, "*in place*: append x; returns the vector", [](void* s, Args& a) { vv(s).push_back(a[0].f); return O(static_cast<Vec*>(s)); });
  api.method(Vt, "extend", {Vt}, Vt, "*in place*: append another vector; returns the vector", [](void* s, Args& a) {
    V tail = av(a[0]); vv(s).insert(vv(s).end(), tail.begin(), tail.end()); return O(static_cast<Vec*>(s)); });
  api.method(Vt, "copy", {}, Vt, "independent copy", [](void* s, Args&) { return vecv(vv(s)); });
  api.method(Vt, "slice", {"int", "int"}, Vt, "elements [start, stop) (numpy slicing; negative indexes count from the end, out-of-range clamps)", [](void* s, Args& a) {
    int64_t n = int64_t(vv(s).size());
    auto norm = [n](int64_t i) { if (i < 0) i += n; return std::clamp<int64_t>(i, 0, n); };
    int64_t b = norm(a[0].i), e = norm(a[1].i);
    return vecv(e > b ? V(vv(s).begin() + b, vv(s).begin() + e) : V{});
  });
  api.method(Vt, "take", {Vt}, Vt, "elements at the given (float, integral) indexes", [](void* s, Args& a) {
    V r;
    for (double d : av(a[0])) {
      if (d != std::floor(d)) fail("Vec.take: index " + jnum(d) + " is not integral");
      r.push_back(vv(s)[index_arg(int64_t(d), vv(s).size(), "Vec.take")]);
    }
    return vecv(std::move(r));
  });
  api.method(Vt, "where", {Vt}, Vt, "elements whose mask entry is nonzero (boolean indexing x[mask])", [](void* s, Args& a) {
    same_len(vv(s), av(a[0]), "where");
    V r;
    for (size_t i = 0; i < vv(s).size(); ++i) if (av(a[0])[i] != 0 && !std::isnan(av(a[0])[i])) r.push_back(vv(s)[i]);
    return vecv(std::move(r));
  });
  api.method(Vt, "find", {Vt}, Vt, "indexes whose mask entry is nonzero (numpy.flatnonzero)", [](void* s, Args&) {
    V r;
    for (size_t i = 0; i < vv(s).size(); ++i) if (vv(s)[i] != 0 && !std::isnan(vv(s)[i])) r.push_back(double(i));
    return vecv(std::move(r));
  });
  // arithmetic
  api.method(Vt, "add", {Vt}, Vt, "elementwise sum", [](void* s, Args& a) { return zip(vv(s), av(a[0]), "add", std::plus<double>()); });
  api.method(Vt, "add", {"float"}, Vt, "add a scalar", [](void* s, Args& a) { double c = a[0].f; return each(vv(s), [c](double x) { return x + c; }); });
  api.method(Vt, "sub", {Vt}, Vt, "elementwise difference", [](void* s, Args& a) { return zip(vv(s), av(a[0]), "sub", std::minus<double>()); });
  api.method(Vt, "sub", {"float"}, Vt, "subtract a scalar", [](void* s, Args& a) { double c = a[0].f; return each(vv(s), [c](double x) { return x - c; }); });
  api.method(Vt, "rsub", {"float"}, Vt, "scalar minus each element", [](void* s, Args& a) { double c = a[0].f; return each(vv(s), [c](double x) { return c - x; }); });
  api.method(Vt, "mul", {Vt}, Vt, "elementwise product", [](void* s, Args& a) { return zip(vv(s), av(a[0]), "mul", std::multiplies<double>()); });
  api.method(Vt, "mul", {"float"}, Vt, "scale by a scalar", [](void* s, Args& a) { double c = a[0].f; return each(vv(s), [c](double x) { return x * c; }); });
  api.method(Vt, "div", {Vt}, Vt, "elementwise quotient (IEEE: x/0 is inf or nan)", [](void* s, Args& a) { return zip(vv(s), av(a[0]), "div", std::divides<double>()); });
  api.method(Vt, "div", {"float"}, Vt, "divide by a scalar", [](void* s, Args& a) { double c = a[0].f; return each(vv(s), [c](double x) { return x / c; }); });
  api.method(Vt, "neg", {}, Vt, "negation", [](void* s, Args&) { return each(vv(s), [](double x) { return -x; }); });
  api.method(Vt, "pow", {"float"}, Vt, "elementwise power", [](void* s, Args& a) { double c = a[0].f; return each(vv(s), [c](double x) { return c == 2.0 ? x * x : std::pow(x, c); }); });
  api.op(Vt, "+", "add"); api.op(Vt, "-", "sub"); api.op(Vt, "*", "mul"); api.op(Vt, "neg", "neg");
  api.op(Vt, "r+", "add"); api.op(Vt, "r-", "rsub"); api.op(Vt, "r*", "mul");
  struct Unary { const char* name; double (*f)(double); const char* doc; };
  static const Unary unary[] = {
      {"abs", [](double x) { return std::fabs(x); }, "absolute value"},
      {"sqrt", [](double x) { return std::sqrt(x); }, "square root"},
      {"exp", [](double x) { return std::exp(x); }, "exponential"},
      {"log", [](double x) { return std::log(x); }, "natural logarithm"},
      {"log10", [](double x) { return std::log10(x); }, "base-10 logarithm"},
      {"log1p", [](double x) { return std::log1p(x); }, "log(1 + x)"},
      {"sin", [](double x) { return std::sin(x); }, "sine"},
      {"cos", [](double x) { return std::cos(x); }, "cosine"},
      {"tanh", [](double x) { return std::tanh(x); }, "hyperbolic tangent"},
      {"square", [](double x) { return x * x; }, "x*x"},
      {"floor", [](double x) { return std::floor(x); }, "floor"},
      {"ceil", [](double x) { return std::ceil(x); }, "ceiling"},
      {"round", [](double x) { return std::nearbyint(x); }, "round half to even (numpy.round)"},
      {"isnan", [](double x) { return std::isnan(x) ? 1.0 : 0.0; }, "1.0 where NaN, else 0.0"},
  };
  for (const auto& u : unary) {
    auto f = u.f;
    api.method(Vt, u.name, {}, Vt, std::string("elementwise ") + u.doc, [f](void* s, Args&) { return each(vv(s), f); });
  }
  api.method(Vt, "clip", {"float", "float"}, Vt, "clamp each element to [lo, hi]", [](void* s, Args& a) {
    double lo = a[0].f, hi = a[1].f; return each(vv(s), [lo, hi](double x) { return std::min(std::max(x, lo), hi); }); });
  api.method(Vt, "apply", {"string"}, Vt, "evaluate an expression in `x` for each element: `v.apply(\"exp(-x^2/2)\")`", [](void* s, Args& a) {
    ExprCompiler c({"x"});
    ExprProgram p = c.compile(a[0].s);
    return each(vv(s), [&p](double x) { return expr_eval(p, &x); });
  });
  api.method(Vt, "apply", {"string", "string", Vt}, Vt, "evaluate an expression in `x` and a second named vector elementwise: `x.apply(\"x*w\", \"w\", w)`", [](void* s, Args& a) {
    ExprCompiler c({"x", a[1].s});
    ExprProgram p = c.compile(a[0].s);
    const V& w = av(a[2]);
    same_len(vv(s), w, "apply");
    V r(vv(s).size());
    for (size_t i = 0; i < r.size(); ++i) { double xs[2] = {vv(s)[i], w[i]}; r[i] = expr_eval(p, xs); }
    return vecv(std::move(r));
  });
  // comparisons -> masks
  struct Cmp { const char* name; bool (*f)(double, double); const char* doc; };
  static const Cmp cmps[] = {{"gt", [](double x, double c) { return x > c; }, ">"},  {"ge", [](double x, double c) { return x >= c; }, ">="},
                             {"lt", [](double x, double c) { return x < c; }, "<"},  {"le", [](double x, double c) { return x <= c; }, "<="},
                             {"eq", [](double x, double c) { return x == c; }, "=="}, {"ne", [](double x, double c) { return x != c; }, "!="}};
  for (const auto& c : cmps) {
    auto f = c.f;
    api.method(Vt, c.name, {"float"}, Vt, std::string("mask (1.0/0.0) of x ") + c.doc + " c", [f](void* s, Args& a) {
      double k = a[0].f; return each(vv(s), [f, k](double x) { return f(x, k) ? 1.0 : 0.0; }); });
    api.method(Vt, c.name, {Vt}, Vt, std::string("elementwise mask of x ") + c.doc + " y", [f](void* s, Args& a) {
      return zip(vv(s), av(a[0]), "compare", [f](double x, double y) { return f(x, y) ? 1.0 : 0.0; }); });
  }
  // reductions
  api.method(Vt, "sum", {}, "float", "sum (compensated)", [](void* s, Args&) { return F(vsum(vv(s))); });
  api.method(Vt, "prod", {}, "float", "product", [](void* s, Args&) { double p = 1; for (double x : vv(s)) p *= x; return F(p); });
  api.method(Vt, "mean", {}, "float", "arithmetic mean", [](void* s, Args&) { need_nonempty(vv(s), "mean"); return F(vmean(vv(s))); });
  api.method(Vt, "var", {"int"}, "float", "variance with ddof (0 population, 1 sample)", [](void* s, Args& a) { return F(vvar(vv(s), a[0].i)); });
  api.method(Vt, "std", {"int"}, "float", "standard deviation with ddof", [](void* s, Args& a) { return F(std::sqrt(vvar(vv(s), a[0].i))); });
  api.method(Vt, "min", {}, "float", "minimum (NaN if any NaN)", [](void* s, Args&) {
    need_nonempty(vv(s), "min"); double m = INFINITY; for (double x : vv(s)) { if (std::isnan(x)) return F(NAN); m = std::min(m, x); } return F(m); });
  api.method(Vt, "max", {}, "float", "maximum (NaN if any NaN)", [](void* s, Args&) {
    need_nonempty(vv(s), "max"); double m = -INFINITY; for (double x : vv(s)) { if (std::isnan(x)) return F(NAN); m = std::max(m, x); } return F(m); });
  api.method(Vt, "argmin", {}, "int", "index of the first minimum", [](void* s, Args&) { need_nonempty(vv(s), "argmin"); return I(std::min_element(vv(s).begin(), vv(s).end()) - vv(s).begin()); });
  api.method(Vt, "argmax", {}, "int", "index of the first maximum", [](void* s, Args&) { need_nonempty(vv(s), "argmax"); return I(std::max_element(vv(s).begin(), vv(s).end()) - vv(s).begin()); });
  api.method(Vt, "median", {}, "float", "median", [](void* s, Args&) { V c = vv(s); std::sort(c.begin(), c.end()); return F(quantile_sorted(c, 0.5)); });
  api.method(Vt, "quantile", {"float"}, "float", "quantile q in [0,1], linear interpolation (numpy default)", [](void* s, Args& a) {
    V c = vv(s); std::sort(c.begin(), c.end()); return F(quantile_sorted(c, a[0].f)); });
  api.method(Vt, "norm", {}, "float", "Euclidean norm", [](void* s, Args&) { return F(to_eigen(vv(s)).norm()); });
  api.method(Vt, "dot", {Vt}, "float", "inner product", [](void* s, Args& a) { same_len(vv(s), av(a[0]), "dot"); return F(to_eigen(vv(s)).dot(to_eigen(av(a[0])))); });
  api.method(Vt, "count", {}, "int", "number of nonzero, non-NaN entries (count of a mask)", [](void* s, Args&) {
    int64_t c = 0; for (double x : vv(s)) c += (x != 0 && !std::isnan(x)); return I(c); });
  api.method(Vt, "all_finite", {}, "bool", "true when no element is NaN or infinite", [](void* s, Args&) {
    for (double x : vv(s)) if (!std::isfinite(x)) return B(false); return B(true); });
  api.method(Vt, "cumsum", {}, Vt, "cumulative sum", [](void* s, Args&) { V r = vv(s); for (size_t i = 1; i < r.size(); ++i) r[i] += r[i - 1]; return vecv(std::move(r)); });
  api.method(Vt, "diff", {}, Vt, "first differences x[i+1]-x[i]", [](void* s, Args&) {
    V r; for (size_t i = 1; i < vv(s).size(); ++i) r.push_back(vv(s)[i] - vv(s)[i - 1]); return vecv(std::move(r)); });
  api.method(Vt, "sort", {}, Vt, "ascending copy (NaN last)", [](void* s, Args&) {
    V r = vv(s); std::stable_sort(r.begin(), r.end(), [](double x, double y) { return std::isnan(y) ? !std::isnan(x) : x < y; }); return vecv(std::move(r)); });
  api.method(Vt, "argsort", {}, Vt, "indexes that sort ascending (stable)", [](void* s, Args&) {
    const V& x = vv(s); std::vector<size_t> idx(x.size()); std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [&x](size_t i, size_t j) { return std::isnan(x[j]) ? !std::isnan(x[i]) : x[i] < x[j]; });
    V r(idx.begin(), idx.end()); return vecv(std::move(r)); });
  api.method(Vt, "reverse", {}, Vt, "reversed copy", [](void* s, Args&) { V r(vv(s).rbegin(), vv(s).rend()); return vecv(std::move(r)); });
  api.method(Vt, "unique", {}, Vt, "sorted distinct values", [](void* s, Args&) {
    V r = vv(s); std::sort(r.begin(), r.end()); r.erase(std::unique(r.begin(), r.end()), r.end()); return vecv(std::move(r)); });
  api.method(Vt, "json", {}, "string", "JSON array text", [](void* s, Args&) { return S(jvec(vv(s))); });
  api.method(Vt, "fmt", {"int"}, "string", "space-separated values with fixed decimals", [](void* s, Args& a) {
    std::string out; for (size_t i = 0; i < vv(s).size(); ++i) out += (i ? " " : "") + fmt_fixed(vv(s)[i], a[0].i); return S(out); });

  // ---------------------------------------------------------------------------------------------------- StrVec
  api.section("Text lists", "`StrVec` is a list of strings: column names, labels, path lists.");
  api.type(St, "List of strings.");
  api.fn("strs", {}, St, "empty list", [](Args&) { return strsv({}); });
  api.fn("strs", {"string"}, St, "comma-separated names, trimmed: `strs(\"x1, x2\")`", [](Args& a) {
    std::vector<std::string> r; for (auto& p : split_text(a[0].s, ",")) { auto t = trim(p); if (!t.empty()) r.push_back(t); } return strsv(std::move(r)); });
  api.fn("strs", {"int", "string"}, St, "n copies of one string (a label column)", [](Args& a) {
    if (a[0].i < 0) fail("strs: negative count"); return strsv(std::vector<std::string>(size_t(a[0].i), a[1].s)); });
  api.fn("split", {"string", "string"}, St, "split text on a separator (no trimming)", [](Args& a) { return strsv(split_text(a[0].s, a[1].s)); });
  api.method(St, "len", {}, "int", "length", [](void* s, Args&) { return I(int64_t(static_cast<StrVec*>(s)->v.size())); });
  api.method(St, "at", {"int"}, "string", "element i", [](void* s, Args& a) { auto& v = static_cast<StrVec*>(s)->v; return S(v[index_arg(a[0].i, v.size(), "StrVec.at")]); });
  api.method(St, "push", {"string"}, St, "*in place*: append; returns the list", [](void* s, Args& a) { static_cast<StrVec*>(s)->v.push_back(a[0].s); return O(static_cast<StrVec*>(s)); });
  api.method(St, "contains", {"string"}, "bool", "membership", [](void* s, Args& a) { auto& v = static_cast<StrVec*>(s)->v; return B(std::find(v.begin(), v.end(), a[0].s) != v.end()); });
  api.method(St, "index", {"string"}, "int", "first position or -1", [](void* s, Args& a) {
    auto& v = static_cast<StrVec*>(s)->v; auto it = std::find(v.begin(), v.end(), a[0].s); return I(it == v.end() ? -1 : it - v.begin()); });
  api.method(St, "join", {"string"}, "string", "join with a separator", [](void* s, Args& a) {
    std::string out; auto& v = static_cast<StrVec*>(s)->v; for (size_t i = 0; i < v.size(); ++i) out += (i ? a[0].s : "") + v[i]; return S(out); });
  api.method(St, "json", {}, "string", "JSON array text", [](void* s, Args&) { return S(jstrs(static_cast<StrVec*>(s)->v)); });

  // ---------------------------------------------------------------------------------------------------- Mat
  api.section("Matrices", "`Mat` is a dense float64 matrix. `*` is the matrix product (numpy `@`) when both sides are matrices, a "
                          "matrix-vector product with a Vec, and scaling with a float; use `hadamard` for the elementwise product.");
  api.type(Mt, "Dense float64 matrix. Operators: `+`, `-` (Mat), `*` (Mat, Vec or float), unary `-`, `float * Mat`.");
  api.fn("mat", {"int", "int"}, Mt, "rows x cols zeros", [](Args& a) { if (a[0].i < 0 || a[1].i < 0) fail("mat: negative size"); return matv(Eigen::MatrixXd::Zero(a[0].i, a[1].i)); });
  api.fn("mat", {Vt, "int", "int"}, Mt, "reshape a vector, row-major, to rows x cols", [](Args& a) {
    const V& d = av(a[0]);
    if (a[1].i < 0 || a[2].i < 0 || size_t(a[1].i * a[2].i) != d.size()) fail("mat: " + std::to_string(d.size()) + " values do not fill " + std::to_string(a[1].i) + "x" + std::to_string(a[2].i));
    Eigen::MatrixXd m(a[1].i, a[2].i);
    for (int64_t r = 0; r < a[1].i; ++r) for (int64_t c = 0; c < a[2].i; ++c) m(r, c) = d[size_t(r * a[2].i + c)];
    return matv(std::move(m));
  });
  api.fn("mat", {"string"}, Mt, "parse rows separated by `;` (or JSON nested arrays): `mat(\"1 2; 3 4\")`", [](Args& a) {
    std::string t = a[0].s;
    std::vector<V> rows;
    if (t.find(';') != std::string::npos) {
      for (auto& r : split_text(t, ";")) if (!trim(r).empty()) rows.push_back(parse_numbers(r));
    } else {  // nested JSON: split on "]"
      for (auto& r : split_text(t, "]")) { V x = parse_numbers(r); if (!x.empty()) rows.push_back(x); }
    }
    if (rows.empty()) return matv(Eigen::MatrixXd(0, 0));
    Eigen::MatrixXd m(rows.size(), rows[0].size());
    for (size_t r = 0; r < rows.size(); ++r) {
      if (rows[r].size() != rows[0].size()) fail("mat: row " + std::to_string(r) + " has " + std::to_string(rows[r].size()) + " values, expected " + std::to_string(rows[0].size()));
      for (size_t c = 0; c < rows[r].size(); ++c) m(r, c) = rows[r][c];
    }
    return matv(std::move(m));
  });
  api.fn("eye", {"int"}, Mt, "n x n identity", [](Args& a) { if (a[0].i < 0) fail("eye: negative size"); return matv(Eigen::MatrixXd::Identity(a[0].i, a[0].i)); });
  api.fn("diag", {Vt}, Mt, "diagonal matrix", [](Args& a) { return matv(Eigen::MatrixXd(to_eigen(av(a[0])).asDiagonal())); });
  auto colstack = [](std::vector<const V*> cols) {
    size_t n = cols[0]->size();
    for (auto* c : cols) if (c->size() != n) fail("cols: columns have different lengths");
    Eigen::MatrixXd m(n, cols.size());
    for (size_t j = 0; j < cols.size(); ++j) for (size_t i = 0; i < n; ++i) m(i, j) = (*cols[j])[i];
    return matv(std::move(m));
  };
  api.fn("cols", {Vt}, Mt, "one-column matrix", [colstack](Args& a) { return colstack({&av(a[0])}); });
  api.fn("cols", {Vt, Vt}, Mt, "column_stack of two vectors", [colstack](Args& a) { return colstack({&av(a[0]), &av(a[1])}); });
  api.fn("cols", {Vt, Vt, Vt}, Mt, "column_stack of three vectors", [colstack](Args& a) { return colstack({&av(a[0]), &av(a[1]), &av(a[2])}); });
  api.fn("cols", {Vt, Vt, Vt, Vt}, Mt, "column_stack of four vectors", [colstack](Args& a) { return colstack({&av(a[0]), &av(a[1]), &av(a[2]), &av(a[3])}); });
  api.method(Mt, "rows", {}, "int", "row count", [](void* s, Args&) { return I(mm(s).rows()); });
  api.method(Mt, "cols", {}, "int", "column count", [](void* s, Args&) { return I(mm(s).cols()); });
  api.method(Mt, "at", {"int", "int"}, "float", "element (i, j)", [](void* s, Args& a) {
    return F(mm(s)(index_arg(a[0].i, size_t(mm(s).rows()), "Mat.at row"), index_arg(a[1].i, size_t(mm(s).cols()), "Mat.at column"))); });
  api.method(Mt, "set", {"int", "int", "float"}, Mt, "*in place*: element (i, j) = x; returns the matrix", [](void* s, Args& a) {
    mm(s)(index_arg(a[0].i, size_t(mm(s).rows()), "Mat.set row"), index_arg(a[1].i, size_t(mm(s).cols()), "Mat.set column")) = a[2].f; return O(static_cast<Mat*>(s)); });
  api.method(Mt, "row", {"int"}, Vt, "row i", [](void* s, Args& a) { return vecv(Eigen::VectorXd(mm(s).row(index_arg(a[0].i, size_t(mm(s).rows()), "Mat.row")).transpose())); });
  api.method(Mt, "col", {"int"}, Vt, "column j", [](void* s, Args& a) { return vecv(Eigen::VectorXd(mm(s).col(index_arg(a[0].i, size_t(mm(s).cols()), "Mat.col")))); });
  api.method(Mt, "t", {}, Mt, "transpose", [](void* s, Args&) { return matv(mm(s).transpose()); });
  api.method(Mt, "mul", {Mt}, Mt, "matrix product (numpy @)", [](void* s, Args& a) {
    need_shape(mm(s).cols() == am(a[0]).rows(), "Mat.mul: shapes " + shape(mm(s)) + " and " + shape(am(a[0])) + " do not align"); return matv(mm(s) * am(a[0])); });
  api.method(Mt, "mul", {Vt}, Vt, "matrix-vector product", [](void* s, Args& a) {
    need_shape(size_t(mm(s).cols()) == av(a[0]).size(), "Mat.mul: " + shape(mm(s)) + " times a vector of length " + std::to_string(av(a[0]).size())); return vecv(Eigen::VectorXd(mm(s) * to_eigen(av(a[0])))); });
  api.method(Mt, "mul", {"float"}, Mt, "scale", [](void* s, Args& a) { return matv(mm(s) * a[0].f); });
  api.method(Mt, "add", {Mt}, Mt, "elementwise sum", [](void* s, Args& a) { need_shape(mm(s).rows() == am(a[0]).rows() && mm(s).cols() == am(a[0]).cols(), "Mat.add: shapes differ"); return matv(mm(s) + am(a[0])); });
  api.method(Mt, "sub", {Mt}, Mt, "elementwise difference", [](void* s, Args& a) { need_shape(mm(s).rows() == am(a[0]).rows() && mm(s).cols() == am(a[0]).cols(), "Mat.sub: shapes differ"); return matv(mm(s) - am(a[0])); });
  api.method(Mt, "neg", {}, Mt, "negation", [](void* s, Args&) { return matv(-mm(s)); });
  api.method(Mt, "hadamard", {Mt}, Mt, "elementwise product", [](void* s, Args& a) { need_shape(mm(s).rows() == am(a[0]).rows() && mm(s).cols() == am(a[0]).cols(), "Mat.hadamard: shapes differ"); return matv(mm(s).cwiseProduct(am(a[0]))); });
  api.op(Mt, "+", "add"); api.op(Mt, "-", "sub"); api.op(Mt, "*", "mul"); api.op(Mt, "neg", "neg"); api.op(Mt, "r*", "mul");
  api.method(Mt, "hstack", {Mt}, Mt, "columns of this then other", [](void* s, Args& a) {
    need_shape(mm(s).rows() == am(a[0]).rows(), "Mat.hstack: row counts differ"); Eigen::MatrixXd r(mm(s).rows(), mm(s).cols() + am(a[0]).cols()); r << mm(s), am(a[0]); return matv(std::move(r)); });
  api.method(Mt, "vstack", {Mt}, Mt, "rows of this then other", [](void* s, Args& a) {
    need_shape(mm(s).cols() == am(a[0]).cols(), "Mat.vstack: column counts differ"); Eigen::MatrixXd r(mm(s).rows() + am(a[0]).rows(), mm(s).cols()); r << mm(s), am(a[0]); return matv(std::move(r)); });
  api.method(Mt, "add_col", {Vt}, Mt, "new matrix with the vector appended as a last column", [](void* s, Args& a) {
    need_shape(size_t(mm(s).rows()) == av(a[0]).size() || mm(s).size() == 0, "Mat.add_col: length differs from the row count");
    Eigen::MatrixXd r(Eigen::Index(av(a[0]).size()), mm(s).cols() + 1); if (mm(s).cols()) r.leftCols(mm(s).cols()) = mm(s); r.col(mm(s).cols()) = to_eigen(av(a[0])); return matv(std::move(r)); });
  api.method(Mt, "with_intercept", {}, Mt, "new matrix with a leading column of ones (sm.add_constant)", [](void* s, Args&) {
    Eigen::MatrixXd r(mm(s).rows(), mm(s).cols() + 1); r.col(0).setOnes(); r.rightCols(mm(s).cols()) = mm(s); return matv(std::move(r)); });
  api.method(Mt, "block", {"int", "int", "int", "int"}, Mt, "sub-matrix (row, col, nrows, ncols)", [](void* s, Args& a) {
    if (a[0].i < 0 || a[1].i < 0 || a[2].i < 0 || a[3].i < 0 || a[0].i + a[2].i > mm(s).rows() || a[1].i + a[3].i > mm(s).cols()) fail("Mat.block: out of range for " + shape(mm(s)));
    return matv(mm(s).block(a[0].i, a[1].i, a[2].i, a[3].i)); });
  api.method(Mt, "take_rows", {Vt}, Mt, "rows at the given indexes", [](void* s, Args& a) {
    const V& idx = av(a[0]); Eigen::MatrixXd r(Eigen::Index(idx.size()), mm(s).cols());
    for (size_t i = 0; i < idx.size(); ++i) r.row(Eigen::Index(i)) = mm(s).row(index_arg(int64_t(idx[i]), size_t(mm(s).rows()), "Mat.take_rows"));
    return matv(std::move(r)); });
  api.method(Mt, "flatten", {}, Vt, "row-major values", [](void* s, Args&) {
    V r; for (Eigen::Index i = 0; i < mm(s).rows(); ++i) for (Eigen::Index j = 0; j < mm(s).cols(); ++j) r.push_back(mm(s)(i, j)); return vecv(std::move(r)); });
  api.method(Mt, "col_means", {}, Vt, "mean of each column", [](void* s, Args&) { return vecv(Eigen::VectorXd(mm(s).colwise().mean().transpose())); });
  api.method(Mt, "col_sums", {}, Vt, "sum of each column", [](void* s, Args&) { return vecv(Eigen::VectorXd(mm(s).colwise().sum().transpose())); });
  api.method(Mt, "row_sums", {}, Vt, "sum of each row", [](void* s, Args&) { return vecv(Eigen::VectorXd(mm(s).rowwise().sum())); });
  api.method(Mt, "sum", {}, "float", "sum of all elements", [](void* s, Args&) { return F(mm(s).sum()); });
  api.method(Mt, "max_abs_diff", {Mt}, "float", "max |a - b| (shapes must match)", [](void* s, Args& a) {
    need_shape(mm(s).rows() == am(a[0]).rows() && mm(s).cols() == am(a[0]).cols(), "Mat.max_abs_diff: shapes differ");
    return F(mm(s).size() ? (mm(s) - am(a[0])).cwiseAbs().maxCoeff() : 0.0); });
  api.method(Mt, "json", {}, "string", "nested JSON arrays, row-major", [](void* s, Args&) { return S(jmat(mm(s))); });

  // ---------------------------------------------------------------------------------------------------- Opts
  api.section("Options and results", "Operations take an optional `Opts` (named options, validated: an unknown key is an error) and "
                                     "return a `Res` (named, typed result fields). Field and option names follow pycli's JSON names.");
  api.type(Ot, "Named options builder: `opts().set(\"method\", \"BFGS\").set(\"tol\", 1e-8)`.");
  api.fn("opts", {}, Ot, "empty options", [](Args&) { return O(make<Opts>()); });
  auto setter = [&api](const std::string& type, std::function<Item(const Value&)> conv, const std::string& doc) {
    api.method(Ot, "set", {"string", type}, Ot, doc, [conv](void* s, Args& a) {
      auto* o = static_cast<Opts*>(s); o->put(a[0].s, conv(a[1])); return O(o); });
  };
  setter("float", [](const Value& v) { Item it; it.kind = Item::NUM; it.f = v.f; return it; }, "*in place*: float option");
  setter("int", [](const Value& v) { Item it; it.kind = Item::INT; it.i = v.i; return it; }, "*in place*: int option");
  setter("string", [](const Value& v) { Item it; it.kind = Item::STR; it.s = v.s; return it; }, "*in place*: string option");
  setter("bool", [](const Value& v) { Item it; it.kind = Item::BOOL; it.b = v.b; return it; }, "*in place*: bool option");
  setter(Vt, [](const Value& v) { Item it; it.kind = Item::VEC; it.vec = obj<Vec>(v).v; return it; }, "*in place*: vector option");
  setter(St, [](const Value& v) { Item it; it.kind = Item::STRS; it.strs = obj<StrVec>(v).v; return it; }, "*in place*: name-list option");
  setter(Mt, [](const Value& v) { Item it; it.kind = Item::MAT; it.mat = obj<Mat>(v).m; return it; }, "*in place*: matrix option");
  api.method(Ot, "has", {"string"}, "bool", "whether a key is set", [](void* s, Args& a) {
    for (auto& kv : static_cast<Opts*>(s)->items) if (kv.first == a[0].s) return B(true); return B(false); });

  // ---------------------------------------------------------------------------------------------------- Res
  api.type(Rt, "Typed result fields. Reading a missing field or the wrong kind is a run-time error that lists the fields.");
  api.method(Rt, "num", {"string"}, "float", "float field (an int field widens)", [](void* s, Args& a) { return item_value(static_cast<Res*>(s)->get(a[0].s, Item::NUM), Item::NUM); });
  api.method(Rt, "int", {"string"}, "int", "int field", [](void* s, Args& a) { return item_value(static_cast<Res*>(s)->get(a[0].s, Item::INT), Item::INT); });
  api.method(Rt, "text", {"string"}, "string", "string field", [](void* s, Args& a) { return item_value(static_cast<Res*>(s)->get(a[0].s, Item::STR), Item::STR); });
  api.method(Rt, "flag", {"string"}, "bool", "bool field", [](void* s, Args& a) { return item_value(static_cast<Res*>(s)->get(a[0].s, Item::BOOL), Item::BOOL); });
  api.method(Rt, "vec", {"string"}, Vt, "vector field", [](void* s, Args& a) { return item_value(static_cast<Res*>(s)->get(a[0].s, Item::VEC), Item::VEC); });
  api.method(Rt, "strs", {"string"}, St, "name-list field", [](void* s, Args& a) { return item_value(static_cast<Res*>(s)->get(a[0].s, Item::STRS), Item::STRS); });
  api.method(Rt, "mat", {"string"}, Mt, "matrix field", [](void* s, Args& a) { return item_value(static_cast<Res*>(s)->get(a[0].s, Item::MAT), Item::MAT); });
  api.method(Rt, "res", {"string"}, Rt, "nested result field", [](void* s, Args& a) { return item_value(static_cast<Res*>(s)->get(a[0].s, Item::RES), Item::RES); });
  api.method(Rt, "has", {"string"}, "bool", "whether a field exists", [](void* s, Args& a) { return B(static_cast<Res*>(s)->find(a[0].s) != nullptr); });
  api.method(Rt, "keys", {}, St, "field names in order", [](void* s, Args&) {
    std::vector<std::string> k; for (auto& kv : static_cast<Res*>(s)->items) k.push_back(kv.first); return strsv(std::move(k)); });
  api.method(Rt, "status", {}, "string", "the `status` field (\"OK\", \"NO_CONVERGENCE\", ...)", [](void* s, Args&) {
    const Item* it = static_cast<Res*>(s)->find("status"); return S(it && it->kind == Item::STR ? it->s : ""); });
  api.method(Rt, "ok", {}, "bool", "status == \"OK\"", [](void* s, Args&) {
    const Item* it = static_cast<Res*>(s)->find("status"); return B(it && it->kind == Item::STR && it->s == "OK"); });
  api.method(Rt, "json", {}, "string", "the whole result as JSON text", [](void* s, Args&) { return S(jres(*static_cast<Res*>(s))); });

  // ---------------------------------------------------------------------------------------------------- utilities
  api.section("Utilities", "Formatting and numeric helpers that the DSL's built-ins (`str`, `float`, `int`, `sqrt`, `pow`, ...) do not cover.");
  api.fn("fmt", {"float", "int"}, "string", "fixed decimals: `fmt(3.14159, 2)` is \"3.14\"", [](Args& a) { return S(fmt_fixed(a[0].f, a[1].i)); });
  api.fn("fmt_g", {"float", "int"}, "string", "printf %.Ng (significant digits)", [](Args& a) {
    if (a[1].i < 1 || a[1].i > 17) fail("fmt_g: digits must be in [1, 17]"); char b[64]; std::snprintf(b, sizeof b, "%.*g", int(a[1].i), a[0].f); return S(b); });
  api.fn("pad", {"string", "int"}, "string", "left-justify to a width (negative width right-justifies)", [](Args& a) {
    std::string s = a[0].s; size_t w = size_t(std::llabs(a[1].i)); if (s.size() >= w) return S(s);
    return S(a[1].i < 0 ? std::string(w - s.size(), ' ') + s : s + std::string(w - s.size(), ' ')); });
  api.fn("nan", {}, "float", "quiet NaN", [](Args&) { return F(NAN); });
  api.fn("inf", {}, "float", "+infinity", [](Args&) { return F(INFINITY); });
  api.fn("pi", {}, "float", "π", [](Args&) { return F(M_PI); });
  api.fn("is_nan", {"float"}, "bool", "x is NaN", [](Args& a) { return B(std::isnan(a[0].f)); });
  api.fn("is_finite", {"float"}, "bool", "x is neither NaN nor infinite", [](Args& a) { return B(std::isfinite(a[0].f)); });
  api.fn("exp", {"float"}, "float", "e^x", [](Args& a) { return F(std::exp(a[0].f)); });
  api.fn("log", {"float"}, "float", "natural log", [](Args& a) { return F(std::log(a[0].f)); });
  api.fn("log10", {"float"}, "float", "base-10 log", [](Args& a) { return F(std::log10(a[0].f)); });
  api.fn("tan", {"float"}, "float", "tangent", [](Args& a) { return F(std::tan(a[0].f)); });
  api.fn("atan2", {"float", "float"}, "float", "atan2(y, x)", [](Args& a) { return F(std::atan2(a[0].f, a[1].f)); });
  api.fn("hypot", {"float", "float"}, "float", "sqrt(x^2 + y^2) without overflow", [](Args& a) { return F(std::hypot(a[0].f, a[1].f)); });
  api.fn("eval", {"string"}, "float", "evaluate a constant expression (pycli expression grammar)", [](Args& a) {
    ExprCompiler c({}); ExprProgram p = c.compile(a[0].s); return F(expr_eval(p, nullptr)); });
  api.fn("eval", {"string", "string", "float"}, "float", "evaluate an expression in one named variable", [](Args& a) {
    ExprCompiler c({a[1].s}); ExprProgram p = c.compile(a[0].s); double x = a[2].f; return F(expr_eval(p, &x)); });
  api.fn("eval", {"string", St, Vt}, "float", "evaluate an expression with named variables bound to values", [](Args& a) {
    const auto& names = obj<StrVec>(a[1]).v; const V& vals = av(a[2]);
    if (names.size() != vals.size()) fail("eval: " + std::to_string(names.size()) + " names but " + std::to_string(vals.size()) + " values");
    ExprCompiler c(names); ExprProgram p = c.compile(a[0].s); return F(expr_eval(p, vals.data())); });

  register_out(api);
}

}  // namespace lmnum
