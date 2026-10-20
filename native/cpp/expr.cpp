#include "expr.hpp"
#include "host.hpp"
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/erf.hpp>
#include <cmath>
#include <cstdlib>
#include <limits>

namespace lmnum {
namespace {

struct FnDef { const char* name; int arity; bool reducer; };
const FnDef FNS[] = {
    {"abs", 1, false},   {"acos", 1, false},  {"asin", 1, false},  {"atan", 1, false},    {"atan2", 2, false},
    {"beta", 2, false},  {"ceiling", 1, false}, {"choose", 2, false}, {"cos", 1, false},  {"cosh", 1, false},
    {"digamma", 1, false}, {"dnorm", 1, false}, {"exp", 1, false},  {"expm1", 1, false},  {"floor", 1, false},
    {"gamma", 1, false}, {"lbeta", 2, false},  {"lchoose", 2, false}, {"lgamma", 1, false}, {"log", 1, false},
    {"log10", 1, false}, {"log1p", 1, false},  {"log2", 1, false},   {"max", 1, true},     {"mean", 1, true},
    {"min", 1, true},    {"plogis", 1, false}, {"pmax", 2, false},   {"pmin", 2, false},   {"pnorm", 1, false},
    {"prod", 1, true},   {"qnorm", 1, false},  {"round", 1, false},  {"sign", 1, false},   {"sin", 1, false},
    {"sinh", 1, false},  {"sqrt", 1, false},   {"sum", 1, true},     {"tan", 1, false},    {"tanh", 1, false},
    {"trunc", 1, false},
};
int find_fn(const std::string& n) {
  for (int i = 0; i < int(sizeof FNS / sizeof FNS[0]); ++i)
    if (n == FNS[i].name) return i;
  return -1;
}

double lchoose(double n, double k) { return std::lgamma(n + 1) - std::lgamma(k + 1) - std::lgamma(n - k + 1); }
double call1(const char* n, double x) {
  switch (n[0]) {
    case 'a':
      if (!std::strcmp(n, "abs")) return std::fabs(x);
      if (!std::strcmp(n, "acos")) return std::acos(x);
      if (!std::strcmp(n, "asin")) return std::asin(x);
      return std::atan(x);
    case 'c':
      if (!std::strcmp(n, "ceiling")) return std::ceil(x);
      if (!std::strcmp(n, "cos")) return std::cos(x);
      return std::cosh(x);
    case 'd':
      if (!std::strcmp(n, "digamma")) return boost::math::digamma(x);
      return std::exp(-0.5 * x * x) / std::sqrt(2 * M_PI);  // dnorm
    case 'e':
      if (!std::strcmp(n, "exp")) return std::exp(x);
      return std::expm1(x);
    case 'f': return std::floor(x);
    case 'g': return std::tgamma(x);
    case 'l':
      if (!std::strcmp(n, "lgamma")) return std::lgamma(x);
      if (!std::strcmp(n, "log")) return std::log(x);
      if (!std::strcmp(n, "log10")) return std::log10(x);
      if (!std::strcmp(n, "log1p")) return std::log1p(x);
      return std::log2(x);
    case 'p':
      if (!std::strcmp(n, "plogis")) return 1.0 / (1.0 + std::exp(-x));
      return 0.5 * std::erfc(-x / std::sqrt(2.0));  // pnorm
    case 'q':
      if (!(x > 0 && x < 1)) return x == 0 ? -INFINITY : x == 1 ? INFINITY : NAN;
      return -std::sqrt(2.0) * boost::math::erfc_inv(2 * x);
    case 'r': return std::nearbyint(x);
    case 's':
      if (!std::strcmp(n, "sign")) return x > 0 ? 1.0 : x < 0 ? -1.0 : x;
      if (!std::strcmp(n, "sin")) return std::sin(x);
      if (!std::strcmp(n, "sinh")) return std::sinh(x);
      return std::sqrt(x);
    case 't':
      if (!std::strcmp(n, "tan")) return std::tan(x);
      if (!std::strcmp(n, "tanh")) return std::tanh(x);
      return std::trunc(x);
  }
  return NAN;
}
double call2(const char* n, double a, double b) {
  if (!std::strcmp(n, "atan2")) return std::atan2(a, b);
  if (!std::strcmp(n, "beta")) return std::exp(std::lgamma(a) + std::lgamma(b) - std::lgamma(a + b));
  if (!std::strcmp(n, "lbeta")) return std::lgamma(a) + std::lgamma(b) - std::lgamma(a + b);
  if (!std::strcmp(n, "choose")) return std::nearbyint(std::exp(lchoose(a, b)));
  if (!std::strcmp(n, "lchoose")) return lchoose(a, b);
  if (!std::strcmp(n, "pmax")) return std::isnan(a) || std::isnan(b) ? NAN : std::max(a, b);
  return std::isnan(a) || std::isnan(b) ? NAN : std::min(a, b);  // pmin
}

// AST node, stored flat in ExprProgram::code (children are indexes).
struct Parser {
  const std::string& t;
  const std::vector<std::string>& scalars;
  const std::vector<std::string>& vectors;
  ExprProgram& p;
  size_t pos = 0;
  int reducer_depth = 0;

  [[noreturn]] void error(const std::string& what, size_t at) const {
    std::string near = t.substr(at > 10 ? at - 10 : 0, 20);
    fail("expression \"" + t + "\": " + what + " at offset " + std::to_string(at) + " (near \"" + near + "\")");
  }
  void ws() { while (pos < t.size() && (t[pos] == ' ' || t[pos] == '\t' || t[pos] == '\n')) ++pos; }
  bool take(const char* s) {
    ws();
    size_t n = std::strlen(s);
    if (t.compare(pos, n, s) == 0) {
      // do not split <= into < and =
      if (n == 1 && (s[0] == '<' || s[0] == '>' || s[0] == '=' || s[0] == '!') && pos + 1 < t.size() && t[pos + 1] == '=') return false;
      pos += n;
      return true;
    }
    return false;
  }
  int emit(ExprProgram::Ins ins) { p.code.push_back(ins); return int(p.code.size()) - 1; }
  // Each node stores children in slot (a), index (b), fn (c) fields: we reuse Ins fields as child links.
  int node(ExprProgram::Code c, int a = -1, int b = -1, int d = -1) {
    ExprProgram::Ins ins{c};
    ins.slot = a; ins.index = b; ins.fn = d;
    return emit(ins);
  }
  int parse_or() { int l = parse_and(); while (take("|")) l = node(ExprProgram::OR, l, parse_and()); return l; }
  int parse_and() { int l = parse_rel(); while (take("&")) l = node(ExprProgram::AND, l, parse_rel()); return l; }
  int parse_rel() {
    int l = parse_sum();
    if (take("<=")) return node(ExprProgram::LE, l, parse_sum());
    if (take(">=")) return node(ExprProgram::GE, l, parse_sum());
    if (take("==")) return node(ExprProgram::EQ, l, parse_sum());
    if (take("!=")) return node(ExprProgram::NE, l, parse_sum());
    if (take("<")) return node(ExprProgram::LT, l, parse_sum());
    if (take(">")) return node(ExprProgram::GT, l, parse_sum());
    return l;
  }
  int parse_sum() {
    int l = parse_product();
    for (;;) {
      if (take("+")) l = node(ExprProgram::ADD, l, parse_product());
      else if (take("-")) l = node(ExprProgram::SUB, l, parse_product());
      else return l;
    }
  }
  int parse_product() {
    int l = parse_unary();
    for (;;) {
      if (take("*")) l = node(ExprProgram::MUL, l, parse_unary());
      else if (take("/")) l = node(ExprProgram::DIV, l, parse_unary());
      else return l;
    }
  }
  int parse_unary() {
    if (take("-")) return node(ExprProgram::NEG, parse_unary());
    if (take("+")) return parse_unary();
    return parse_power();
  }
  int parse_power() {
    int l = parse_atom();
    if (take("^")) return node(ExprProgram::POW, l, parse_unary());
    return l;
  }
  int parse_atom() {
    ws();
    if (pos >= t.size()) error("unexpected end", pos);
    size_t start = pos;
    if (take("(")) {
      int x = parse_or();
      if (!take(")")) error("expected ')'", pos);
      return x;
    }
    char c = t[pos];
    if ((c >= '0' && c <= '9') || c == '.') {
      const char* b = t.c_str() + pos;
      char* e = nullptr;
      double v = std::strtod(b, &e);
      if (e == b) error("bad number", pos);
      pos += size_t(e - b);
      ExprProgram::Ins ins{ExprProgram::NUM};
      ins.value = v;
      return emit(ins);
    }
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_') {
      while (pos < t.size() && (std::isalnum(static_cast<unsigned char>(t[pos])) || t[pos] == '_' || t[pos] == '.')) ++pos;
      std::string name = t.substr(start, pos - start);
      ws();
      if (pos < t.size() && t[pos] == '(') return parse_call(name, start);
      for (size_t i = 0; i < scalars.size(); ++i)
        if (scalars[i] == name) { ExprProgram::Ins ins{ExprProgram::VAR}; ins.slot = int(i); return emit(ins); }
      for (size_t i = 0; i < vectors.size(); ++i)
        if (vectors[i] == name) {
          if (take("[")) {
            ws();
            size_t d = pos;
            while (pos < t.size() && std::isdigit(static_cast<unsigned char>(t[pos]))) ++pos;
            if (d == pos) error("vector index must be an integer literal", d);
            int idx = std::atoi(t.substr(d, pos - d).c_str());
            if (!take("]")) error("expected ']'", pos);
            ExprProgram::Ins ins{ExprProgram::IDX};
            ins.slot = int(i); ins.index = idx;
            return emit(ins);
          }
          if (reducer_depth == 0) error("vector '" + name + "' needs an index name[i] or a reducer such as sum(" + name + ")", start);
          ExprProgram::Ins ins{ExprProgram::IDX};
          ins.slot = int(i); ins.index = -1;  // current element inside a reducer
          return emit(ins);
        }
      if (name == "pi") { ExprProgram::Ins ins{ExprProgram::NUM}; ins.value = M_PI; return emit(ins); }
      if (name == "e") { ExprProgram::Ins ins{ExprProgram::NUM}; ins.value = M_E; return emit(ins); }
      std::string known;
      for (auto& s : scalars) known += (known.empty() ? "" : ", ") + s;
      for (auto& s : vectors) known += (known.empty() ? "" : ", ") + s + "[]";
      error("unknown name '" + name + "' (known: " + (known.empty() ? "none" : known) + ")", start);
    }
    error(std::string("unexpected character '") + c + "'", pos);
  }
  int parse_call(const std::string& name, size_t start) {
    take("(");
    if (name == "ifelse") {
      int c = parse_or();
      if (!take(",")) error("ifelse needs three arguments", pos);
      int a = parse_or();
      if (!take(",")) error("ifelse needs three arguments", pos);
      int b = parse_or();
      if (!take(")")) error("expected ')'", pos);
      return node(ExprProgram::IFELSE, c, a, b);
    }
    int f = find_fn(name);
    if (f < 0) error("unknown function '" + name + "'", start);
    std::vector<int> args;
    bool reducer = FNS[f].reducer;
    if (reducer) ++reducer_depth;
    if (!take(")")) {
      for (;;) {
        args.push_back(parse_or());
        if (take(")")) break;
        if (!take(",")) error("expected ',' or ')'", pos);
      }
    }
    if (reducer) --reducer_depth;
    // min/max with two arguments are the elementwise pmin/pmax
    if (reducer && args.size() == 2 && (name == "min" || name == "max")) {
      ExprProgram::Ins ins{ExprProgram::CALL2};
      ins.slot = args[0]; ins.index = args[1]; ins.fn = find_fn(name == "min" ? "pmin" : "pmax");
      return emit(ins);
    }
    if (int(args.size()) != FNS[f].arity) error(name + " takes " + std::to_string(FNS[f].arity) + " argument(s)", start);
    ExprProgram::Ins ins{reducer ? ExprProgram::VREDUCE : FNS[f].arity == 1 ? ExprProgram::CALL1 : ExprProgram::CALL2};
    ins.fn = f;
    ins.slot = args[0];
    ins.index = args.size() > 1 ? args[1] : -1;
    return emit(ins);
  }
};

struct Evaluator {
  const ExprProgram& p;
  const double* vars;
  const VecRef* vecs;
  long elem = -1;  // current reducer element

  size_t reducer_length(int n) const {
    const auto& ins = p.code[size_t(n)];
    size_t len = 0;
    auto merge = [&](size_t l) {
      if (len == 0) len = l;
      else if (l != len) fail("expression \"" + p.text + "\": vectors in one reducer have different lengths");
    };
    switch (ins.code) {
      case ExprProgram::IDX: if (ins.index < 0) merge(vecs[ins.slot].n); return len;
      case ExprProgram::NUM: case ExprProgram::VAR: case ExprProgram::VREDUCE: return 0;
      case ExprProgram::NEG: case ExprProgram::CALL1: return reducer_length(ins.slot);
      case ExprProgram::IFELSE: {
        for (int c : {ins.slot, ins.index, ins.fn}) { size_t l = reducer_length(c); if (l) merge(l); }
        return len;
      }
      default: {
        for (int c : {ins.slot, ins.index}) if (c >= 0) { size_t l = reducer_length(c); if (l) merge(l); }
        return len;
      }
    }
  }
  double eval(int n) {
    const auto& ins = p.code[size_t(n)];
    switch (ins.code) {
      case ExprProgram::NUM: return ins.value;
      case ExprProgram::VAR: return vars[ins.slot];
      case ExprProgram::IDX: {
        const VecRef& v = vecs[ins.slot];
        size_t i = ins.index >= 0 ? size_t(ins.index) : size_t(elem);
        if (i >= v.n) fail("expression \"" + p.text + "\": index " + std::to_string(i) + " beyond vector length " + std::to_string(v.n));
        return v.data[i];
      }
      case ExprProgram::NEG: return -eval(ins.slot);
      case ExprProgram::ADD: return eval(ins.slot) + eval(ins.index);
      case ExprProgram::SUB: return eval(ins.slot) - eval(ins.index);
      case ExprProgram::MUL: return eval(ins.slot) * eval(ins.index);
      case ExprProgram::DIV: return eval(ins.slot) / eval(ins.index);
      case ExprProgram::POW: {
        double a = eval(ins.slot), b = eval(ins.index);
        if (b == 2.0) return a * a;
        return std::pow(a, b);
      }
      case ExprProgram::LT: return eval(ins.slot) < eval(ins.index);
      case ExprProgram::LE: return eval(ins.slot) <= eval(ins.index);
      case ExprProgram::GT: return eval(ins.slot) > eval(ins.index);
      case ExprProgram::GE: return eval(ins.slot) >= eval(ins.index);
      case ExprProgram::EQ: return eval(ins.slot) == eval(ins.index);
      case ExprProgram::NE: return eval(ins.slot) != eval(ins.index);
      case ExprProgram::AND: return (eval(ins.slot) != 0) && (eval(ins.index) != 0);
      case ExprProgram::OR: return (eval(ins.slot) != 0) || (eval(ins.index) != 0);
      case ExprProgram::IFELSE: return eval(ins.slot) != 0 ? eval(ins.index) : eval(ins.fn);
      case ExprProgram::CALL1: return call1(FNS[ins.fn].name, eval(ins.slot));
      case ExprProgram::CALL2: return call2(FNS[ins.fn].name, eval(ins.slot), eval(ins.index));
      case ExprProgram::VREDUCE: {
        size_t len = reducer_length(ins.slot);
        long saved = elem;
        const char* name = FNS[ins.fn].name;
        bool is_sum = !std::strcmp(name, "sum"), is_mean = !std::strcmp(name, "mean"), is_prod = !std::strcmp(name, "prod"), is_min = !std::strcmp(name, "min");
        double acc = is_prod ? 1.0 : is_sum || is_mean ? 0.0 : (is_min ? INFINITY : -INFINITY);
        size_t count = len ? len : 1;
        for (size_t i = 0; i < count; ++i) {
          elem = long(i);
          double x = eval(ins.slot);
          if (is_sum || is_mean) acc += x;
          else if (is_prod) acc *= x;
          else if (std::isnan(x)) acc = NAN;
          else if (!std::isnan(acc)) acc = is_min ? std::min(acc, x) : std::max(acc, x);
        }
        elem = saved;
        return is_mean ? acc / double(count) : acc;
      }
    }
    return NAN;
  }
};

}  // namespace

ExprCompiler::ExprCompiler(std::vector<std::string> scalars, std::vector<std::string> vectors)
    : scalars_(std::move(scalars)), vectors_(std::move(vectors)) {}

int ExprCompiler::scalar_slot(const std::string& name) const {
  for (size_t i = 0; i < scalars_.size(); ++i)
    if (scalars_[i] == name) return int(i);
  return -1;
}

ExprProgram ExprCompiler::compile(const std::string& text) const {
  ExprProgram p;
  p.text = text;
  if (text.size() > 65536) fail("expression is longer than 65536 bytes");
  Parser ps{text, scalars_, vectors_, p};
  int root = ps.parse_or();
  ps.ws();
  if (ps.pos != text.size()) ps.error("unexpected text", ps.pos);
  // The root is evaluated by index; keep it last by recording it in a trailing NUM-free marker.
  ExprProgram::Ins ins{ExprProgram::NUM};
  ins.slot = root;
  ins.value = NAN;
  p.code.push_back(ins);  // sentinel: code.back().slot is the root
  return p;
}

double expr_eval(const ExprProgram& p, const double* vars, const VecRef* vecs) {
  Evaluator ev{p, vars, vecs};
  return ev.eval(p.code.back().slot);
}

std::vector<std::string> expr_names(const std::string& text) {
  std::vector<std::string> out;
  size_t i = 0;
  while (i < text.size()) {
    char c = text[i];
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_') {
      size_t s = i;
      while (i < text.size() && (std::isalnum(static_cast<unsigned char>(text[i])) || text[i] == '_' || text[i] == '.')) ++i;
      std::string name = text.substr(s, i - s);
      size_t j = i;
      while (j < text.size() && text[j] == ' ') ++j;
      bool call = j < text.size() && text[j] == '(';
      if (!call && name != "pi" && name != "e" && std::find(out.begin(), out.end(), name) == out.end()) out.push_back(name);
    } else if ((c >= '0' && c <= '9') || c == '.') {
      ++i;
      while (i < text.size() && (std::isdigit(static_cast<unsigned char>(text[i])) || text[i] == '.')) ++i;
      if (i < text.size() && (text[i] == 'e' || text[i] == 'E')) {
        ++i;
        if (i < text.size() && (text[i] == '+' || text[i] == '-')) ++i;
        while (i < text.size() && std::isdigit(static_cast<unsigned char>(text[i]))) ++i;
      }
    } else {
      ++i;
    }
  }
  return out;
}

}  // namespace lmnum

namespace lmnum {
std::vector<VecRef> ExprBindings::refs() const {
  std::vector<VecRef> r;
  for (auto& v : vector_values) r.push_back(VecRef{v.data(), v.size()});
  return r;
}
ExprBindings expr_bindings(const Opts& o) {
  ExprBindings b;
  for (auto& kv : o.items) {
    if (kv.first.empty() || kv.first[0] != '$') continue;
    std::string name = kv.first.substr(1);
    if (name.empty()) fail("option '$' needs a name after the dollar sign");
    const Item* it = o.find(kv.first);  // marks the key used
    if (it->kind == Item::NUM || it->kind == Item::INT) { b.scalar_names.push_back(name); b.scalar_values.push_back(it->kind == Item::NUM ? it->f : double(it->i)); }
    else if (it->kind == Item::VEC) { b.vector_names.push_back(name); b.vector_values.push_back(it->vec); }
    else fail("option '" + kv.first + "' must be a float, int or Vec (an expression binding)");
  }
  return b;
}
}  // namespace lmnum
