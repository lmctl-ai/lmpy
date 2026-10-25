#include "host.hpp"
#include <algorithm>
#include <charconv>
#include <sstream>

namespace lmpy {

std::vector<std::unique_ptr<Obj>>& arena() { static std::vector<std::unique_ptr<Obj>> a; return a; }

const char* item_kind_name(Item::Kind k) {
  static const char* n[] = {"float", "int", "string", "bool", "Vec", "StrVec", "Mat", "Res"};
  return n[k];
}

// ---- Opts ---------------------------------------------------------------------------------------------------------
void Opts::put(const std::string& key, Item item) {
  for (size_t i = 0; i < items.size(); ++i)
    if (items[i].first == key) { items[i].second = std::move(item); return; }
  items.emplace_back(key, std::move(item));
  used.push_back(false);
}
const Item* Opts::find(const std::string& key) const {
  for (size_t i = 0; i < items.size(); ++i)
    if (items[i].first == key) { used[i] = true; return &items[i].second; }
  return nullptr;
}
static void opt_kind(const std::string& key, const Item& it, const char* want) {
  fail("option '" + key + "' must be " + want + "; found " + item_kind_name(it.kind));
}
double Opts::num(const std::string& key, double def) const {
  const Item* it = find(key);
  if (!it) return def;
  if (it->kind == Item::NUM) return it->f;
  if (it->kind == Item::INT) return double(it->i);
  opt_kind(key, *it, "a number (num or int)");
}
int64_t Opts::integer(const std::string& key, int64_t def) const {
  const Item* it = find(key);
  if (!it) return def;
  if (it->kind == Item::INT) return it->i;
  opt_kind(key, *it, "an int (set it with .int)");
}
std::string Opts::str(const std::string& key, const std::string& def) const {
  const Item* it = find(key);
  if (!it) return def;
  if (it->kind == Item::STR) return it->s;
  opt_kind(key, *it, "a string");
}
bool Opts::flag(const std::string& key, bool def) const {
  const Item* it = find(key);
  if (!it) return def;
  if (it->kind == Item::BOOL) return it->b;
  opt_kind(key, *it, "a bool");
}
const std::vector<double>* Opts::vec(const std::string& key) const {
  const Item* it = find(key);
  if (!it) return nullptr;
  if (it->kind == Item::VEC) return &it->vec;
  opt_kind(key, *it, "a Vec");
}
const std::vector<std::string>* Opts::strs(const std::string& key) const {
  const Item* it = find(key);
  if (!it) return nullptr;
  if (it->kind == Item::STRS) return &it->strs;
  opt_kind(key, *it, "a StrVec");
}
const Eigen::MatrixXd* Opts::mat(const std::string& key) const {
  const Item* it = find(key);
  if (!it) return nullptr;
  if (it->kind == Item::MAT) return &it->mat;
  opt_kind(key, *it, "a Mat");
}
void Opts::finish(const std::string& op) const {
  std::string unknown;
  for (size_t i = 0; i < items.size(); ++i)
    if (!used[i]) unknown += (unknown.empty() ? "" : ", ") + items[i].first;
  if (!unknown.empty()) fail(op + ": unknown option(s) " + unknown + " (see `lmpy --api-md` for the options of " + op + ")");
}
const Opts& no_opts() { static Opts o; return o; }

// ---- Res ----------------------------------------------------------------------------------------------------------
Res& Res::put(const std::string& k, Item it) {
  for (auto& kv : items)
    if (kv.first == k) { kv.second = std::move(it); return *this; }
  items.emplace_back(k, std::move(it));
  return *this;
}
const Item* Res::find(const std::string& k) const {
  for (auto& kv : items)
    if (kv.first == k) return &kv.second;
  return nullptr;
}
std::string Res::keys_text() const {
  std::string out;
  for (auto& kv : items) out += (out.empty() ? "" : ", ") + kv.first + ":" + item_kind_name(kv.second.kind);
  return out;
}
const Item& Res::get(const std::string& k, Item::Kind want) const {
  const Item* it = find(k);
  if (!it) fail("result has no field '" + k + "' (fields: " + keys_text() + ")");
  if (it->kind != want && !(want == Item::NUM && it->kind == Item::INT))
    fail("result field '" + k + "' is " + item_kind_name(it->kind) + ", not " + item_kind_name(want));
  return *it;
}

// ---- JSON ---------------------------------------------------------------------------------------------------------
std::string jnum(double v) {
  if (!std::isfinite(v)) return "null";
  if (v == 0) return std::signbit(v) ? "-0.0" : "0";
  char buf[64];
  auto r = std::to_chars(buf, buf + sizeof buf, v);
  return std::string(buf, r.ptr);
}
std::string jstr(const std::string& s) { return dslvm::json_escape(s); }
std::string jvec(const std::vector<double>& v) {
  std::string out = "[";
  for (size_t i = 0; i < v.size(); ++i) out += (i ? "," : "") + jnum(v[i]);
  return out + "]";
}
std::string jstrs(const std::vector<std::string>& v) {
  std::string out = "[";
  for (size_t i = 0; i < v.size(); ++i) out += (i ? "," : "") + jstr(v[i]);
  return out + "]";
}
std::string jmat(const Eigen::MatrixXd& m) {
  std::string out = "[";
  for (Eigen::Index r = 0; r < m.rows(); ++r) {
    out += r ? ",[" : "[";
    for (Eigen::Index c = 0; c < m.cols(); ++c) out += (c ? "," : "") + jnum(m(r, c));
    out += "]";
  }
  return out + "]";
}
std::string jitem(const Item& it) {
  switch (it.kind) {
    case Item::NUM: return jnum(it.f);
    case Item::INT: return std::to_string(it.i);
    case Item::STR: return jstr(it.s);
    case Item::BOOL: return it.b ? "true" : "false";
    case Item::VEC: return jvec(it.vec);
    case Item::STRS: return jstrs(it.strs);
    case Item::MAT: return jmat(it.mat);
    case Item::RES: return it.res ? jres(*it.res) : "null";
  }
  return "null";
}
std::string jres(const Res& r) {
  std::string out = "{";
  for (size_t i = 0; i < r.items.size(); ++i) out += (i ? "," : "") + jstr(r.items[i].first) + ":" + jitem(r.items[i].second);
  return out + "}";
}

Eigen::VectorXd to_eigen(const std::vector<double>& v) { return Eigen::Map<const Eigen::VectorXd>(v.data(), Eigen::Index(v.size())); }
std::vector<double> from_eigen(const Eigen::VectorXd& v) { return std::vector<double>(v.data(), v.data() + v.size()); }
size_t index_arg(int64_t i, size_t n, const char* what) {
  int64_t k = i < 0 ? int64_t(n) + i : i;
  if (k < 0 || k >= int64_t(n)) fail(std::string(what) + ": index " + std::to_string(i) + " is out of range for length " + std::to_string(n));
  return size_t(k);
}

// ---- Api ----------------------------------------------------------------------------------------------------------
void Api::section(const std::string& name, const std::string& doc) {
  current = name;
  if (!section_docs.count(name)) { sections.push_back(name); section_docs[name] = doc; }
}
void Api::type(const std::string& name, const std::string& doc) {
  auto& t = types[name];
  t.doc = doc;
  t.section = current;
}
void Api::fn(const std::string& name, std::vector<std::string> params, const std::string& returns, const std::string& doc, Fn f) {
  reg.function(name, params, returns, std::move(f));
  functions[name].push_back({std::move(params), returns, doc, current});
}
void Api::method(const std::string& type, const std::string& name, std::vector<std::string> params, const std::string& returns, const std::string& doc, Meth m) {
  if (!types.count(type)) throw std::logic_error("method on undeclared type " + type);
  reg.method(type, name, params, returns, std::move(m));
  types[type].methods[name].push_back({std::move(params), returns, doc, current});
}
void Api::op(const std::string& type, const std::string& o, const std::string& method) { types[type].operators[o] = method; }
void Api::conversion(const std::string& from, const std::string& to, const std::string& via) { conversions.push_back({from, to, via}); }
void Api::enumeration(const std::string& name, std::vector<std::string> values) { enums[name] = std::move(values); }
void Api::ctx(const std::string& name, const std::string& type, const std::string& doc) {
  reg.context(name, type);
  context[name] = type;
  section_docs["context:" + name] = doc;
}

static std::string jparams(const std::vector<std::string>& p) { return jstrs(p); }

std::string Api::schema_json() const {
  std::ostringstream o;
  auto overloads = [&](const std::vector<Overload>& list, const std::string& indent) {
    o << "[";
    for (size_t i = 0; i < list.size(); ++i)
      o << (i ? ",\n" + indent + "  " : "") << "{\"params\": " << jparams(list[i].params) << ", \"returns\": " << jstr(list[i].returns) << "}";
    o << "]";
  };
  o << "{\n  \"dsl.schema\": 2,\n  \"name\": \"lmpy\",\n  \"types\": {";
  bool first = true;
  for (auto& [tname, t] : types) {
    o << (first ? "\n" : ",\n") << "    " << jstr(tname) << ": {\n      \"methods\": {";
    first = false;
    bool fm = true;
    for (auto& [mname, list] : t.methods) {
      o << (fm ? "\n" : ",\n") << "        " << jstr(mname) << ": ";
      fm = false;
      overloads(list, "        ");
    }
    o << (t.methods.empty() ? "}" : "\n      }");
    if (!t.operators.empty()) {
      o << ",\n      \"operators\": {";
      bool fo = true;
      for (auto& [k, v] : t.operators) { o << (fo ? "" : ", ") << jstr(k) << ": " << jstr(v); fo = false; }
      o << "}";
    }
    o << "\n    }";
  }
  o << "\n  },\n  \"functions\": {";
  first = true;
  for (auto& [fname, list] : functions) {
    o << (first ? "\n" : ",\n") << "    " << jstr(fname) << ": ";
    first = false;
    overloads(list, "    ");
  }
  o << "\n  }";
  if (!enums.empty()) {
    o << ",\n  \"enums\": {";
    first = true;
    for (auto& [k, v] : enums) { o << (first ? "\n" : ",\n") << "    " << jstr(k) << ": " << jstrs(v); first = false; }
    o << "\n  }";
  }
  if (!conversions.empty()) {
    o << ",\n  \"conversions\": [";
    for (size_t i = 0; i < conversions.size(); ++i)
      o << (i ? ",\n" : "\n") << "    {\"from\": " << jstr(conversions[i].from) << ", \"to\": " << jstr(conversions[i].to) << ", \"via\": " << jstr(conversions[i].via) << "}";
    o << "\n  ]";
  }
  o << ",\n  \"context\": {";
  first = true;
  for (auto& [k, v] : context) { o << (first ? "" : ", ") << jstr(k) << ": " << jstr(v); first = false; }
  o << "}\n}\n";
  return o.str();
}

static std::string sig(const std::string& name, const Api::Overload& ov) {
  std::string s = name + "(";
  for (size_t i = 0; i < ov.params.size(); ++i) s += (i ? ", " : "") + ov.params[i];
  return s + ") -> " + ov.returns;
}

std::string Api::summary() const {
  std::ostringstream o;
  o << "context:\n";
  for (auto& [k, v] : context) o << "  " << k << ": " << v << "\n";
  o << "functions:\n";
  for (auto& [n, list] : functions)
    for (auto& ov : list) o << "  " << sig(n, ov) << "\n";
  for (auto& [t, def] : types) {
    o << t << ":\n";
    for (auto& [n, list] : def.methods)
      for (auto& ov : list) o << "  " << sig(n, ov) << "\n";
  }
  return o.str();
}

std::string Api::markdown() const {
  std::ostringstream o;
  o << "# lmpy DSL API reference\n\n"
       "Generated by `lmpy --api-md` from the host registry (the same definitions produce `schema/lmpy.dsl.json` and the\n"
       "registry the bytecode is bound to). Do not edit by hand: change the C++ definition and regenerate.\n\n"
       "The lmcode std v1 catalog (`str_*`, `math_*`, `int_list`/`float_list`/`str_list`/`str_map`, `file_*`) is also available in\n"
       "every block: the lmcode-cpp runtime installs it and lmscript (0.2.420 and later) declares it, so it is not listed here or in\n"
       "the schema. Reference: https://github.com/lmctl-ai/lmcode-cpp/blob/main/docs/std.md\n\n";
  o << "## Context\n\n";
  for (auto& [k, v] : context) {
    auto it = section_docs.find("context:" + k);
    o << "- `" << k << ": " << v << "` — " << (it == section_docs.end() ? "" : it->second) << "\n";
  }
  o << "\n";
  for (auto& sec : sections) {
    o << "## " << sec << "\n\n" << section_docs.at(sec) << "\n\n";
    bool any = false;
    for (auto& [n, list] : functions)
      for (auto& ov : list)
        if (ov.section == sec) {
          if (!any) { o << "### Functions\n\n| Signature | Meaning |\n|---|---|\n"; any = true; }
          o << "| `" << sig(n, ov) << "` | " << ov.doc << " |\n";
        }
    if (any) o << "\n";
    for (auto& [t, def] : types) {
      bool here = def.section == sec;
      bool members = false;
      for (auto& [n, list] : def.methods)
        for (auto& ov : list) members |= ov.section == sec;
      if (!here && !members) continue;
      o << "### " << (here ? "Type " : "More methods of ") << "`" << t << "`\n\n";
      if (here && !def.doc.empty()) o << def.doc << "\n\n";
      if (here && !def.operators.empty()) {
        o << "Operators: ";
        bool f = true;
        for (auto& [k, v] : def.operators) { o << (f ? "" : ", ") << "`" << k << "` → `" << v << "`"; f = false; }
        o << "\n\n";
      }
      o << "| Method | Meaning |\n|---|---|\n";
      for (auto& [n, list] : def.methods)
        for (auto& ov : list)
          if (ov.section == sec) o << "| `" << sig(n, ov) << "` | " << ov.doc << " |\n";
      o << "\n";
    }
  }
  if (!enums.empty()) {
    o << "## Enums\n\n";
    for (auto& [k, v] : enums) {
      o << "- `" << k << "`: ";
      for (size_t i = 0; i < v.size(); ++i) o << (i ? ", " : "") << "`" << k << "." << v[i] << "`";
      o << "\n";
    }
    o << "\n";
  }
  if (!conversions.empty()) {
    o << "## Conversions\n\n";
    for (auto& c : conversions) o << "- `" << c.from << "` → `" << c.to << "` via `" << c.via << "`\n";
    o << "\n";
  }
  return o.str();
}

}  // namespace lmpy
