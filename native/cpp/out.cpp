#include "out.hpp"
#include "frame.hpp"
#include <iostream>

namespace lmpy {

void Out::put(const std::string& key, std::string json) {
  for (auto& kv : values)
    if (kv.first == key) { kv.second = std::move(json); return; }
  values.emplace_back(key, std::move(json));
}

std::string Out::result_json() const {
  std::string o = "{\"values\": {";
  for (size_t i = 0; i < values.size(); ++i) o += (i ? ", " : "") + jstr(values[i].first) + ": " + values[i].second;
  size_t passed = 0;
  std::string failures;
  for (auto& c : checks) {
    if (c.ok) ++passed;
    else failures += (failures.empty() ? "" : ", ") + std::string("{\"name\": ") + jstr(c.name) + ", \"detail\": " + jstr(c.detail) + "}";
  }
  o += "}, \"checks\": {\"passed\": " + std::to_string(passed) + ", \"failed\": " + std::to_string(checks.size() - passed) + ", \"failures\": [" + failures + "]}}";
  return o;
}

std::string Out::failure_text() const {
  std::string t;
  size_t n = 0;
  for (auto& c : checks)
    if (!c.ok) { ++n; t += (t.empty() ? "" : "; ") + c.name + (c.detail.empty() ? "" : " (" + c.detail + ")"); }
  return n ? std::to_string(n) + " check(s) failed: " + t : "";
}

void register_out(Api& api) {
  const std::string Ot = "Out";
  api.section("Reporting", "The `out` context object collects named values and checks. The host prints them as the single JSON "
                           "result line `{\"ok\": true, \"result\": {\"values\": {...}, \"checks\": {...}}}`; a failed check makes the run an "
                           "application failure (exit 4) whose message names every failed check.");
  api.type(Ot, "Result collector (the context variable `out`).");
  api.ctx("out", Ot, "values and checks returned to the caller as JSON");
  auto& o = the_out;
  api.method(Ot, "put", {"string", "float"}, "void", "record a float", [o](void*, Args& a) { o().put(a[0].s, jnum(a[1].f)); return Value::none(); });
  api.method(Ot, "put", {"string", "int"}, "void", "record an int", [o](void*, Args& a) { o().put(a[0].s, std::to_string(a[1].i)); return Value::none(); });
  api.method(Ot, "put", {"string", "string"}, "void", "record a string", [o](void*, Args& a) { o().put(a[0].s, jstr(a[1].s)); return Value::none(); });
  api.method(Ot, "put", {"string", "bool"}, "void", "record a bool", [o](void*, Args& a) { o().put(a[0].s, a[1].b ? "true" : "false"); return Value::none(); });
  api.method(Ot, "put", {"string", "Vec"}, "void", "record a vector (JSON array)", [o](void*, Args& a) { o().put(a[0].s, jvec(obj<Vec>(a[1]).v)); return Value::none(); });
  api.method(Ot, "put", {"string", "StrVec"}, "void", "record a list of strings", [o](void*, Args& a) { o().put(a[0].s, jstrs(obj<StrVec>(a[1]).v)); return Value::none(); });
  api.method(Ot, "put", {"string", "Mat"}, "void", "record a matrix (nested arrays)", [o](void*, Args& a) { o().put(a[0].s, jmat(obj<Mat>(a[1]).m)); return Value::none(); });
  api.method(Ot, "put", {"string", "Res"}, "void", "record a whole result object", [o](void*, Args& a) { o().put(a[0].s, jres(obj<Res>(a[1]))); return Value::none(); });
  api.method(Ot, "put", {"string", "Frame"}, "void", "record a table as {columns, rows, data}", [o](void*, Args& a) { o().put(a[0].s, frame_to_json(obj<Frame>(a[1]))); return Value::none(); });
  api.method(Ot, "check", {"string", "bool"}, "bool", "record a named check; returns the condition", [o](void*, Args& a) {
    o().checks.push_back({a[0].s, a[1].b, ""}); return B(a[1].b); });
  api.method(Ot, "close", {"string", "float", "float", "float"}, "bool", "check |got - want| <= tol (NaN fails)", [o](void*, Args& a) {
    double got = a[1].f, want = a[2].f, tol = a[3].f;
    bool ok = std::fabs(got - want) <= tol;
    o().checks.push_back({a[0].s, ok, ok ? "" : "got " + jnum(got) + ", want " + jnum(want) + " +- " + jnum(tol)});
    return B(ok);
  });
  api.method(Ot, "close_rel", {"string", "float", "float", "float"}, "bool", "check |got - want| <= rtol * max(|want|, 1e-300)", [o](void*, Args& a) {
    double got = a[1].f, want = a[2].f, rtol = a[3].f;
    bool ok = std::fabs(got - want) <= rtol * std::max(std::fabs(want), 1e-300);
    o().checks.push_back({a[0].s, ok, ok ? "" : "got " + jnum(got) + ", want " + jnum(want) + " (rtol " + jnum(rtol) + ")"});
    return B(ok);
  });
  api.method(Ot, "close", {"string", "Vec", "Vec", "float"}, "bool", "check every |got[i] - want[i]| <= tol (lengths must match)", [o](void*, Args& a) {
    const auto& g = obj<Vec>(a[1]).v; const auto& w = obj<Vec>(a[2]).v;
    bool ok = g.size() == w.size();
    std::string detail = ok ? "" : "lengths " + std::to_string(g.size()) + " and " + std::to_string(w.size());
    for (size_t i = 0; ok && i < g.size(); ++i)
      if (!(std::fabs(g[i] - w[i]) <= a[3].f)) { ok = false; detail = "element " + std::to_string(i) + ": got " + jnum(g[i]) + ", want " + jnum(w[i]); }
    o().checks.push_back({a[0].s, ok, detail});
    return B(ok);
  });
  api.method(Ot, "log", {"string"}, "void", "write a line to stderr (stdout is reserved for the JSON result)", [](void*, Args& a) {
    std::cerr << a[0].s << "\n"; return Value::none(); });
  api.method(Ot, "fail", {"string"}, "void", "stop now with an application failure (exit 4)", [](void*, Args& a) -> Value {
    throw dslvm::Error("APP_FAIL", a[0].s); });
}

}  // namespace lmpy
