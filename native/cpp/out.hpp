// The `out` context object: what a DSL chunk reports back to the lmscript caller.
#pragma once
#include "host.hpp"

namespace lmpy {

struct Out {
  static constexpr const char* TYPE = "Out";
  std::vector<std::pair<std::string, std::string>> values;  // key -> JSON text, in first-put order
  struct Check { std::string name; bool ok; std::string detail; };
  std::vector<Check> checks;
  void put(const std::string& key, std::string json);
  std::string result_json() const;  // {"values": {...}, "checks": {"passed": n, "failed": m, "failures": [...]}}
  std::string failure_text() const; // "" when every check passed
  void reset() { values.clear(); checks.clear(); }
};
Out& the_out();
void register_out(Api& api);
std::string fmt_fixed(double x, int64_t digits);

}  // namespace lmpy
