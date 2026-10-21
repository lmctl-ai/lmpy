// lmpy: one self-contained executable that runs lmscript `_dsl_` bytecode against a numerical host library
// (linear algebra, statistics, optimisation, signal processing, tables, models) with no Python, R or Docker.
#include "host.hpp"
#include "out.hpp"
#include "version.hpp"
#include <cstring>
#include <iostream>

using namespace lmpy;

static int usage(int code) {
  (code ? std::cerr : std::cout)
      << "usage: lmpy --dsl FILE [FILE ...]   run bytecode compiled by lmscript (`_dsl_ schema=\"lmpy.dsl.json\"`)\n"
         "       lmpy --schema [--json]          print the API summary, or the exact schema bytes the bytecode binds to\n"
         "       lmpy --dsl-signatures           canonical signatures (diff with `lmscript --dsl-signatures SCHEMA`)\n"
         "       lmpy --api-md                   the API reference as Markdown\n"
         "       lmpy --version\n";
  return code;
}

int main(int argc, char** argv) {
  Api api;
  try {
    register_core(api);
    register_frame(api);
    register_linalg(api);
    register_families(api);
  } catch (const std::exception& e) {
    std::cerr << "lmpy: registry error: " << e.what() << "\n";
    return 70;
  }
  if (argc < 2) return usage(2);
  std::string cmd = argv[1];
  if (cmd == "--help" || cmd == "-h") return usage(0);
  if (cmd == "--version") { std::cout << "lmpy " << LMPY_VERSION << " (dsl bytecode v" << dslvm::VERSION << ")\n"; return 0; }
  if (cmd == "--schema") {
    if (argc == 2) { std::cout << api.summary(); return 0; }
    if (argc == 3 && std::string(argv[2]) == "--json") { std::cout << api.schema_json(); return 0; }
    return usage(2);
  }
  if (cmd == "--dsl-signatures" && argc == 2) {
    for (auto& s : api.reg.signatures()) std::cout << s << "\n";
    return 0;
  }
  if (cmd == "--api-md" && argc == 2) { std::cout << api.markdown(); return 0; }
  if (cmd != "--dsl" || argc < 3) return usage(2);
  for (int i = 2; i < argc; ++i)
    if (!argv[i][0] || std::strncmp(argv[i], "--", 2) == 0) return usage(2);

  const std::string schema = api.schema_json();
  Out out_ctx_tag;  // the context pointer; methods use the_out()
  dslvm::Context ctx;
  ctx.set("out", &out_ctx_tag);
  for (int i = 2; i < argc; ++i) {
    dslvm::Program program;
    try {
      program = dslvm::Program::load(dslvm::read_file(argv[i]), schema, api.reg);
    } catch (const dslvm::Error& e) {
      return dslvm::report_error("load", e);
    }
    the_out().reset();
    try {
      program.run(ctx, api.reg);
    } catch (const dslvm::Error& e) {
      if (!the_out().values.empty()) std::cerr << "lmpy: values before the failure: " << the_out().result_json() << "\n";
      return dslvm::report_error(e.code == "APP_FAIL" ? "app" : "run", e);
    }
    std::string failed = the_out().failure_text();
    if (!failed.empty()) {
      std::cerr << "lmpy: " << the_out().result_json() << "\n";
      return dslvm::report_app_failure(failed);
    }
    dslvm::report_ok(the_out().result_json());
  }
  return 0;
}
