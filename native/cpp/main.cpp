// lmpy: one self-contained executable that runs lmscript `_lmcode_` bytecode against a numerical host library
// (linear algebra, statistics, optimisation, signal processing, tables, models) with no Python, R or Docker.
#include "host.hpp"
#include "out.hpp"
#include "version.hpp"
#include <cstring>
#include <iostream>
#include <optional>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

using namespace lmpy;

static int usage(int code) {
  (code ? std::cerr : std::cout)
      << "usage: lmpy [--dry-run] --lmcode FILE [FILE ...]   run bytecode compiled by lmscript (`_lmcode_ schema=\"lmpy.dsl.json\"`)\n"
         "       lmpy --lmcode-schema [--json]      print the API summary, or the exact schema bytes the bytecode binds to\n"
         "                                       (lmscript: _lmcode_ cli=\"lmpy\" reads it; --schema is an older alias)\n"
         "       lmpy --lmcode-signatures           canonical signatures (diff with `lmscript --lmcode-signatures SCHEMA`)\n"
         "       lmpy --api-md                   the API reference as Markdown\n"
         "       lmpy --version\n";
  return code ? dslvm::failure_frame("cli", dslvm::Error("CLI_USAGE", "invalid lmpy arguments")) : 0;
}

int main(int argc, char** argv) {
#ifdef _WIN32
  // text mode would turn \n into \r\n and break the byte-exact schema and the terminal result frame
  _setmode(_fileno(stdout), _O_BINARY);
  _setmode(_fileno(stderr), _O_BINARY);
#endif
  Api api;
  try {
    register_core(api);
    register_frame(api);
    register_linalg(api);
    register_families(api);
  } catch (const std::exception& e) {
    std::cerr << "lmpy: registry error: " << e.what() << "\n";
    return dslvm::failure_frame("load", dslvm::Error("LOAD_BAD_SCHEMA", e.what()));
  }
  if (argc < 2) return usage(2);
  std::string cmd = argv[1];
  if (cmd == "--help" || cmd == "-h") return usage(0);
  if (cmd == "--version") { std::cout << "lmpy " << LMPY_VERSION << "\n"; return 0; }
  if (cmd == "--lmcode-schema" || cmd == "--schema") {
    if (argc == 2) { std::cout << api.summary(); return 0; }
    if (argc == 3 && std::string(argv[2]) == "--json") { std::cout << api.schema_json(); return 0; }
    return usage(2);
  }
  if (cmd == "--lmcode-signatures" && argc == 2) {
    for (auto& s : api.reg.signatures()) std::cout << s << "\n";
    return 0;
  }
  if (cmd == "--api-md" && argc == 2) { std::cout << api.markdown(); return 0; }
  bool dry = cmd == "--dry-run";
  int first = dry ? 3 : 2;
  if (argc <= first || std::string(argv[first - 1]) != "--lmcode") return usage(2);
  std::vector<std::string> files;
  for (int i = first; i < argc; ++i) {
    if (!argv[i][0] || std::strncmp(argv[i], "--", 2) == 0) return usage(2);
    files.push_back(argv[i]);
  }
  std::optional<dslvm::InvocationResults> holder;
  try {
    if (dry && !std::getenv("LMCODE_INVOCATION_FILE"))
      holder.emplace(dslvm::InvocationResults::from_command(files, true));
    else holder.emplace(dslvm::InvocationResults::from_command(files));
  } catch (const dslvm::Error& e) { return dslvm::failure_frame("cli", e); }
  auto& results = *holder;
  if (dry && !results.dry())
    return dslvm::failure_frame("cli", dslvm::Error("CLI_USAGE", "--dry-run contradicts real-mode metadata"));
  const std::string schema = api.schema_json();
  Out out_ctx_tag;
  dslvm::Context ctx;
  ctx.set("out", &out_ctx_tag);
  dslvm::ResultContract contract;
  try { contract = dslvm::ResultContract::from_schema(schema); }
  catch (const dslvm::Error& e) { results.fail(std::nullopt, "load", e); return results.write_frame(std::cout); }
  std::vector<std::pair<size_t, dslvm::Program>> programs;
  // A malformed later part must not leave earlier file/model effects behind.
  for (const auto& part : results.registrations()) {
    try { programs.emplace_back(part.index, dslvm::Program::load(dslvm::read_file(part.file), schema, api.reg)); }
    catch (const dslvm::Error& e) { results.fail(part.index, "load", e); return results.write_frame(std::cout); }
    catch (const std::exception& e) { results.fail(part.index, "load", dslvm::Error(e.what())); return results.write_frame(std::cout); }
  }
  for (auto& [index, program] : programs) {
    the_out().reset();
    try {
      auto& sink = results.select(index, api.reg, contract);
      if (!results.dry()) {
        program.run(ctx, api.reg);
        const auto failed = the_out().failure_text();
        if (!failed.empty()) {
          std::cerr << "lmpy: " << the_out().result_json() << "\n";
          results.fail(index, "app", dslvm::Error("APP_FAIL", failed)); break;
        }
        // Preserve the native values/checks payload; explicit lmcode_return takes precedence.
        if (!sink.written()) {
          auto payload = dslvm::invocation_detail::parse(the_out().result_json());
          sink.set(dslvm::Value::object("Json", &payload));
        }
      }
      results.complete(index);
    } catch (const dslvm::Error& e) {
      if (!the_out().values.empty()) std::cerr << "lmpy: values before the failure: " << the_out().result_json() << "\n";
      results.fail(index, e.code == "APP_FAIL" ? "app" : "run", e); break;
    } catch (const std::exception& e) { results.fail(index, "run", dslvm::Error(e.what())); break; }
  }
  return results.write_frame(std::cout);
}
