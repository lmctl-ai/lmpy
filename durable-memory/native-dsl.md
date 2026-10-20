# lmnum: pycli as one static native executable driven by `_dsl_` bytecode

Status (2026-10-20): foundation built on branch `native-dsl`. This document sets the direction and the conventions every
op family follows. It supersedes the language choice in [native-plan.md](native-plan.md) (Rust, lmapi ops). That plan's
op mapping, clean-room rules (§4) and tolerance tiers (§3) still apply.

## 1. What changes

Today's pycli is an lmapi CLI: lmscript calls `pycli.op({...})`, a C++ front end validates the call, and a Python child
runs it inside Docker. lmnum replaces the whole chain with the `_dsl_` model used by lmcad and lmhtml:

    examples/dsl/x.lms                          native/build/lmnum (static, no Python/R/Docker)
    prog = _dsl_ schema="lmnum.dsl.json";   ──▶  lmnum --dsl x.dslb
      let f = read_csv("data.csv");              loads the bytecode, binds it to the embedded schema (SHA-256),
      let fit = ols(f, "y", strs("x1,x2"));      runs it, then prints one JSON line:
      out.put("r2", fit.num("r2"));              {"ok": true, "result": {"values": {...}, "checks": {...}}}
    end _dsl_
    v = lmnum.run(prog, "x");   # savedsl + system([lmnum, "--dsl", file]) + json_parse

- **One program, many steps.** A DSL chunk chains operations in native memory: read → transform → fit → check → report.
  There is no process, JSON or file round trip per step. pycli needed `_to` twins and safetensors artifacts to pass arrays
  between calls; in lmnum, arrays are values (`Vec`, `Mat`, `Frame`).
- **Typed at compile time.** lmscript checks every call against the schema before anything runs (wrong argument type,
  unknown method, misspelt function = located compile error, exit 2).
- **No isolation boundary to maintain.** The DSL accepts no caller code beyond what the host offers (R33: the bytecode is
  trusted like code). The expression strings (`"100*(y-x^2)^2"`) are parsed by the host's closed grammar (pycli's
  `expr-spec.json`), not by an interpreter.

## 2. Build, run, test

    native/scripts/fetch-deps.sh          # pinned sources from native/deps.lock (sha256), builds HiGHS static; network only here
    cmake -S native -B native/build -G Ninja && cmake --build native/build
    native/test.sh                        # static check, schema/API drift, signature parity, every examples/dsl/*.lms
    LMNUM=$PWD/native/build/lmnum lmscript examples/dsl/lstsq-fit.lms

- The binary is fully static (glibc static, libstdc++ static) and is about 4 MB with the core families.
- `lmnum --schema --json` prints the schema the bytecode binds to. It is generated from the registry.
  `native/schema/lmnum.dsl.json` is a committed copy, and `test.sh` fails when it drifts.
- `lmnum --api-md` prints the API reference. `native/API.md` is a committed copy and is drift-checked too.
- `lmnum --dsl-signatures` must equal `lmscript --dsl-signatures native/schema/lmnum.dsl.json`.
- Development-only oracle: `native/.deps/oracle-venv` has pycli's exact pins (numpy 2.3.3, scipy 1.16.2, pandas 2.3.3,
  statsmodels 0.14.6, scikit-learn 1.7.2) for pinning expected values. `Rscript` is on the host too. Neither is used at
  run time or by the gate.

## 3. Libraries (pure C/C++, permissive, statically linked)

| Library | Use | Licence |
|---|---|---|
| Eigen 3.4.0 (`EIGEN_MPL2_ONLY`) | dense/sparse linear algebra, decompositions, matrix functions | MPL-2.0 (unmodified files; source available upstream) |
| Boost 1.86 (header-only parts) | Boost.Math special functions and distributions, quadrature, root finding, odeint, Boost.Graph | BSL-1.0 |
| pocketfft (C++ header) | FFT (the same code numpy/scipy use) | BSD-3-Clause |
| nanoflann 1.6.1 | kd-tree k-nearest neighbours | BSD-2-Clause |
| LBFGSpp 0.3.0 | L-BFGS / L-BFGS-B | MIT |
| HiGHS 1.7.2 (static archive) | linear and mixed-integer programming (scipy's `linprog(method="highs")` solver) | MIT |

Adding a library: pin it in `native/deps.lock` (url, sha256), extract it in `fetch-deps.sh`, add its include path in
`CMakeLists.txt`, and add its notice to `THIRD_PARTY_NOTICES.md`. GPL, LGPL-only, Numerical Recipes and ACM-licensed
code are never allowed (native-plan §4). Nothing may need a GUI, a display or a shared library at run time.

## 4. Source layout and how to add a family

    native/cpp/host.hpp      Api builder (one definition → registry + schema + API.md), objects (Vec, StrVec, Mat, Opts, Res), JSON
    native/cpp/core.cpp      Vec, StrVec, Mat, Opts, Res, utilities
    native/cpp/out.cpp       the `out` context (put/check/close/log/fail)
    native/cpp/expr.cpp      expression engine (pycli expr-spec grammar; scalars, vectors, reducers)
    native/cpp/frame.cpp     Frame basics (CSV, columns, filter/mutate/select/sort)   frame_ops.cpp: groupby/merge/rolling/resample, array files
    native/cpp/linalg.cpp    dense linear algebra                                    sparse.cpp
    native/cpp/stats.cpp     distributions, special functions, RNG, tests
    native/cpp/optimize.cpp  minimize, least squares, roots, quadrature, linprog
    native/cpp/ode.cpp interp.cpp signal.cpp spatial.cpp graph.cpp models.cpp ml.cpp
    native/cpp/families.cpp  calls every register_* in a fixed order

Rules for a family file:

1. Expose one `void register_<family>(Api& api)`. Start it with `api.section("Title", "one-paragraph description")`.
   Declare new object types with `api.type(...)`, then members with `api.fn` / `api.method`. Every member has a one-line
   doc string, because API.md is generated from them.
2. **Names.** Functions are lower_snake verbs or nouns named after the numpy/scipy/pandas/statsmodels/sklearn concept
   (`linspace`, `read_csv`, `minimize`, `least_squares`, `ols`, `dist`, `find_peaks`). The doc string names the pycli op it
   replaces. Do not use DSL built-in names as host functions: `sin cos sqrt abs floor pow min max clamp lerp ease ease_in
   ease_out int float str range`.
3. **Options** go in one trailing `Opts` argument. Also provide an overload without it when the defaults are useful.
   - Option keys are pycli's parameter names (`method`, `tol`, `maxiter`, `rtol`, `t_eval`, ...).
   - Read every key through `Opts` getters, then call `o.finish("op")`, so unknown keys are errors.
   - **Named constants and data in expressions:** an option key that starts with `$` binds a name in that operation's
     expressions. `opts().set("$beta", 0.3)` gives a scalar; `.set("$x", xs)` gives a data vector, read as `x[i]` or
     through reducers (`sum((y - a*exp(-b*x))^2)`).
4. **Results.** Return a `Res` whose field names and kinds follow pycli's JSON result (`x`, `fun`, `status`, `nfev`, `par`,
   `coefficients`, ...).
   - Use nested `Res` (`r.res("coefficients").vec("estimate")`) where pycli nests objects.
   - Use `Vec`/`Mat` where pycli returns arrays.
   - Keep pycli's `status` strings (`OK`, `NO_CONVERGENCE`, `RANK_DEFICIENT`, ...) for numerical outcomes.
   - Bad input (wrong shapes, unknown method, non-finite data where finite is required) is a `fail(...)` run-time error
     whose message names the operation and the problem.
   - A typed object (e.g. `Dist`, `Model`, `Filter`, `Interp`, `Graph`) is better than a `Res` when the result is used
     again: evaluating a distribution, predicting, filtering.
5. **Numerics.** Single-threaded, deterministic, IEEE (no fast-math; `-ffp-contract=off`). Random numbers come only from an
   explicit integer seed. Iterative methods return a status and never hang: keep pycli's caps (maxiter, max_evals).
6. **Testing.** Port the family's examples (§5). Pin expected values from the oracle venv in the example itself, with the
   tolerance tier of native-plan §3. Then run `native/test.sh`.

## 5. Examples

- `examples/dsl/<name>.lms` is the `_dsl_` version of `examples/<name>.lms`.
  - Same base name, same data files (`../data/...`), same known answers and tolerances.
  - The checks move into the DSL as `out.check` / `out.close`. A failed check exits 4 and lmscript prints the host's
    message.
  - The lmscript part keeps the human-readable output of the original and ends with the same "passed" line.
- `examples/dsl/task-*.lms` are new multi-step workflows. Each chains several families in one DSL program; that is the
  point of the model.
- `examples/dsl/lmnum.lms` is the shared runner (`lmnum.run(prog, name)`). It prints `SKIPPED` and exits 0 when lmnum
  is not installed.
- The `oracle-*` and `cross-*` originals compare pycli with sibling tools (rcli, statsrust, lmda) at run time. Their DSL
  versions instead assert the pinned known answers those comparisons established (from the oracle venv or R),
  because the point here is lmnum's own correctness.
