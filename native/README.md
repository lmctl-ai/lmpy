# lmpy

`lmpy` is pycli's numerical toolbox as **one C++17 executable**. Every library is linked in except the system C library
(glibc), so it needs no Python, R, Docker, GUI or other shared library at run time. Programs are written as lmscript `_lmcode_` blocks. lmscript type-checks a block against
[`schema/lmpy.dsl.json`](schema/lmpy.dsl.json) and compiles it to bytecode; `lmpy --dsl FILE` runs the bytecode and
prints one JSON result line.

```lmscript
import "lmpy.lms";                       // examples/dsl/lmpy.lms: savedsl + system + json_parse
prog = _lmcode_ schema="../../native/schema/lmpy.dsl.json";
let f = read_csv("../data/tut-ols.csv");
let fit = ols(f, "y", strs("x1, x2"));
out.put("r2", fit.num("r2"));
out.close("slope", fit.res("coefficients").vec("estimate").at(1), 0.82797, 0.0001);
end _lmcode_
v = lmpy.run(prog, "ols");               // {"r2": ...}; a failed out.close/out.check exits 4 with the check's name
```

## Build

```sh
native/scripts/fetch-deps.sh                     # pinned sources (native/deps.lock, sha256-checked); builds HiGHS static
cmake -S native -B native/build -G Ninja && cmake --build native/build
native/test.sh                                   # glibc-only check, schema/API drift, signature parity, every examples/dsl/*.lms
```

You need a C++17 compiler (GCC 13 here), CMake ≥ 3.24, Ninja and network access for the fetch step. Running the examples
needs `lmscript` from `@lmctl-ai/lmctl` ≥ 0.2.420 (the `_lmcode_` block keyword and the built-in lmcode std catalog). The result is `native/build/lmpy`. `ldd` lists only glibc.

## Command line

| Command | Output |
|---|---|
| `lmpy --dsl FILE [FILE ...]` | runs the bytecode files in order; one JSON line per file: `{"ok": true, "result": {"values": {...}, "checks": {...}}}` or `{"ok": false, "phase": "load\|run\|app", "code": ..., "message": "dsl line N:C: ...", "line": N}` |
| `lmpy --dsl-schema [--json]` | API summary, or the exact schema bytes the bytecode binds to |
| `lmpy --dsl-signatures` | canonical signatures; must equal `lmscript --dsl-signatures schema/lmpy.dsl.json` |
| `lmpy --api-md` | the generated API reference ([API.md](API.md)) |
| `lmpy --version` | exact release identity, `lmpy X.Y.Z` |

Exit codes: 0 ok, 2 usage (lmscript compile errors are also 2), 3 load or run-time error, 4 application failure (`out.fail` or
a failed check).

## Documents

- [API.md](API.md): every type, function and method, generated from the registry.
- [../durable-memory/native-dsl.md](../durable-memory/native-dsl.md): design, libraries, conventions for adding a family,
  example rules.
- [../examples/dsl/](../examples/dsl/): the `_lmcode_` versions of the pycli recipes, plus multi-step `task-*` workflows.
