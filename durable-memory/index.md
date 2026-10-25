# pycli durable memory

- [status.md](status.md): hand-off state of the Python/Docker pycli v0.1.0 and the rules that apply (commit identity, CI).
- [native-dsl.md](native-dsl.md): **current direction**: lmpy, one static C/C++ executable that runs lmscript `_lmcode_`
  bytecode (no Python, R or Docker); build/test, libraries, conventions for op families and examples.
- [native-plan.md](native-plan.md): the earlier Rust/lmapi design; its op mapping, tolerance tiers and clean-room rules
  still apply, its language choice is superseded by native-dsl.md.
