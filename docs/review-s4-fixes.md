# S4 review fixes

Reviews read in full: REVIEW-S4-Opus.md, REVIEW-S4-Fable.md, REVIEW-S4-Astra.md
in ~/repos/cppwrapper/pycli. No finding declined or deferred.

## Dispositions

* Astra major1: nonzero rows and objective normalized; fixed native cutoff1e-12;
  C++ rejects normalized coefficient ratios below1e-9 and scaled RHS above1e15.
  Original-data relative primal/bound/integrality and LP KKT checks reject false
  solutions as NUMERICAL_FAILURE. Duals are restored to original row/objective
  units. Tests include all tiny-row reproducers (LP/MILP, bounded/unbounded form),
  random wide-dynamic-range LPs checked by exact Fraction vertex enumeration,
  and injected false native primals/duals to prove the certificate rejects them.
* Opus majors1/2: finite abs>1e15 in c/A/b/bounds rejected before spawning with
  VALUE_EXCEEDS_SOLVER_LIMIT; normalized RHS capped too. Only explicit null or
  signed Inf tokens open bounds. Model error remains MODEL_ERROR; INFEASIBLE
  requires native model status8. The finite-but-huge exact reproductions are
  tested with a missing interpreter, proving preflight rather than child failure.
* Opus minor3 / Fable3: native model status9 maps to INFEASIBLE_OR_UNBOUNDED,
  including unbounded integer problems; no disambiguating solve or guessed answer.
  Native code retained as highs_model_status; unknown codes fail conservatively.
* Fable1: every failed/native/budget/nonfinite/event-cap integration is marked
  termination:failed / partial_trajectory:true. Blow-up metadata regression.
* Fable2: integer components within1e-6 rounded; -0 normalized; pre-round error
  retained and returned vector rechecked against original constraints/bounds.
* Fable4: initial-time zero events excluded even when terminal; later roots still
  terminate. Event indices are JSON integers / I64 tensors. Ball/coarse t_eval,
  initial threshold directions and later sine crossing fixtures cover the paths.
* Fable5: fixed 60x5 multi-knapsack asserts NODE_LIMIT/no primal at node0, a native
  TIME_LIMIT (model13), integer incumbent components, and maximize equality-dual
  sensitivity. This complements the pre-native tiny-budget branch.
* Fable6: schema/README warn about default10s RK45 on stiff systems; use
  BDF/Radau/LSODA. outputs_in_artifact is explicit; schema explains vector refs
  and array_inspect. Read form overflows recommend _to rather than writing.
* Astra minor2: generated uniform grid deduplicated with exact endpoints;
  effective_output_points exposed; forward/reverse large-offset fixtures.
* Opus access note: read opt_linprog/ode_solve + write opt_linprog_to/ode_solve_to
  with required out, matching S1. Readonly small solves work; read overflow never
  writes. Write readonly/dryrun checks leave no artifacts. Existing recipes stay
  on read ops because their outputs fit inline.

## Test-first evidence

Review regressions on the original image exited1: tiny coefficients returned
false optima/unbounded, huge inputs reached missing Python, false certificates
were accepted, readonly failed, failure/grid/event/integer metadata were wrong.
Separate corrected harness runs reproduced short-grid, initial event, float
index and signed-zero failures. A first harness helper had duplicate kwargs;
that helper was corrected before relying on its results. Source-worker fault
injection passes independently of the C++ frontend; final image tests cover both.

## Final acceptance (unpiped commands)

* ./scripts/build.sh test: exit0.
* docker run --rm --network=none pycli:test: **24/24, exit0, 592.68s**.
  New review suite:12 groups,52.65s; original S4:16 groups,37.39s. Parser ASan/UBSan,
  all S0-S3 regressions,24-op schema/docs,notices and packaging gates pass.
* ./scripts/build.sh runtime: exit0.
* All seven recipes via production bin/pycli, each real/check/dry-run:21 exits0.
* Production-shim readonly small solves exit0; read overflow exits1 with
  OUTPUT_TOO_LARGE/no artifact; _to readonly exits6/no artifact. Event artifact
  indices are I64 and the decay threshold time agrees with analytic log(2).
* Hardened cap harness:exit0; all four calls domain OK, no crash/timeout; peak
  **105209856 bytes (100.34MiB)** including probe arrays, responses below2KiB.
* Final-runtime schema/docs check:exit0. Sequential timing probe:exit0,40 calls
  domain OK,all exit0. Full raw samples and shared-host load telemetry retained.

Runtime **527MB /115249079 content bytes**; test **1.06GB /222443444 content bytes**.
No wheel/package was added. Image IDs/layers:measurement-s4-review-packaging.json;
cap command/results:measurement-s4-review-caps.json; recipes:measurement-s4-review-
recipes.json; policy/artifact checks:measurement-s4-review-policy.json; timing
samples:measurement-s4-review-shim.json and timings.md. Measurements are for
allocation fixtures and a shared host, not worst-case MILP complexity or global
ODE accuracy guarantees. amd64 remains unverified.

Temporary worker probes and copied development binaries were removed; superseded
images and label-scoped disposable builder cache cleaned (exit0). Final runtime
and test images retained. All findings fixed; no finding deferred/declined.
