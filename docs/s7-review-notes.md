# S7 review fixes

All ranked findings in Opus, Fable, Astra, Argon and Kimi are addressed. The only
scope deferrals are the two explicitly requested by Lead: Argon F4's table/array
bridge and F7's seeded sampling, concretely proposed in s8-bridge-proposal.md.

| Review finding | Resolution / regression evidence |
|---|---|
| Opus 1 | C++ checks 2*m*(k+exclude_self)<=1900000 before Python; missing-interpreter 200000x1000 reproducer. |
| Opus 2 | Candidate counting precedes expensive prominence/width work; read candidates<=225, candidate*window<=20M, explicit smaller wlen required above cap; compute prominences once; cooperative guard. Mock refuses any expensive call in both saw reproductions. |
| Opus 3 | Adaptive per-query k searches stop at max_pairs; no all-pairs count. Mock makes query_ball_point forbidden and limits search calls. |
| Astra 1 | Stable max-scaled long-double Minkowski norm and exact brute search for finite p>2, work cap before spawn; lower-p scaled KD validates ranges, indices and distances. Identity/membership regressions from 1e-100 to 1e100. |
| Astra 2 / Argon F6 power | Window ENBW correction makes total_power invariant under zero padding; independent density/spectrum and boxcar energy checks, Welch/Hann/Hamming. |
| Astra 3 | In-domain finiteness checked before applying intentional outside NaNs; tiny-node linear reproducer returns NUMERICAL_FAILURE. |
| Astra 4 / Argon F1 | Narrow per-tensor NaN masks in worker plus independent C++ reconstruction of outside mask; finite-only default retained for all other outputs. |
| Astra 5 / Argon F2 | Integral-only write rejected BAD_PARAM before spawn; scalar integral remains available read-only. |
| Astra fuzzers | Seed-71015 primary 147 and secondary 15 cases retained as standalone NumPy-reference regression; high-p identity/radius added. Extreme interpolation may honestly return NUMERICAL_FAILURE. |
| Fable 1 | VI symmetry tolerance 1e-12*maxabs, symmetrize and report vi_symmetrised; hand quadratic distance oracle. |
| Fable 2 | Bounded undefined row indices, sampled-width wording, coherence at powered bins, op-level inline/file caps. |
| Kimi 1 | Removed false Akima/Makima bound promise; method-specific nonuniform hand fixture includes negative and above-plateau values. |
| Kimi 2 | Bessel phase-normalisation wn documented; order2/order4 hand magnitudes and changed-frequency tests. |
| Kimi 3 | rel_height half prominence, sampled crossings; asymmetric-base fixture gives height4/prominence2/width0.75. |
| Kimi 4 | total_power documented only for Welch/periodogram, per-method field tests. |
| Kimi 5 | Hand square-plus-center Voronoi region/ridge membership including infinity, inline/artifact; signed CSD bin32 = +0.128i. |
| Argon F3 | One seven-field artifacts[name] descriptor across all writes S1-S7; no aliases. Small lstsq vectors also stored. Failure diagnostics preserved with bounded arrays/bytes. Cross-slice contract regression, migrated callers/schema/docs. |
| Argon F5 | RBF polynomial degree/minimum term count and ellip rp<rs validated C++; degree -1 deliberately disables polynomial (SciPy API), not an invalid degree. |
| Argon F6 coherence | Default min(256,n/2), >=2 segments required; explicit single segment BAD_PARAM. Independent unrelated-signal case must not be all ones. |
| Argon F8 | Resource/shape/parameter error codes aligned; radius inline count preflight; time_limit seconds default10 >0..30 accepted throughout guarded S3-S7, independent hard child cap includes imports. |
| Argon F9 | All six S7 staging names added to async signal cleanup; killed child/staging regression. |
| Argon F10 | Stable extended-precision quadratic roots of actual rounded SOS, independently checked with 80-digit Decimal; reachable pole-one UNSTABLE documented honestly. |

Test-first evidence: d88a02d introduced review reproducers (baseline exit1, 28 failing
subtests plus 10 errors); 6542b3f added the independently failing SOS precision case.
Coverage-only numerical/doc fixtures that were already correct pass on baseline;
the schema assertions fail until its false claims are corrected. Resource guards
are tested both at the front-end boundary and by instrumenting expensive worker calls.
