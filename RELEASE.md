# Releasing pycli (human checklist; nothing is automatic)

pycli publishes no image and no package. A release is a git tag on a green tree plus the artefacts a human chooses to hand out.
Every step below is done by a person; CI only verifies.

1. **Tree is green on both architectures.** `.github/workflows/ci.yml` passed on `ubuntu-24.04-arm` and `ubuntu-24.04` for the commit
   you are releasing. The amd64 job is the only amd64 verification that exists (README "What pycli is not"); do not release amd64
   wording that the job has not proven.
2. **Locks are current.** The last "Weekly lock drift" run reported no drift for both locks; if it did, regenerate the lock files in a
   build (`scripts/lock.py` on the downloaded wheels), update `licenses/manifest.json` and `THIRD_PARTY_NOTICES.md`
   (`scripts/notices.py`, gated by `tests/notices_inventory.py`), and rerun CI.
3. **Version.** Bump the version in exactly one place, `CMakeLists.txt` `project(pycli VERSION x.y.z)`; regenerate the schema
   (`pycli --schema > cli.schema.dev.json` from the test image build), move the CHANGELOG "Unreleased" entries under `## [x.y.z] - date`,
   and run `python3 tests/docs_gate.py` (it checks schema, CMake, CHANGELOG and `py_info.provenance.version` agree).
4. **Acceptance on the release host, unpiped, exit codes recorded in docs/timings.md:**
   `./scripts/build.sh test`; `docker run --rm --pull=never --network=none pycli:test`; `./scripts/build.sh runtime`;
   `scripts/run-recipes.sh` (every recipe in real, `--check` and `--dry-run` modes through the host shim); `python3 scripts/no-secrets.py`.
5. **Provenance.** `pycli py_info` from the runtime image must show `provenance.git_sha` equal to the release commit without `-dirty`,
   the pinned `lmcpp_ref`, and the two lock SHA-256 values equal to `sha256sum docker/requirements-*.lock`.
6. **Documentation.** README timing table re-measured on an idle host (`uptime` load below 1 on the 4-core host; otherwise state the
   load), `docs/timings.md` acceptance paragraph, `docs/reference.md` regenerated, `docs/parity.md` and `docs/oracle-findings.md` current.
7. **Tag and push.** `git tag -a vX.Y.Z -m "pycli X.Y.Z"`; `git push origin main --tags`. The repository stays private unless the owner
   decides otherwise (docs/decisions.md: push / visibility decisions belong to the user).
8. **Hand-out (optional, manual).** If an image is to be shared, `docker save pycli:runtime | zstd` and publish the archive together with
   its SHA-256 and `THIRD_PARTY_NOTICES.md`; never `docker push` from CI. Consumers rebuild with `scripts/build.sh runtime` otherwise.
9. **Clean up.** `./scripts/clean-builder.sh`, `docker image prune -f`, remove scratch directories below HOME.
