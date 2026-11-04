# Native invocation envelope (0.1.2)

The current lmpy is native C++ under native/, not the legacy Python child frontend
under src/. Build with native/scripts/fetch-deps.sh and native/test.sh.

SDK synced at 93bddf1dadff84791ea90e17d81a7af85a842f40 using native/scripts/sync-sdk.sh.
The host validates invocation metadata, loads all programs before any execution,
then selects/completes each part and writes one LMCODE_RETURN_BEGIN/END frame.
Dry metadata or --dry-run loads without running callbacks. Contradictory real
metadata plus --dry-run is refused. Errors: 2 CLI/metadata, 3 load/run, 4 app/output.

Legacy out.put/check/close payloads remain result.values/result.checks. Each part
gets a reset collector; failed checks and out.fail are app failures. Explicit
lmcode_return_set/put/null takes precedence over the default collector result.
Dry results are null. All JSON passes SDK result validation/serialization.

Examples use run_lmcode and its decoded result; the shared runner retains LMPY
selection and LMPY_VALUES_OUT parity dumps. No stdout parsing or exports remain.
README rendering requires compatible local lmpy/lmdraw and never downloads hosts.
Tests use a native recording executable, Windows npm .cmd, and explicit UTF-8.
The legacy pycli/Docker sources/tests are not part of this protocol port.
