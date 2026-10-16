"""Documentation gates: the README operation tables, the generated reference (statuses, tool codes, caps), the version string and the
release sections cannot drift from the code. Pure Python, no container: it reads the checked-in schema, the sources and the documents
(the schema itself is gated against the binary by tests/schema_docs.py)."""
import json, pathlib, re, subprocess, sys, unittest
ROOT = pathlib.Path(__file__).resolve().parents[1]
BIN = pathlib.Path(sys.argv.pop(1)).resolve() if len(sys.argv) > 1 and not sys.argv[1].startswith('-') else None
SCHEMA = json.loads((ROOT / 'cli.schema.dev.json').read_text())
README = (ROOT / 'README.md').read_text()
OPS = [o['name'] for o in SCHEMA['operations']]


def contract_rows(text):
    """Operation rows of the contract tables: | `op` | read/write | ...; timing tables have numbers in the second cell and do not count."""
    rows = {}
    for number, line in enumerate(text.splitlines(), 1):
        m = re.match(r'^\|\s*`([a-z][a-z0-9_]*)`\s*\|\s*(read|write)\s*\|', line)
        if m:
            rows.setdefault(m.group(1), []).append(number)
    return rows


class Readme(unittest.TestCase):
    def test_every_operation_has_exactly_one_contract_row(self):
        rows = contract_rows(README)
        self.assertEqual([op for op in OPS if op not in rows], [], 'operations without a README contract row')
        self.assertEqual({op: lines for op, lines in rows.items() if len(lines) != 1}, {}, 'operations with more than one contract row')
        self.assertEqual(sorted(set(rows) - set(OPS)), [], 'README rows that are not operations')

    def test_access_matches_the_schema(self):
        access = {o['name']: o['access'] for o in SCHEMA['operations']}
        for number, line in enumerate(README.splitlines(), 1):
            m = re.match(r'^\|\s*`([a-z][a-z0-9_]*)`\s*\|\s*(read|write)\s*\|', line)
            if m:
                self.assertEqual(m.group(2), access[m.group(1)], (number, m.group(1)))

    def test_release_sections_exist(self):
        for heading in ('## Quick start', '## Conventions', '## Errors, statuses and caps', '## What pycli is not', '## Timings'):
            self.assertIn(heading, README, heading)
        self.assertIn('docs/reference.md', README)
        self.assertIn('UNVERIFIED', README)  # amd64 honesty until the first CI run

    def test_every_parameter_is_documented(self):
        for o in SCHEMA['operations']:
            for p in o['params']:
                self.assertTrue(p.get('description'), (o['name'], p['name']))


class Reference(unittest.TestCase):
    def test_generated_reference_matches_the_code(self):
        r = subprocess.run([sys.executable, str(ROOT / 'scripts/generate-reference.py'), '--check'], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)

    def test_every_documented_status_or_code_appears_in_the_reference(self):
        reference = (ROOT / 'docs/reference.md').read_text()
        for status in ('OK', 'TIME_LIMIT', 'NUMERICAL_FAILURE', 'OUTPUT_TOO_LARGE', 'INPUT_TOO_LARGE', 'NEGATIVE_CYCLE', 'OUT_OF_RANGE', 'PY_ERROR', 'TOOL_CRASH'):
            self.assertIn('| `%s` |' % status, reference, status)
        for code, exit_code in (('VALIDATION_ERROR', 2), ('PARSE_ERROR', 3), ('NOT_FOUND', 4), ('CONFLICT', 5), ('TIMEOUT', 7), ('SERVICE_UNAVAILABLE', 8), ('COMMAND_FAILED', 1)):
            self.assertIn('| `%s` | %d |' % (code, exit_code), reference, code)


class Version(unittest.TestCase):
    def test_one_version_string_everywhere(self):
        version = SCHEMA['version']
        self.assertRegex(version, r'^\d+\.\d+\.\d+$')
        cmake = (ROOT / 'CMakeLists.txt').read_text()
        self.assertIn('project(pycli VERSION %s ' % version, cmake)
        changelog = (ROOT / 'CHANGELOG.md').read_text()
        self.assertRegex(changelog, r'^## \[%s\]' % re.escape(version), 'CHANGELOG.md must start its first release section with the schema version')
        self.assertTrue((ROOT / 'RELEASE.md').exists())
        if BIN is not None:
            info = json.loads(subprocess.run([str(BIN), 'py_info'], capture_output=True, text=True, check=True).stdout)
            provenance = info['provenance']
            self.assertEqual(provenance['version'], version)
            self.assertRegex(provenance['git_sha'], r'^([0-9a-f]{40}(-dirty)?|unknown)$')
            self.assertRegex(provenance['lock_sha256_aarch64'], r'^[0-9a-f]{64}$')
            self.assertRegex(provenance['lock_sha256_amd64'], r'^[0-9a-f]{64}$')
            self.assertRegex(provenance['lmcpp_ref'], r'^[0-9a-f]{40}$')
            self.assertEqual(provenance['worker_sha256'], info['worker_sha256'])
            import hashlib
            self.assertEqual(provenance['lock_sha256_aarch64'], hashlib.sha256((ROOT / 'docker/requirements-aarch64.lock').read_bytes()).hexdigest())
            self.assertEqual(provenance['lock_sha256_amd64'], hashlib.sha256((ROOT / 'docker/requirements-amd64.lock').read_bytes()).hexdigest())


if __name__ == '__main__':
    unittest.main()
