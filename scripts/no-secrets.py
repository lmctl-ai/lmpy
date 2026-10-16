#!/usr/bin/env python3
"""Fail when a tracked file contains something that looks like a credential. Runs in CI and locally: python3 scripts/no-secrets.py"""
import pathlib, re, subprocess, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
PATTERNS = [
    ('AWS access key', re.compile(r'\bAKIA[0-9A-Z]{16}\b')),
    ('AWS secret key assignment', re.compile(r'(?i)aws_secret_access_key\s*[=:]\s*[A-Za-z0-9/+=]{40}')),
    ('private key block', re.compile(r'-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY')),
    ('GitHub token', re.compile(r'\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36,}\b|\bgithub_pat_[A-Za-z0-9_]{40,}\b')),
    ('Slack token', re.compile(r'\bxox[baprs]-[A-Za-z0-9-]{10,}\b')),
    ('OpenAI / Anthropic style key', re.compile(r'\bsk-(?:ant-)?[A-Za-z0-9_-]{32,}\b')),
    ('Google API key', re.compile(r'\bAIza[0-9A-Za-z_-]{35}\b')),
    ('npm token', re.compile(r'\bnpm_[A-Za-z0-9]{36}\b')),
    ('password assignment', re.compile(r'(?i)\b(?:password|passwd|secret_key|api_key|apikey|auth_token)\s*[=:]\s*["\'][^"\'\s]{8,}["\']')),
    ('basic auth URL', re.compile(r'https?://[^/\s:]+:[^/\s@]{4,}@')),
]
SKIP_SUFFIXES = ('.json', '.lock', '.safetensors', '.npz', '.npy')  # measurement data and hash locks: long base64/hex blobs, no secrets by construction
files = subprocess.run(['git', '-C', str(ROOT), 'ls-files', '-z'], capture_output=True, check=True).stdout.decode().split('\0')
hits = []
for name in files:
    if not name or name.endswith(SKIP_SUFFIXES) or name == 'scripts/no-secrets.py':
        continue
    path = ROOT / name
    if not path.is_file():
        continue
    try:
        text = path.read_text(errors='replace')
    except OSError:
        continue
    for number, line in enumerate(text.splitlines(), 1):
        for label, pattern in PATTERNS:
            if pattern.search(line):
                hits.append('%s:%d: %s' % (name, number, label))
if hits:
    print('\n'.join(hits), file=sys.stderr)
    sys.exit(1)
print('no credential-like strings in %d tracked files' % len([f for f in files if f]))
