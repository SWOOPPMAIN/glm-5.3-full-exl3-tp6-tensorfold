#!/usr/bin/env python3
"""Validate snapshot hashes, published links, and accidental private artifacts.

This is a bounded packaging check, not an inference or full secret-scanning audit.
It prints offending paths and rule names, never matched credential contents.
"""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    failures = []
    records = json.loads((ROOT / 'provenance/imports.json').read_text())['files']
    for item in records:
        path = ROOT / item['path']
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            failures.append((item['path'], 'import hash mismatch'))
    listed = subprocess.check_output(
        ['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'],
        cwd=ROOT).decode().split('\0')
    paths = [ROOT / p for p in listed if p]
    patterns = {
        'private key': re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
        'GitHub token': re.compile(rb'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b'),
        'HF token': re.compile(rb'\bhf_[A-Za-z0-9]{30,}\b'),
        'private workspace': re.compile(rb'/home/[A-Za-z0-9_-]+/(?:worktrees|\.ssh|\.config)/'),
    }
    for path in paths:
        relative = str(path.relative_to(ROOT))
        if path.is_symlink():
            failures.append((relative, 'symlink'))
            continue
        data = path.read_bytes()
        if path.suffix in {'.safetensors', '.gguf', '.so', '.cubin', '.pem', '.key', '.log', '.pyc'} or path.name == '.env':
            failures.append((relative, 'private or generated artifact'))
        if path.stat().st_size > 5 * 1024**2:
            failures.append((relative, 'unexpected large file'))
        for label, pattern in patterns.items():
            if path != Path(__file__).resolve() and pattern.search(data):
                failures.append((relative, label))
        if path.suffix == '.json':
            json.loads(data)
        if path.suffix == '.py':
            compile(data, str(path), 'exec')
        # Imported upstream notices may refer to their original tree. Check our docs.
        if path.suffix == '.md' and not (relative.startswith('experimental/tensorfold/') and relative != 'experimental/tensorfold/README.md') and not relative.startswith('runtime/vllm/rocenante/'):
            for target in re.findall(r'\[[^\]]*\]\(([^\s)]+)\)', data.decode()):
                if '://' in target or target.startswith('#'):
                    continue
                target = target.split('#')[0]
                if target and not (path.parent / target).exists():
                    failures.append((relative, 'missing relative link: ' + target))
    print(json.dumps({'passed': not failures, 'files': len(paths),
                      'import_hashes': len(records), 'failures': failures}, indent=2))
    return bool(failures)


if __name__ == '__main__':
    sys.exit(main())
