#!/usr/bin/env python3
"""Install the source-pinned P24 dispatch helper into both vLLM trees."""
import hashlib
import json
from pathlib import Path
import py_compile

PREVIOUS = '77a9e99f23177c798da2b769f992a246ce4aabd9c615df7cc8cc206d97234d73'
ROOTS = (Path('/usr/local/lib/python3.12/dist-packages/vllm'),
         Path('/opt/glm53-full/vllm/vllm'))


def install(source, roots=ROOTS):
    payload = source.read_bytes()
    candidate = hashlib.sha256(payload).hexdigest()
    targets = [root/'amos_grouped_prefill.py' for root in roots]
    # Validate the entire destination set before changing any file.
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in targets}
    if any(value not in (PREVIOUS, candidate) for value in before.values()):
        raise ValueError('P24 requires the exact verified E3 dispatch source')
    compile(payload, str(source), 'exec')
    for target in targets:
        target.write_bytes(payload)
        py_compile.compile(str(target), doraise=True)
    return {'before': before, 'after_sha256': candidate,
            'scope': 'Dispatch only; native/E3 kernel and weight bytes unchanged'}


if __name__ == '__main__':
    print(json.dumps(install(Path(__file__).with_name('amos_grouped_prefill.py'))))
