#!/usr/bin/env python3
"""Extend the current opt-in recorder; preserve every model execution hook."""
import hashlib
import json
from pathlib import Path
import shutil

OLD = 'af6412e082741643d36bc008fd10d3d8b2c1a145953825e0c8d73a3294809a57'
HOOK = '401b6c84ec169406d29dc027be20245e80297a9a1016163f432f4a1de1955694'


def main():
    source = Path(__file__).with_name('amos_shared_audit.py')
    roots = [Path('/usr/local/lib/python3.12/dist-packages/vllm'),
             Path('/opt/glm53-full/vllm/vllm')]
    for root in roots:
        assert hashlib.sha256((root/'amos_shared_audit.py').read_bytes()).hexdigest() == OLD
        hook = root/'model_executor/kernels/linear/mxfp8/b12x.py'
        assert hashlib.sha256(hook.read_bytes()).hexdigest() == HOOK
    compile(source.read_text(), str(source), 'exec')
    for root in roots:
        shutil.copy2(source, root/'amos_shared_audit.py')
    print(json.dumps({'recorder_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                      'model_hook_unchanged': True, 'requires_explicit_control': True}))


if __name__ == '__main__':
    main()
