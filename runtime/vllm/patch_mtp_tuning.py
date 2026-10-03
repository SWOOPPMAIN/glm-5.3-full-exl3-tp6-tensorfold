#!/usr/bin/env python3
"""Patch only the final depth-choice wrapper of the pinned phase controller."""
import hashlib
from pathlib import Path
import shutil
import sys

ORIGINAL_SHA256='b7730101518d7afe84b8b03d42640ce5a8ff83c341ed9114cee80f17bd36708b'
NAME='amos_request_phase_mtp.py'


def patched(raw):
    if hashlib.sha256(raw).hexdigest()!=ORIGINAL_SHA256:
        raise ValueError('Unknown request/phase controller; refusing patch')
    before='        controller.choose([live[r] for r in scheduled_ids if r in live], live)\n'
    after=('        from vllm.amos_mtp_tuning import choose as tuning_choose\n'
           '        tuning_choose(controller, [live[r] for r in scheduled_ids if r in live], live)\n')
    source=raw.decode()
    assert source.count(before)==1
    source=source.replace(before,after)
    compile(source,NAME,'exec')
    return source.encode()


if __name__=='__main__':
    root=Path(sys.argv[1]);p=root/NAME
    p.write_bytes(patched(p.read_bytes()))
    shutil.copy2(Path(__file__).with_name('amos_mtp_tuning.py'),root/'amos_mtp_tuning.py')
