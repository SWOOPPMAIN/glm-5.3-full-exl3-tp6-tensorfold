#!/usr/bin/env python3
"""Prepare an opt-in E3 hook against the exact P20 EXL3 source.

Does not stage E3, launch containers, or change serving settings. The grouped
runtime must be packaged separately with its original license and source.
"""
import hashlib
import json
from pathlib import Path
import shutil
import sys

RELATIVE = 'model_executor/layers/quantization/exl3.py'
EXPECTED = '5fafc04fac6618ea0d5a04b9ad06002ee1a7657399f7066b4e727649f5179817'
FIRST_INTEGRATION = '8e6a32e15fed5de89f6adbabff3c47e76433c3978d67e121c0a669b775801fdb'
ANCHOR = '        runtime = self._mixed_rank_sliced_runtime(layer, x, topk_ids)\n'
HOOK = (
    '        from vllm.amos_grouped_prefill import apply_if_prefill\n'
    '        grouped = apply_if_prefill(layer, x, topk_weights, topk_ids,\n'
    '            max_decode_m=_positive_env_int("VLLM_EXL3_TRELLIS_MAX_M", 32))\n'
    '        if grouped is not None:\n'
    '            return grouped\n')


def patch(root):
    path = root/RELATIVE
    raw = path.read_bytes()
    before = hashlib.sha256(raw).hexdigest()
    if before == FIRST_INTEGRATION:
        # Repair only the exact first candidate; verify recovery of the pinned
        # source before applying the new ordering. No serving image rollback.
        raw = raw.decode().replace(HOOK, '', 1).encode()
    if hashlib.sha256(raw).hexdigest() != EXPECTED:
        raise ValueError('E3 hook requires the pinned P20 native EXL3 source')
    source = raw.decode()
    if source.count(ANCHOR) != 1:
        raise ValueError('Native mixed EXL3 dispatch anchor changed')
    # The first eager profile pass must create the native immutable runtime:
    # vLLM later warms its decode route-pack kernels before KV sizing. E3 may
    # replace the prefill arithmetic only after this initialization has run.
    replacement = ANCHOR + HOOK
    source = source.replace(ANCHOR, replacement)
    compile(source, str(path), 'exec')
    helper = Path(__file__).with_name('amos_grouped_prefill.py')
    compile(helper.read_text(), str(helper), 'exec')
    path.write_text(source)
    shutil.copy2(helper, root/'amos_grouped_prefill.py')
    return dict(native_before=before,
                native_after=hashlib.sha256(path.read_bytes()).hexdigest(),
                helper=hashlib.sha256(helper.read_bytes()).hexdigest(),
                serving_enabled=False, gpu_qualified=False)


if __name__ == '__main__':
    print(json.dumps(patch(Path(sys.argv[1]))))
