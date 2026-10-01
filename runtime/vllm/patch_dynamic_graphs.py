#!/usr/bin/env python3
"""Patch the pinned V1 target runner, dispatcher and calibration hook."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

INPUT_HASHES = {
    'v1/cudagraph_dispatcher.py': '51f632753af729aeee6ce4a802d9d74b64f22eef02a524f36e2a24b07ef0cf63',
    'v1/worker/gpu_model_runner.py': 'cda2020e5aad42e617e3fac04902acde33990ccb78dd899113e8309628a0ef3a',
    'v1/core/sched/scheduler.py': 'a08faf89d37ef7115942253b0ecfbaaf980ef1e8586973b024540bc6d86281f3',
    'config/vllm.py': 'b400c810e21ebc66d5bdcc33cb6852031287feaf167e1d37f7b80ff8ee7b1fbf',
}


def replace(text, old, new, count=1):
    if text.count(old) != count:
        raise ValueError(f'Expected {count} occurrences of {old[:100]!r}')
    return text.replace(old, new)


def dispatcher_patch(text):
    text = replace(text, 'from collections.abc import Set as AbstractSet\n',
        'from collections.abc import Set as AbstractSet\nfrom vllm import amos_dynamic_graphs\n')
    text = replace(text, '        self.keys_initialized = True\n\n    def dispatch(',
        '        amos_dynamic_graphs.initialize(self, cudagraph_mode, uniform_decode_query_len)\n'
        '        self.keys_initialized = True\n\n    def dispatch(')
    text = replace(text,
        '        invalid_modes: AbstractSet[CUDAGraphMode] | None = None,\n',
        '        invalid_modes: AbstractSet[CUDAGraphMode] | None = None,\n'
        '        uniform_decode_query_len: int | None = None,\n')
    text = replace(text,
        '''        batch_desc = self._create_padded_batch_descriptor(
            num_tokens, normalized_uniform, has_lora, effective_num_active_loras
        )''',
        '''        batch_desc = amos_dynamic_graphs.exact_descriptor(
            self, num_tokens, uniform_decode_query_len, normalized_uniform, has_lora
        )
        if batch_desc is None:
            batch_desc = self._create_padded_batch_descriptor(
                num_tokens, normalized_uniform, has_lora, effective_num_active_loras
            )''')
    # Identical token counts can have different request counts. Every TP rank
    # must capture those graphs in the same order for collective registration.
    text = replace(text, 'key=lambda d: (d.num_tokens, d.num_active_loras),',
        'key=lambda d: (d.num_tokens, d.num_active_loras, d.num_reqs or 0, d.uniform),')
    return text


def runner_patch(text):
    text = replace(text, 'from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher\n',
        'from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher\n'
        'from vllm import amos_dynamic_graphs\n')
    text = replace(text, '            uniform_decode_query_len=self.uniform_decode_query_len,\n',
        '            uniform_decode_query_len=amos_dynamic_graphs.actual_query_len(\n'
        '                self, max_num_scheduled_tokens),\n')
    text = replace(text,
        '                invalid_modes={CUDAGraphMode.FULL} if disable_full else None,\n',
        '                invalid_modes={CUDAGraphMode.FULL} if disable_full else None,\n'
        '                uniform_decode_query_len=max_num_scheduled_tokens,\n')
    text = replace(text,
        '        single_request_prefill: bool = False,\n        run_drafter: bool = True,\n',
        '        single_request_prefill: bool = False,\n        run_drafter: bool = True,\n'
        '        uniform_decode_query_len: int | None = None,\n')
    text = replace(text,
        '        max_query_len = self.uniform_decode_query_len if uniform_decode else num_tokens\n',
        '        decode_query_len = (self.uniform_decode_query_len\n'
        '                            if uniform_decode_query_len is None else uniform_decode_query_len)\n'
        '        max_query_len = decode_query_len if uniform_decode else num_tokens\n')
    text = replace(text, '                uniform_decode=desc.uniform,\n',
        '                uniform_decode=desc.uniform,\n'
        '                uniform_decode_query_len=amos_dynamic_graphs.capture_query_len(self, desc),\n', count=2)
    return text


def scheduler_patch(text):
    return replace(text,
        '    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:\n',
        '    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:\n'
        '        from vllm.amos_dynamic_graphs import apply_calibration\n'
        '        apply_calibration(self)\n')


def config_patch(text):
    return replace(text,
        '        logger.warning_once(\n'
        '            "Dynamic speculative decoding changes the target verification "\n',
        '        from vllm.amos_dynamic_graphs import allow_dynamic_full_graphs\n'
        '        if allow_dynamic_full_graphs(self):\n'
        '            return\n\n'
        '        logger.warning_once(\n'
        '            "Dynamic speculative decoding changes the target verification "\n')


def patch(root):
    transforms = {'v1/cudagraph_dispatcher.py': dispatcher_patch,
                  'v1/worker/gpu_model_runner.py': runner_patch,
                  'v1/core/sched/scheduler.py': scheduler_patch,
                  'config/vllm.py':config_patch}
    prepared = {}
    for relative, transform in transforms.items():
        path = root / relative
        source = path.read_bytes()
        if hashlib.sha256(source).hexdigest() != INPUT_HASHES[relative]:
            raise ValueError(f'Source differs from pinned runtime: {path}')
        prepared[path] = transform(source.decode())
        compile(prepared[path], str(path), 'exec')
    # Validate every file before writing any of them.
    for path, text in prepared.items():
        path.write_text(text)
    shutil.copy2(Path(__file__).with_name('amos_dynamic_graphs.py'), root/'amos_dynamic_graphs.py')
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in prepared}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('vllm_root', type=Path)
    args = p.parse_args()
    print(json.dumps(patch(args.vllm_root), indent=2))
