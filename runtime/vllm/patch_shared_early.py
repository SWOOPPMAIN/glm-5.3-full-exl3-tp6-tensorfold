#!/usr/bin/env python3
"""Prepare an early shared-expert launch with explicit fork/join events.

CPU preparation only until isolated CUDA and full-model qualification pass.
Preserves synchronous execution, DBO slots, and consumer allocation lifetime.
"""
import ast
import hashlib
from pathlib import Path

RELATIVE = Path('model_executor/layers/fused_moe/runner/shared_experts.py')
BEFORE = '1576aa768d343706dc3ee35f78339a517cc94a6982f07b50a6c57ea842393e14'

INIT_SUFFIX = '''
        # AMOS P25: all event objects exist before graph capture. One pair per
        # DBO slot; disabled auxiliary streams retain synchronous execution.
        self._early_input = [None, None]
        self._early_output = [None, None]
        self._early_input_ready = [
            torch.cuda.Event(enable_timing=False) if self._stream is not None else None
            for _ in range(2)
        ]
        self._early_output_ready = [
            torch.cuda.Event(enable_timing=False) if self._stream is not None else None
            for _ in range(2)
        ]
'''

EARLY = '''    def maybe_sync_shared_experts_stream(
        self,
        shared_experts_input: torch.Tensor,
    ):
        order = self._determine_shared_experts_order(shared_experts_input)
        if order != SharedExpertsOrder.MULTI_STREAM_OVERLAPPED:
            return
        assert self._stream is not None
        slot = self._output_idx
        if (self._early_input[slot] is not None or self._early_output[slot] is not None
                or self._output[slot] is not None):
            raise RuntimeError("Shared-expert slot was not consumed before reuse")
        shared_experts_input.record_stream(self._stream)
        input_ready = self._early_input_ready[slot]
        output_ready = self._early_output_ready[slot]
        input_ready.record(current_stream())
        with torch.cuda.stream(self._stream):
            self._stream.wait_event(input_ready)
            self._early_output[slot] = self._layer(shared_experts_input)
            output_ready.record(self._stream)
        self._early_input[slot] = shared_experts_input
'''

JOIN = '''    def _join_early_shared_experts(
        self,
        shared_experts_input: torch.Tensor,
    ) -> torch.Tensor:
        slot = self._output_idx
        if self._early_input[slot] is not shared_experts_input or self._early_output[slot] is None:
            raise RuntimeError("Shared-expert join requires its matching early launch")
        consumer = current_stream()
        consumer.wait_event(self._early_output_ready[slot])
        output = self._early_output[slot]
        output.record_stream(consumer)
        self._early_input[slot] = None
        self._early_output[slot] = None
        return output
'''


def render(source):
    if hashlib.sha256(source.encode()).hexdigest() != BEFORE:
        raise ValueError('Early shared experts require the exact P24 source')
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SharedExperts')
    methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    lines = source.splitlines(keepends=True)
    init = methods['__init__']
    replacements = {
        '__init__': ''.join(lines[init.lineno-1:init.end_lineno]).rstrip()+'\n'+INIT_SUFFIX,
        'maybe_sync_shared_experts_stream': EARLY,
        '_run_in_aux_stream': JOIN,
    }
    for name in sorted(replacements, key=lambda k: methods[k].lineno, reverse=True):
        node = methods[name]
        lines[node.lineno-1:node.end_lineno] = [replacements[name]]
    result = ''.join(lines)
    assert result.count('self._run_in_aux_stream(') == 1
    result = result.replace('self._run_in_aux_stream(', 'self._join_early_shared_experts(')
    compile(result, str(RELATIVE), 'exec')
    return result


def install(roots):
    # Validate all destinations before changing either installed/source tree.
    outputs = {root/RELATIVE: render((root/RELATIVE).read_text()) for root in roots}
    for path, content in outputs.items():
        path.write_text(content)
    return {str(path): hashlib.sha256(content.encode()).hexdigest()
            for path, content in outputs.items()}
