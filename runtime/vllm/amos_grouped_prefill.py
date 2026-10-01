"""Opt-in E3 prefill dispatch for the pinned native TP6 EXL3 representation.

The native decoder, original weight storage and global router outputs remain
the source of truth. E3 owns separate scratch for each CUDA stream. GPU and
full-model qualification are required before enabling this in serving.
"""
import os

# One policy for generation and teacher scoring. The native prefill kernel
# handles short tails more efficiently than E3's routing and launch overhead.
NATIVE_PREFILL_MAX_ROWS = 512


def apply_if_prefill(layer, x, weights, ids, *, max_decode_m):
    if os.environ.get('AMOS_TP6_E3_PREFILL') != '1':
        return None
    capacity = int(layer.exl3_max_num_batched_tokens)
    if max_decode_m <= 0 or capacity <= 0:
        raise ValueError('E3 dispatch requires positive decode and batch capacities')
    rows = int(x.shape[0])
    if rows <= min(max_decode_m, capacity):
        return None
    if os.environ.get('AMOS_EXL3_TP6_PIECES') != '1':
        raise ValueError('E3 dispatch requires the verified TP6 piece loader')
    if rows > capacity:
        raise ValueError('E3 batch exceeds the configured capacity')
    if (layer.exl3_hidden_size != 6144 or
            layer.exl3_intermediate_size_per_partition != 512):
        raise ValueError('E3 requires full-GLM 6144x512 expert pieces')
    if rows <= NATIVE_PREFILL_MAX_ROWS:
        # Keep max_decode_m unchanged: the caller selects its native PREFILL
        # plan above that boundary, rather than misusing a decode-sized arena.
        return None
    from vllm.amos_e3 import runtime
    return runtime.apply(layer, x, weights, ids, stream_scratch=True)
