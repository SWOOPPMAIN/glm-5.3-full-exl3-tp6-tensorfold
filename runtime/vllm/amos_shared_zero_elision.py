"""Shared experts with canonical partial sums and verified empty-shard elision.

Four 512-channel partials preserve the established reduction grouping. The
remaining two TP6 ranks skip both projections and the activation only after
their loaded MXFP8 weights have been verified to be zero. Routed EXL3 pieces,
attention geometry, original checkpoints and the collective stay unchanged.
"""
import os
import re

CANONICAL_AXIS = {'original_size': 2048, 'padded_size': 3072,
                  'tp_size': 6, 'local_size': 512}
PATTERN = re.compile(r'model\.layers\.(\d+)\.mlp\.shared_experts')


def enabled():
    value = os.getenv('AMOS_TP6_SHARED_ZERO_ELISION', '0')
    if value not in ('0', '1'):
        raise ValueError('Invalid shared zero-elision setting')
    if value == '1' and (os.getenv('AMOS_TP6_SHARED_384') != '1'
                         or os.getenv('AMOS_EXL3_TP6_PIECES') != '1'):
        raise ValueError('Shared zero elision requires the independent full-GLM TP6 plan')
    return value == '1'


def axis(previous):
    return dict(CANONICAL_AXIS) if enabled() else dict(previous)


def selected(prefix):
    match = PATTERN.fullmatch(prefix)
    return match is not None and 3 <= int(match[1]) <= 78


def configure(prefix, intermediate_size, hidden_size):
    if not enabled() or not selected(prefix):
        return False
    from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
    world, rank = get_tensor_model_parallel_world_size(), get_tensor_model_parallel_rank()
    if world != 6 or not 0 <= rank < 6 or intermediate_size != 3072 or hidden_size != 6144:
        raise ValueError('Shared zero-elision geometry mismatch')
    return rank >= 4


def verify(layer, weight, scale):
    prefix = getattr(layer, 'prefix', '')
    base, _, projection = prefix.rpartition('.')
    if not enabled() or not selected(base):
        return
    if projection not in ('gate_up_proj', 'down_proj'):
        raise ValueError('Unexpected shared projection')
    from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
    rank = get_tensor_model_parallel_rank()
    if get_tensor_model_parallel_world_size() != 6 or not 0 <= rank < 6:
        raise ValueError('Shared zero-elision requires TP6')
    expected = (1024, 6144) if projection == 'gate_up_proj' else (6144, 512)
    if tuple(weight.shape) != expected or tuple(scale.shape) != (expected[0], expected[1]//32):
        raise ValueError('Shared zero-elision weight layout mismatch')
    if rank >= 4:
        import torch
        if weight.dtype != torch.float8_e4m3fn or scale.dtype not in (torch.uint8, torch.float8_e8m0fnu):
            raise ValueError('Shared zero-elision requires loaded MXFP8 values and scales')
        # Both signs of zero are valid; exponent 255 is not a finite scale.
        if bool((weight.view(torch.uint8) & 127).count_nonzero()) or bool((scale.view(torch.uint8)==255).any()):
            raise ValueError('Refusing to elide a nonzero or nonfinite shared shard')
        layer.amos_shared_zero_verified = True
