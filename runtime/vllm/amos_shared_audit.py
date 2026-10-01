"""Bounded, opt-in tensor recording for held TP6 shared-expert diagnosis.

Records the first selected prefill call per projection and revision. It does
not replace tensors, modify weights, or create distributed state. Decode and
CUDA graph capture never record. The operator creates the control file only
after verifying both maintenance holds and an empty queue.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import time

ENABLED = os.getenv('AMOS_TP6_SHARED_AUDIT', '0') == '1'
CONTROL = Path('/root/.cache/amos-shared-audit-control.json')
ROOT = Path('/root/.cache/amos-shared-audit')
PATTERN = re.compile(r'model\.layers\.(3|40|77)\.mlp\.shared_experts\.(gate_up_proj|down_proj)')
_checked_at = 0.0
_control = None
_revision = None
_written = set()
_bytes = 0


def validate_control(value):
    if (value.get('schema') != 1 or not isinstance(value.get('revision'), str)
            or not re.fullmatch(r'[a-z0-9-]{1,64}', value['revision'])
            or value.get('layers') != [3, 40, 77]
            or type(value.get('min_rows')) is not int
            or type(value.get('max_rows')) is not int
            or not 256 <= value['min_rows'] <= value['max_rows'] <= 1536
            or type(value.get('max_bytes_per_rank')) is not int
            or not 1 <= value['max_bytes_per_rank'] <= 64 * 1024**2):
        raise ValueError('Invalid bounded shared-expert audit control')
    return value


def read_control():
    global _checked_at, _control
    now = time.monotonic()
    if now - _checked_at >= .25:
        _checked_at = now
        try:
            _control = validate_control(json.loads(CONTROL.read_text()))
        except FileNotFoundError:
            _control = None
    return _control


def projection_shape(projection, in_features, out_features):
    """Accept the original 384-wide and current 512-wide shared shards."""
    allowed = {'gate_up_proj': {(6144, 768), (6144, 1024)},
               'down_proj': {(384, 6144), (512, 6144)}}
    shape = (in_features, out_features)
    if shape not in allowed.get(projection, set()):
        raise ValueError('Unexpected full-GLM shared-expert audit geometry')
    return shape


def capture(layer, x, packed, output):
    global _revision, _written, _bytes
    if not ENABLED or x.ndim != 2 or not 256 <= x.shape[0] <= 1536:
        return
    match = PATTERN.fullmatch(getattr(layer, 'prefix', ''))
    if match is None:
        return
    import torch
    if not x.is_cuda or torch.cuda.is_current_stream_capturing():
        return
    control = read_control()
    if control is None or not control['min_rows'] <= x.shape[0] <= control['max_rows']:
        return
    if _revision != control['revision']:
        _revision, _written, _bytes = control['revision'], set(), 0
    prefix = layer.prefix
    if prefix in _written:
        return
    expected_k, expected_n = projection_shape(match[2], packed.in_features, packed.out_features)
    if (x.dtype != torch.bfloat16 or output.dtype != torch.bfloat16
            or tuple(x.shape) != (x.shape[0], expected_k)
            or tuple(output.shape) != (x.shape[0], expected_n)
            or packed.in_features != expected_k or packed.out_features != expected_n):
        raise ValueError('Unexpected full-GLM shared-expert audit shape')
    tensors = {'input': x, 'output': output, 'weight_values': packed.weight.values,
               'weight_scale_rows': packed.weight.scale_rows}
    amount = sum(t.numel() * t.element_size() for t in tensors.values())
    if _bytes + amount > control['max_bytes_per_rank']:
        raise ValueError('Shared-expert audit byte budget exceeded')
    from safetensors.torch import save_file
    from vllm.distributed import get_tensor_model_parallel_rank
    rank = get_tensor_model_parallel_rank()
    if not 0 <= rank < 6:
        raise ValueError('Shared-expert audit requires TP6 ranks')
    directory = ROOT / _revision / f'rank{rank}'
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / (prefix + '.safetensors')
    manifest = target.with_suffix('.json')
    if target.exists() or manifest.exists():
        raise FileExistsError('Audit revision already contains this projection')
    metadata = {'schema': 1, 'revision': _revision, 'rank': rank, 'prefix': prefix,
                'tensor_bytes': amount, 'tensors': {
                    name: {'dtype': str(t.dtype), 'shape': list(t.shape)}
                    for name, t in tensors.items()}}
    # Float8 exponent storage is serialized as raw bytes for safetensors
    # compatibility; the manifest preserves its original shape and dtype.
    cpu = {name: t.detach().contiguous().view(torch.uint8).cpu()
           for name, t in tensors.items()}
    temporary = target.with_suffix('.tmp')
    save_file(cpu, str(temporary))
    metadata['sha256'] = hashlib.sha256(temporary.read_bytes()).hexdigest()
    temporary.replace(target)
    temp_manifest = manifest.with_suffix('.tmp')
    temp_manifest.write_text(json.dumps(metadata, indent=2) + '\n')
    temp_manifest.replace(manifest)
    _written.add(prefix)
    _bytes += amount
