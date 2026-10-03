"""Bounded diagnostic for real TP6 E3 inputs; never changes returned tensors.

The operator arms this only in an owned, drained six-GPU window for a synthetic
request. No control file means no tensor access or CUDA calls. Captures are
private artifacts, not speed measurements. Expiry and byte limits are mandatory.
"""
import hashlib
import json
import logging
import math
from pathlib import Path
import re
import time

CONTROL = Path('/root/.cache/amos-e3-capture-control.json')
ROOT = Path('/root/.cache/amos-e3-capture')
PATTERN = re.compile(r'(?:^|\.)layers\.(\d+)\.mlp\.experts$')
MAX_BYTES = 320 * 1024**2
_checked = 0.0
_control = None
_revision = None
_written = set()
_bytes = 0
_failed = set()
LOG = logging.getLogger('vllm.amos_e3.capture')


def validate(value):
    if (not isinstance(value, dict) or value.get('schema') != 1
            or not isinstance(value.get('revision'), str)
            or not re.fullmatch(r'e3[0-9]+-[a-z0-9-]{1,48}', value['revision'])
            or value.get('input_layers') != [3, 40, 77]
            or type(value.get('rows')) is not int or value['rows'] != 3072
            or type(value.get('max_bytes_per_rank')) is not int
            or not 1 <= value['max_bytes_per_rank'] <= MAX_BYTES
            or type(value.get('armed_at')) not in (float, int)
            or type(value.get('expires_at')) not in (float, int)
            or not math.isfinite(value['armed_at']) or not math.isfinite(value['expires_at'])
            or not 0 < value['expires_at'] - value['armed_at'] <= 180):
        raise ValueError('Invalid bounded E3 capture control')
    return value


def read_control():
    global _checked, _control
    now = time.monotonic()
    if now - _checked >= .25:
        _checked = now
        try:
            _control = validate(json.loads(CONTROL.read_text()))
        except FileNotFoundError:
            _control = None
    if _control is None:
        return None
    wall = time.time()
    if not _control['armed_at'] <= wall < _control['expires_at']:
        return None
    return _control


def capture(layer, x, weights, ids, output):
    """Synchronously copy one call per layer/revision, preserving original output."""
    global _revision, _written, _bytes
    # Keep disabled production work independent of CUDA, including graph replay.
    if x.shape[0] != 3072:
        return
    try:
        control = read_control()
        if control is None or control['revision'] in _failed:
            return
        name = getattr(layer, 'layer_name', '')
        match = PATTERN.search(name)
        if match is None or not 0 <= int(match[1]) < 78:
            return
        if _revision != control['revision']:
            _revision, _written, _bytes = control['revision'], set(), 0
        if name in _written:
            return
        import torch
        if not x.is_cuda or torch.cuda.is_current_stream_capturing():
            return
        from vllm.distributed import get_tensor_model_parallel_rank
        rank = get_tensor_model_parallel_rank()
        assert 0 <= rank < 6
        assert tuple(x.shape) == tuple(output.shape) == (3072, 6144)
        assert tuple(ids.shape) == tuple(weights.shape) == (3072, 8)
        assert x.dtype == output.dtype == torch.bfloat16 and weights.dtype == torch.float32
        mapping = layer.exl3_mixed_trellis['global_to_combined']
        assert mapping.numel() == 256
        tensors = dict(ids=ids, weights=weights, mapping=mapping, bits=layer.glm6_e3_binding['bits'])
        if int(match[1]) in control['input_layers']:
            tensors.update(input=x, output=output)
        amount = sum(t.numel() * t.element_size() for t in tensors.values())
        if _bytes + amount > control['max_bytes_per_rank']:
            raise ValueError('E3 capture byte limit exceeded')
        directory = ROOT / _revision / f'rank{rank}'
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        target = directory / f'layer{int(match[1]):02d}.safetensors'
        manifest = target.with_suffix('.json')
        temporary = target.with_suffix('.tmp')
        if any(p.exists() for p in (target, manifest, temporary)):
            raise FileExistsError('Capture revision already contains this layer')
        cpu = {key: tensor.detach().cpu().contiguous() for key, tensor in tensors.items()}
        global_counts = torch.bincount(cpu['ids'].long().reshape(-1), minlength=256)
        local = cpu['mapping'][cpu['ids'].long()].reshape(-1).long()
        experts = int(layer.glm6_e3_binding['experts'])
        counts = torch.bincount(local[local >= 0], minlength=experts)
        assert len(counts) == experts
        tiles = (counts + 63) // 64
        from safetensors.torch import save_file
        save_file(cpu, str(temporary))
        temporary.chmod(0o600)
        with temporary.open('rb') as handle:
            sha = hashlib.file_digest(handle, 'sha256').hexdigest()
        metadata = dict(schema=1, revision=_revision, rank=rank, layer=int(match[1]),
            layer_name=name, captured_at=time.time(), tensor_bytes=amount, sha256=sha,
            experts=int(layer.glm6_e3_binding['experts']),
            global_route_counts=global_counts.tolist(), local_route_counts=counts.tolist(),
            global_to_combined=cpu['mapping'].reshape(-1).tolist(), bits=cpu['bits'].tolist(),
            actual_local_rows=int(counts.sum()), tile64_count=int(tiles.sum()),
            tile64_rows=int(tiles.sum())*64,
            tensor_sha256={key:hashlib.sha256(t.view(torch.uint8).numpy()).hexdigest() for key,t in cpu.items()},
            tensors={key: dict(dtype=str(t.dtype), shape=list(t.shape)) for key, t in tensors.items()})
        temporary.replace(target)
        with manifest.open('x') as handle:
            json.dump(metadata, handle, indent=2)
            handle.write('\n')
        manifest.chmod(0o600)
        _written.add(name)
        _bytes += amount
    except Exception as error:
        # Diagnostic failure cannot replace a model output. The operator requires
        # every expected record plus absence of error markers before accepting it.
        revision = (_control or {}).get('revision', 'invalid-control')
        if revision not in _failed:
            _failed.add(revision)
            LOG.exception('E3 diagnostic capture failed; revision disabled')
            try:
                ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
                (ROOT / (revision + '.error.json')).write_text(json.dumps(dict(
                    revision=revision, error=type(error).__name__, message=str(error), at=time.time())) + '\n')
            except OSError:
                pass
