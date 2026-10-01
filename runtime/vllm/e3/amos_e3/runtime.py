"""Experimental grouped mixed-trellis CUDA driver adapter.

Weights are views of fastload's existing tier buffers, not a new checkpoint or
ordinary TP6 slicing. Preserve gate/up input rotations independently, K3/K4
packed bytes, global-to-local -1 sentinels, and FP32 route weights. Decode stays
on the existing B12X path. This module is not a quality/speed qualification.
"""
from __future__ import annotations
import ctypes as C
import logging
from pathlib import Path
from .geometry import projection_offsets, scratch_bytes

LOG = logging.getLogger('vllm.glm6_e3')
_MODULES = {}
_SCRATCH = {}
SMEM = 49152  # dual gate/up A pipelines + maximum K4 B pipelines


class DeviceModule:
    def __init__(self, path):
        self.driver = C.CDLL('libcuda.so.1')
        self.check(self.driver.cuInit(0))
        self.module = C.c_void_p()
        self.check(self.driver.cuModuleLoad(C.byref(self.module), str(path).encode()))
        self.functions = {}
        for name in ('gather', 'gateup', 'down'):
            function = C.c_void_p()
            self.check(self.driver.cuModuleGetFunction(C.byref(function), self.module,
                                                      ('glm6_e3_' + name).encode()))
            if name != 'gather':
                # CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8, CUDA13 cuda.h.
                self.check(self.driver.cuFuncSetAttribute(function, 8, SMEM))
            self.functions[name] = function

    @staticmethod
    def check(code):
        if code:
            raise RuntimeError(f'GLM6 E3 CUDA driver error {code}')

    def launch(self, name, grid, args, stream, smem=0):
        storage = [C.c_void_p(value.data_ptr()) if hasattr(value, 'data_ptr')
                   else C.c_float(value) if isinstance(value, float) else C.c_int(value)
                   for value in args]
        params = (C.c_void_p * len(storage))(*[C.addressof(value) for value in storage])
        self.check(self.driver.cuLaunchKernel(self.functions[name],
            C.c_uint(grid[0]), C.c_uint(grid[1]), C.c_uint(1),
            C.c_uint(256), C.c_uint(1), C.c_uint(1), C.c_uint(smem),
            C.c_void_p(stream), params, None))


def bind(layer):
    import torch
    if hasattr(layer, 'glm6_e3_binding'):
        return layer.glm6_e3_binding
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError('E3 layer binding must precede CUDA capture')
    mixed = layer.exl3_mixed_trellis
    h = int(layer.exl3_hidden_size)
    i = int(layer.exl3_intermediate_size_per_partition)
    if h != 6144 or i != 512 or mixed['trellis_codebook'] != 'mcg':
        raise ValueError('E3 candidate supports full-GLM 6144x512 exact MCG fragments only')
    rotations = mixed['rotations']
    ne = sum(len(ids) for ids in mixed['tier_ids'])
    pointers = {key: [] for key in ('gate', 'up', 'down', 'gate_suh', 'up_suh',
                                    'gate_svh', 'up_svh', 'down_suh', 'down_svh')}
    bits_table = []
    offset = 0
    for bits, ids, tier in zip(mixed['tier_bits'], mixed['tier_ids'], mixed['tiers'], strict=True):
        n = len(ids)
        # Prepared B12X arrays are zero-copy int32 views of packed int16 tiles.
        w13, w2 = tier.w13, tier.w2
        if (not w13.is_contiguous() or not w2.is_contiguous()
                or w13.numel() * w13.element_size() != 2*n*h*i*bits//8
                or w2.numel() * w2.element_size() != n*h*i*bits//8
                or tier.w13_layout != 'trellis_t256_proj'):
            raise ValueError('E3 requires un-repacked projection-major trellis buffers')
        addresses = projection_offsets(bits, n, h, i)
        for name, base in (('gate', w13), ('up', w13), ('down', w2)):
            pointers[name].extend(base.data_ptr() + byte for byte in addresses[name])
        bits_table.extend([bits] * n)
        offset += n
    for name, tensor, shape in (
        ('gate_suh', rotations.gate_suh, (ne, h)),
        ('up_suh', rotations.up_suh, (ne, h)),
        ('down_svh', rotations.down_svh, (ne, h)),
        ('intermediate', rotations.intermediate, (ne, 3*i))):
        if tensor.dtype != torch.float16 or tuple(tensor.shape) != shape or not tensor.is_contiguous():
            raise ValueError(f'E3 invalid independent rotation storage {name}')
        for expert in range(ne):
            base = tensor.data_ptr() + expert * tensor.stride(0) * tensor.element_size()
            if name == 'intermediate':
                for projection, extra in (('gate_svh', 0), ('up_svh', i), ('down_suh', 2*i)):
                    pointers[projection].append(base + extra * tensor.element_size())
            else:
                pointers[name].append(base)
    device = rotations.gate_suh.device
    pointers = {name: torch.tensor(values, dtype=torch.int64, device=device)
                for name, values in pointers.items()}
    pointers['bits'] = torch.tensor(bits_table, dtype=torch.int32, device=device)
    pointers.update(hidden=h, intermediate=i, experts=ne, owner=mixed)
    layer.glm6_e3_binding = pointers
    return pointers


def route_tables(topk_ids, topk_weights, mapping, experts):
    import torch
    m, k = topk_ids.shape
    local = mapping[topk_ids.to(torch.long)].reshape(-1).to(torch.long)
    keys = torch.where(local >= 0, local, experts)
    sorted_keys, order = torch.sort(keys, stable=True)
    counts = torch.zeros(experts + 1, dtype=torch.int32, device=topk_ids.device)
    counts.scatter_add_(0, keys, torch.ones_like(keys, dtype=torch.int32))
    counts = counts[:experts]
    cumulative = counts.cumsum(0)
    offsets = cumulative - counts
    tiles = (counts + 63) // 64
    tile_cum = tiles.cumsum(0)
    tile_off = tile_cum - tiles
    seg = torch.arange((m*k + 63)//64 + experts, device=topk_ids.device)
    exp = torch.searchsorted(tile_cum, seg, right=True).clamp(max=experts - 1)
    tile = seg - tile_off[exp]
    return dict(row_token=(order // k).contiguous(),
        row_expert=sorted_keys.to(torch.int32), row_route=order.contiguous(),
        row_weight=topk_weights.reshape(-1)[order].to(torch.float32),
        num_rows=cumulative[-1:].to(torch.int32), num_segs=tile_cum[-1:].to(torch.int32),
        seg_expert=exp.to(torch.int32), seg_row0=(offsets[exp] + tile*64).to(torch.int32),
        seg_rows=(counts[exp] - tile*64).clamp(min=0, max=64).to(torch.int32))


def apply(layer, x, weights, ids, *, grid_cap=512, stream_scratch=False):
    import torch
    m, h = x.shape
    k = int(ids.shape[1])
    capacity = int(layer.exl3_max_num_batched_tokens)
    if (x.dtype != torch.bfloat16 or not x.is_contiguous() or k != 8
            or weights.dtype != torch.float32 or m > capacity):
        raise ValueError('E3 expects contiguous BF16, FP32 routes, top8, within batch capacity')
    binding = bind(layer)
    device = x.device
    key = (device.index, h, binding['intermediate'], capacity, k)
    # Explicit isolation variant; default keeps original production behavior.
    if stream_scratch:
        key += (torch.cuda.current_stream(device).cuda_stream,)
    if key not in _SCRATCH:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('E3 scratch must be allocated before graph capture')
        rows = capacity * k
        _SCRATCH[key] = dict(
            gate=torch.empty((rows, h), dtype=torch.float16, device=device),
            up=torch.empty((rows, h), dtype=torch.float16, device=device),
            intermediate=torch.empty((rows, 2*binding['intermediate']), dtype=torch.float16, device=device),
            output=torch.empty((capacity, h), dtype=torch.float32, device=device))
        LOG.info('GLM6 E3 mixed-fragment prefill initialized capacity=%d topk=%d shape=%dx%d scratch=%.1f MiB',
                 capacity, k, h, binding['intermediate'], scratch_bytes(capacity, k, h, binding['intermediate'])/2**20)
    if device.index not in _MODULES:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('E3 module must be loaded before graph capture')
        _MODULES[device.index] = DeviceModule(Path(__file__).with_name('grouped_fragments.cubin'))
    module = _MODULES[device.index]
    scratch = _SCRATCH[key]
    routes = route_tables(ids, weights, layer.exl3_mixed_trellis['global_to_combined'], binding['experts'])
    output = scratch['output'][:m]
    stream = torch.cuda.current_stream(device).cuda_stream
    module.launch('gather', (min(1024, (m*k + 7)//8), h//128),
        [x, routes['row_token'], routes['row_expert'], binding['gate_suh'], binding['up_suh'],
         scratch['gate'], scratch['up'], routes['num_rows'], h], stream)
    grid_y = routes['seg_expert'].numel() if grid_cap is None else min(grid_cap, routes['seg_expert'].numel())
    segments = [routes[name] for name in ('seg_expert', 'seg_row0', 'seg_rows', 'num_segs')]
    module.launch('gateup', (binding['intermediate']//128, grid_y),
        [scratch['gate'], scratch['up'], binding['gate'], binding['up'], binding['gate_svh'],
         binding['up_svh'], binding['down_suh'], scratch['intermediate'], *segments,
         binding['bits'], h, binding['intermediate'], float('inf')], stream, SMEM)
    from .native_epilogues import activate, ordered_sum
    # Gate rotation storage is dead after FC1; up rotation storage is dead
    # after FC1 too. Reuse them for activation and router-order FC2 output.
    activate(layer, scratch['intermediate'], scratch['gate'], routes['row_expert'], m)
    module.launch('down', (h//256, grid_y),
        [scratch['gate'], binding['down'], binding['down_svh'], scratch['up'],
         routes['row_route'], routes['row_weight'], *segments, binding['bits'],
         binding['intermediate'], h], stream, SMEM)
    ordered_sum(layer, scratch['up'], output, weights, ids)
    return output.to(x.dtype)
