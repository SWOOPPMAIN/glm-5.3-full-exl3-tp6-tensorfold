"""TP6 short-batch output shards derived from unchanged online MXFP8 weights.

The original layer, checkpoint loader, quantizer and full packed projection stay
intact. Only target QKV-A and indexer WQ-B get an additional packed row shard.
Both see replicated inputs before the existing indexer query-splitting stage.
"""
import os
import re

import torch

WORLD = 6
MAX_ROWS = 20
DECODE_ROWS = (1,2,3,4,5,6,8,9,10,12,15,16,20)
_PREFIX = re.compile(r'^model\.layers\.(\d+)\.self_attn\.(fused_qkv_a_proj|indexer\.wq_b)$')
_SHAPES = {'fused_qkv_a_proj':(2624,6144), 'indexer.wq_b':(4096,2048)}
INDEX_PATTERN = 'FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS'


def enabled():
    value = os.getenv('AMOS_TP6_DECODE_PROJECTIONS','0')
    if value not in ('0','1'):
        raise ValueError('AMOS_TP6_DECODE_PROJECTIONS must be 0 or 1')
    return value == '1'


def projection_kind(prefix):
    match = _PREFIX.fullmatch(prefix)
    # The MTP layer and all other projections retain their original path.
    return match[2] if match and 0 <= int(match[1]) < 78 else None


def quantized_shard(weight, scales, rank):
    """Copy exact FP8 bytes and block scales; only the final rank gets zeros."""
    if type(rank) is not int or not 0 <= rank < WORLD:
        raise ValueError('Expected TP6 rank 0..5')
    if weight.ndim != 2 or weight.dtype != torch.float8_e4m3fn:
        raise ValueError('Expected original row-major MXFP8 E4M3 weights')
    n,k = weight.shape
    if k % 32 or scales.dtype != torch.uint8 or tuple(scales.shape) != (n,k//32):
        raise ValueError('Expected original unswizzled block32 scales')
    local_n = ((n+WORLD*64-1)//(WORLD*64))*64
    values = torch.zeros((local_n,k),dtype=weight.dtype,device=weight.device)
    local_scales = torch.full((local_n,k//32),127,dtype=scales.dtype,device=scales.device)
    begin = rank*local_n
    count = max(0,min(local_n,n-begin))
    if count:
        values[:count].copy_(weight[begin:begin+count])
        local_scales[:count].copy_(scales[begin:begin+count])
    return values,local_scales


def prepare(layer, weight, scales, mxfp8):
    if not enabled():
        return
    kind = projection_kind(getattr(layer,'prefix',''))
    if kind is None:
        return
    if tuple(weight.shape) != _SHAPES[kind] or getattr(layer,'bias',None) is not None:
        raise ValueError('Decode projection requires the pinned full-GLM geometry without bias')
    from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
    if get_tensor_model_parallel_world_size() != WORLD:
        raise ValueError('Decode projection requires exactly six tensor-parallel ranks')
    values,local_scales = quantized_shard(weight,scales,get_tensor_model_parallel_rank())
    layer.amos_decode_projection = mxfp8.pack_weight(values,local_scales)
    layer.amos_decode_projection_kind = kind


def apply_if_small(layer, input_2d, bias, mxfp8, stream, gather=None):
    """Called inside the existing opaque linear op; runtime shape stays dynamic."""
    shard = getattr(layer,'amos_decode_projection',None)
    if shard is None or not 0 < input_2d.shape[0] <= MAX_ROWS:
        return None
    if bias is not None:
        raise ValueError('Decode projection does not support bias')
    if gather is None:
        from vllm.distributed import tensor_model_parallel_all_gather
        gather = tensor_model_parallel_all_gather
    local = mxfp8.mm(input_2d,shard,expected_m=int(input_2d.shape[0]),stream=stream)
    output = gather(local,dim=-1)
    n = int(layer.b12x_mxfp8_packed_weight.out_features)
    return output[:,:n].contiguous()


def warmup(model, mxfp8):
    if not enabled():
        return 0
    layers = [m for m in model.modules() if getattr(m,'amos_decode_projection',None) is not None]
    eligible = [m for m in model.modules() if projection_kind(getattr(m,'prefix',''))]
    if len(eligible) != len(layers):
        raise RuntimeError('An eligible decode projection was not packed after loading')
    if not layers:
        raise RuntimeError('Decode projections enabled but the target model has no prepared shards')
    # IndexCache's S layers reuse earlier indices and instantiate no indexer.
    expected = {f'model.layers.{i}.self_attn.fused_qkv_a_proj' for i in range(78)}
    expected.update(f'model.layers.{i}.self_attn.indexer.wq_b'
                    for i,kind in enumerate(INDEX_PATTERN) if kind == 'F')
    actual = {layer.prefix for layer in layers}
    if actual != expected:
        raise RuntimeError(f'Decode projection set differs: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}')
    seen = set()
    calls = 0
    for layer in layers:
        packed = layer.amos_decode_projection
        signature = (int(packed.out_features),int(packed.in_features))
        if signature in seen:
            continue
        seen.add(signature)
        device = packed.weight.values.device
        for rows in DECODE_ROWS:
            x = torch.zeros((rows,int(packed.in_features)),device=device,dtype=torch.bfloat16)
            mxfp8.mm(x,packed,expected_m=rows,stream=torch.cuda.current_stream().cuda_stream)
            calls += 1
    from vllm.logger import init_logger
    init_logger(__name__).info('AMOS TP6 decode projections: %d target projections, %d geometries, %d warmups, rows<=%d',
        len(layers),len(seen),calls,MAX_ROWS)
    return calls
