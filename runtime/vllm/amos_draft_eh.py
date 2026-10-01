"""Short-batch TP6 output sharding of the unchanged BF16 draft EH matrix.

Keep the original nn.Linear and checkpoint loader. The opaque op reads a view
of its existing rows; no persistent weight copy or requantization is needed.
Large draft-prefill batches retain the full matrix. Target layers are untouched.
"""
import os
import torch
import torch.nn.functional as F

MAX_ROWS = 20
DECODE_ROWS = (1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 20)
_REGISTERED = False


def enabled():
    value = os.getenv('AMOS_TP6_DRAFT_EH', '0')
    if value not in ('0', '1'):
        raise ValueError('AMOS_TP6_DRAFT_EH must be 0 or 1')
    return value == '1'


def validate_config(config, prefix):
    parallel = config.parallel_config
    draft = config.speculative_config.draft_model_config.hf_config
    target = config.speculative_config.target_model_config.hf_text_config
    if (prefix != 'model.layers.78' or target.model_type != 'glm_moe_dsa'
            or draft.model_type != 'deepseek_mtp'
            or draft.hidden_size != 6144 or draft.num_hidden_layers != 78
            or parallel.tensor_parallel_size != 6
            or parallel.pipeline_parallel_size != 1
            or parallel.data_parallel_size != 1
            or parallel.decode_context_parallel_size != 1
            or config.scheduler_config.max_num_seqs != 4):
        raise ValueError('Draft EH sharding requires pinned full GLM TP6/PP1/DP1/DCP1/C4')


def configure(config, prefix):
    if not enabled():
        return -1
    validate_config(config, prefix)
    from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
    if get_tensor_model_parallel_world_size() != 6:
        raise ValueError('Draft EH process group must have six ranks')
    register()
    return get_tensor_model_parallel_rank()


def project(x, weight, rank, gather=None):
    if (type(rank) is not int or not 0 <= rank < 6 or x.ndim != 2
            or tuple(weight.shape) != (6144, 12288) or x.shape[1] != 12288
            or weight.dtype != torch.bfloat16 or x.dtype != torch.bfloat16
            or weight.device != x.device or not weight.is_contiguous()):
        raise ValueError('Draft EH requires original contiguous BF16 6144x12288 weights')
    if not 0 < x.shape[0] <= MAX_ROWS:
        return F.linear(x, weight)
    if gather is None:
        from vllm.distributed import tensor_model_parallel_all_gather
        gather = tensor_model_parallel_all_gather
    local = F.linear(x, weight[rank*1024:(rank+1)*1024])
    return gather(local, dim=-1)


def _forward(x: torch.Tensor, weight: torch.Tensor, rank: int) -> torch.Tensor:
    return project(x, weight, rank)


def _fake(x: torch.Tensor, weight: torch.Tensor, rank: int) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], 6144))


def register():
    global _REGISTERED
    if not _REGISTERED:
        from vllm.utils.torch_utils import direct_register_custom_op
        direct_register_custom_op('amos_tp6_draft_eh', _forward, fake_impl=_fake)
        _REGISTERED = True


def apply(layer, x, rank):
    if rank < 0:
        return layer(x)
    return torch.ops.vllm.amos_tp6_draft_eh(x, layer.weight, rank)


def warmup(model):
    if not enabled():
        return
    layers = [m for m in model.modules() if getattr(m, 'amos_draft_eh_rank', -1) >= 0]
    if len(layers) != 1:
        raise RuntimeError('Expected exactly one configured GLM draft EH projection')
    layer = layers[0]
    weight = layer.eh_proj.weight
    rank = layer.amos_draft_eh_rank
    if (tuple(weight.shape) != (6144, 12288) or weight.dtype != torch.bfloat16
            or not weight.is_cuda or not weight.is_contiguous() or layer.eh_proj.bias is not None):
        raise ValueError('Loaded draft EH weight differs from the qualified geometry')
    with torch.inference_mode():
        for rows in DECODE_ROWS:
            x = torch.zeros((rows, 12288), device=weight.device, dtype=weight.dtype)
            F.linear(x, weight[rank*1024:(rank+1)*1024])
    torch.cuda.synchronize(weight.device)
    from vllm.logger import init_logger
    init_logger(__name__).info('AMOS TP6 draft EH: one projection, rank%d, %d local GEMM warmups, rows<=%d',
                              rank, len(DECODE_ROWS), MAX_ROWS)
