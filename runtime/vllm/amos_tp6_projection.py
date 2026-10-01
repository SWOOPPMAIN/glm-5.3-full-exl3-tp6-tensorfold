"""Opt-in full-GLM TP6 output sharding of its otherwise replicated QKV-A GEMM.

Preserves checkpoint rows and the existing online MXFP8 quantizer. All ranks
gather the projected rows before the original Q/KV split and normalization.
"""
import torch

from vllm.distributed import (get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size, tensor_model_parallel_all_gather)
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.tp6_projection_layout import load_component


class TP6FusedQkvAProjection(ReplicatedLinear):
    def __init__(self, input_size, output_size, quant_config=None, prefix=''):
        if input_size != 6144 or list(output_size) != [2048,576]:
            raise ValueError('TP6 QKV-A sharding is limited to full GLM geometry')
        if get_tensor_model_parallel_world_size() != 6:
            raise ValueError('TP6 QKV-A sharding requires exactly six ranks')
        self.shard_rank = get_tensor_model_parallel_rank()
        self.loaded_components = set()
        super().__init__(input_size,448,bias=False,quant_config=quant_config,prefix=prefix)
        if set(dict(self.named_parameters())) != {'weight'} or self.weight.dtype != torch.bfloat16:
            raise ValueError('TP6 QKV-A needs the original BF16 checkpoint and online dense quantizer')
        self.weight.data.zero_()

    def weight_loader(self, param, loaded_weight, shard_id=None):
        if shard_id not in (0,1) or shard_id in self.loaded_components:
            raise ValueError('Expected each Q/KV checkpoint component exactly once')
        if loaded_weight.dtype != torch.bfloat16 or param.dtype != torch.bfloat16:
            raise ValueError('TP6 QKV-A weight loading must precede online quantization')
        load_component(param.data,loaded_weight,self.shard_rank,shard_id)
        self.loaded_components.add(shard_id)

    def forward(self, input_):
        if self.loaded_components != {0,1}:
            raise RuntimeError('QKV-A checkpoint loading is incomplete')
        output,_ = super().forward(input_)
        gathered = tensor_model_parallel_all_gather(output,dim=-1)
        return gathered[...,:2624].contiguous(),None
