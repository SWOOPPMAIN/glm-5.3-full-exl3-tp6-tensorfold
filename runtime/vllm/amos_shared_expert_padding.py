"""Separate GLM shared MXFP8 width from the unchanged EXL3 piece width.

Geometry is also used by kindlingai/glm-5.3-full-exl3-tp6 at 0ecf21eb,
runtime/tp6-fragments/patch_vllm.py. This uses our native virtual-TP plan and
loaders, preserving 66 attention heads and existing expert placement.
"""
import os

AXIS = {'original_size':2048, 'padded_size':2304, 'tp_size':6, 'local_size':384}


def enabled():
    value=os.getenv('AMOS_TP6_SHARED_384','0')
    if value not in ('0','1'):
        raise ValueError('AMOS_TP6_SHARED_384 must be 0 or 1')
    return value=='1'


def validate_model(config):
    target=getattr(config,'model_type',None)=='glm_moe_dsa'
    draft=(getattr(config,'model_type',None)=='deepseek_mtp'
           and getattr(config,'architectures',None)==['DeepSeekMTPModel']
           and getattr(config,'n_predict',None)==1)
    if (not (target or draft)
            or config.hidden_size!=6144 or config.num_hidden_layers!=78
            or config.n_routed_experts!=256 or config.n_shared_experts!=1
            or os.getenv('AMOS_EXL3_TP6_PIECES')!='1'):
        raise ValueError('Shared384 requires full GLM5.3 target or its native MTP config with original TP6 EXL3 pieces')


def shared_axis(model_config, parallel_config, original_size, count, previous):
    if not enabled():
        return previous
    validate_model(model_config.hf_text_config)
    p=parallel_config
    if (p.tensor_parallel_size!=6 or p.pipeline_parallel_size!=1
            or p.data_parallel_size!=1 or p.prefill_context_parallel_size!=1
            or p.decode_context_parallel_size!=1
            or original_size!=2048 or count!=1 or previous is not None):
        raise ValueError('Shared384 requires pinned TP6/DCP1 and original shared width2048')
    return dict(AXIS)


def shared_width(config, tp_size, previous):
    if not enabled():
        return previous
    validate_model(config)
    plan=getattr(config,'vllm_virtual_tp_plan',{})
    if (tp_size!=6 or previous!=3072 or plan.get('shared_expert_intermediate_size')!=AXIS
            or plan.get('moe_intermediate_size',{}).get('local_size')!=512
            or plan.get('moe_intermediate_size',{}).get('padded_size')!=3072):
        raise ValueError('Shared384 constructor and virtual TP plan disagree')
    return AXIS['padded_size']
