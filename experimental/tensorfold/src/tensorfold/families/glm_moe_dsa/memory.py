"""Full-model admission estimates, validated against actual storage after load.

Stored rank files include replicated tensors. Count the loader's retained GPU
shards and transformed norms, not the total file sizes. This estimate excludes
allocator/runtime overhead; an explicit reserve and live host guard are required.
"""
import math


def weight_plan(reader):
    cfg=reader.config;rank=reader.rank
    real=cfg.head_range(rank)[1]-cfg.head_range(rank)[0]
    def stored(name,shape,dtype='BF16'):
        meta=reader.tensor_meta(name)
        if meta['dtype']!=dtype or tuple(meta['shape'])!=tuple(shape):
            raise ValueError('Memory plan encountered unqualified tensor: '+name)
        return math.prod(shape)*({'BF16':2,'F16':2,'I16':2,'F32':4}[dtype])
    vocab=sum(stored(n,(154880,6144)) for n in ('model.embed_tokens.weight','lm_head.weight'))
    vocab=2*25856*6144*2+stored('model.norm.weight',(6144,))
    mtp=sum(stored('model.layers.78.'+n+'.weight',(6144,)) for n in ('enorm','hnorm','shared_head.norm'))
    stored('model.layers.78.eh_proj.weight',(6144,12288));mtp+=1024*12288*2
    layers=[]
    for layer in range(79):
        prefix=f'model.layers.{layer}.'
        norms=sum(stored(prefix+n+'.weight',(6144,)) for n in ('input_layernorm','post_attention_layernorm'))
        att=prefix+'self_attn.'
        attention=sum(stored(att+n,s) for n,s in (
            ('q_a_proj.weight',(2048,6144)),('kv_a_proj_with_mqa.weight',(576,6144)),
            ('q_a_layernorm.weight',(2048,)),('kv_a_layernorm.weight',(512,))))
        stored(att+'q_b_proj.weight',(16384,2048));attention+=11*256*2048*2
        stored(att+'kv_b_proj.weight',(64*448,512));attention+=11*448*512*2
        stored(att+'o_proj.weight',(6144,16384));attention+=6144*real*256*2
        indexer=0
        if cfg.indexer_source(layer)==layer:
            indexer=sum(stored(att+'indexer.'+n,s) for n,s in (
                ('wq_b.weight',(4096,2048)),('wk.weight',(128,6144)),('weights_proj.weight',(32,6144))))
            for n in ('k_norm.weight','k_norm.bias'):stored(att+'indexer.'+n,(128,))
            indexer+=2*128*4  # Original BF16 normalization values retained as FP32.
        global_width=12288 if layer<3 else 2048
        width=2048 if layer<3 else (512 if rank<4 else 0)
        ff=prefix+'mlp.'+('' if layer<3 else 'shared_experts.')
        for n,s in (('gate_proj.weight',(global_width,6144)),('up_proj.weight',(global_width,6144)),('down_proj.weight',(6144,global_width))):stored(ff+n,s)
        dense=3*width*6144*2
        routed=router=0
        if layer>=3:
            router=stored(prefix+'mlp.gate.weight',(256,6144))+stored(prefix+'mlp.gate.e_score_correction_bias',(256,),'F32')
            bits=reader.tiers[str(layer)]['k']
            if len(bits)!=256 or bits.count(3)!=192 or bits.count(4)!=64:raise ValueError('Quantization tiers changed')
            count=0
            for expert in range(256):
                for original_rank in range(4):
                    if (4*expert+original_rank)%6!=rank:continue
                    count+=1
                    for name,k,n in (('gate_proj',6144,512),('up_proj',6144,512),('down_proj',512,6144)):
                        ep=prefix+f'mlp.experts.{expert}.{name}.rank{original_rank}.'
                        routed+=stored(ep+'trellis',(k//16,n//16,16*bits[expert]),'I16')
                        routed+=stored(ep+'suh',(k,),'F16')+stored(ep+'svh',(n,),'F16')
            routed+=count*(3*8+3*4)+256*4  # Device pointer/width tables and global-to-local routing.
        item=dict(layer=layer,norms=norms,attention=attention,indexer=indexer,dense_shared=dense,router=router,routed=routed)
        item['total']=sum(v for k,v in item.items() if k!='layer');layers.append(item)
    return dict(rank=rank,vocabulary=vocab,mtp=mtp,layers=layers,total=vocab+mtp+sum(x['total'] for x in layers),
                stored_file_bytes=sum(x['bytes'] for x in reader.files.values()),
                scope='Retained GPU tensor payload; original marker validation occurs in the actual loader; no runtime/allocator reserve included')


def tensor_storage_bytes(obj,*,device_type=None,skip=()):
    """Unique tensor storage, including tensors retained in nested lists."""
    import torch
    seen_objects=set();seen_storage=set();total=0
    def walk(x):
        nonlocal total
        if id(x) in seen_objects:return
        seen_objects.add(id(x))
        if isinstance(x,torch.Tensor):
            if device_type is not None and x.device.type!=device_type:return
            storage=x.untyped_storage();key=(x.device,storage._cdata)
            if key not in seen_storage:seen_storage.add(key);total+=storage.nbytes()
        elif isinstance(x,dict):
            for k,v in x.items():
                if k not in skip:walk(v)
        elif isinstance(x,(list,tuple)):
            for v in x:walk(v)
        elif hasattr(x,'__dict__'):
            for k,v in vars(x).items():
                if k not in skip:walk(v)
    walk(obj);return total


def workspace_plan(config,rank,rows,capacity,*,logit_rows=17,expert_chunk_rows=128,bulk_min_rows=None,attention_part_rows=128,skip_empty_attention=False):
    """Instantiate actual scratch shapes on Torch's non-allocating meta device."""
    import torch
    from types import SimpleNamespace as NS
    from .experts import RoutedLayer
    from .model import ModelWorkspace
    from .reduction_plan import reduction_plan
    def layer(index):
        norm=torch.empty(6144,dtype=torch.bfloat16,device='meta')
        if index<3:ffn=NS(width=2048)
        else:
            count=sum((4*e+p)%6==rank for e in range(256) for p in range(4))
            ex=RoutedLayer(NS(dims=6144,width=512,count=count),torch.empty(256,dtype=torch.int32,device='meta'))
            ffn=NS(gate=torch.empty((256,6144),dtype=torch.bfloat16,device='meta'),shared=NS(weights=NS(width=512 if rank<4 else 0)),routed=ex)
        return NS(layer=index,rank=rank,input_norm=norm,attention=NS(weights=NS(real_heads=9 if rank==5 else 11)),ffn=NS(weights=ffn))
    weights=NS(layers=[layer(i) for i in range(79)],vocab=NS(norm=torch.empty(6144,device='meta')))
    workspace=ModelWorkspace(weights,rows,capacity,logit_rows=logit_rows,expert_chunk_rows=expert_chunk_rows,
                             attention_part_rows=attention_part_rows,skip_empty_attention=skip_empty_attention)
    scratch=tensor_storage_bytes(workspace,device_type='meta',skip=('weights','layer'))
    caches=capacity*(79*656+sum(config.indexer_source(i)==i for i in range(79))*132)
    collective=reduction_plan(rows,bulk_min_rows=bulk_min_rows)['total'];rope=capacity*64*2
    return dict(rows=rows,capacity=capacity,logit_rows=logit_rows,expert_chunk_rows=expert_chunk_rows,bulk_min_rows=bulk_min_rows,
                attention_part_rows=attention_part_rows,skip_empty_attention=skip_empty_attention,workspace=scratch,cache=caches,
                collective=collective,rope=rope,total=scratch+caches+collective+rope)
