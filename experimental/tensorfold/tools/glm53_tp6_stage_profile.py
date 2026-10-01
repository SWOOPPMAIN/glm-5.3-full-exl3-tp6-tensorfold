"""Temporary eager CUDA-event instrumentation for the full-model diagnostic.

Intervals include stream waits and host launch gaps. Attention contains its
indexer/projection subcategories; never sum those inclusive categories together.
Uninstrumented whole-pass timing remains the performance comparison.
"""
from collections import defaultdict
import functools
import torch


class StageProfile:
    def __init__(self):self.originals=[];self.events=[]

    def wrap(self,owner,name,label):
        original=getattr(owner,name);self.originals.append((owner,name,original))
        @functools.wraps(original)
        def measured(*args,**kwargs):
            category=label(args) if callable(label) else label
            begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            begin.record();value=original(*args,**kwargs);end.record()
            self.events.append((category,begin,end));return value
        setattr(owner,name,measured)

    def __enter__(self):
        from tensorfold.families.glm_moe_dsa import mla,indexer,decoder,mlp,model,dense
        from tensorfold.cuda.exl3 import experts
        self.wrap(mla.MLA,'forward','attention_inclusive')
        self.wrap(indexer.Indexer,'forward','indexer_inclusive')
        self.wrap(mla,'linear','attention_projection')
        self.wrap(mla,'absorb_query','attention_absorb')
        self.wrap(mla,'attend','attention_selected')
        self.wrap(mla,'expand_value','attention_expand')
        self.wrap(mla,'apply_rope','attention_rope')
        self.wrap(mla,'write_mla','attention_cache_write')
        self.wrap(mla,'rms','attention_norms')
        self.wrap(dense,'linear','indexer_projection')
        self.wrap(mlp.Dense,'forward','dense_shared_ffn')
        self.wrap(mlp,'gate_projection','router_projection')
        self.wrap(mlp,'top8','router_top8')
        self.wrap(mlp,'combine','moe_combine')
        self.wrap(decoder.TP6Reduction,'sum_into','tp6_sum_inclusive')
        self.wrap(decoder,'hidden_rms','decoder_norms')
        self.wrap(model,'hidden_rms','final_norm')
        ext=experts._ext()
        for name,category in [('group','expert_grouping'),('rot_in','expert_input_rotation'),
                              ('gateup_epilogue','expert_gateup_epilogue'),('down_combine','expert_down_combine')]:
            self.wrap(ext,name,category)
        self.wrap(ext,'grouped',lambda args:'expert_gateup_compute' if args[10]==2 else 'expert_down_compute')
        return self

    def __exit__(self,*exc):
        for owner,name,original in reversed(self.originals):setattr(owner,name,original)
        self.originals.clear()

    def result(self):
        torch.cuda.synchronize();groups=defaultdict(list)
        for category,begin,end in self.events:groups[category].append(begin.elapsed_time(end))
        result={k:dict(calls=len(v),total_ms=sum(v),mean_ms=sum(v)/len(v)) for k,v in groups.items()}
        self.events.clear();return result
