"""Full-GLM TP6 local MLA forward, with original BF16 projections.

Returns a rank's BF16 output-projection contribution. Six-rank reduction and
residual/MLP execution belong to the model engine and are not performed here.
"""
import torch
from .attention import (AttentionScratch,load_head_weights,apply_rope,absorb_query,attend,expand_value)
from .cache import write_mla
from .dense import linear
from .indexer import Indexer,IndexerWeights,IndexerForwardScratch
from .norms import rms


class MLAWeights:
    def __init__(self,reader,layer,rank,device='cuda'):
        self.layer,self.rank=layer,rank
        self.source=reader.config.indexer_source(layer)
        first,last,slots=reader.config.head_range(rank)
        self.real_heads=last-first
        prefix=f'model.layers.{layer}.self_attn.'

        def read(name,shape):
            meta=reader.tensor_meta(prefix+name)
            if meta['dtype']!='BF16' or tuple(meta['shape'])!=shape:
                raise ValueError('Expected original full GLM BF16 attention weights')
            return reader.read_tensor(prefix+name)

        self.qkv_a=torch.cat((read('q_a_proj.weight',(2048,6144)),
                              read('kv_a_proj_with_mqa.weight',(576,6144))),0).to(device)
        self.q_norm=read('q_a_layernorm.weight',(2048,)).to(device)
        self.kv_norm=read('kv_a_layernorm.weight',(512,)).to(device)
        qb=read('q_b_proj.weight',(16384,2048)).view(64,256,2048)
        self.q_b=torch.zeros((slots,256,2048),dtype=torch.bfloat16,device=device)
        self.q_b[:last-first].copy_(qb[first:last]);self.q_b=self.q_b.view(slots*256,2048)
        self.wk,self.wv=load_head_weights(reader,layer,rank,device)
        # Padded heads have no output columns, so stale/NaN padding cannot contribute.
        self.o=read('o_proj.weight',(6144,16384))[:,first*256:last*256].contiguous().to(device)
        self.indexer=Indexer(IndexerWeights(reader,layer,device)) if self.source==layer else None


class SelectionState:
    """One forward's most recent full-indexer selection, separate for target/MTP.

    The graph planner captures full-indexer publication before shared consumers.
    On eager calls, scope identity and row-metadata pointers reject stale reuse.
    Graph replays must refresh every full-indexer producer in the captured plan.
    """
    def __init__(self,rows,device):
        if type(rows) is not int or not 1<=rows<=3072:
            raise ValueError('Invalid full GLM selection capacity')
        self.rows=rows
        self.tokens=torch.empty((rows,2048),dtype=torch.int32,device=device)
        self.counts=torch.empty(rows,dtype=torch.int32,device=device)
        self.source,self.scope,self.metadata=None,None,None

    def publish(self,source,scope,positions,bases,tokens,counts):
        rows=positions.shape[0]
        if scope is None or not 1<=rows<=self.rows or tokens.shape!=(rows,2048) or counts.shape!=(rows,):
            raise ValueError('Invalid selection publication')
        self.tokens[:rows].copy_(tokens);self.counts[:rows].copy_(counts)
        self.source,self.scope=source,scope
        self.metadata=(rows,positions.data_ptr(),bases.data_ptr())

    def require(self,source,scope,positions,bases):
        if (scope is None or self.source!=source or self.scope is not scope
                or self.metadata!=(positions.shape[0],positions.data_ptr(),bases.data_ptr())):
            raise ValueError('Shared index selection belongs to another layer/forward/row layout')
        n=positions.shape[0]
        return self.tokens[:n],self.counts[:n]


class MLAScratch:
    def __init__(self,rows,context_capacity,device,*,real_heads=11,attention_part_rows=128,skip_empty_attention=False):
        if type(rows) is not int or not 1<=rows<=3072 or real_heads not in (9,11):
            raise ValueError('Invalid full GLM TP6 MLA scratch geometry')
        self.rows,self.real_heads=rows,real_heads
        def bf(*shape):return torch.empty(shape,dtype=torch.bfloat16,device=device)
        self.qkv=bf(rows,2624);self.q_lora=bf(rows,2048);self.kv_lora=bf(rows,512)
        self.q=bf(rows,11,256);self.q_rotated=bf(rows,11,256)
        self.k_rope=bf(rows,64);self.k_rotated=bf(rows,64);self.q_rope=bf(rows,11,64)
        self.q_absorbed=bf(rows,11,512);self.attended=bf(rows,11,512);self.expanded=bf(rows,11,256)
        self.o_input=bf(rows,real_heads*256);self.output=bf(rows,6144)
        self.attention=AttentionScratch(rows,device,real_heads,part_rows=attention_part_rows,skip_empty=skip_empty_attention)
        self.indexer=IndexerForwardScratch(rows,context_capacity,device)


class MLA:
    def __init__(self,weights):self.weights=weights

    def forward(self,hidden,positions,bases,slots,cache,table,scratch,selection,*,scope,visible_tokens=None):
        """Normalized hidden rows to one TP rank's unreduced output contribution.

        Cache/request admission checks belong to the planner. Active write slots
        are unique; -1 skips a write. Existing slots remain private to this layer.
        """
        w=self.weights;rows=hidden.shape[0]
        if (hidden.shape!=(rows,6144) or not 1<=rows<=scratch.rows or rows>selection.rows or scope is None
                or scratch.real_heads!=w.real_heads or cache.layer!=w.layer
                or (cache.index_keys is not None)!=(w.indexer is not None)):
            raise ValueError('MLA layer/cache/scratch/selection geometry mismatch')
        if w.indexer is None:
            tokens,counts=selection.require(w.source,scope,positions,bases)
        qkv=scratch.qkv[:rows];ql=scratch.q_lora[:rows];kl=scratch.kv_lora[:rows]
        linear(hidden,w.qkv_a,qkv)
        rms(qkv[:,:2048],w.q_norm,ql);rms(qkv[:,2048:2560],w.kv_norm,kl)
        q=scratch.q[:rows];qr=scratch.q_rotated[:rows]
        linear(ql,w.q_b,q.view(rows,2816))
        kr=scratch.k_rope[:rows];krr=scratch.k_rotated[:rows]
        kr.copy_(qkv[:,2560:])
        apply_rope(q,kr,positions,table,qr,krr,real_heads=w.real_heads)
        write_mla(kl,krr,slots,cache)
        if w.indexer is not None:
            tokens,counts=w.indexer.forward(hidden,ql,positions,bases,slots,cache.index_keys,
                                             cache.index_scales,table,scratch.indexer,visible_tokens=visible_tokens)
            selection.publish(w.layer,scope,positions,bases,tokens,counts)
            tokens,counts=selection.require(w.source,scope,positions,bases)
        qa=scratch.q_absorbed[:rows];qrope=scratch.q_rope[:rows]
        absorb_query(qr,w.wk,qa,real_heads=w.real_heads);qrope.copy_(qr[:,:,192:])
        ao=scratch.attended[:rows]
        attend(qa,qrope,cache.latent,cache.rope,tokens,counts,positions,bases,ao,scratch.attention,
               latent_scales=cache.scales)
        expanded=scratch.expanded[:rows]
        expand_value(ao,w.wv,expanded,real_heads=w.real_heads)
        oi=scratch.o_input[:rows]
        oi.view(rows,w.real_heads,256).copy_(expanded[:,:w.real_heads])
        return linear(oi,w.o,scratch.output[:rows])
