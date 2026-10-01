"""Full target/MTP forward assembly; request scheduling/admission is separate.

There is deliberately no serving registration yet. Loading all original weights
and caches requires a measured fleet admission plan, not a component-test cap.
All returned tensors are borrowed workspace views. Sampling/draft verification
must retain the needed rows before another pass reuses the same workspace.
"""
import torch
import triton
import triton.language as tl

from .decoder import Decoder,DecoderWeights,DecoderWorkspace,TP6Reduction
from .dense import linear
from .mla import SelectionState
from .norms import hidden_rms
from .vocab import original_norm,VocabWeights,Vocabulary,VocabScratch


class MTPWeights:
    def __init__(self,reader,device='cuda'):
        self.rank=reader.rank
        prefix='model.layers.78.'
        self.enorm=original_norm(reader,prefix+'enorm.weight',device)
        self.hnorm=original_norm(reader,prefix+'hnorm.weight',device)
        self.norm=original_norm(reader,prefix+'shared_head.norm.weight',device)
        name=prefix+'eh_proj.weight';meta=reader.tensor_meta(name)
        if meta['dtype']!='BF16' or tuple(meta['shape'])!=(6144,12288):
            raise ValueError('Expected original BF16 MTP embedding/hidden projection')
        self.eh=reader.read_rows(name,1024*self.rank,1024*(self.rank+1),device)


class MTPScratch:
    def __init__(self,rows,device):
        if type(rows) is not int or not 1<=rows<=3072:
            raise ValueError('Invalid MTP workspace capacity')
        self.rows=rows;self.chunk=min(rows,128)
        def bf(*shape):return torch.empty(shape,dtype=torch.bfloat16,device=device)
        self.masked=bf(self.chunk,6144)
        self.norm_embedding=bf(self.chunk,6144);self.norm_hidden=bf(self.chunk,6144)
        self.residual=bf(self.chunk,6144)
        self.concat=bf(self.chunk,12288)
        self.local=bf(rows,1024);self.mixed=bf(rows,6144)


@triton.jit
def _mask_initial(EMBED,POS,OUT):
    row=tl.program_id(0);col=tl.program_id(1)*256+tl.arange(0,256)
    x=tl.load(EMBED+row*6144+col)
    tl.store(OUT+row*6144+col,tl.where(tl.load(POS+row)==0,0.,x))


class MTPMix:
    def __init__(self,weights,reduction):
        if not isinstance(reduction,TP6Reduction) or reduction.rank!=weights.rank:
            raise ValueError('MTP mix requires its real TP6 collective owner')
        self.weights,self.reduction=weights,reduction

    def forward(self,embedding,previous_hidden,positions,scratch):
        w=self.weights;rows=embedding.shape[0]
        if (embedding.shape!=(rows,6144) or previous_hidden.shape!=embedding.shape
                or not 1<=rows<=scratch.rows or positions.shape!=(rows,) or positions.dtype!=torch.int64
                or not positions.is_cuda or not positions.is_contiguous() or positions.device!=w.eh.device
                or not all(t.is_cuda and t.is_contiguous() and t.device==w.eh.device
                           and t.dtype==torch.bfloat16 for t in (embedding,previous_hidden,scratch.mixed))):
            raise ValueError('Invalid original MTP embedding/hidden mix inputs')
        for start in range(0,rows,scratch.chunk):
            stop=min(start+scratch.chunk,rows);n=stop-start
            masked=scratch.masked[:n];en=scratch.norm_embedding[:n];hn=scratch.norm_hidden[:n]
            _mask_initial[(n,24)](embedding[start:stop],positions[start:stop],masked)
            hidden_rms(masked,w.enorm,en,scratch.residual[:n])
            hidden_rms(previous_hidden[start:stop],w.hnorm,hn,scratch.residual[:n])
            concat=scratch.concat[:n]
            concat[:,:6144].copy_(en);concat[:,6144:].copy_(hn)
            linear(concat,w.eh,scratch.local[start:stop])
        return self.reduction.gather_columns_into(scratch.local[:rows],scratch.mixed[:rows])


class ModelWeights:
    def __init__(self,reader,device='cuda',*,on_layer=None):
        self.config,self.rank=reader.config,reader.rank
        self.vocab=VocabWeights(reader,device)
        layers=[]
        for layer in range(79):
            layers.append(DecoderWeights(reader,layer,device))
            if on_layer is not None:on_layer(layer)
        self.layers=tuple(layers)
        self.mtp=MTPWeights(reader,device)


class ModelWorkspace:
    def __init__(self,weights,rows,context_capacity,*,logit_rows=128,expert_chunk_rows=128,attention_part_rows=128,skip_empty_attention=False):
        device=weights.vocab.norm.device
        self.weights,self.rows=weights,rows
        self.decoder=DecoderWorkspace(weights.layers[0],weights.layers[3],rows,context_capacity,expert_chunk_rows=expert_chunk_rows,
                                      attention_part_rows=attention_part_rows,skip_empty_attention=skip_empty_attention)
        self.vocab=VocabScratch(rows,device,logit_rows=logit_rows)
        self.mtp=MTPScratch(rows,device)
        self.target_selection=SelectionState(rows,device)
        self.mtp_selection=SelectionState(rows,device)
        # Keep final results outside decoder scratch: the next target/MTP pass
        # may consume these tensors while rebinding and overwriting that scratch.
        self.hidden=torch.empty((rows,6144),dtype=torch.bfloat16,device=device)
        self.residual=torch.empty_like(self.hidden)


class TargetForward:
    def __init__(self,layers,norm,reduction):
        if (len(layers)!=78 or [w.layer for w in layers]!=list(range(78))
                or not isinstance(reduction,TP6Reduction)
                or any(w.rank!=reduction.rank for w in layers)):
            raise ValueError('Target requires all78 original layers in order on one TP6 rank')
        self.layers=tuple(Decoder(w,reduction) for w in layers)
        self.norm=norm

    def forward(self,hidden,positions,bases,slots,caches,table,workspace,selection,out,residual_out,*,scope,visible_tokens=None):
        if (len(caches)!=78 or [c.layer for c in caches]!=list(range(78))
                or scope is None):
            raise ValueError('Target requires78 independently owned layer caches and a pass scope')
        residual=None
        for decoder,cache in zip(self.layers,caches):
            workspace.bind(decoder.weights)
            hidden,residual=decoder.forward(hidden,residual,positions,bases,slots,cache,table,
                                             workspace,selection,scope=scope,visible_tokens=visible_tokens)
        return hidden_rms(hidden,self.norm,out,residual_out,residual=residual)[0]


class FullModel:
    def __init__(self,weights,reduction):
        if len(weights.layers)!=79 or weights.layers[78].layer!=78:
            raise ValueError('Full model needs78 target layers and original MTP layer78')
        self.weights=weights
        self.vocab=Vocabulary(weights.vocab,reduction)
        self.target=TargetForward(weights.layers[:78],weights.vocab.norm,reduction)
        self.mix=MTPMix(weights.mtp,reduction)
        self.draft=Decoder(weights.layers[78],reduction)

    def _workspace(self,workspace,rows):
        if workspace.weights is not self.weights or not 1<=rows<=workspace.rows:
            raise ValueError('Model workspace belongs to another model or row capacity')

    def target_forward(self,token_ids,positions,bases,slots,caches,table,workspace,*,scope,visible_tokens=None):
        self._workspace(workspace,len(token_ids))
        embedding=self.vocab.embed(token_ids,workspace.vocab)
        rows=len(token_ids)
        return self.target.forward(embedding,positions,bases,slots,caches,table,workspace.decoder,
            workspace.target_selection,workspace.hidden[:rows],workspace.residual[:rows],scope=scope,visible_tokens=visible_tokens)

    def mtp_forward(self,token_ids,previous_hidden,positions,bases,slots,cache,table,workspace,*,scope,visible_tokens=None):
        """Return post-final-norm hidden for BOTH logits and the next draft step."""
        self._workspace(workspace,len(token_ids))
        embedding=self.vocab.embed(token_ids,workspace.vocab)
        mixed=self.mix.forward(embedding,previous_hidden,positions,workspace.mtp)
        workspace.decoder.bind(self.draft.weights)
        hidden,residual=self.draft.forward(mixed,None,positions,bases,slots,cache,table,
            workspace.decoder,workspace.mtp_selection,scope=scope,visible_tokens=visible_tokens)
        rows=len(token_ids)
        return hidden_rms(hidden,self.weights.mtp.norm,workspace.hidden[:rows],
                           workspace.residual[:rows],residual=residual)[0]

    def logits(self,normalized_hidden,workspace):
        self._workspace(workspace,len(normalized_hidden))
        return self.vocab.project(normalized_hidden,workspace.vocab)
