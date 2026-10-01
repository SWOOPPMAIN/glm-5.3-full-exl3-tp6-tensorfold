"""Original BF16 vocabulary shards and bounded, all-rank logits.

The engine selects which normalized hidden rows need logits. A 3072-token
prefill therefore does not allocate a 3072 x vocabulary matrix. Token values
must be validated by the request planner before any captured embedding call.
"""
import torch
import triton
import triton.language as tl

from .decoder import TP6Reduction
from .dense import linear

VOCAB=154880
WIDTH=25856
LOGIT_ROWS=128


def original_norm(reader,name,device):
    meta=reader.tensor_meta(name)
    if meta['dtype']!='BF16' or tuple(meta['shape'])!=(6144,):
        raise ValueError('Expected original BF16 hidden norm: '+name)
    return reader.read_tensor(name,device)


class VocabWeights:
    def __init__(self,reader,device='cuda'):
        self.rank=reader.rank
        self.start,self.stop,self.width=reader.config.vocab_range(self.rank)
        if reader.config.vocab!=VOCAB or self.width!=WIDTH:
            raise ValueError('Unqualified full GLM vocabulary partition')
        tables=[]
        for name in ('model.embed_tokens.weight','lm_head.weight'):
            meta=reader.tensor_meta(name)
            if meta['dtype']!='BF16' or tuple(meta['shape'])!=(VOCAB,6144):
                raise ValueError('Expected original BF16 vocabulary table: '+name)
            local=reader.read_rows(name,self.start,self.stop,device)
            if self.stop-self.start!=self.width:
                padded=torch.zeros((self.width,6144),dtype=torch.bfloat16,device=device)
                padded[:len(local)].copy_(local)
                local=padded
            tables.append(local)
        self.embedding,self.head=tables
        self.norm=original_norm(reader,'model.norm.weight',device)


class VocabScratch:
    def __init__(self,rows,device,*,logit_rows=LOGIT_ROWS):
        if type(rows) is not int or not 1<=rows<=3072 or type(logit_rows) is not int or not 1<=logit_rows<=LOGIT_ROWS:
            raise ValueError('Invalid bounded vocabulary workspace')
        self.rows,self.logit_rows=rows,logit_rows
        self.local_embedding=torch.empty((rows,6144),dtype=torch.bfloat16,device=device)
        self.embedding=torch.empty_like(self.local_embedding)
        self.projected=torch.empty((logit_rows,WIDTH),dtype=torch.bfloat16,device=device)
        self.local_logits=torch.empty((logit_rows,WIDTH),dtype=torch.float32,device=device)
        self.gather=torch.empty((6*logit_rows,WIDTH),dtype=torch.float32,device=device)
        self.logits=torch.empty((logit_rows,VOCAB),dtype=torch.float32,device=device)


@triton.jit
def _embedding(IDS,WEIGHT,OUT,START:tl.constexpr,STOP:tl.constexpr):
    row=tl.program_id(0);col=tl.program_id(1)*256+tl.arange(0,256)
    token=tl.load(IDS+row)
    value=tl.load(WEIGHT+(token-START)*6144+col,(token>=START)&(token<STOP),0.)
    tl.store(OUT+row*6144+col,value)


@triton.jit
def _logits(BF,OUT,ROWS:tl.constexpr,REAL:tl.constexpr,WIDTH:tl.constexpr):
    i=tl.program_id(0)*256+tl.arange(0,256)
    x=tl.load(BF+i,i<ROWS*WIDTH,0.).to(tl.float32)
    tl.store(OUT+i,tl.where(i%WIDTH<REAL,x,-float('inf')),i<ROWS*WIDTH)


class Vocabulary:
    def __init__(self,weights,reduction):
        if not isinstance(reduction,TP6Reduction) or reduction.rank!=weights.rank:
            raise ValueError('Vocabulary requires its real TP6 collective owner')
        self.weights,self.reduction=weights,reduction

    def embed(self,token_ids,scratch):
        w=self.weights
        if (token_ids.ndim!=1 or not 1<=len(token_ids)<=scratch.rows
                or token_ids.dtype!=torch.int64 or not token_ids.is_cuda or not token_ids.is_contiguous()
                or token_ids.device!=w.embedding.device or scratch.embedding.device!=w.embedding.device):
            raise ValueError('Expected contiguous CUDA token IDs and matching vocabulary scratch')
        rows=len(token_ids);local=scratch.local_embedding[:rows]
        _embedding[(rows,24)](token_ids,w.embedding,local,w.start,w.stop)
        return self.reduction.sum_into(local,scratch.embedding[:rows])

    def project(self,hidden,scratch):
        """Return borrowed FP32 logits in global token order, excluding padding.

        Match the native default head contract: original BF16 projection output
        followed by conversion to FP32 for sampling. No extra final norm here.
        """
        w=self.weights;rows=hidden.shape[0]
        if not 1<=rows<=scratch.logit_rows or scratch.projected.device!=w.head.device:
            raise ValueError('Select bounded normalized hidden rows for vocabulary projection')
        projected=scratch.projected[:rows];local=scratch.local_logits[:rows]
        linear(hidden,w.head,projected)
        _logits[(triton.cdiv(rows*WIDTH,256),)](projected,local,rows,w.stop-w.start,w.width)
        return self.reduction.gather_columns_into(local,scratch.logits[:rows],gather=scratch.gather)
