"""Original full-GLM TP6 feed-forward contributions, before TP reduction.

The first three layers shard 12288 dense channels evenly. Shared experts keep
the original four 512-channel shards on ranks 0..3; ranks 4/5 contribute zero.
The deployed SM121 gate rounds its projection to BF16, then routes with FP32
scores, bias for selection only, and unscaled probabilities.
The local MoE epilogue rounds routed output, scales, then adds shared output in
BF16. The model engine must reduce this result exactly once across six ranks.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from .dense import linear
from .experts import RoutedLayer
from .compiled import require_experts
from tensorfold.cuda.exl3.experts import ACT_F32


@triton.jit(do_not_specialize=['ROWS'])
def _gate(X,W,OUT,BOUND,ROWS,BOUNDS:tl.constexpr):
    r=tl.program_id(0)*16+tl.arange(0,16)
    e=tl.program_id(1)*64+tl.arange(0,64)
    d=tl.arange(0,64)
    total=tl.zeros((16,64),tl.float32)
    correction=tl.zeros((16,64),tl.float32)
    if BOUNDS: magnitude=tl.zeros((16,64),tl.float32)
    # Do not carry a tensor-core accumulator through the whole hidden width.
    # FP32 compensation preserves near-midpoint BF16 logits before top-k.
    for start in range(0,6144,64):
        k=start+d
        x=tl.load(X+r[:,None]*6144+k[None,:],r[:,None]<ROWS,0.)
        w=tl.load(W+e[None,:]*6144+k[:,None])
        partial=tl.dot(x,w)
        if BOUNDS:magnitude=magnitude+tl.dot(tl.abs(x),tl.abs(w))
        value=libdevice.sub_rn(partial,correction)
        updated=libdevice.add_rn(total,value)
        correction=libdevice.sub_rn(libdevice.sub_rn(updated,total),value)
        total=updated
    tl.store(OUT+r[:,None]*256+e[None,:],total,r[:,None]<ROWS)
    if BOUNDS:
        # Conservative margin for 64-term FP32 dot reductions plus compensated
        # accumulation. The positive magnitude sum's own rounding is covered by
        # the spare factor over (64+2)*unit_roundoff. Bounds are per output.
        bound=magnitude*(128.*(2.**-24)/(1.-256.*(2.**-24)))
        tl.store(BOUND+r[:,None]*256+e[None,:],bound,r[:,None]<ROWS)


@triton.jit
def _route_candidates(RAW,BOUND,BIAS,CANDIDATES):
    row=tl.program_id(0);e=tl.arange(0,256)
    value=tl.load(RAW+row*256+e);bound=tl.load(BOUND+row*256+e)
    lower=(value-bound).to(tl.bfloat16).to(tl.float32)
    upper=(value+bound).to(tl.bfloat16).to(tl.float32)
    bias=tl.load(BIAS+e)
    low_score=1./(1.+tl.exp(-lower))+bias
    high_score=1./(1.+tl.exp(-upper))+bias
    remaining=low_score
    threshold=0.
    for _ in tl.static_range(8):
        threshold=tl.max(remaining,0)
        picked=tl.min(tl.where(remaining==threshold,e,256),0)
        remaining=tl.where(e==picked,float('-inf'),remaining)
    # Monotonic sigmoid bounds plus a guard for its FP32 approximation and the
    # bias addition. Every expert that could enter the exact top8 is retained.
    margin=8.*(2.**-24)*(1.+tl.abs(bias)+tl.abs(threshold))
    tl.store(CANDIDATES+row*256+e,high_score>=threshold-margin)


@triton.jit
def _round_gate(X,W,RAW,BOUND,OUT,CANDIDATES,FILTER:tl.constexpr):
    row=tl.program_id(0);e=tl.program_id(1)*32+tl.arange(0,32)
    raw=tl.load(RAW+row*256+e);bound=tl.load(BOUND+row*256+e)
    low=(raw-bound).to(tl.bfloat16).to(tl.float32)
    high=(raw+bound).to(tl.bfloat16).to(tl.float32)
    uncertain=low!=high
    if FILTER:uncertain=uncertain&tl.load(CANDIDATES+row*256+e)
    tl.store(OUT+row*256+e,raw)
    # No host read, allocation, or shape-dependent choice: capture/verification
    # batches use the same per-row decision and reduction as a single token.
    while tl.sum(uncertain.to(tl.int32),0)>0:
        expert=tl.min(tl.where(uncertain,e,256),0)
        k=tl.arange(0,8192)
        x=tl.load(X+row*6144+k,k<6144,0.).to(tl.float64)
        w=tl.load(W+expert*6144+k,k<6144,0.).to(tl.float64)
        precise=tl.sum(x*w,0)
        value=precise.to(tl.float32)
        bits=value.to(tl.uint32,bitcast=True)
        # Avoid double rounding FP64 -> FP32 -> BF16 when the intermediate
        # FP32 value is exactly halfway between two BF16 values.
        halfway=(bits&0xffff)==0x8000
        above=precise>value.to(tl.float64);below=precise<value.to(tl.float64)
        negative=(bits&0x80000000)!=0
        increment=tl.where(negative,below,above)
        direct_bits=(bits&0xffff0000)+tl.where(increment,0x10000,0).to(tl.uint32)
        direct=direct_bits.to(tl.float32,bitcast=True)
        rounded=tl.where(halfway&(above|below),direct,value)
        tl.store(OUT+row*256+expert,rounded)
        uncertain=uncertain&(e!=expert)


def gate_projection(x,weight,out,*,scratch=None,bias=None,candidates=None):
    """BF16 projection, optionally exact only for potential router selections.

    With bias/candidates, unselected logits may keep their bounded FP32 result.
    Only top8 routing may consume that mode's output. Without them, every
    ambiguous column receives the high-precision correction.
    """
    rows=x.shape[0]
    if (x.shape!=(rows,6144) or weight.shape!=(256,6144) or out.shape!=(rows,256)
            or not 1<=rows<=3072 or out.dtype not in (torch.bfloat16,torch.float32)
            or x.dtype!=torch.bfloat16 or weight.dtype!=torch.bfloat16
            or not all(t.is_cuda and t.is_contiguous() and t.device==x.device for t in (x,weight,out))):
        raise ValueError('Invalid original full GLM router projection')
    if out.dtype==torch.bfloat16:
        if (not isinstance(scratch,tuple) or len(scratch)!=2
                or not all(t.shape==(rows,256) and t.dtype==torch.float32 and t.is_cuda
                           and t.device==x.device and t.is_contiguous() for t in scratch)
                or scratch[0].data_ptr()==scratch[1].data_ptr()):
            raise ValueError('BF16 router projection requires independent FP32 value/bound scratch')
        raw,bound=scratch
        filtered=bias is not None
        if filtered:
            if (bias.shape!=(256,) or bias.dtype!=torch.float32 or candidates is None
                    or candidates.shape!=(rows,256) or candidates.dtype!=torch.bool
                    or not all(t.is_cuda and t.device==x.device and t.is_contiguous() for t in (bias,candidates))):
                raise ValueError('Invalid selection-bound scratch or router bias')
        elif candidates is not None:raise ValueError('Candidate scratch requires router bias')
        _gate[(triton.cdiv(rows,16),4)](x,weight,raw,bound,rows,True,
                                      num_warps=4,num_stages=2,enable_fp_fusion=False)
        if filtered:_route_candidates[(rows,)](raw,bound,bias,candidates,num_warps=4,enable_fp_fusion=False)
        _round_gate[(rows,8)](x,weight,raw,bound,out,candidates if filtered else out,filtered,
                             num_warps=8,num_stages=1,enable_fp_fusion=False)
    else:
        if bias is not None or candidates is not None:raise ValueError('Routing correction requires BF16 output')
        _gate[(triton.cdiv(rows,16),4)](x,weight,out,out,rows,False,
                                      num_warps=4,num_stages=2,enable_fp_fusion=False)
    return out


@triton.jit
def _silu(GU, OUT, N, WIDTH:tl.constexpr, B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    row=i//WIDTH;col=i%WIDTH
    gate=tl.load(GU+row*(2*WIDTH)+col,i<N,0.).to(tl.float32)
    up=tl.load(GU+row*(2*WIDTH)+WIDTH+col,i<N,0.).to(tl.float32)
    # Native SiluAndMul rounds SiLU to BF16 before the multiply by up.
    silu=libdevice.div_rn(gate,1.+libdevice.exp(-gate)).to(tl.bfloat16).to(tl.float32)
    tl.store(OUT+i,silu*up,i<N)


def silu_and_mul(gate_up,out):
    rows,width=out.shape
    if (gate_up.shape!=(rows,2*width) or width not in (512,2048)
            or not 1<=rows<=3072 or not all(t.is_cuda and t.is_contiguous()
                and t.dtype==torch.bfloat16 and t.device==out.device for t in (gate_up,out))):
        raise ValueError('Invalid original BF16 SwiGLU tensors')
    _silu[(triton.cdiv(rows*width,256),)](gate_up,out,rows*width,width,256,
                                       enable_fp_fusion=False)
    return out


@triton.jit
def _top8(LOGITS,BIAS,IDS,PROBS):
    row=tl.program_id(0);e=tl.arange(0,256);slot=tl.arange(0,8)
    x=tl.load(LOGITS+row*256+e)
    score=1./(1.+tl.exp(-x))
    score=tl.where((score==score)&(tl.abs(score)!=float('inf')),score,0.)
    biased=score+tl.load(BIAS+e)
    weights=tl.full((8,),0.,tl.float32)
    indices=tl.full((8,),0,tl.int32)
    total=0.
    for k in tl.static_range(8):
        best=tl.max(biased,0)
        expert=tl.min(tl.where(biased==best,e,256),0)
        value=tl.sum(tl.where(e==expert,score,0.),0)
        weights=tl.where(slot==k,value,weights)
        indices=tl.where(slot==k,expert,indices)
        total=total+value
        biased=tl.where(e==expert,float('-inf'),biased)
    scale=1./tl.where(total>0.,total,1.)
    tl.store(IDS+row*8+slot,indices.to(tl.int64))
    tl.store(PROBS+row*8+slot,weights*scale)


def top8(logits,bias,ids,probabilities):
    """Descending biased-score order, lower expert ID wins exact ties.

    Bias must be finite. As in the native router, invalid sigmoid values become
    zero and an all-zero row keeps zero weights. No routed scale is applied.
    """
    rows=logits.shape[0]
    if (logits.shape!=(rows,256) or bias.shape!=(256,) or ids.shape!=(rows,8)
            or probabilities.shape!=(rows,8) or not 1<=rows<=3072
            or ids.dtype!=torch.int64 or any(t.dtype!=torch.float32 for t in (logits,bias,probabilities))
            or not all(t.is_cuda and t.is_contiguous() and t.device==logits.device
                       for t in (logits,bias,ids,probabilities))):
        raise ValueError('Invalid full GLM top8 router tensors')
    _top8[(rows,)](logits,bias,ids,probabilities,num_warps=4,enable_fp_fusion=False)
    return ids,probabilities


@triton.jit
def _combine(ROUTED,SHARED,OUT,N,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    r=tl.load(ROUTED+i,i<N,0.).to(tl.bfloat16).to(tl.float32)
    scaled=(r*2.5).to(tl.bfloat16).to(tl.float32)
    shared=tl.load(SHARED+i,i<N,0.).to(tl.float32)
    tl.store(OUT+i,scaled+shared,i<N)


def combine(routed,shared,out):
    if (routed.shape!=shared.shape or shared.shape!=out.shape or out.ndim!=2 or out.shape[1]!=6144
            or not 1<=out.shape[0]<=3072 or routed.dtype!=torch.float32
            or shared.dtype!=torch.bfloat16 or out.dtype!=torch.bfloat16
            or not all(t.is_cuda and t.is_contiguous() and t.device==out.device for t in (routed,shared,out))):
        raise ValueError('Invalid local routed/shared combination')
    _combine[(triton.cdiv(out.numel(),256),)](routed,shared,out,out.numel(),256,enable_fp_fusion=False)
    return out


def _read(reader,name,shape,dtype='BF16'):
    meta=reader.tensor_meta(name)
    if meta['dtype']!=dtype or tuple(meta['shape'])!=shape:
        raise ValueError('Unexpected original full GLM feed-forward weight: '+name)
    return reader.read_tensor(name)


class DenseWeights:
    def __init__(self,reader,layer,rank,device='cuda'):
        if type(layer) is not int or not 0<=layer<=78 or type(rank) is not int or not 0<=rank<6:
            raise ValueError('Expected full GLM target/draft layer and TP6 rank')
        self.layer,self.rank=layer,rank
        shared=layer>=3
        global_width=2048 if shared else 12288
        start=rank*(512 if shared else 2048)
        self.width=512 if shared and rank<4 else (0 if shared else 2048)
        self.device=torch.device(device)
        prefix=f'model.layers.{layer}.mlp.'+('shared_experts.' if shared else '')
        # Even empty shards verify stored tensor metadata before eliding work.
        weights=[]
        for name,shape in (('gate_proj.weight',(global_width,6144)),('up_proj.weight',(global_width,6144)),
                           ('down_proj.weight',(6144,global_width))):
            meta=reader.tensor_meta(prefix+name)
            if meta['dtype']!='BF16' or tuple(meta['shape'])!=shape:
                raise ValueError('Invalid original dense/shared geometry')
            if self.width:
                value=reader.read_tensor(prefix+name)
                weights.append((value[:,start:start+self.width] if name.startswith('down')
                                else value[start:start+self.width]).contiguous())
        self.gate_up=torch.cat(weights[:2],0).to(device) if self.width else None
        self.down=weights[2].to(device) if self.width else None
        if self.width:self.device=self.down.device


class DenseScratch:
    """Reusable for all layers of the same width, private to one stream/graph."""
    def __init__(self,rows,width,device):
        if type(rows) is not int or not 1<=rows<=3072 or width not in (0,512,2048):
            raise ValueError('Invalid dense/shared scratch geometry')
        self.rows,self.width=rows,width
        self.gate_up=torch.empty((rows,2*width),dtype=torch.bfloat16,device=device)
        self.activation=torch.empty((rows,width),dtype=torch.bfloat16,device=device)
        self.output=torch.empty((rows,6144),dtype=torch.bfloat16,device=device)


class Dense:
    def __init__(self,weights):self.weights=weights

    def forward(self,hidden,scratch):
        w=self.weights;rows=hidden.shape[0]
        if (hidden.shape!=(rows,6144) or not 1<=rows<=scratch.rows or scratch.width!=w.width
                or hidden.dtype!=torch.bfloat16 or not hidden.is_cuda or not hidden.is_contiguous()
                or scratch.output.device!=hidden.device):
            raise ValueError('Invalid dense/shared input or scratch')
        out=scratch.output[:rows]
        if not w.width:return out.zero_()
        gu=scratch.gate_up[:rows];act=scratch.activation[:rows]
        linear(hidden,w.gate_up,gu);silu_and_mul(gu,act)
        return linear(act,w.down,out)


class MoEWeights:
    def __init__(self,reader,layer,device='cuda'):
        if type(layer) is not int or not 3<=layer<=78:
            raise ValueError('Expected original routed layer3..78')
        require_experts()
        self.layer,self.rank=layer,reader.rank
        prefix=f'model.layers.{layer}.mlp.gate.'
        self.gate=_read(reader,prefix+'weight',(256,6144)).to(device)
        self.bias=_read(reader,prefix+'e_score_correction_bias',(256,),'F32').to(device)
        if not bool(torch.isfinite(self.bias).all()):raise ValueError('Router correction bias must be finite')
        self.shared=Dense(DenseWeights(reader,layer,reader.rank,device))
        self.routed=RoutedLayer.load(reader,layer,device)


class MoEScratch:
    def __init__(self,weights,rows,*,expert_chunk_rows=128):
        device=weights.gate.device
        self.weights,self.rows=weights,rows
        self.shared=DenseScratch(rows,weights.shared.weights.width,device)
        self.routed=weights.routed.scratch(rows,chunk_rows=expert_chunk_rows)
        self.gate_output=torch.empty((rows,256),dtype=torch.bfloat16,device=device)
        self.gate_bounds=torch.empty((rows,256),dtype=torch.float32,device=device)
        self.gate_candidates=torch.empty((rows,256),dtype=torch.bool,device=device)
        self.logits=torch.empty((rows,256),dtype=torch.float32,device=device)
        self.ids=torch.empty((rows,8),dtype=torch.int64,device=device)
        self.probabilities=torch.empty((rows,8),dtype=torch.float32,device=device)
        self.output=torch.empty((rows,6144),dtype=torch.bfloat16,device=device)


class MoE:
    def __init__(self,weights):self.weights=weights

    def forward(self,hidden,scratch):
        w=self.weights;rows=hidden.shape[0]
        if scratch.weights is not w or not 1<=rows<=scratch.rows:
            raise ValueError('MoE scratch belongs to another layer or is too small')
        logits=scratch.logits[:rows];ids=scratch.ids[:rows];probs=scratch.probabilities[:rows]
        # Native GateLinear on SM121 takes the BF16 ReplicatedLinear fallback,
        # then casts to its requested FP32 output dtype. Keep that rounding.
        gate_projection(hidden,w.gate,scratch.gate_output[:rows],scratch=(logits,scratch.gate_bounds[:rows]),
                        bias=w.bias,candidates=scratch.gate_candidates[:rows])
        logits.copy_(scratch.gate_output[:rows])
        top8(logits,w.bias,ids,probs)
        shared=w.shared.forward(hidden,scratch.shared)
        routed=w.routed.routed(hidden,ids,probs,scratch.routed,act_mode=ACT_F32)
        return combine(routed,shared,scratch.output[:rows])
