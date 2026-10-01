"""Full GLM decoder assembly with two explicit six-rank sums.

The local MLA and MLP adapters never perform collectives themselves. This layer
owns their reduction order. Distributed qualification is required before this
module can be registered as a serving backend.
"""
import torch
import torch.distributed as dist
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from .mla import MLA, MLAWeights, MLAScratch
from .mlp import Dense, DenseWeights, DenseScratch, MoE, MoEWeights, MoEScratch
from .experts import EXPERT_CHUNK_ROWS,validate_chunk_rows
from .norms import hidden_rms, _overlap
from .reduction_plan import reduction_plan


@triton.jit
def _sum_six(GATHER,OUT,N):
    col=tl.program_id(0)*256+tl.arange(0,256)
    value=tl.load(GATHER+col,col<N,0.).to(tl.float32)
    for rank in tl.static_range(1,6):
        part=tl.load(GATHER+rank*N+col,col<N,0.).to(tl.float32)
        value=libdevice.add_rn(value,part)
    tl.store(OUT+col,value,col<N)


@triton.jit
def _pad_rows(INPUT,OUTPUT,N,PADDED):
    col=tl.program_id(0)*1024+tl.arange(0,1024)
    value=tl.load(INPUT+col,col<N,0.)
    tl.store(OUTPUT+col,value,col<PADDED)


@triton.jit
def _join_columns(GATHER,OUT,ROWS:tl.constexpr,WIDTH:tl.constexpr,COLUMNS:tl.constexpr):
    row=tl.program_id(0);col=tl.program_id(1)*256+tl.arange(0,256)
    source=col//WIDTH;local=col%WIDTH
    value=tl.load(GATHER+(source*ROWS+row)*WIDTH+local,col<COLUMNS,0.)
    tl.store(OUT+row*COLUMNS+col,value,col<COLUMNS)


class TP6Reduction:
    """Use an explicitly supplied, already initialized six-member NCCL group.

    Construction does not create a communicator. The engine must create the
    group on an unloaded fleet and use identical layer/row ordering on all ranks.
    """
    world_size=6

    def __init__(self,group,rank):
        if (group is None or not dist.is_initialized() or type(rank) is not int
                or not 0<=rank<6 or dist.get_world_size(group)!=6
                or dist.get_rank(group)!=rank or dist.get_backend(group)!='nccl'):
            raise ValueError('An explicit six-member NCCL group and matching rank are required')
        self.group,self.rank=group,rank
        self.gather=None
        self.send=self.shard=None
        self.bulk_min_rows=None

    def prepare(self,rows,device,*,bulk_min_rows=None):
        """Allocate once before capture; the same workspace serves both sums."""
        plan=reduction_plan(rows,bulk_min_rows=bulk_min_rows)
        if torch.device(device).type!='cuda':
            raise ValueError('Invalid TP6 collective workspace')
        self.gather=torch.empty((plan['gather_rows'],6144),dtype=torch.bfloat16,device=device)
        self.bulk_min_rows=bulk_min_rows
        self.send=self.shard=None
        if plan['shard_rows']:
            self.send=torch.empty((plan['padded_rows'],6144),dtype=torch.bfloat16,device=device)
            self.shard=torch.empty((plan['shard_rows'],6144),dtype=torch.bfloat16,device=device)
        return self

    def sum_into(self,local,out):
        if (local.ndim!=2 or not 1<=local.shape[0]<=3072 or local.shape[1]!=6144
                or out.shape!=local.shape
                or not all(t.is_cuda and t.is_contiguous() and t.dtype==torch.bfloat16
                           and t.device==local.device for t in (local,out))
                or _overlap(local,out) or self.gather is None
                or self.gather.device!=local.device or self.gather.shape[0]<6*local.shape[0]
                or any(_overlap(t,arena) for t in (local,out)
                       for arena in (self.gather,self.send,self.shard) if arena is not None)):
            raise ValueError('TP6 sum requires separate contiguous BF16 hidden buffers')
        # A BF16 all-reduce may change its intermediate rounding with NCCL's
        # topology/size-dependent algorithm. Gather identical BF16 contributions
        # and add rank0..5 in FP32, then round once. Every row uses the same order.
        rows=local.shape[0]
        if self.bulk_min_rows is not None and rows>=self.bulk_min_rows:
            # Rank d receives the d-th contiguous row shard from sources0..5.
            # Each element still uses the identical _sum_six FP32 rank order.
            # Only the BF16 rounded shard is subsequently replicated. Never
            # substitute NCCL reduce_scatter: its arithmetic order can differ.
            shard_rows=triton.cdiv(rows,6);padded_rows=6*shard_rows
            receive=self.gather[:padded_rows]
            if rows==padded_rows:send=local
            else:
                send=self.send[:padded_rows]
                _pad_rows[(triton.cdiv(send.numel(),1024),)](local,send,local.numel(),send.numel())
            dist.all_to_all_single(receive,send,group=self.group)
            shard=self.shard[:shard_rows]
            _sum_six[(triton.cdiv(shard.numel(),256),)](receive,shard,shard.numel(),enable_fp_fusion=False)
            # receive is dead after the sum, so reuse its storage for all-gather.
            dist.all_gather_into_tensor(receive,shard,group=self.group)
            out.copy_(receive[:rows])
        else:
            gathered=self.gather[:6*rows]
            dist.all_gather_into_tensor(gathered,local,group=self.group)
            _sum_six[(triton.cdiv(local.numel(),256),)](gathered,out,local.numel(),enable_fp_fusion=False)
        return out

    def gather_columns_into(self,local,out,*,gather=None):
        """Join rank-local columns; optionally trim the last rank's padding.

        EH projection uses the existing BF16 reduction arena. Vocabulary logits
        supply a separate bounded FP32 arena. All ranks receive identical rows.
        """
        arena=self.gather if gather is None else gather
        if (local.ndim!=2 or out.ndim!=2 or not 1<=local.shape[0]<=3072
                or local.shape[1]<1 or out.shape[0]!=local.shape[0]
                or not 5*local.shape[1]<out.shape[1]<=6*local.shape[1]
                or arena is None or arena.numel()<6*local.numel()
                or local.dtype not in (torch.bfloat16,torch.float32)
                or not all(t.is_cuda and t.is_contiguous() and t.dtype==local.dtype
                           and t.device==local.device for t in (local,out,arena))
                or _overlap(local,out) or _overlap(local,arena) or _overlap(out,arena)):
            raise ValueError('TP6 column gather requires distinct matching contiguous buffers')
        rows,width=local.shape
        packed=arena.view(-1)[:6*local.numel()].view(6*rows,width)
        dist.all_gather_into_tensor(packed,local,group=self.group)
        _join_columns[(rows,triton.cdiv(out.shape[1],256))](packed,out,rows,width,out.shape[1])
        return out


class DecoderWeights:
    def __init__(self,reader,layer,device='cuda'):
        if type(layer) is not int or not 0<=layer<=78:
            raise ValueError('Expected target layer0..77 or MTP layer78')
        self.layer,self.rank=layer,reader.rank
        norms=[]
        for name in ('input_layernorm','post_attention_layernorm'):
            key=f'model.layers.{layer}.{name}.weight'
            meta=reader.tensor_meta(key)
            if meta['dtype']!='BF16' or tuple(meta['shape'])!=(6144,):
                raise ValueError('Expected original BF16 hidden normalization weights')
            norms.append(reader.read_tensor(key).to(device))
        self.input_norm,self.post_norm=norms
        self.attention=MLA(MLAWeights(reader,layer,reader.rank,device))
        self.ffn=(Dense(DenseWeights(reader,layer,reader.rank,device)) if layer<3
                  else MoE(MoEWeights(reader,layer,device)))


class DecoderScratch:
    """One layer's execution buffers; not yet a whole-model memory planner."""
    def __init__(self,weights,rows,context_capacity,*,expert_chunk_rows=EXPERT_CHUNK_ROWS):
        expert_chunk_rows=validate_chunk_rows(expert_chunk_rows)
        device=weights.input_norm.device
        self.weights,self.rows=weights,rows
        self.attention=MLAScratch(rows,context_capacity,device,
                                  real_heads=weights.attention.weights.real_heads)
        # Attention and both TP sums retain the full model batch. Once the
        # attention contribution has been reduced, its output buffer is dead
        # and can hold the local FFN result. Bound the other FFN temporaries to
        # the existing expert chunk size rather than every prompt row.
        self.ffn_rows=min(rows,expert_chunk_rows)
        self.ffn=(DenseScratch(self.ffn_rows,weights.ffn.weights.width,device) if weights.layer<3
                  else MoEScratch(weights.ffn.weights,self.ffn_rows,expert_chunk_rows=expert_chunk_rows))
        self.normalized=torch.empty((rows,6144),dtype=torch.bfloat16,device=device)
        self.input_residual=torch.empty_like(self.normalized)
        self.post_residual=torch.empty_like(self.normalized)
        self.reduced=torch.empty_like(self.normalized)


class DecoderWorkspace(DecoderScratch):
    """One sequential model pass's buffers, reused by target and MTP layers.

    Bind before every layer. Binding only changes Python owners, never allocates
    or changes a tensor address, so complete-layer CUDA graphs can share it.
    Concurrent passes/streams must have separate workspaces or be serialized.
    Returned layer outputs are borrowed until the next forward on this workspace.
    """
    def __init__(self,dense_weights,moe_weights,rows,context_capacity,*,expert_chunk_rows=EXPERT_CHUNK_ROWS):
        if (not 0<=dense_weights.layer<3 or not 3<=moe_weights.layer<=78
                or dense_weights.rank!=moe_weights.rank
                or dense_weights.input_norm.device!=moe_weights.input_norm.device
                or dense_weights.attention.weights.real_heads!=moe_weights.attention.weights.real_heads):
            raise ValueError('Model workspace needs matching original dense and MoE rank weights')
        super().__init__(dense_weights,rows,context_capacity,expert_chunk_rows=expert_chunk_rows)
        self.rank=dense_weights.rank
        self.dense=self.ffn
        self.moe=MoEScratch(moe_weights.ffn.weights,self.ffn_rows,expert_chunk_rows=expert_chunk_rows)
        self.bind(dense_weights)

    def bind(self,weights):
        if (weights.rank!=self.rank or weights.input_norm.device!=self.normalized.device
                or weights.attention.weights.real_heads!=self.attention.real_heads
                or not 0<=weights.layer<=78):
            raise ValueError('Decoder workspace belongs to another rank/device/head geometry')
        w=weights.ffn.weights
        if weights.layer<3:
            if w.width!=self.dense.width:raise ValueError('Dense scratch width changed')
            ffn=self.dense
        else:
            ex=w.routed.weights;arena=self.moe.routed
            if (w.shared.weights.width!=self.moe.shared.width
                    or (ex.dims,ex.width,ex.count)!=(6144,512,arena.kernel.count_experts)
                    or w.gate.device!=self.normalized.device
                    or w.routed.global_to_local.device!=self.normalized.device):
                raise ValueError('Original routed/shared scratch geometry changed')
            self.moe.weights=w
            arena.layer=w.routed
            ffn=self.moe
        self.weights,self.ffn=weights,ffn
        return self

    def nbytes(self):
        """Actual unique buffer storage, excluding all weights and caches."""
        seen=set();total=0
        def count(obj):
            nonlocal total
            if isinstance(obj,torch.Tensor):
                storage=obj.untyped_storage();key=(obj.device,storage.data_ptr())
                if key not in seen:seen.add(key);total+=storage.nbytes()
            elif hasattr(obj,'__dict__'):
                for key,value in vars(obj).items():
                    if key not in ('weights','layer'):count(value)
        for obj in (self.attention,self.dense,self.moe,self.normalized,
                    self.input_residual,self.post_residual,self.reduced):count(obj)
        return total


class Decoder:
    def __init__(self,weights,reduction):
        if not isinstance(reduction,TP6Reduction) or reduction.rank!=weights.rank:
            raise ValueError('Decoder requires its real TP6 collective owner')
        self.weights,self.reduction=weights,reduction

    def forward(self,hidden,residual,positions,bases,slots,cache,table,scratch,selection,*,scope,visible_tokens=None):
        w=self.weights;rows=hidden.shape[0]
        if scratch.weights is not w or not 1<=rows<=scratch.rows:
            raise ValueError('Decoder scratch belongs to another layer or row capacity')
        normalized=scratch.normalized[:rows]
        before=scratch.input_residual[:rows];after=scratch.post_residual[:rows]
        reduced=scratch.reduced[:rows]
        hidden_rms(hidden,w.input_norm,normalized,before,residual=residual)
        local=w.attention.forward(normalized,positions,bases,slots,cache,table,
                                   scratch.attention,selection,scope=scope,visible_tokens=visible_tokens)
        self.reduction.sum_into(local,reduced)
        hidden_rms(reduced,w.post_norm,normalized,after,residual=before)
        local=scratch.attention.output[:rows]
        for start in range(0,rows,scratch.ffn_rows):
            stop=min(start+scratch.ffn_rows,rows)
            local[start:stop].copy_(w.ffn.forward(normalized[start:stop],scratch.ffn))
        self.reduction.sum_into(local,reduced)
        # MLP output joins the skip connection at the NEXT layer/final norm.
        return reduced,after
