"""Original BF16 projections with fixed tensor-core reduction order per row.

K traversal is fixed; immutable model plans may choose output tiles by row count. Caller-owned outputs permit
CUDA graph replay and allow one bounded workspace to be reused across layers.
No weight conversion or quantization is performed.
"""
import torch
import triton
import triton.language as tl

from .projection_plan import LinearTile,REFERENCE_TILE,current_projection_plan


@triton.jit(do_not_specialize=['M'])
def _linear(X, W, OUT, M, K: tl.constexpr, N: tl.constexpr):
    r = tl.program_id(0)*16+tl.arange(0,16)
    n = tl.program_id(1)*64+tl.arange(0,64)
    d = tl.arange(0,64)
    acc = tl.zeros((16,64),tl.float32)
    for start in range(0,K,64):
        k = start+d
        x = tl.load(X+r[:,None]*K+k[None,:],(r[:,None]<M)&(k[None,:]<K),0.)
        w = tl.load(W+n[None,:]*K+k[:,None],(n[None,:]<N)&(k[:,None]<K),0.)
        acc = tl.dot(x,w,acc)
    tl.store(OUT+r[:,None]*N+n[None,:],acc,(r[:,None]<M)&(n[None,:]<N))


@triton.jit(do_not_specialize=['M'])
def _linear_tiled(X,W,OUT,M,K:tl.constexpr,N:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr):
    r=tl.program_id(0)*BM+tl.arange(0,BM)
    n=tl.program_id(1)*BN+tl.arange(0,BN)
    d=tl.arange(0,64)
    acc=tl.zeros((BM,BN),tl.float32)
    for start in range(0,K,64):
        k=start+d
        x=tl.load(X+r[:,None]*K+k[None,:],(r[:,None]<M)&(k[None,:]<K),0.)
        w=tl.load(W+n[None,:]*K+k[:,None],(n[None,:]<N)&(k[:,None]<K),0.)
        acc=tl.dot(x,w,acc)
    tl.store(OUT+r[:,None]*N+n[None,:],acc,(r[:,None]<M)&(n[None,:]<N))


def linear(x, weight, out, *, tile=None):
    """BF16 ``x @ weight.T`` into preallocated BF16/FP32 output."""
    if x.ndim != 2 or weight.ndim != 2:
        raise ValueError('Dense projections require matrices')
    m,k=x.shape; n,wk=weight.shape
    if (not 1<=m<=3072 or k!=wk or k<=0 or n<=0 or out.shape!=(m,n)
            or x.dtype!=torch.bfloat16 or weight.dtype!=torch.bfloat16
            or out.dtype not in (torch.bfloat16,torch.float32)
            or not all(t.is_cuda and t.is_contiguous() and t.device==x.device for t in (x,weight,out))):
        raise ValueError('Invalid original BF16 projection tensors')
    tile=current_projection_plan().tile(m) if tile is None else tile
    if type(tile) is not LinearTile:
        raise ValueError('Expected an admitted immutable projection tile')
    if tile==REFERENCE_TILE:
        # Keep the previously qualified kernel intact as an independent oracle.
        _linear[(triton.cdiv(m,16),triton.cdiv(n,64))](x,weight,out,m,k,n,num_warps=4,num_stages=2,
                                                   enable_fp_fusion=False)
    else:
        _linear_tiled[(triton.cdiv(m,tile.m),triton.cdiv(n,tile.n))](x,weight,out,m,k,n,tile.m,tile.n,
                         num_warps=4,num_stages=tile.stages,enable_fp_fusion=False)
    return out
