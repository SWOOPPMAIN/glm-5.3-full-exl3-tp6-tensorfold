"""Batch-independent RMS arithmetic and the full-GLM BF16 residual contract."""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _rms(X,W,Y,R,STRIDE:tl.constexpr,D:tl.constexpr):
    r=tl.program_id(0)*2+tl.arange(0,2)[:,None]
    if D==2048:
        col=tl.arange(0,1024)[None,:]
        partial=tl.full((2,1024),0.,tl.float32)
        for offset in tl.range(0,D,1024):
            x=tl.load(X+r*STRIDE+offset+col,r<R,0.).to(tl.float32)
            partial=tl.where(r<R,partial+x*x,partial)
        total=tl.sum(partial,1)[:,None]
        for offset in tl.range(0,D,1024):
            x=tl.load(X+r*STRIDE+offset+col,r<R,0.).to(tl.float32)
            w=tl.load(W+offset+col).to(tl.float32)
            tl.store(Y+r*D+offset+col,x*libdevice.rsqrt(total/D+1e-5)*w,r<R)
    else:
        col=tl.arange(0,D)[None,:]
        x=tl.load(X+r*STRIDE+col,r<R,0.).to(tl.float32)
        w=tl.load(W+col).to(tl.float32)
        total=tl.sum(tl.where(r<R,x*x,0.),1)[:,None]
        tl.store(Y+r*D+col,x*libdevice.rsqrt(total/D+1e-5)*w,r<R)


def rms(x,weight,out):
    rows,dim=x.shape
    if (dim not in (512,2048) or weight.shape!=(dim,) or out.shape!=x.shape or x.stride(1)!=1
            or not 1<=rows<=3072 or not weight.is_contiguous() or not out.is_contiguous()
            or not all(t.is_cuda and t.device==x.device and t.dtype==torch.bfloat16 for t in (x,weight,out))):
        raise ValueError('Invalid full GLM LoRA RMSNorm tensors')
    _rms[(triton.cdiv(rows,2),)](x,weight,out,rows,x.stride(0),dim,num_warps=8,num_stages=1)
    return out


@triton.jit
def _hidden_rms(X,W,Y,RES,SUM,XS:tl.constexpr,RS:tl.constexpr,ADD:tl.constexpr):
    row=tl.program_id(0);col=tl.arange(0,8192)
    x=tl.load(X+row*XS+col,col<6144,0.).to(tl.float32)
    if ADD:
        residual=tl.load(RES+row*RS+col,col<6144,0.).to(tl.float32)
        # The skip connection is stored in BF16 before its variance is taken.
        x=(x+residual).to(tl.bfloat16).to(tl.float32)
    weight=tl.load(W+col,col<6144,0.).to(tl.float32)
    variance=tl.sum(x*x,0)/6144.
    value=x*libdevice.rsqrt(variance+1e-5)*weight
    tl.store(SUM+row*6144+col,x,col<6144)
    tl.store(Y+row*6144+col,value,col<6144)


def _extent(tensor):
    """Conservative byte interval for a positive-stride tensor view."""
    start=tensor.data_ptr()
    span=1+sum((size-1)*stride for size,stride in zip(tensor.shape,tensor.stride()))
    return start,start+span*tensor.element_size()


def _overlap(a,b):
    a0,a1=_extent(a);b0,b1=_extent(b)
    return a0<b1 and b0<a1


def hidden_rms(x,weight,out,residual_out,*,residual=None):
    """Return normalized rows and their BF16 skip connection, without allocation.

    The first decoder layer copies ``x`` to ``residual_out``. Later layers add
    their incoming branch to ``residual`` before normalization. The same fixed
    reduction tree handles 1..3072 rows, including graph replay. Writable buffers
    must be separate from inputs and each other; the engine owns their lifetime.
    """
    if x.ndim!=2:
        raise ValueError('Expected full GLM hidden rows')
    rows,dim=x.shape
    inputs=(x,weight) if residual is None else (x,weight,residual)
    tensors=(*inputs,out,residual_out)
    if (dim!=6144 or not 1<=rows<=3072 or weight.shape!=(6144,)
            or out.shape!=x.shape or residual_out.shape!=x.shape
            or x.stride(1)!=1 or x.stride(0)<6144
            or not all(t.is_contiguous() for t in (weight,out,residual_out))
            or not all(t.is_cuda and t.device==x.device and t.dtype==torch.bfloat16 for t in tensors)
            or (residual is not None and (residual.shape!=x.shape or residual.stride(1)!=1
                                          or residual.stride(0)<6144))):
        raise ValueError('Invalid full GLM hidden RMS/residual tensors')
    if _overlap(out,residual_out) or any(_overlap(dst,src) for dst in (out,residual_out) for src in inputs):
        raise ValueError('Hidden RMS requires disjoint writable buffers')
    _hidden_rms[(rows,)](x,weight,out,residual if residual is not None else x,residual_out,
                         x.stride(0),residual.stride(0) if residual is not None else x.stride(0),
                         residual is not None,num_warps=8,num_stages=1,enable_fp_fusion=False)
    return out,residual_out
