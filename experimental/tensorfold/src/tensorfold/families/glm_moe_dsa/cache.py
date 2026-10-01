"""Compact full-GLM MLA and full-indexer caches, with explicit layer ownership.

Each MLA token uses512 FP8 bytes+4 FP32 scales+64 BF16 RoPE values (656B).
An owning indexer adds128 FP8 bytes+1 FP32 scale. Allocation/admission, request
extent validation, and speculative commit/reclamation remain engine duties.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


class LayerCache:
    def __init__(self,layer,capacity,device,*,indexer):
        if type(layer) is not int or not 0<=layer<=78 or type(capacity) is not int or capacity<1:
            raise ValueError('Invalid full GLM cache owner/capacity')
        self.layer,self.capacity=layer,capacity
        self.latent=torch.empty((capacity,512),dtype=torch.float8_e4m3fn,device=device)
        self.scales=torch.empty((capacity,4),dtype=torch.float32,device=device)
        self.rope=torch.empty((capacity,64),dtype=torch.bfloat16,device=device)
        self.index_keys=torch.empty((capacity,128),dtype=torch.float8_e4m3fn,device=device) if indexer else None
        self.index_scales=torch.empty(capacity,dtype=torch.float32,device=device) if indexer else None

    def nbytes(self):
        return sum(t.numel()*t.element_size() for t in
                   (self.latent,self.scales,self.rope,self.index_keys,self.index_scales) if t is not None)


@triton.jit
def _write(L,R,SLOTS,LC,LS,RC,CAP:tl.constexpr):
    row,block=tl.program_id(0),tl.program_id(1)
    slot=tl.load(SLOTS+row).to(tl.int64)
    if slot>=0 and slot<CAP:
        d=tl.arange(0,128)
        x=tl.load(L+row*512+block*128+d).to(tl.float32)
        # Match the deployed native MLA cache: ordinary FP32 scales and FLT_MIN.
        # The separate indexer cache uses UE8M0; the two formats must not be conflated.
        scale=tl.maximum(libdevice.div_rn(tl.max(tl.abs(x),0),448.),1.1754943508222875e-38)
        tl.store(LC+slot*512+block*128+d,tl.minimum(tl.maximum(libdevice.div_rn(x,scale),-448.),448.))
        tl.store(LS+slot*4+block,scale)
        if block==0:
            dr=tl.arange(0,64)
            tl.store(RC+slot*64+dr,tl.load(R+row*64+dr))


def write_mla(latent,rotated_rope,slots,cache):
    rows=latent.shape[0]
    if (latent.shape!=(rows,512) or rotated_rope.shape!=(rows,64) or slots.shape!=(rows,)
            or slots.dtype!=torch.int64 or latent.dtype!=torch.bfloat16 or rotated_rope.dtype!=torch.bfloat16
            or not all(t.is_cuda and t.is_contiguous() and t.device==latent.device for t in
                       (latent,rotated_rope,slots,cache.latent,cache.scales,cache.rope))):
        raise ValueError('Invalid compact MLA cache write tensors')
    _write[(rows,4)](latent,rotated_rope,slots,cache.latent,cache.scales,cache.rope,cache.capacity,num_warps=1)
