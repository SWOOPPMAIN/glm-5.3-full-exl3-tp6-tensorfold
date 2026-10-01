"""Independent CPU nearest-value BF16 rounding oracle for finite FP64 values.

Some tensor conversions round FP64 through FP32. This oracle enumerates the
finite BF16 value grid, finds the adjacent values, and compares FP64 distances.
It therefore does not depend on the GPU repair's midpoint/bit-tail algorithm.
"""
from functools import lru_cache


@lru_cache(maxsize=1)
def _grid():
    import numpy as np
    codes=np.arange(65536,dtype=np.uint32)
    values=(codes<<16).view(np.float32)
    finite=np.isfinite(values);codes,values=codes[finite],values[finite].astype(np.float64)
    order=np.argsort(values,kind='stable')
    return codes[order].astype(np.uint16),values[order]


def bf16_nearest(value):
    import numpy as np
    import torch
    raw=value.detach().double().cpu().numpy();codes,grid=_grid()
    if not np.isfinite(raw).all() or np.any(raw<grid[0]) or np.any(raw>grid[-1]):
        raise ValueError('Reference expects finite values inside BF16 range')
    upper=np.searchsorted(grid,raw,side='left').clip(1,len(grid)-1);lower=upper-1
    dl=raw-grid[lower];du=grid[upper]-raw
    pick_lower=(dl<du)|((dl==du)&((codes[lower]&1)==0))
    result=np.where(pick_lower,codes[lower],codes[upper]).astype(np.uint16)
    return torch.from_numpy(result.copy()).view(torch.bfloat16).to(value.device)
