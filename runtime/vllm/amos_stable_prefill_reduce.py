"""Keep TP6 hidden-state sum arithmetic fixed while compute batches grow.

NCCL's BF16 reduction ordering depends on message size. Fixed row tiles retain
the qualified summation partition independently of the larger compute batch.
Use the existing communicator and stream; never initialize another group.
"""
import torch

ROWS = 1536
HIDDEN = 6144


def apply(comm, x, *, in_place=False, tail_reduce=None):
    if (comm is None or comm.disabled or comm.world_size != 6
            or x.ndim != 2 or x.shape[1] != HIDDEN or x.shape[0] <= ROWS
            or x.dtype != torch.bfloat16):
        return None
    if not x.is_cuda or not x.is_contiguous():
        raise ValueError('Stable TP6 reduction requires contiguous CUDA hidden-state rows')
    out = x if in_place else torch.empty_like(x)
    for start in range(0, x.shape[0], ROWS):
        stop = min(start+ROWS, x.shape[0])
        if not in_place and stop-start < ROWS and tail_reduce is not None:
            # A short final tile may qualify for the existing RoCE backend.
            # Re-enter ordinary dispatch at a size that cannot recurse here.
            out[start:stop].copy_(tail_reduce(x[start:stop]))
            continue
        result = comm.all_reduce(x[start:stop], out_tensor=out[start:stop])
        if result is None:
            raise RuntimeError('The existing TP6 NCCL communicator did not perform the reduction')
    return out
