"""Experimental exact TP6 draft argmax with aligned value/index exchange.

The common LM-head GEMM is outside this helper. Original, contiguous vocabulary
only; no added tokens, logit scaling, or soft capping. Not enabled in serving.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _local_argmax(X, Pair, WIDTH: tl.constexpr, VALID: tl.constexpr,
                  OFFSET: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    idx = tl.arange(0, BLOCK)
    valid = idx < VALID
    values = tl.load(X + row * WIDTH + idx, mask=valid, other=-float('inf')).to(tl.float32)
    nan = (values != values) & valid
    has_nan = tl.sum(nan.to(tl.int32), 0) > 0
    maximum = tl.max(tl.where(nan, -float('inf'), values), 0)
    winner = tl.min(tl.where(valid & tl.where(has_nan, nan, values == maximum),
                             idx, 2147483647), 0)
    value = tl.where(has_nan, float('nan'), maximum)
    tl.store(Pair + row * 4, value)
    tl.store(Pair + row * 4 + 1, (winner + OFFSET).to(tl.float32))
    tl.store(Pair + row * 4 + 2, 0.)
    tl.store(Pair + row * 4 + 3, 0.)


@triton.jit
def _global_argmax(Pairs, Out):
    row = tl.program_id(0)
    ranks = tl.arange(0, 8)
    values = tl.load(Pairs + row * 24 + ranks * 4, ranks < 6, other=-float('inf'))
    tokens = tl.load(Pairs + row * 24 + ranks * 4 + 1, ranks < 6,
                     other=2147483647.).to(tl.int32)
    nan = (values != values) & (ranks < 6)
    has_nan = tl.sum(nan.to(tl.int32), 0) > 0
    maximum = tl.max(tl.where(nan, -float('inf'), values), 0)
    winner = tl.min(tl.where((ranks < 6) & tl.where(has_nan, nan, values == maximum),
                             tokens, 2147483647), 0)
    tl.store(Out + row, winner.to(tl.int64))


def select(logits, shard_indices, gather):
    """Return exactly the full-vocabulary argmax, including first-index ties."""
    if (logits.ndim != 2 or not logits.is_cuda or not logits.is_contiguous()
            or logits.dtype not in (torch.bfloat16, torch.float32)
            or not 1 <= logits.shape[0] <= 4 or logits.shape[1] != 25824
            or shard_indices.num_added_elements_padded != 0
            or shard_indices.num_org_elements_padded != 25824
            or shard_indices.org_vocab_start_index not in range(0, 154944, 25824)
            or shard_indices.num_org_elements != min(
                25824, 154880 - shard_indices.org_vocab_start_index)):
        raise ValueError('Unsupported TP6 GLM vocabulary geometry')
    rows = logits.shape[0]
    pair = torch.empty((rows, 4), device=logits.device, dtype=torch.float32)
    out = torch.empty((rows,), device=logits.device, dtype=torch.int64)
    _local_argmax[(rows,)](logits, pair, 25824, shard_indices.num_org_elements,
                           shard_indices.org_vocab_start_index, 32768, num_warps=16)
    gathered = gather(pair, dim=-1)
    _global_argmax[(rows,)](gathered, out, num_warps=4)
    return out
