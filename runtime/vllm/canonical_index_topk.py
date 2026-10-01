"""Experimental deterministic finalization of B12X paged tiled top-k.

Keep the native scorer and exact cutoff. Retain scores above the cutoff,
break equal-score ties by logical token ID, then emit in logical token order.
Map to physical cache slots only after selection. This module is not installed
in serving: its restricted contract matches the paged carry-chain path only.
"""
import os
import torch
import triton
import triton.language as tl

_VERIFY_WITH_ORACLE = os.environ.get('AMOS_TP6_INDEXER_ORACLE', '0')
if _VERIFY_WITH_ORACLE not in ('0', '1'):
    raise ValueError('AMOS_TP6_INDEXER_ORACLE must be 0 or 1')


@triton.jit
def _pack(ids, values):
    return (ids.to(tl.uint64) << 32) | values.to(tl.uint32, bitcast=True).to(tl.uint64)


@triton.jit
def _refine(Logits, Lengths, Values, Indices, CarryValues, CarryIndices, Packed,
            K: tl.constexpr, BQ: tl.constexpr, BK: tl.constexpr,
            NK: tl.constexpr, OFFSET: tl.constexpr, EXTENT: tl.constexpr,
            FIRST: tl.constexpr, SCAN: tl.constexpr):
    row = tl.program_id(0)
    x = tl.arange(0, K)
    ids = tl.load(Indices + row * K + x)
    values = tl.load(Values + row * K + x)
    valid = ids >= 0
    cutoff = tl.min(tl.where(valid, values, float('inf')), 0)
    n_valid = tl.sum(valid.to(tl.int32), 0)
    strict = valid & (values > cutoff)
    n_strict = tl.sum(strict.to(tl.int32), 0)
    sentinel: tl.constexpr = 0xffffffffffffffff
    # Initialize once, then compact strict winners into disjoint slots. The
    # final emit kernel sorts the completed list, so no first sort is needed.
    tl.store(Packed + row * K + x, tl.full((K,), sentinel, tl.uint64))
    tl.debug_barrier()
    strict_dest = tl.cumsum(strict.to(tl.int32), 0) - 1
    tl.store(Packed + row * K + strict_dest, _pack(ids, values), strict)
    written = n_strict
    if not FIRST:
        ci = tl.load(CarryIndices + row * K + x)
        cv = tl.load(CarryValues + row * K + x)
        # Carry is emitted by the preceding canonical chunk, hence is already
        # ordered by logical ID; all of its IDs precede this chunk's IDs.
        cp = _pack(ci, cv)
        carry_take = (ci >= 0) & (cv == cutoff)
        carry_dest = written + tl.cumsum(carry_take.to(tl.int32), 0) - 1
        tl.store(Packed + row * K + carry_dest, cp, carry_take & (carry_dest < n_valid))
        written += tl.sum(carry_take.to(tl.int32), 0)
    length = tl.minimum(tl.maximum(tl.load(Lengths + row) - OFFSET, 0), EXTENT)
    row_base = (row // BQ) * NK * BQ * BK + (row % BQ) * BK
    s = tl.arange(0, SCAN)
    start = 0
    while (start < length) & (written < n_valid):
        local_id = start + s
        addresses = row_base + (local_id // BK) * BQ * BK + local_id % BK
        value = tl.load(Logits + addresses, local_id < length, other=-float('inf'))
        take = (local_id < length) & (value == cutoff)
        n_tie = tl.sum(take.to(tl.int32), 0)
        if n_tie > 0:
            dest = written + tl.cumsum(take.to(tl.int32), 0) - 1
            tl.store(Packed + row * K + dest, _pack(local_id + OFFSET, value),
                     take & (dest < n_valid))
            written += n_tie
        start += SCAN


@triton.jit
def _emit(Packed, Values, Indices, Pages,
          K: tl.constexpr, PHYSICAL: tl.constexpr,
          PAGE_STRIDE: tl.constexpr, PAGE_SIZE: tl.constexpr):
    row = tl.program_id(0)
    x = tl.arange(0, K)
    packed = tl.sort(tl.load(Packed + row * K + x), descending=False)
    ids = (packed >> 32).to(tl.int32)
    values = (packed & 0xffffffff).to(tl.uint32).to(tl.float32, bitcast=True)
    valid = ids >= 0
    values = tl.where(valid, values, -float('inf'))
    if PHYSICAL:
        page = tl.load(Pages + row * PAGE_STRIDE + ids // PAGE_SIZE, valid, other=-1)
        ids = tl.where(valid & (page >= 0), page * PAGE_SIZE + ids % PAGE_SIZE, -1)
    tl.store(Values + row * K + x, values)
    tl.store(Indices + row * K + x, ids)


def canonicalize(*, tile_logits, lengths, values, indices, block_q, block_k,
                 num_k_tiles, input_index_offset, input_extent,
                 carry_values=None, carry_indices=None, is_first=True,
                 output_page_table=None, output_page_size=64):
    """Finalize zero-start supertiles; carries must be prior canonical outputs."""
    rows, k = indices.shape
    assert k > 0 and k & (k - 1) == 0
    assert values.shape == indices.shape and values.dtype == torch.float32
    assert indices.dtype == torch.int32 and values.is_contiguous() and indices.is_contiguous()
    assert lengths.shape == (rows,) and lengths.dtype == torch.int32 and lengths.is_contiguous()
    assert input_extent > 0 and input_extent <= num_k_tiles * block_k
    assert tile_logits.dtype == torch.float32 and tile_logits.is_contiguous()
    assert tile_logits.numel() >= triton.cdiv(rows, block_q) * num_k_tiles * block_q * block_k
    assert is_first or (carry_values is not None and carry_indices is not None and input_index_offset > 0)
    if output_page_table is not None:
        assert output_page_table.shape[0] == rows and output_page_table.dtype == torch.int32
        assert output_page_table.stride(1) == 1 and output_page_size > 0
    packed = torch.empty((rows, k), dtype=torch.uint64, device=indices.device)
    _refine[(rows,)](tile_logits, lengths, values, indices,
                    values if is_first else carry_values, indices if is_first else carry_indices,
                    packed, k, block_q, block_k, num_k_tiles, input_index_offset, input_extent,
                    is_first, 1024, num_warps=8)
    _emit[(rows,)](packed, values, indices,
                  lengths if output_page_table is None else output_page_table,
                  k, output_page_table is not None,
                  0 if output_page_table is None else output_page_table.stride(0),
                  output_page_size, num_warps=8)
    return values, indices


def wrap_tiled_topk(native):
    """Fail closed on untested selector contracts; never silently change paths."""
    def wrapped(**kw):
        assert kw.get('zero_row_start') and kw.get('k_start') is None
        assert kw.get('k_end') is None and kw.get('lengths') is not None
        assert kw.get('extent_splits', 1) == 1 and kw.get('output_row_stride') in (None, 1)
        assert kw.get('tile_k_offset', 0) == kw.get('output_row_base', 0) == 0
        assert kw.get('input_index_offset', 0) == kw.get('output_index_offset', 0)
        assert kw.get('num_k_tiles', 0) > 0 and kw.get('input_extent', 0) > 0
        page_table = kw.get('output_page_table')
        values, indices = native(**{**kw, 'output_page_table': None})
        result = canonicalize(tile_logits=kw['tile_logits'], lengths=kw['lengths'],
                            values=values, indices=indices, block_q=kw['block_q'], block_k=kw['block_k'],
                            num_k_tiles=kw['num_k_tiles'], input_index_offset=kw.get('input_index_offset', 0),
                            input_extent=kw['input_extent'], carry_values=kw.get('carry_values'),
                            carry_indices=kw.get('carry_indices'), is_first=kw.get('is_first', True),
                            output_page_table=page_table, output_page_size=kw.get('output_page_size', 64))
        if _VERIFY_WITH_ORACLE == '1':
            from b12x.attention.dsa_indexer.canonical_index_oracle import oracle_tiled_topk, matches
            expected_values, expected_indices = oracle_tiled_topk(**kw)
            torch._assert_async(matches(*result, expected_values, expected_indices).all(),
                                'Independent canonical top-k oracle mismatch')
        return result
    return wrapped
