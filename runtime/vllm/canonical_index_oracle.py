"""Independent, bounded PyTorch stable-sort oracle for paged top-k.

Diagnostic only: no native radix selector or canonical Triton kernel is used.
Process one 32-row scoring tile at a time to bound temporary allocation.
"""
import torch
import torch.nn.functional as F


def oracle_tiled_topk(*, tile_logits, k_start, lengths, topk, block_q, block_k,
                     num_k_tiles, input_index_offset=0, input_extent=0,
                     output_index_offset=0, zero_row_start=False,
                     carry_values=None, carry_indices=None, is_first=True,
                     output_page_table=None, output_page_size=64,
                     output_values=None, output_indices=None, **unused):
    assert k_start is None and zero_row_start
    assert input_index_offset == output_index_offset
    assert input_extent > 0 and block_q == 32
    assert unused.get('extent_splits', 1) == 1
    assert unused.get('output_row_stride') in (None, 1)
    assert unused.get('output_row_base', 0) == unused.get('tile_k_offset', 0) == 0
    assert unused.get('k_end') is None
    rows = lengths.numel()
    # Output arguments are intentionally ignored: the oracle must never
    # overwrite candidate output storage before comparing it.
    values = torch.empty((rows, topk), device=tile_logits.device, dtype=torch.float32)
    indices = torch.empty((rows, topk), device=tile_logits.device, dtype=torch.int32)
    sentinel = torch.iinfo(torch.int32).max
    tiles = tile_logits[:((rows+block_q-1)//block_q)*num_k_tiles*block_q*block_k].view(-1, num_k_tiles, block_q, block_k)
    logical = torch.arange(input_extent, device=tile_logits.device, dtype=torch.int64)+input_index_offset
    for start in range(0, rows, block_q):
        stop = min(start+block_q, rows)
        count = stop-start
        scores = tiles[start//block_q].permute(1, 0, 2).reshape(block_q, -1)[:count, :input_extent]
        ids = logical[None, :].expand(count, -1)
        scores = scores.masked_fill(ids >= lengths[start:stop, None], -torch.inf)
        if not is_first:
            assert carry_values is not None and carry_indices is not None
            # Sort carry IDs independently; their slots precede all new IDs.
            old_ids, old_order = carry_indices[start:stop].masked_fill(carry_indices[start:stop] < 0, sentinel).sort(dim=1, stable=True)
            old_scores = carry_values[start:stop].gather(1, old_order).masked_fill(old_ids == sentinel, -torch.inf)
            scores = torch.cat((old_scores, scores), dim=1)
            ids = torch.cat((old_ids.to(torch.int64), ids), dim=1)
        if scores.shape[1] < topk:
            pad = topk-scores.shape[1]
            scores = F.pad(scores, (0, pad), value=-torch.inf)
            ids = F.pad(ids, (0, pad), value=sentinel)
        order = scores.argsort(dim=1, descending=True, stable=True)[:, :topk]
        best_values = scores.gather(1, order)
        best_ids = ids.gather(1, order).masked_fill(~torch.isfinite(best_values), sentinel)
        best_ids, order = best_ids.sort(dim=1, stable=True)
        best_values = best_values.gather(1, order)
        valid = best_ids != sentinel
        if output_page_table is not None:
            safe_ids = best_ids.masked_fill(~valid, 0)
            pages = output_page_table[start:stop].gather(1, safe_ids//output_page_size)
            best_ids = pages*output_page_size + safe_ids%output_page_size
            valid = valid & (pages >= 0)
        indices[start:stop].copy_(best_ids.masked_fill(~valid, -1))
        values[start:stop].copy_(best_values)
    return values, indices


def matches(candidate_values, candidate_indices, oracle_values, oracle_indices):
    return ((candidate_values == oracle_values) &
            (candidate_indices == oracle_indices)).all(dim=1)
