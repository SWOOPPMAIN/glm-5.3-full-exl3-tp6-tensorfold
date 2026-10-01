#!/usr/bin/env python3
"""Initialize finite synthetic inputs for pinned paged-indexer prewarm only."""
import hashlib
import json
from pathlib import Path

EXPECTED = 'd07f35dd8e757d311ac094f470d975ce3ff9527823e629fd923e7bdd7ebe7799'
RELATIVE = 'model_executor/layers/sparse_attn_indexer.py'


def transform(raw):
    if hashlib.sha256(raw).hexdigest() != EXPECTED:
        raise ValueError('Finite indexer prewarm requires pinned P23r6 source')
    source = raw.decode()
    start = source.index('def _prewarm_b12x_paged_indexer_prefill(')
    end = source.index('\ndef ', start+1)
    part = source[start:end]
    left = part.index('    q_warm = ')
    right = part.index('    seq_lens = ', left)
    part = part[:left]+'''    # Prewarm compiles shapes; profile-run tensors may contain arbitrary
    # bytes or NaNs. Never read them as numerical fixtures or mutate live KV.
    q_warm = torch.zeros(
        (q_rows, *tuple(q_quant.shape[1:])),
        dtype=q_quant.dtype,
        device=q_quant.device,
    )
    weights_warm = torch.full(
        (q_rows, num_q_heads),
        1.0 / num_q_heads,
        dtype=torch.float32,
        device=q_quant.device,
    )
'''+part[right:]
    left = part.index('    kv_cache_warm = ')
    right = part.index('    topk_indices = ', left)
    part = part[:left]+'''    # Every synthetic page-table entry points to page0. Supply one finite
    # packed page: contiguous FP8 zero keys, then FP32 unit scales.
    kv_cache_warm = torch.zeros(
        (1, _B12X_PAGED_INDEX_PAGE_SIZE,
         _B12X_PAGED_INDEX_HEAD_DIM + _B12X_PAGED_INDEX_SCALE_BYTES),
        dtype=torch.uint8,
        device=q_quant.device,
    )
    kv_cache_warm.view(-1)[
        _B12X_PAGED_INDEX_PAGE_SIZE * _B12X_PAGED_INDEX_HEAD_DIM:
    ].view(torch.float32).fill_(1.0)
'''+part[right:]
    result = source[:start]+part+source[end:]
    compile(result, RELATIVE, 'exec')
    return result


def main():
    roots = [Path('/usr/local/lib/python3.12/dist-packages/vllm'), Path('/opt/glm53-full/vllm/vllm')]
    prepared = [(root, transform((root/RELATIVE).read_bytes())) for root in roots]
    for root, source in prepared:
        (root/RELATIVE).write_text(source)
    print(json.dumps(dict(before=EXPECTED, after=hashlib.sha256(prepared[0][1].encode()).hexdigest(),
                          scope='Synthetic paged-indexer prewarm inputs only; real inference and oracle unchanged')))


if __name__ == '__main__':
    main()
