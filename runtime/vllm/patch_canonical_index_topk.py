#!/usr/bin/env python3
"""Prepare canonical logical selection in pinned B12X physical-slot prefills."""
import hashlib
import json
from pathlib import Path
import shutil

EXPECTED = '0672ac4b3c10fe78d3f9cd2b8b72fe28828a30ecb5bd5add454ecfd7f4afc933'
RELATIVE = 'attention/dsa_indexer/paged.py'


def transform(raw):
    if hashlib.sha256(raw).hexdigest() != EXPECTED:
        raise ValueError('Canonical indexer requires pinned P23r4 paged source')
    source = raw.decode()
    anchor = 'from b12x.attention.dsa_indexer.tiled_topk import (\n    run_row_topk,\n    run_tiled_topk,\n)\n'
    assert source.count(anchor) == 1
    source = source.replace(anchor, anchor +
        'from b12x.attention.dsa_indexer.canonical_index_topk import wrap_tiled_topk\n'
        '\n_canonical_tiled_topk = wrap_tiled_topk(run_tiled_topk)\n')
    prefix, source = source.split('def index_topk_fp8(', 1)
    anchor = '    for chunk_idx in range(num_chunks):\n'
    assert source.count(anchor) == 1
    source = source.replace(anchor,
        '    # DCP1 physical-slot prefills use the carry chain. Canonicalize logical\n'
        '    # token selection before mapping pages; fused decode is unchanged.\n'
        '    selected_tiled_topk = (\n'
        '        _canonical_tiled_topk\n'
        '        if use_shared_prefill_scorer and output_physical_slots\n'
        '        else run_tiled_topk\n'
        '    )\n\n' + anchor)
    anchor = '        else:\n            run_tiled_topk(\n                tile_logits=tile_logits,\n                k_start=None,\n'
    assert source.count(anchor) == 1
    source = source.replace(anchor, anchor.replace('run_tiled_topk(', 'selected_tiled_topk('))
    source = prefix + 'def index_topk_fp8(' + source
    compile(source, RELATIVE, 'exec')
    return source


def main():
    import importlib.util
    spec = importlib.util.find_spec('b12x')
    root = Path(spec.origin).parent
    target = root/RELATIVE
    transformed = transform(target.read_bytes())
    helper = Path(__file__).with_name('canonical_index_topk.py')
    compile(helper.read_bytes(), str(helper), 'exec')
    shutil.copy2(helper, target.with_name(helper.name))
    target.write_text(transformed)
    print(json.dumps(dict(before=EXPECTED, after=hashlib.sha256(transformed.encode()).hexdigest(),
                          helper=hashlib.sha256(helper.read_bytes()).hexdigest(),
                          scope='Shared-page physical-slot prefill only; native scorer retained',
                          source_weights_changed=False)))


if __name__ == '__main__':
    main()
