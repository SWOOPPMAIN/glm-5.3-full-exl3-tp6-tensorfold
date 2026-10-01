#!/usr/bin/env python3
"""Pin the B12X persistent-CTA TMA store fix and invalidate compiled kernels.

With one epilogue tile, the original kernel omitted commit/wait operations
before a persistent CTA reused shared output memory for its next work tile.
The existing PipelineTmaStore path provides the required completion wait and
thread barrier. Arithmetic, tile selection and quantization stay unchanged.

NVIDIA's completion semantics:
https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-wait-group
"""
import hashlib
import json
from pathlib import Path
import sys

SOURCE_SHA256 = 'f92d4e1e73a20dd801db200aaa6d89e7463c8b319ac304a19d2d13b194efec4e'
BEFORE = '''                    has_multi_epi_store = cutlass.const_expr(
                        not (
                            self.epi_stage == 1 and epi_rest_m == 1 and epi_rest_n == 1
                        )
                    )'''
AFTER = '''                    # Persistent CTAs must finish an asynchronous TMA store
                    # before reusing sC, even with one epilogue tile per GEMM tile.
                    has_multi_epi_store = cutlass.const_expr(
                        not (
                            self.epi_stage == 1 and epi_rest_m == 1 and epi_rest_n == 1
                        )
                        or (
                            not self.single_work_tile_per_cta
                            and not self.use_m1_non_tma_c
                            and not self.quantize_c
                            and self.split_k_slices == 1
                        )
                    )'''
KEY_BEFORE = '''        return (
            self._n,
            self._k,'''
KEY_AFTER = '''        return (
            "amos-persistent-tma-store-v1",
            self._n,
            self._k,'''


def transform(raw, path):
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError('B12X source differs from pinned P18: ' + str(path))
    source = raw.decode()
    for before, after in ((BEFORE, AFTER), (KEY_BEFORE, KEY_AFTER)):
        if source.count(before) != 1:
            raise ValueError('B12X patch anchor changed: ' + str(path))
        source = source.replace(before, after)
    # All derived launch keys call super().compile_key(), invalidating the
    # persistent cache for every dense variant that shares DenseGemmKernel.
    compile(source, str(path), 'exec')
    return source


def patch(paths):
    prepared = {path: transform(path.read_bytes(), path) for path in paths}
    for path, source in prepared.items():
        path.write_text(source)
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


if __name__ == '__main__':
    print(json.dumps(patch([Path(arg) for arg in sys.argv[1:]])))
