#!/usr/bin/env python3
"""Compile the pinned SM121 experts extension without exposing or opening a GPU.

Run in a bounded CPU-only container. This is a build, not a GPU test. The explicit
architecture here replaces device discovery only; CUDA sources are unmodified.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    assert os.environ.get('NVIDIA_VISIBLE_DEVICES') == 'void'
    assert not list(Path('/dev').glob('nvidia*'))
    limit = Path('/sys/fs/cgroup/memory.max').read_text().strip()
    assert limit != 'max' and int(limit) <= 4 * 2**30
    assert os.environ.get('MAX_JOBS') == '1'
    import torch
    import tensorfold.cuda.exl3.experts as experts
    from torch.utils.cpp_extension import load
    assert not torch.cuda.is_initialized()
    root = Path(experts.__file__).parent
    sources = [root / f for f in ('experts.cpp', 'experts.cu', 'experts_cb0.cu',
                                  'experts_cb1.cu', 'experts_cb2.cu')]
    report = dict(phase='compiling', started_at=time.time(), cuda_initialized=False,
                  architecture='sm_121', max_jobs=1,
                  sources={s.name: hashlib.sha256(s.read_bytes()).hexdigest()
                           for s in sources + [root/'decode.cuh', root/'experts_grouped.cuh']})
    a.output.write_text(json.dumps(report, indent=2)+'\n')
    try:
        module = load(name='tensorfold_exl3_experts_v1', sources=list(map(str, sources)),
                      extra_cuda_cflags=['-O3', '-lineinfo', '-gencode=arch=compute_121,code=sm_121'],
                      verbose=True)
        assert not torch.cuda.is_initialized()
        binary = Path(module.__file__)
        report.update(phase='complete', passed=True, binary=str(binary),
                      binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                      binary_bytes=binary.stat().st_size, finished_at=time.time())
    except Exception as exc:
        report.update(phase='failed', passed=False, error=f'{type(exc).__name__}: {exc}',
                      finished_at=time.time())
        raise
    finally:
        a.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
