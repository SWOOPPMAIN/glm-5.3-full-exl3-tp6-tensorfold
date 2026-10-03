#!/usr/bin/env python3
"""Add the CPU-checked ngram history fix to the current E35 source overlay.

This prepares an image archive only. No controller, serving method, kernel,
weight, precision, allocation or running container changes here.
"""
import argparse
import copy
import io
import json
from pathlib import Path
import tarfile
import time

from ngram_scatter_fix import patch_source
from squash_patch_layers import add_bytes, digest, encoded

BASE = 'sha256:286e0a42c8caa3a7d45a76f006bd400e391e161b6ec9f0bbdc9dacb7dfb0d20e'
CHECKED = '9a2772e4c32c19a78e9781db93b9445fc8e20a6aa9e6e26f967ba9ced108fda0'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('base-archive', 'source', 'checks', 'output', 'receipt'):
        p.add_argument('--'+name, required=True, type=Path)
    a = p.parse_args()
    assert not a.output.exists() and not a.receipt.exists()
    source = a.source.read_bytes()
    candidate = patch_source(source)
    checks = json.loads(a.checks.read_bytes())
    assert checks['source_sha256'] == digest(source)
    assert checks['candidate_sha256'] == digest(candidate) == CHECKED
    assert checks['candidate_all_passed'] and checks['history_rows_checked'] == 2048
    assert not checks['gpu_used'] and not checks['proposal_matching_kernel_changed']
    compile(candidate, 'ngram_proposer_gpu.py', 'exec')
    with tarfile.open(a.base_archive) as src:
        manifests = json.load(src.extractfile('manifest.json'))
        assert len(manifests) == 1
        manifest = manifests[0]
        raw = src.extractfile(manifest['Config']).read()
        assert 'sha256:'+digest(raw) == BASE
        layer = src.extractfile(manifest['Layers'][-1]).read()
    original = json.loads(raw)
    assert len(original['rootfs']['diff_ids']) == 125
    assert original['rootfs']['diff_ids'][-1] == 'sha256:'+digest(layer)
    files = {}
    with tarfile.open(fileobj=io.BytesIO(layer)) as src:
        for member in src.getmembers():
            assert member.isfile() and member.mode == 0o644
            assert member.uid == member.gid == 0
            assert member.name not in files and '..' not in Path(member.name).parts
            files[member.name] = src.extractfile(member).read()
    assert len(files) == 11
    original_files = dict(files)
    added = []
    for root in ('usr/local/lib/python3.12/dist-packages/vllm', 'opt/glm53-full/vllm/vllm'):
        name = root+'/v1/spec_decode/ngram_proposer_gpu.py'
        assert name not in files
        files[name] = candidate
        added.append(name)
    assert all(files[k] == v for k,v in original_files.items())
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w', format=tarfile.PAX_FORMAT) as dst:
        for name,data in sorted(files.items()):
            add_bytes(dst, name, data)
    layer = buffer.getvalue()
    config = copy.deepcopy(original)
    config['rootfs']['diff_ids'][-1] = 'sha256:'+digest(layer)
    assert not config['history'][-1].get('empty_layer')
    config['history'][-1]['created_by'] = 'Preserve E35 overlay; add source-pinned ngram history-scatter fix.'
    assert config['config'] == original['config']
    assert config['rootfs']['diff_ids'][:-1] == original['rootfs']['diff_ids'][:-1]
    raw = encoded(config)
    image = 'sha256:'+digest(raw)
    cn, ln = 'blobs/sha256/'+digest(raw), 'blobs/sha256/'+digest(layer)
    tag = 'amos/glm53-exl3-tp6:ngram-history-'+CHECKED[:16]
    with tarfile.open(a.output, 'w') as dst:
        add_bytes(dst, 'manifest.json', encoded([dict(Config=cn,RepoTags=[tag],Layers=manifest['Layers'][:-1]+[ln])]))
        add_bytes(dst, cn, raw)
        add_bytes(dst, ln, layer)
    receipt = dict(phase='image_prepared_not_deployed', at=time.time(),
        base_image=BASE, image=image, layers=125, tag=tag,
        archive=str(a.output), archive_sha256=digest(a.output.read_bytes()),
        base_archive_sha256=digest(a.base_archive.read_bytes()),
        source_sha256=digest(source), candidate_sha256=CHECKED,
        checks_sha256=digest(a.checks.read_bytes()),
        files={n:dict(sha256=digest(data),size=len(data),mode=0o644,uid=0,gid=0) for n,data in files.items()},
        added_files=added, preserved_existing_files=sorted(original_files),
        runtime_config_change={}, weights_or_precision_changed=False,
        gpu_used=False, deployed=False,
        limitations=['CPU source checks only; CUDA proposal, TP6 scheduling, target verification, numerical gates and serving timing remain.'])
    a.receipt.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:receipt[k] for k in ('phase','base_image','image','archive_sha256')}))


if __name__ == '__main__':
    main()
