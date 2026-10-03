#!/usr/bin/env python3
"""Add a source-only E3 diagnostic to the exact selected serving image."""
import argparse
import copy
import io
import json
from pathlib import Path
import tarfile
import time

from squash_patch_layers import digest, encoded, add_bytes

BASE = 'sha256:2e947348fd26b8d58e4535e0321126935069a5e521784fa573f19fb560975d9d'
SOURCE_SHA = '2651478eee38d054cd163df0bd01e219515cd244fcacb9e354b1395e55dbc00b'


def patch(original):
    assert digest(original) == SOURCE_SHA
    old = b'    return output.to(x.dtype)\n'
    assert original.count(old) == 1
    return original.replace(old, b'    result = output.to(x.dtype)\n'
        b'    from .diagnostic_capture import capture\n'
        b'    capture(layer, x, weights, ids, result)\n'
        b'    return result\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-archive', type=Path, required=True)
    p.add_argument('--runtime-source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    a = p.parse_args(); assert not a.output.exists() and not a.receipt.exists()
    here = Path(__file__).resolve().parent
    with tarfile.open(a.base_archive) as archive:
        manifests = json.load(archive.extractfile('manifest.json')); assert len(manifests) == 1
        manifest = manifests[0]; raw = archive.extractfile(manifest['Config']).read()
        assert 'sha256:' + digest(raw) == BASE
    original = json.loads(raw); assert len(original['rootfs']['diff_ids']) == 125
    new = copy.deepcopy(original)
    changed = patch(a.runtime_source.read_bytes())
    diagnostic = (here/'amos_e3_capture.py').read_bytes()
    files = {}
    for root in ('usr/local/lib/python3.12/dist-packages/vllm', 'opt/glm53-full/vllm/vllm'):
        files[root+'/amos_e3/runtime.py'] = changed
        files[root+'/amos_e3/diagnostic_capture.py'] = diagnostic
    for name, data in files.items(): compile(data, name, 'exec')
    layer = io.BytesIO()
    with tarfile.open(fileobj=layer, mode='w', format=tarfile.PAX_FORMAT) as out:
        for name, data in sorted(files.items()): add_bytes(out, name, data)
    blob = layer.getvalue(); new['rootfs']['diff_ids'].append('sha256:'+digest(blob))
    new['history'].append(dict(created=original['created'], created_by='Bounded opt-in E3 route/input capture; source only, no math changes.'))
    assert sum(not h.get('empty_layer') for h in new['history']) == 126
    assert new['config'] == original['config']
    config = encoded(new); image = 'sha256:'+digest(config)
    config_name = 'blobs/sha256/'+digest(config); layer_name = 'blobs/sha256/'+digest(blob)
    tag = 'amos/glm53-exl3-tp6:e3-capture-'+digest(diagnostic)[:16]
    with tarfile.open(a.output, mode='w') as out:
        add_bytes(out, 'manifest.json', encoded([dict(Config=config_name, RepoTags=[tag], Layers=manifest['Layers']+[layer_name])]))
        add_bytes(out, config_name, config); add_bytes(out, layer_name, blob)
    receipt = dict(at=time.time(), base_image=BASE, image=image, tag=tag, layers=126,
        base_archive_sha256=digest(a.base_archive.read_bytes()), archive=str(a.output), archive_sha256=digest(a.output.read_bytes()),
        files={name:dict(sha256=digest(data), size=len(data), mode=0o644, uid=0, gid=0) for name,data in files.items()},
        source_runtime_sha256=SOURCE_SHA, gpu_used=False, deployed=False, runtime_config_change={},
        method='One source-only child layer; original runtime return is captured and returned unchanged.')
    a.receipt.write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({k:receipt[k] for k in ('base_image','image','tag','layers')}))


if __name__ == '__main__': main()
