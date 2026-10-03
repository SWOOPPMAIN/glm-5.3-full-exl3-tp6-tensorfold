#!/usr/bin/env python3
"""Replace only the current source overlay with a forward boundary comparison.

Keep all earlier layers, row32/64 binaries and image configuration identical.
Replacing the last overlay avoids Docker's layer-depth limit. Requires the
retained E33 delta archive; this is not a clean-machine serving-image build.
"""
import argparse
import copy
import io
import json
from pathlib import Path
import tarfile
import time

from squash_patch_layers import digest,encoded,add_bytes

BASE='sha256:b4988201229054893df527528c38e8091d4d5392d44c314406e66493ec9300da'
DISPATCH='bea184c0a42967bf52e1306b1fecedb2eaa4533451402b98180e6e42046ec0fb'


def patch_dispatch(data):
    assert digest(data)==DISPATCH
    old=b'# One policy for generation and teacher scoring. The native prefill kernel\n# handles short tails more efficiently than E3\'s routing and launch overhead.\nNATIVE_PREFILL_MAX_ROWS = 512\n'
    assert data.count(old)==1
    data=data.replace(old,b'# The required boundary control is shared by generation and teacher scoring.\n')
    old=b'    if rows <= NATIVE_PREFILL_MAX_ROWS:\n'
    assert data.count(old)==1
    data=data.replace(old,b'    from vllm.amos_e3 import boundary\n    selected = boundary.latch(layer.layer_name)\n    if rows <= selected["native_max_rows"]:\n')
    old=b'        return None\n    from vllm.amos_e3 import policy as runtime\n    return runtime.apply(layer, x, weights, ids, stream_scratch=True)\n'
    assert data.count(old)==1
    return data.replace(old,b'        boundary.acknowledge(layer.layer_name, rows, "native")\n        return None\n    from vllm.amos_e3 import policy as runtime\n    result = runtime.apply(layer, x, weights, ids, stream_scratch=True)\n    boundary.acknowledge(layer.layer_name, rows, "e3")\n    return result\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-archive',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--receipt',required=True,type=Path);a=p.parse_args()
    assert not a.output.exists() and not a.receipt.exists()
    with tarfile.open(a.base_archive) as src:
        manifests=json.load(src.extractfile('manifest.json'));assert len(manifests)==1
        manifest=manifests[0];raw=src.extractfile(manifest['Config']).read();assert 'sha256:'+digest(raw)==BASE
        old_layer=src.extractfile(manifest['Layers'][-1]).read()
    original=json.loads(raw);assert len(original['rootfs']['diff_ids'])==125
    assert 'sha256:'+digest(old_layer)==original['rootfs']['diff_ids'][-1]
    files={}
    with tarfile.open(fileobj=io.BytesIO(old_layer)) as src:
        for member in src.getmembers():
            assert member.isfile() and member.mode==0o644 and member.uid==member.gid==0
            assert member.name not in files and '..' not in Path(member.name).parts
            files[member.name]=src.extractfile(member).read()
    assert len(files)==9;old_files=dict(files)
    boundary=Path(__file__).with_name('amos_e3_boundary.py').read_bytes()
    roots=('usr/local/lib/python3.12/dist-packages/vllm','opt/glm53-full/vllm/vllm')
    changed=[]
    for root in roots:
        name=root+'/amos_grouped_prefill.py';files[name]=patch_dispatch(files[name]);changed.append(name)
        files[root+'/amos_e3/boundary.py']=boundary
    assert all(files[n]==data for n,data in old_files.items() if n not in changed)
    for name,data in files.items():
        if name.endswith('.py'):compile(data,name,'exec')
    buf=io.BytesIO()
    with tarfile.open(fileobj=buf,mode='w',format=tarfile.PAX_FORMAT) as dst:
        for name,data in sorted(files.items()):add_bytes(dst,name,data)
    layer=buf.getvalue();new=copy.deepcopy(original)
    new['rootfs']['diff_ids'][-1]='sha256:'+digest(layer)
    assert not new['history'][-1].get('empty_layer')
    new['history'][-1]['created_by']='Preserve E33 source overlay and kernels; add required uniform native/E3 boundary control.'
    assert new['config']==original['config'] and new['rootfs']['diff_ids'][:-1]==original['rootfs']['diff_ids'][:-1]
    config=encoded(new);image='sha256:'+digest(config);cn='blobs/sha256/'+digest(config);ln='blobs/sha256/'+digest(layer)
    tag='amos/glm53-exl3-tp6:e3-boundary-'+digest(boundary)[:16]
    with tarfile.open(a.output,'w') as dst:
        add_bytes(dst,'manifest.json',encoded([dict(Config=cn,RepoTags=[tag],Layers=manifest['Layers'][:-1]+[ln])]))
        add_bytes(dst,cn,config);add_bytes(dst,ln,layer)
    receipt=dict(at=time.time(),base_image=BASE,image=image,tag=tag,layers=125,
        archive=str(a.output),archive_sha256=digest(a.output.read_bytes()),base_archive_sha256=digest(a.base_archive.read_bytes()),
        files={n:dict(sha256=digest(data),size=len(data),mode=0o644,uid=0,gid=0) for n,data in files.items()},
        changed_existing_files=changed,preserved_existing_files=[n for n in old_files if n not in changed],
        required_control='/root/.cache/amos-e3-boundary.json on all six ranks; qualified row32 policy also required',
        runtime_config_change={},weights_or_precision_changed=False,gpu_used=False,deployed=False)
    a.receipt.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:receipt[k] for k in ('base_image','image','layers','archive_sha256')}))


if __name__=='__main__':main()
