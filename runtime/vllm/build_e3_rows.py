#!/usr/bin/env python3
"""Build a pinned row32/64 comparison image; no weights or CUDA compilation."""
import argparse
import copy
import io
import json
from pathlib import Path
import tarfile
import time

from squash_patch_layers import digest,encoded,add_bytes

BASE='sha256:340a9bae07ab134120249cf0224108fda7b3706720724bd1b2fe705245e7ce20'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-archive',type=Path,required=True)
    p.add_argument('--probe-source',type=Path,required=True)
    p.add_argument('--row32-cuda-source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--receipt',type=Path,required=True)
    a=p.parse_args();assert not a.output.exists() and not a.receipt.exists()
    here=Path(__file__).resolve().parent
    with tarfile.open(a.base_archive) as archive:
        manifests=json.load(archive.extractfile('manifest.json'));assert len(manifests)==1
        manifest=manifests[0];raw=archive.extractfile(manifest['Config']).read();assert 'sha256:'+digest(raw)==BASE
    original=json.loads(raw);assert len(original['rootfs']['diff_ids'])==124
    new=copy.deepcopy(original)
    probe=json.loads((a.probe_source/'manifest.json').read_text());assert probe['image']==BASE
    runtime=(a.probe_source/'row32/runtime.py').read_bytes()
    binary=(a.probe_source/'row32/grouped_fragments.cubin').read_bytes()
    assert digest(runtime)==probe['probe_files']['row32/runtime.py']
    assert digest(binary)==probe['probe_files']['row32/grouped_fragments.cubin']=='319dca90efb470b3e7eae708365c10f123c9f04751e06f1680c9df7bca2ce1ad'
    old=b"Path(__file__).with_name('grouped_fragments.cubin')";assert runtime.count(old)==1
    runtime=runtime.replace(old,b"Path(__file__).with_name('grouped_fragments_row32.cubin')")
    cuda=a.row32_cuda_source.read_bytes();assert digest(cuda)=='db50b3885b825537e3af84c3605d13c533f07e160de8f3bc31fc27441aafc8d6'
    dispatch=(here/'amos_grouped_prefill.py').read_bytes()
    assert digest(dispatch)=='edad95d4cdbfa8d62456dba1a9d3e433fedde8b985e4b33c4f456dbae45d84cf'
    old=b'from vllm.amos_e3 import runtime';assert dispatch.count(old)==1
    dispatch=dispatch.replace(old,b'from vllm.amos_e3 import policy as runtime')
    policy=(here/'amos_e3_policy.py').read_bytes()
    files={}
    for root in ('usr/local/lib/python3.12/dist-packages/vllm','opt/glm53-full/vllm/vllm'):
        files[root+'/amos_grouped_prefill.py']=dispatch
        files[root+'/amos_e3/policy.py']=policy
        files[root+'/amos_e3/runtime_row32.py']=runtime
        files[root+'/amos_e3/grouped_fragments_row32.cubin']=binary
    files['opt/amos-tp6/e3-row32-source/grouped_fragments.cu']=cuda
    for name,data in files.items():
        if name.endswith('.py'):compile(data,name,'exec')
    layer=io.BytesIO()
    with tarfile.open(fileobj=layer,mode='w',format=tarfile.PAX_FORMAT) as out:
        for name,data in sorted(files.items()):add_bytes(out,name,data)
    blob=layer.getvalue();new['rootfs']['diff_ids'].append('sha256:'+digest(blob))
    new['history'].append(dict(created=original['created'],created_by='Exact-tested E3 row32 candidate plus explicit required row64/32 policy; preserve original weights.'))
    assert sum(not h.get('empty_layer') for h in new['history'])==125 and new['config']==original['config']
    config=encoded(new);image='sha256:'+digest(config);cn='blobs/sha256/'+digest(config);ln='blobs/sha256/'+digest(blob)
    tag='amos/glm53-exl3-tp6:e3-rows-'+digest(policy)[:16]
    with tarfile.open(a.output,mode='w') as out:
        add_bytes(out,'manifest.json',encoded([dict(Config=cn,RepoTags=[tag],Layers=manifest['Layers']+[ln])]))
        add_bytes(out,cn,config);add_bytes(out,ln,blob)
    receipt=dict(at=time.time(),base_image=BASE,image=image,tag=tag,layers=125,archive=str(a.output),
        archive_sha256=digest(a.output.read_bytes()),base_archive_sha256=digest(a.base_archive.read_bytes()),
        files={n:dict(sha256=digest(d),size=len(d),mode=0o644,uid=0,gid=0) for n,d in files.items()},
        probe_manifest_sha256=digest((a.probe_source/'manifest.json').read_bytes()),
        runtime_config_change={},gpu_used=False,deployed=False,
        required_control='/root/.cache/amos-e3-rows.json on all six ranks; no missing-file fallback',
        weights_or_precision_changed=False)
    a.receipt.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:receipt[k] for k in ('base_image','image','layers','archive_sha256')}))


if __name__=='__main__':main()
