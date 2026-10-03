#!/usr/bin/env python3
"""Build one small source-only Docker layer above the exact retained image.

No GPU work or base re-export. A Docker classic delta archive requires the
exact lower chain on the destination; every loaded image/file is verified.
"""
import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import tarfile
import time

from patch_mtp_tuning import patched
from squash_patch_layers import digest, encoded, add_bytes

BASE='sha256:7176241a30ba8349fccb979ad7a35d0eba422d23017543da8c0147b28cb0c1cd'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-archive',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--receipt',required=True,type=Path)
    a=p.parse_args();assert not a.output.exists() and not a.receipt.exists()
    here=Path(__file__).resolve().parent
    with tarfile.open(a.base_archive) as archive:
        manifests=json.load(archive.extractfile('manifest.json'));assert len(manifests)==1
        manifest=manifests[0];raw=archive.extractfile(manifest['Config']).read()
        assert 'sha256:'+digest(raw)==BASE
    original=json.loads(raw);assert len(original['rootfs']['diff_ids'])==124
    new=copy.deepcopy(original)
    source={n:(here/n).read_bytes() for n in ['amos_mtp_tuning.py','patch_mtp_tuning.py','amos_request_phase_mtp.py']}
    changed=patched(source['amos_request_phase_mtp.py'])
    files={}
    for root in ['usr/local/lib/python3.12/dist-packages/vllm','opt/glm53-full/vllm/vllm']:
        files[root+'/amos_request_phase_mtp.py']=changed
        files[root+'/amos_mtp_tuning.py']=source['amos_mtp_tuning.py']
    for name in ['amos_mtp_tuning.py','patch_mtp_tuning.py']:
        files['opt/amos-tp6/mtp-tuning/'+name]=source[name]
    for name,data in files.items():compile(data,name,'exec')
    layer=io.BytesIO()
    with tarfile.open(fileobj=layer,mode='w',format=tarfile.PAX_FORMAT) as out:
        directory=tarfile.TarInfo('opt/amos-tp6/mtp-tuning');directory.type=tarfile.DIRTYPE;directory.mode=0o755
        out.addfile(directory)
        for name,data in sorted(files.items()):add_bytes(out,name,data)
    blob=layer.getvalue();diff='sha256:'+digest(blob)
    new['rootfs']['diff_ids'].append(diff)
    assert not any(x.startswith('AMOS_MTP_TUNING_CONTROL=') for x in new['config']['Env'])
    new['config']['Env'].append('AMOS_MTP_TUNING_CONTROL=1')
    new['history'].append(dict(created=original['created'],created_by='Install pinned request-phase MTP tuning control; source-only patch layer.'))
    assert sum(not x.get('empty_layer') for x in new['history'])==125
    config=encoded(new);image='sha256:'+digest(config)
    config_name='blobs/sha256/'+digest(config);layer_name='blobs/sha256/'+digest(blob)
    tag='amos/glm53-exl3-tp6:mtp-tuning-'+digest(source['amos_mtp_tuning.py'])[:16]
    with tarfile.open(a.output,mode='w') as out:
        add_bytes(out,'manifest.json',encoded([dict(Config=config_name,RepoTags=[tag],Layers=manifest['Layers']+[layer_name])]))
        add_bytes(out,config_name,config);add_bytes(out,layer_name,blob)
    receipt=dict(at=time.time(),base_image=BASE,image=image,tag=tag,layers=125,
        base_archive_sha256=digest(a.base_archive.read_bytes()),archive=str(a.output),archive_sha256=digest(a.output.read_bytes()),
        files={n:dict(sha256=digest(d),size=len(d),mode=0o644,uid=0,gid=0) for n,d in files.items()},
        source_sha256={n:digest(d) for n,d in source.items()},runtime_config_change={'Env':['AMOS_MTP_TUNING_CONTROL=1']},
        gpu_used=False,deployed=False,method='Exact124-layer base plus one verified regular-file source layer, no base rebuild.')
    a.receipt.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:receipt[k] for k in ['base_image','image','tag','layers']}))


if __name__=='__main__':main()
