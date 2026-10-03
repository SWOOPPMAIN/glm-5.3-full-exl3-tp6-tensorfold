#!/usr/bin/env python3
"""Squash simple, verified source-patch layers without exporting the large base.

Only regular files and directories are supported. Whiteouts, links, special
files and type replacements are rejected. Runtime image configuration is kept
byte-for-byte equivalent as JSON; lower layer digests are unchanged. The delta
requires that lower chain already exists in Docker's classic layer store.
"""
import argparse
import copy
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def add_bytes(archive, name, data):
    member = tarfile.TarInfo(name)
    member.size = len(data)
    member.mode = 0o644
    archive.addfile(member, io.BytesIO(data))


def merge(layers):
    entries = {}
    for data in layers:
        seen = set()
        with tarfile.open(fileobj=io.BytesIO(data)) as archive:
            for member in archive:
                path = PurePosixPath(member.name)
                assert not path.is_absolute() and '..' not in path.parts
                assert str(path) == member.name.rstrip('/') and str(path) != '.'
                assert not any(p.startswith('.wh.') for p in path.parts)
                assert member.isfile() or member.isdir(), 'Unsupported layer entry'
                assert not member.linkname
                assert not any('overlay' in k for k in member.pax_headers)
                name = str(path)
                assert name not in seen, 'Duplicate member within a layer'
                seen.add(name)
                if name in entries:
                    assert entries[name][0].type == member.type, 'Type replacement'
                entries[name] = (copy.copy(member), archive.extractfile(member).read() if member.isfile() else None)
    for name in entries:
        for parent in PurePosixPath(name).parents:
            if str(parent) in entries:
                assert entries[str(parent)][0].isdir(), 'File used as a parent'
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w', format=tarfile.PAX_FORMAT) as out:
        for name in sorted(entries, key=lambda n: (len(PurePosixPath(n).parts), n)):
            member, data = entries[name]
            out.addfile(member, None if data is None else io.BytesIO(data))
    result = buf.getvalue()
    with tarfile.open(fileobj=io.BytesIO(result)) as check:
        assert set(check.getnames()) == set(entries)
        for member in check:
            original, data = entries[member.name]
            for field in ('type', 'size', 'mode', 'uid', 'gid', 'uname', 'gname', 'mtime', 'pax_headers'):
                assert getattr(member, field) == getattr(original, field), (member.name, field)
            if data is not None:
                assert check.extractfile(member).read() == data
    files = {name: dict(sha256=digest(data), size=len(data), mode=member.mode, uid=member.uid, gid=member.gid)
             for name, (member, data) in entries.items() if data is not None}
    return result, files


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, action='append', required=True)
    p.add_argument('--keep-layers', type=int, required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists() and not a.receipt.exists()
    available, sources, final = {}, [], None
    for path in a.archive:
        with tarfile.open(path) as archive:
            manifest = json.load(archive.extractfile('manifest.json'))
            assert len(manifest) == 1
            m = manifest[0]
            config_bytes = archive.extractfile(m['Config']).read()
            config = json.loads(config_bytes)
            assert m['Config'].split('/')[-1].removesuffix('.json') == digest(config_bytes)
            ids = config['rootfs']['diff_ids']
            assert len(ids) == len(m['Layers'])
            if final:
                assert ids[:len(final[1]['rootfs']['diff_ids'])] == final[1]['rootfs']['diff_ids']
            names = set(archive.getnames())
            for name, diff in zip(m['Layers'], ids):
                if name in names:
                    data = archive.extractfile(name).read()
                    assert 'sha256:' + digest(data) == diff
                    available[diff] = data
            sources.append(dict(path=str(path), sha256=digest(path.read_bytes()), image='sha256:'+digest(config_bytes)))
            final = (m, config, config_bytes)
    m, old, config_bytes = final
    ids = old['rootfs']['diff_ids']
    assert 0 < a.keep_layers < len(ids)
    layer, files = merge([available[d] for d in ids[a.keep_layers:]])
    new = copy.deepcopy(old)
    new['rootfs']['diff_ids'] = ids[:a.keep_layers] + ['sha256:'+digest(layer)]
    history, count = [], 0
    for entry in old['history']:
        if not entry.get('empty_layer'):
            if count == a.keep_layers:
                break
            count += 1
        history.append(entry)
    assert count == a.keep_layers
    history.append(dict(created=old['created'], created_by='Squash verified source-patch layers; see accompanying provenance receipt.'))
    new['history'] = history
    assert sum(not h.get('empty_layer') for h in history) == len(new['rootfs']['diff_ids'])
    assert new['config'] == old['config']
    raw = encoded(new)
    name = 'blobs/sha256/'+digest(raw)
    layer_name = 'blobs/sha256/'+digest(layer)
    manifest = [dict(Config=name, RepoTags=[a.tag], Layers=m['Layers'][:a.keep_layers]+[layer_name])]
    with tarfile.open(a.output, mode='w') as archive:
        add_bytes(archive, 'manifest.json', encoded(manifest))
        add_bytes(archive, name, raw)
        add_bytes(archive, layer_name, layer)
    receipt = dict(sources=sources, image='sha256:'+digest(raw), tag=a.tag,
        original_image='sha256:'+digest(config_bytes), original_layers=len(ids),
        preserved_lower_layers=a.keep_layers, layers=len(new['rootfs']['diff_ids']),
        merged_layers=ids[a.keep_layers:], merged_diff_id='sha256:'+digest(layer),
        runtime_config_unchanged=True, files=files, archive=str(a.output),
        archive_sha256=digest(a.output.read_bytes()), gpu_used=False, deployed=False)
    a.receipt.write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({k:receipt[k] for k in ('image','original_layers','layers','archive_sha256')}))


if __name__ == '__main__':
    main()
