#!/usr/bin/env python3
"""Bounded CPU uprobes for actual RoCE/NCCL sizes, with no model/image edits.

Run as root on each owned Spark. start launches a 100-second observer; stop or
expiry removes only its own trace instance/events. Numeric arguments only.
"""
import argparse
import hashlib
import json
import mmap
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import time

TRACE = Path('/sys/kernel/tracing')
STATE = Path('/run/amos-comm-sizes')


def symbol(path, name):
    """Resolve an ELF64 AArch64 function to a file offset, including local symbols."""
    with path.open('rb') as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as b:
        h = struct.unpack_from('<16sHHIQQQIHHHHHH', b)
        assert h[0][:6] == b'\x7fELF\x02\x01' and h[2] == 183
        assert h[11] == 64 and h[12] > 0
        sections = [struct.unpack_from('<IIQQQQIIQQ', b, h[6]+i*h[11]) for i in range(h[12])]
        matches = set()
        for section in sections:
            if section[1] not in (2, 11):
                continue
            assert section[9] == 24
            strings = sections[section[6]]
            for off in range(section[4], section[4]+section[5], section[9]):
                n, info, _, index, value, size = struct.unpack_from('<IBBHQQ', b, off)
                if not n or not index or info & 15 != 2:
                    continue
                start = strings[4]+n; end = b.find(b'\0', start, strings[4]+strings[5])
                if b[start:end].decode() != name:
                    continue
                target = sections[index]
                assert target[3] <= value < target[3]+target[5] and size > 0
                matches.add((target[4]+value-target[3], size))
        assert len(matches) == 1, (name, matches)
        offset, size = matches.pop()
        return dict(symbol=name, file_offset=offset, function_size=size,
                    sha256=hashlib.sha256(b).hexdigest())


def parse_hist(text, keys):
    assert 'Dropped: 0' in text, 'Incomplete histogram'
    result = []
    for line in text.splitlines():
        if not line.lstrip().startswith('{'):
            continue
        values = dict(re.findall(r'([a-zA-Z_][a-zA-Z_0-9]*)\s*:\s*(\d+)', line))
        assert set(values) == set(keys) | {'hitcount'}, line
        result.append({k: int(v) for k, v in values.items()})
    totals = re.search(r'Totals:\s*Hits:\s*(\d+)', text)
    assert totals and sum(r['hitcount'] for r in result) == int(totals[1])
    return result


def save(path, data):
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(data, indent=2)+'\n'); tmp.replace(path)


def event_command(command):
    # Text append mode seeks to EOF in FileIO.__init__; tracefs rejects lseek.
    # Write without seeking or truncating other owners' dynamic events.
    fd=os.open(TRACE/'uprobe_events',os.O_WRONLY|os.O_APPEND)
    try:
        data=command.encode();assert os.write(fd,data)==len(data)
    finally:
        os.close(fd)


def inspect(cid):
    c = json.loads(subprocess.check_output(['docker','inspect','amos-glm53-exl3-tp6']))[0]
    assert c['Id'] == cid and c['State']['Running']
    pids = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).split()
    found = []
    for pid in pids:
        assert pid.isdigit() and cid in Path('/proc',pid,'cgroup').read_text()
        paths = {l.split()[-1] for l in Path('/proc',pid,'maps').read_text().splitlines()
                 if 'roce_proxy-' in l or 'libnccl.so' in l}
        if paths:
            assert len(paths) == 2
            found.append((int(pid), paths))
    assert len(found) == 1
    pid, paths = found[0]
    tid_filter = ' || '.join('common_pid == '+p.name for p in sorted(Path('/proc',str(pid),'task').iterdir()))
    probes = []
    for path in sorted(paths):
        local = Path('/proc',str(pid),'root')/path.lstrip('/')
        if 'roce_proxy-' in path:
            probes.append(dict(kind='roce', path=str(local), keys=['nbytes'],
                fetch='nbytes=%x2:u32', **symbol(local,'post_op')))
        else:
            probes.append(dict(kind='nccl', path=str(local), keys=['count','dtype'],
                fetch='count=%x2:u64 dtype=%x3:u32', **symbol(local,'ncclAllReduce')))
    return dict(cid=cid,image=c['Image'],pid=pid,tid_filter=tid_filter,probes=probes)


def remove(group, instance, added):
    for probe in added:
        event = instance/'events'/group/probe['kind']
        if event.exists():
            (event/'enable').write_text('0')
            # Removing the instance also removes its histogram triggers.
    if instance.exists():
        instance.rmdir()
    for probe in reversed(added):
        event_command('-:'+group+'/'+probe['kind']+'\n')


def snapshot(report, instance):
    return {p['kind']:parse_hist((instance/'events'/report['group']/p['kind']/'hist').read_text(),p['keys'])
            for p in report['probes']}


def hold(label, cid):
    path = STATE/(label+'.json'); report = json.loads(path.read_text())
    assert report['phase'] == 'starting' and report['cid'] == cid
    group = 'amos_'+label.replace('-','_'); instance = TRACE/'instances'/group
    assert not instance.exists() and not (TRACE/'events'/group).exists()
    added = []
    try:
        report.update(inspect(cid),group=group,observer_pid=os.getpid(),started_at=time.time())
        instance.mkdir(); (instance/'buffer_size_kb').write_text('4')
        (instance/'tracing_on').write_text('0')
        for p in report['probes']:
            definition = 'p:'+group+'/'+p['kind']+' '+p['path']+':'+hex(p['file_offset'])+' '+p['fetch']+'\n'
            event_command(definition)
            added.append(p)
            event = instance/'events'/group/p['kind']
            (event/'trigger').write_text('hist:keys='+','.join(p['keys'])+':size=1024 if '+report['tid_filter'])
        report.update(phase='armed',armed_at=time.time(),duration_limit_seconds=100)
        save(path,report)
        until = time.monotonic()+100
        while not (STATE/(label+'.stop')).exists() and time.monotonic() < until:
            assert Path('/proc',str(report['pid'])).exists(), 'Observed process exited'
            time.sleep(.2)
        report['stop_requested'] = (STATE/(label+'.stop')).exists()
        for p in report['probes']:
            event=instance/'events'/group/p['kind']
            (event/'trigger').write_text('hist:keys='+','.join(p['keys'])+':size=1024:pause if '+report['tid_filter'])
        report['histograms'] = snapshot(report,instance)
        report.update(phase='stopped',finished_at=time.time())
    except BaseException as exc:
        report.update(phase='failed',error=type(exc).__name__+': '+str(exc))
        raise
    finally:
        try:
            remove(group,instance,added)
            report['cleanup_complete'] = True
        finally:
            save(path,report)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['inspect','start','hold','snapshot','stop'])
    p.add_argument('--label',required=True);p.add_argument('--cid',required=True)
    a=p.parse_args();assert re.fullmatch(r'comm[0-9]+-sizes',a.label)
    assert re.fullmatch(r'[a-f0-9]{64}',a.cid) and os.geteuid()==0
    if a.action=='inspect':
        print(json.dumps(inspect(a.cid)));return
    STATE.mkdir(mode=0o700,exist_ok=True);path=STATE/(a.label+'.json')
    if a.action=='start':
        with path.open('x') as f:json.dump(dict(phase='starting',cid=a.cid),f)
        log=(STATE/(a.label+'.log')).open('x')
        subprocess.Popen([sys.executable,__file__,'hold','--label',a.label,'--cid',a.cid],
                         stdout=log,stderr=log,start_new_session=True)
        print(json.dumps(dict(started=True)));return
    if a.action=='hold':
        hold(a.label,a.cid);return
    report=json.loads(path.read_text());assert report['cid']==a.cid
    if a.action=='stop':
        (STATE/(a.label+'.stop')).touch(exist_ok=True)
        until=time.monotonic()+10
        while report['phase'] in ['starting','armed']:
            assert time.monotonic()<until,'Inspect live observer; do not start another'
            time.sleep(.2);report=json.loads(path.read_text())
    elif report['phase']=='armed':
        assert Path('/proc',str(report['observer_pid'])).exists()
        report['histograms']=snapshot(report,TRACE/'instances'/report['group'])
    print(json.dumps(report))


if __name__=='__main__':main()
