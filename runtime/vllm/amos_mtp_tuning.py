"""Bounded experiments on the existing request/phase MTP choice.

Native accepted-prefix accounting, target verification, token history and
prefill remain unchanged. Policy changes latch between groups of live requests.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import re

CONTROL = Path('/root/.cache/amos-tp6-mtp-tuning.json')


def validate(value):
    if not isinstance(value, dict) or set(value) != {'revision','mode','depth','window','cost_ms'}:
        raise ValueError('Unexpected MTP tuning fields')
    if not isinstance(value['revision'],str) or not re.fullmatch(r'[A-Za-z0-9_.-]{1,80}',value['revision']):
        raise ValueError('Invalid MTP revision')
    mode, depth, window, costs = (value[k] for k in ('mode','depth','window','cost_ms'))
    if mode not in ('original','fixed','costs') or type(depth) is not int or type(window) is not int:
        raise ValueError('Invalid MTP policy')
    if mode in ('original','fixed'):
        if costs is not None or window != 16 or (depth != 0 if mode=='original' else not 1<=depth<=4):
            raise ValueError('Original/fixed policy must retain measured costs and window16')
    else:
        if depth != 0 or window not in (4,8,16,32) or not isinstance(costs,dict) or set(costs) != {'1','2','3','4'}:
            raise ValueError('Candidate costs require C1..C4 and a bounded window')
        for row in costs.values():
            if not isinstance(row,list) or len(row)!=4 or any(type(v) not in (int,float) or not math.isfinite(v) or not 1<v<10000 for v in row):
                raise ValueError('Four finite measured costs are required at each concurrency')
    return value


def choose(controller, scheduled, live, path=CONTROL):
    flag=os.environ.get('AMOS_MTP_TUNING_CONTROL','0')
    if flag not in ('0','1'):
        raise ValueError('Invalid MTP tuning flag')
    if flag == '0':
        return controller.choose(scheduled,live)
    if controller.pending:
        raise RuntimeError('MTP tuning requires completed native acceptance accounting')
    if controller.max_num_spec_tokens != 4 or set(controller.costs) != {1,2,3,4}:
        raise ValueError('MTP tuning requires the pinned TP6 request/phase controller')
    active = any(live.get(key) is state.request and not state.request.is_finished()
                 for key,state in controller.states.items())
    current = getattr(controller,'amos_tuning_control',None)
    if current is None or not active:
        raw = path.read_bytes()
        if len(raw)>4096:
            raise ValueError('Oversized MTP control')
        digest = hashlib.sha256(raw).hexdigest()
        if digest != getattr(controller,'amos_tuning_digest',None):
            candidate = validate(json.loads(raw))
            if active:
                raise RuntimeError('Cannot initialize MTP tuning midway through live requests')
            original = getattr(controller,'amos_tuning_original',None)
            if original is None:
                if controller.observation_window != 16:
                    raise ValueError('Original MTP policy must use qualified window16')
                original = dict(costs=dict(controller.costs),window=controller.observation_window)
            out=path.with_suffix('.applied.json');tmp=out.with_suffix('.tmp')
            tmp.write_text(json.dumps(dict(candidate,sha256=digest,scheduler_pid=os.getpid()),sort_keys=True)+'\n')
            tmp.replace(out)
            controller.amos_tuning_original=original
            controller.costs = ({int(n):tuple(row) for n,row in candidate['cost_ms'].items()}
                                if candidate['mode']=='costs' else dict(original['costs']))
            controller.observation_window = candidate['window']
            controller.states.clear()
            controller.last_signature=None
            controller.amos_tuning_control=current=candidate
            controller.amos_tuning_digest=digest
    selected=controller.choose(scheduled,live)
    if current['mode']=='fixed':
        controller.num_spec_tokens=selected=current['depth']
    return selected
