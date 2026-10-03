#!/usr/bin/env python3
"""Replay frozen MTP workloads against a drained, already-qualified native API.

The operator owns the six-node window, checks host guards and image identity,
and applies/acknowledges each policy externally. This client never changes or
launches a backend. Its image and policy metadata are operator assertions.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import sys
import threading
import time
import urllib.request

from performance import atomic_json,parse_metrics,parse_recall
sys.path.append(str(Path(__file__).resolve().parents[1]/'runtime/vllm'))
from amos_mtp_tuning import validate


class API:
    def __init__(self,base,key_file):
        self.base=base.rstrip('/')
        self.key=key_file.read_text().strip()
        assert self.key and all(33<=ord(c)<=126 for c in self.key)
        self.opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def open(self,path,payload=None,timeout=600):
        return self.opener.open(urllib.request.Request(self.base+path,
            data=None if payload is None else json.dumps(payload).encode(),
            headers={'Content-Type':'application/json','Authorization':'Bearer '+self.key}),timeout=timeout)

    def json(self,path,payload=None):
        with self.open(path,payload) as response:return json.load(response)

    def metrics(self):
        with self.open('/metrics',timeout=15) as response:return parse_metrics(response.read().decode())


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def tokenize(api, body):
    # The pinned /tokenize path drops historical reasoning_content. The chat
    # renderer preserves it, so only /render gives the exact generation prefix.
    return api.json('/v1/chat/completions/render', body)['token_ids']


def stream(api, body, on_first=None):
    start = time.monotonic()
    first, usage, done, reason = None, None, False, None
    answer, thinking, calls, times = '', '', {}, []
    with api.open('/v1/chat/completions', body) as response:
        for line in response:
            if not line.startswith(b'data: '):
                continue
            raw = line[6:].strip()
            if raw == b'[DONE]':
                done = True
                break
            event = json.loads(raw)
            assert 'error' not in event, 'Streaming API error'
            usage = event.get('usage') or usage
            for choice in event.get('choices', []):
                delta = choice.get('delta', {})
                answer += delta.get('content') or ''
                thinking += delta.get('reasoning') or delta.get('reasoning_content') or ''
                for call in delta.get('tool_calls') or []:
                    index = call['index']
                    dst = calls.setdefault(index, {'name': '', 'arguments': ''})
                    for key in dst:
                        dst[key] += call.get('function', {}).get(key) or ''
                if any(delta.get(k) for k in ('content', 'reasoning', 'reasoning_content', 'tool_calls')):
                    now = time.monotonic() - start
                    if first is None:
                        first = now
                        if on_first is not None:
                            on_first()
                    times.append(now)
                reason = choice.get('finish_reason') or reason
    assert done and usage and first is not None, 'Incomplete stream'
    parsed, strict = parse_recall(answer.strip())
    return dict(ttft_seconds=first, elapsed_seconds=time.monotonic()-start, usage=usage,
        stream_events_seconds=times, answer=answer, reasoning=thinking, tool_calls=calls,
        answer_sha256=digest(answer), reasoning_sha256=digest(thinking), finish_reason=reason,
        semantic_pass=parsed == {'result': 42, 'status': 'ok'} and reason == 'stop' and not calls,
        strict_json=strict)


def idle(api):
    end = time.monotonic() + 15
    while True:
        m = api.metrics()
        if m['vllm:num_requests_running'] == m['vllm:num_requests_waiting'] == 0:
            return m
        assert time.monotonic() < end, 'Inspect the still-live requests; do not restart'
        time.sleep(.25)


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    position = (len(values)-1)*fraction
    lo = int(position)
    hi = min(lo+1, len(values)-1)
    return values[lo] + (values[hi]-values[lo])*(position-lo)


def measure(api,items,count):
    before=idle(api);gate=threading.Barrier(count)
    def one(index):
        item=items[index%len(items)];gate.wait(timeout=30)
        r=stream(api,item['payload'])
        tokens=r['usage']['completion_tokens']
        assert 1<tokens<=256 and r['usage']['prompt_tokens']==item['prompt_tokens']
        if item['kind']!='tools':assert tokens==256
        events=r['stream_events_seconds'];assert len(events)>1
        gaps=[b-a for a,b in zip(events,events[1:])]
        r.update(fixture=item['id'],decode_tok_s_estimate=(tokens-1)/(events[-1]-events[0]),
                 p95_stream_gap=percentile(gaps,.95),p99_stream_gap=percentile(gaps,.99),max_stream_gap=max(gaps),
                 tool_stream_present=bool(r['tool_calls']),finished_from_origin=time.monotonic()-start)
        for key in ('answer','reasoning','tool_calls','semantic_pass','strict_json'):r.pop(key,None)
        return r
    start=time.monotonic();peak=0
    with ThreadPoolExecutor(max_workers=count) as pool:
        futures=[pool.submit(one,i) for i in range(count)]
        while not all(f.done() for f in futures):
            peak=max(peak,api.metrics()['vllm:num_requests_running'])
            time.sleep(.25)
        rows=[f.result() for f in futures]
    elapsed=max(r['finished_from_origin'] for r in rows);after=idle(api)
    delta={k:v-before.get(k,0) for k,v in after.items()}
    assert delta['vllm:num_preemptions_total']==0
    assert peak==count, 'Requested concurrency was not observed'
    assert delta['vllm:prefix_cache_queries_total']==sum(r['usage']['prompt_tokens'] for r in rows), 'Unexpected traffic or unsettled cache counters'
    drafts=delta['vllm:spec_decode_num_drafts_total'];accepted=delta['vllm:spec_decode_num_accepted_tokens_total']
    seconds=delta['vllm:request_decode_time_seconds_sum'];assert drafts>0 and seconds>0
    return dict(requests=count,peak_running=peak,rows=rows,metrics_delta=delta,elapsed_seconds=elapsed,
        aggregate_output_tok_s=sum(r['usage']['completion_tokens'] for r in rows)/elapsed,mean_attempted_depth=delta['vllm:spec_decode_num_draft_tokens_total']/drafts,
        accepted_per_draft=accepted/drafts,server_step_ms_estimate=1000*seconds/drafts,
        accepted_per_server_request_decode_second=accepted/seconds,
        accepted_per_wall_second=accepted/elapsed,
        min_available_gib=None,
        cost_limitation='Server decode time includes target, draft and scheduling. Concurrent sums are request-weighted with declining residency, not isolated constant-batch GPU time. Wall throughput counts all output; accepted tokens exclude target bonus tokens.')


def bench(api,image,policy,name,scope,repetitions,expected_depth):
    fixture=json.loads(FIXTURES.read_text());items=fixture['items']
    if scope=='screen':
        cells=[([next(i for i in items if i['id']=='code-0')],1),
               ([i for i in items if i['nominal_context']==0],4)]
    else:
        cells=[([item],1) for item in items]
        cells += [([i for i in items if i['nominal_context']==n],4) for n in (0,8192,32768)]
        if scope=='candidate_matrix':
            cells += [([i for i in items if i['nominal_context']==n],count)
                      for count in (2,3) for n in (0,8192,32768)]
    out=Q/(name+'-bench.json');assert not out.exists()
    report=dict(phase='running',fixture_sha256=hashlib.sha256(FIXTURES.read_bytes()).hexdigest(),
        image=image,policy=policy,scope=scope,repetitions=repetitions,rows=[])
    atomic_json(out,report)
    # Warm each exact prompt and report those requests separately, never as a speed result.
    selected={item['id']:item for group,_ in cells for item in group}
    warm=[]
    for item in selected.values():
        body=dict(item['payload'],max_tokens=32)
        row=stream(api,body);warm.append(dict(fixture=item['id'],usage=row['usage'],ttft_seconds=row['ttft_seconds']))
    report['warmup']=warm;atomic_json(out,report)
    for repeat in range(repetitions):
        # Rotate workload order across visits; policy interleaving is controlled by the caller.
        for group,count in cells[repeat%len(cells):]+cells[:repeat%len(cells)]:
            row=measure(api,group,count);row['repeat']=repeat
            report['rows'].append(row);atomic_json(out,report)
            if expected_depth:
                assert expected_depth-.1 <= row['mean_attempted_depth']<=expected_depth
            print(json.dumps(dict(fixtures=[i['id'] for i in group],requests=count,repeat=repeat,
                aggregate_output_tok_s=row['aggregate_output_tok_s'],mean_depth=row['mean_attempted_depth'],accepted_per_draft=row['accepted_per_draft'])),flush=True)
    report.update(phase='complete',passed=True);atomic_json(out,report)
    return out


def main():
    global Q,FIXTURES
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',required=True)
    p.add_argument('--key-file',required=True,type=Path)
    p.add_argument('--image',required=True)
    p.add_argument('--policy-file',required=True,type=Path)
    p.add_argument('--fixtures',type=Path,default=Path(__file__).with_name('mtp-fixtures.json'))
    p.add_argument('--output-dir',required=True,type=Path)
    p.add_argument('--name',required=True)
    p.add_argument('--scope',choices=['screen','matrix','candidate_matrix'],default='matrix')
    p.add_argument('--repetitions',type=int,default=1)
    a=p.parse_args()
    assert re.fullmatch(r'sha256:[a-f0-9]{64}',a.image)
    assert a.name.replace('-','').isalnum() and 1<=a.repetitions<=3
    policy=validate(json.loads(a.policy_file.read_text()))
    Q=a.output_dir;Q.mkdir(parents=True,exist_ok=True);FIXTURES=a.fixtures
    api=API(a.base,a.key_file);idle(api)
    assert {x['id'] for x in api.json('/v1/models')['data']}=={'glm-5.3'}
    assert api.json('/v1/amos/capacity')=={'context_length':360000,'max_running_requests':4}
    for item in json.loads(FIXTURES.read_text())['items']:
        ids=tokenize(api,item['payload'])
        assert len(ids)==item['prompt_tokens'] and digest(ids)==item['token_sha256']
    out=bench(api,a.image,policy,a.name,a.scope,a.repetitions,policy['depth'])
    result=json.loads(out.read_text())
    result.update(image_and_policy_asserted_by_operator=True,host_guard_checks='Required externally; this API client does not inspect host guards.')
    atomic_json(out,result)


if __name__=='__main__':main()
