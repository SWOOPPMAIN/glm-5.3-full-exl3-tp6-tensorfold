#!/usr/bin/env python3
"""Replay frozen copy/edit/prose workloads against a drained, qualified TP6 API.

Based on the existing MTP measurement harness. Zero copy proposals are a valid
measurement; report them instead of rejecting a non-copying prose workload.
The operator must own maintenance and verify all six host guards externally.
This API client never launches or changes a backend; image/profile are assertions.
The reported draft-slot counter can include worker-trimmed invalid GPU-ngram slots.
"""
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
from pathlib import Path
import re
import json
import threading
import time

from mtp_matrix import API, digest, idle, percentile, stream, tokenize
from performance import atomic_json

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
    seconds=delta['vllm:request_decode_time_seconds_sum'];assert drafts>=0 and seconds>0
    proposed=delta['vllm:spec_decode_num_draft_tokens_total'];assert 0<=accepted<=proposed
    if drafts==0:assert proposed==accepted==0
    return dict(requests=count,peak_running=peak,rows=rows,metrics_delta=delta,elapsed_seconds=elapsed,
        aggregate_output_tok_s=sum(r['usage']['completion_tokens'] for r in rows)/elapsed,mean_attempted_depth=proposed/drafts if drafts else 0,
        accepted_per_draft=accepted/drafts if drafts else 0,server_step_ms_estimate=1000*seconds/drafts if drafts else None,
        reported_draft_slots=proposed,accepted_tokens=accepted,reported_slots_not_accepted=proposed-accepted,accepted_per_reported_slot=accepted/proposed if proposed else None,
        accepted_per_server_request_decode_second=accepted/seconds,
        accepted_per_wall_second=accepted/elapsed,
        min_available_gib=None,
        cost_limitation='Server decode time includes target, draft and scheduling. Concurrent sums are request-weighted with declining residency, not isolated constant-batch GPU time. Wall throughput counts all output; accepted tokens exclude target bonus tokens.')

def bench(api,image,profile,tuning,fixtures,output,label,repetitions):
    assert not output.exists()
    data=json.loads(fixtures.read_bytes());items=data['items']
    assert len(items)==6 and {i['kind'] for i in items}=={'repeat-code','edited-code','prose'}
    cells=[([item],1) for item in items]
    cells.extend(([i for i in items if i['nominal_context']==n],4) for n in (2048,8192))
    report=dict(phase='running',image=image,profile=profile,
        tuning=tuning,fixture_sha256=hashlib.sha256(fixtures.read_bytes()).hexdigest(),
        label=label,repetitions=repetitions,warmup=[],rows=[])
    atomic_json(output,report)
    idle(api)
    for item in items:
        r=stream(api,dict(item['payload'],max_tokens=32))
        assert r['usage']['prompt_tokens']==item['prompt_tokens'] and r['usage']['completion_tokens']==32
        report['warmup'].append(dict(fixture=item['id'],usage=r['usage'],ttft_seconds=r['ttft_seconds']))
        atomic_json(output,report)
    for repeat in range(repetitions):
        for group,count in cells[repeat:]+cells[:repeat]:
            row=measure(api,group,count);row['repeat']=repeat
            report['rows'].append(row);atomic_json(output,report)
            print(json.dumps(dict(profile=report['profile'],fixtures=[i['id'] for i in group],requests=count,
                repeat=repeat,output_tok_s=row['aggregate_output_tok_s'],accepted_tokens=row['accepted_tokens'],
                reported_draft_slots=row['reported_draft_slots'])),flush=True)
    report.update(phase='complete',passed=True);atomic_json(output,report)
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',required=True);p.add_argument('--key-file',type=Path,required=True)
    p.add_argument('--image',required=True);p.add_argument('--profile',choices=['mtp4','ngram-gpu4'],required=True)
    p.add_argument('--tuning-file',type=Path,required=True)
    p.add_argument('--fixtures',type=Path,default=Path(__file__).resolve().parents[1]/'results/copy-drafting-fixtures.json')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--label',required=True)
    p.add_argument('--repetitions',type=int,choices=[1,2,3],default=3);a=p.parse_args()
    assert re.fullmatch(r'sha256:[a-f0-9]{64}',a.image)
    assert re.fullmatch(r'[a-z0-9-]+',a.label)
    api=API(a.base,a.key_file);idle(api)
    assert {x['id'] for x in api.json('/v1/models')['data']}=={'glm-5.3'}
    assert api.json('/v1/amos/capacity')=={'context_length':360000,'max_running_requests':4}
    for item in json.loads(a.fixtures.read_bytes())['items']:
        ids=tokenize(api,item['payload'])
        assert len(ids)==item['prompt_tokens'] and digest(ids)==item['token_sha256']
    out=bench(api,a.image,a.profile,json.loads(a.tuning_file.read_bytes()),a.fixtures,a.output,a.label,a.repetitions)
    value=json.loads(out.read_bytes())
    value.update(image_and_profile_asserted_by_operator=True,host_guard_checks='Required externally; this client does not inspect host guards.',
        counter_limitation='GPU-ngram reported draft slots include invalid slots trimmed by the worker. Slots not accepted are not an exact count of actual rejected copies.')
    atomic_json(out,value)


if __name__=='__main__':main()
