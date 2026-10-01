#!/usr/bin/env python3
"""Original-weight TP6 draft-depth sweep and bounded decode profiling.

Uses original3.25bpw weights, resident804K cache and the existing6-rank group.
Independent serial references gate every output for depths0/1/2/3/4/6/8.
Uninstrumented warm measurements are separate from per-operation CUDA events
and three-step PyTorch kernel traces. Direct request transport, not HTTP timing.
"""
import argparse, faulthandler, hashlib, json, os, time, threading
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rank', type=int, required=True)
    p.add_argument('--port', type=int, required=True)
    for name in ('model', 'manifest', 'output', 'admission'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    assert 0 <= a.rank < 6 and not a.output.exists()
    assert int(Path('/sys/fs/cgroup/memory.max').read_text()) <= 108*2**30
    def available():
        return int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines()
                        if s.startswith('MemAvailable:')))*1024
    assert available() >= 110*2**30
    manifest = json.loads(a.manifest.read_text())
    for path, digest in manifest.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
    deadline = time.monotonic()+90
    while not a.admission.exists():
        assert time.monotonic() < deadline, 'Missing exact-CID guard admission'
        time.sleep(.1)
    import torch
    import torch.distributed as dist
    from tokenizers import Tokenizer
    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.glm_moe_dsa.checkpoint import RankPieces
    from tensorfold.families.glm_moe_dsa.compiled import load_experts
    from tensorfold.families.glm_moe_dsa.decoder import TP6Reduction
    from tensorfold.families.glm_moe_dsa.model import ModelWeights, ModelWorkspace, FullModel
    from tensorfold.families.glm_moe_dsa.memory import weight_plan, workspace_plan, request_plan, tensor_storage_bytes
    from tensorfold.families.glm_moe_dsa.cache import LayerCache
    from tensorfold.families.glm_moe_dsa.attention import rope_table
    from tensorfold.families.glm_moe_dsa.request import RequestEngine, CachePool
    from tensorfold.families.glm_moe_dsa.request_backend import FullModelBackend
    from tensorfold.families.glm_moe_dsa.graph_plan import graph_reserve
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    torch.manual_seed(530619)
    assert torch.cuda.get_device_capability() == (12, 1)
    torch.cuda.set_per_process_memory_fraction(104*2**30/torch.cuda.get_device_properties(0).total_memory)
    capacity = 804000
    report = dict(passed=False, rank=a.rank, cases=[], started_at=time.time(), source_manifest=manifest,
                  scope=__doc__, cache_capacity=capacity, model_rows=3072, expert_chunk_rows=1024,
                  minimum_available_gib=available()/2**30, full_quality_gate=False, serving_throughput=False)
    def save(phase):
        free = available()/2**30
        report.update(phase=phase, updated_at=time.time(), peak_cuda_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                      cuda_allocated_gib=torch.cuda.memory_allocated()/2**30,
                      cuda_reserved_gib=torch.cuda.memory_reserved()/2**30, available_gib=free,
                      minimum_available_gib=min(free, report['minimum_available_gib']),
                      cgroup_peak_gib=int(Path('/sys/fs/cgroup/memory.peak').read_text())/2**30)
        tmp = a.output.with_suffix('.tmp')
        tmp.write_text(json.dumps(report, indent=2)+'\n')
        tmp.replace(a.output)
        print(json.dumps({'rank': a.rank, 'phase': phase, 'checks': len(report['cases']), 'available_gib': free}), flush=True)
    def check(ok, name, **details):
        report['cases'].append(dict(case=name, passed=bool(ok), **details))
        save(name)
        assert ok, name
    def digest(tokens):
        return hashlib.sha256(json.dumps(list(tokens)).encode()).hexdigest()
    faulthandler.dump_traceback_later(600, repeat=True)
    active_component = None
    try:
        store = dist.TCPStore(os.environ['MASTER_ADDR'], a.port, 6, a.rank == 0, timedelta(seconds=240))
        dist.init_process_group('nccl', store=store, rank=a.rank, world_size=6, timeout=timedelta(seconds=240))
        reader = RankPieces(a.model, a.rank)
        plan = weight_plan(reader)
        wp = workspace_plan(reader.config, a.rank, 3072, capacity, logit_rows=17, expert_chunk_rows=1024,
                            bulk_min_rows=256, attention_part_rows=128, skip_empty_attention=True)
        rp = request_plan(3072, 17)
        gp=graph_reserve(24,9)
        assert plan['total']+wp['total']+rp['total']+gp['total']+4*2**30 < 104*2**30
        report['graph_reserve']=gp
        report.update(weight_plan=plan, workspace_plan=wp, request_reserve=rp)
        reduction = TP6Reduction(dist.group.WORLD, a.rank).prepare(3072, 'cuda', bulk_min_rows=256)
        assert tensor_storage_bytes(reduction, device_type='cuda', skip=('group',)) == wp['collective']
        load_experts()
        def progress(layer):
            assert available() >= 12*2**30
            if layer % 4 == 0 or layer == 78:
                save(f'loaded-layer-{layer}')
        weights = ModelWeights(reader, on_layer=progress)
        report['retained_weight_bytes'] = tensor_storage_bytes(weights, device_type='cuda')
        assert report['retained_weight_bytes'] == plan['total']
        model = FullModel(weights, reduction)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        caches = []
        for layer in range(79):
            cache = LayerCache(layer, capacity, 'cuda', indexer=reader.config.indexer_source(layer) == layer)
            cache.latent.zero_()
            cache.scales.fill_(1)
            cache.rope.zero_()
            if cache.index_keys is not None:
                cache.index_keys.zero_()
                cache.index_scales.fill_(1)
            caches.append(cache)
            if layer % 8 == 0:
                torch.cuda.synchronize()
                assert available() >= 10*2**30
                save(f'cache-resident-layer-{layer}')
        table = rope_table(capacity, 'cuda')
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        assert sum(c.nbytes() for c in caches) == wp['cache']
        arena = ModelWorkspace(weights, 3072, capacity, logit_rows=17, expert_chunk_rows=1024,
                               attention_part_rows=128, skip_empty_attention=True)
        assert tensor_storage_bytes(arena, device_type='cuda', skip=('weights', 'layer')) == wp['workspace']
        class CheckedBackend(FullModelBackend):
            samples = 0
            def sample(self, hidden, positions, sampling):
                result = super().sample(hidden, positions, sampling)
                local = torch.tensor(result, dtype=torch.int64, device='cuda')
                gathered = torch.empty((6*len(result),), dtype=torch.int64, device='cuda')
                dist.all_gather_into_tensor(gathered, local)
                assert torch.equal(gathered.view(6, -1), local.expand(6, -1)), 'Ranks sampled different tokens'
                self.samples += 1
                return result
        backend = CheckedBackend(model, caches, table, arena)
        tokenizer = Tokenizer.from_file(str(a.model/'tokenizer.json'))
        from tensorfold.families.glm_moe_dsa.graphs import GraphBackend
        from tensorfold.families.glm_moe_dsa.control import Replica,RequestController,start_command
        from tensorfold.families.glm_moe_dsa.control_sampling import RankZeroSampler
        from tensorfold.cuda.chat_template import ChatTemplate
        from glm53_tp6_profile_metrics import summarize
        from contextlib import nullcontext
        import statistics,gzip
        template=ChatTemplate(a.model)
        texts=[
            'Write a complete Python LRU cache with get and put methods, type annotations, clear documentation, and unit tests covering eviction and updating existing keys. Explain the time complexity.',
            'Explain how a transformer language model processes a prompt and generates an answer. Cover attention, key-value caching, speculative decoding, and the tradeoffs of splitting a model across several GPUs. Use clear paragraphs and concrete examples.',
        ]
        prompts=[tokenizer.encode(template.render([dict(role='user',content=t)],tools=None,enable_thinking=False),add_special_tokens=False).ids for t in texts]
        def serial(prompt,count):
            pool=CachePool(capacity);extent=pool.allocate(len(prompt)+count)
            hidden=backend.target(prompt,0,extent)
            result=backend.sample(hidden[-1:],[len(prompt)],None)
            for i in range(1,count):
                hidden=backend.target([result[-1]],len(prompt)+i-1,extent)
                result.extend(backend.sample(hidden,[len(prompt)+i],None))
            backend.synchronize();pool.release(extent)
            return result
        references=[]
        for i,prompt in enumerate(prompts):
            save(f'independent-serial-{i}')
            references.append(serial(prompt,128))
        report['fixtures']=[dict(text=t,prompt_tokens=len(p),prompt_sha256=digest(p),output_sha256=digest(ref)) for t,p,ref in zip(texts,prompts,references)]
        class Recorder:
            def __init__(self):
                self.enabled=self.trace=False
                self.events=[(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)) for _ in range(2048)]
                for pair in self.events:
                    for event in pair:event.record()
                torch.cuda.synchronize();self.reset()
            def reset(self):self.pending=[];self.host={}
            def call(self,name,fn,*,device=True):
                if not self.enabled and not self.trace:return fn()
                mark=torch.profiler.record_function('tfp19.'+name) if self.trace else nullcontext()
                with mark:
                    pair=None
                    if self.enabled and device:
                        assert len(self.pending)<len(self.events),'Bounded event pool exhausted'
                        pair=self.events[len(self.pending)];pair[0].record()
                    start=time.perf_counter()
                    try:return fn()
                    finally:
                        duration=time.perf_counter()-start
                        if self.enabled:
                            item=self.host.setdefault(name,dict(calls=0,total_s=0.))
                            item['calls']+=1;item['total_s']+=duration
                            if pair is not None:
                                pair[1].record();self.pending.append((name,pair))
            def flush(self):
                torch.cuda.synchronize();device={}
                for name,(start,stop) in self.pending:
                    item=device.setdefault(name,dict(calls=0,total_ms=0.))
                    item['calls']+=1;item['total_ms']+=start.elapsed_time(stop)
                return dict(host=self.host,device_stream_intervals=device,
                    caveat='CUDA event intervals include stream idle/launch/wait time; they are not pure kernel time. Host command/apply ranges nest and must not be added to their child operations. Timed throughput runs have instrumentation disabled.')
        recorder=Recorder()
        leader=RankZeroSampler(dist.group.WORLD,'cuda:0',17)
        def sampling(logits,positions,settings):
            return recorder.call(f'sample.r{len(positions)}',lambda:leader(logits,positions,settings))
        class MeasuredBackend(GraphBackend):
            def target(self,tokens,start,extent):
                return recorder.call(f'target.r{len(tokens)}',lambda:super(MeasuredBackend,self).target(tokens,start,extent))
            def mtp(self,tokens,hidden,start,extent):
                return recorder.call(f'mtp.r{len(tokens)}',lambda:super(MeasuredBackend,self).mtp(tokens,hidden,start,extent))
            def _execute(self,key,**kwargs):
                if key.operation=='head':
                    return recorder.call(f'head.r{key.rows}',lambda:super(MeasuredBackend,self)._execute(key,**kwargs))
                return super()._execute(key,**kwargs)
        class MeasuredReplica(Replica):
            def apply(self,prepared):
                return recorder.call('apply.'+prepared[0],lambda:super(MeasuredReplica,self).apply(prepared),device=False)
            def prepare(self,command):
                return recorder.call('prepare.'+command['op'],lambda:super(MeasuredReplica,self).prepare(command),device=False)
            def fingerprint(self):
                return recorder.call('fingerprint',lambda:super(MeasuredReplica,self).fingerprint(),device=False)
        adapter=MeasuredBackend(model,caches,table,arena,group=dist.group.WORLD,sampler=sampling,max_graphs=24,max_rows=9)
        active_component=adapter
        controller=RequestController(MeasuredReplica(RequestEngine(adapter)),store,a.rank,generation='tfp19_draft_profile',timeout=timedelta(seconds=240))
        def command(packet):
            def execute():
                if a.rank==0:return controller.dispatch(packet)
                controller.follow_once()
            return recorder.call('command.'+packet['op'],execute,device=False)
        sequence=0
        def run_case(fixture,depth,*,events=False,trace=False):
            nonlocal sequence
            sequence+=1;key=f'case-{sequence}';prompt=prompts[fixture]
            captures=adapter.captures
            command(start_command(key,prompt,max_tokens=128,draft_tokens=depth,ignore_eos=True))
            request=controller.replica.core.requests[key]
            begin=time.perf_counter()
            while request.status=='prefill':command(dict(op='step',args=dict(key=key,cancelled=False)))
            first=time.perf_counter();prefill=first-begin
            recorder.reset();recorder.enabled=events
            if trace:
                # Warm position-dependent dispatch on the same live request.
                for _ in range(8):command(dict(op='step',args=dict(key=key,cancelled=False)))
                assert request.status=='decode'
                torch.cuda.synchronize();dist.barrier()
                assert available()>=10*2**30
                recorder.trace=True
                try:
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                                                record_shapes=False,profile_memory=False,with_stack=False) as profile:
                        for _ in range(3):command(dict(op='step',args=dict(key=key,cancelled=False)))
                finally:recorder.trace=False
                path=a.output.parent/f'trace-fixture{fixture}-rank{a.rank}.json'
                profile.export_chrome_trace(str(path))
                raw=path.read_bytes();trace_summary=summarize(json.loads(raw))
                compressed=gzip.compress(raw);archive=path.with_suffix('.json.gz');archive.write_bytes(compressed)
                report.setdefault('traces',[]).append(dict(fixture=fixture,depth=depth,summary=trace_summary,raw_sha256=hashlib.sha256(raw).hexdigest(),archive=archive.name,archive_sha256=hashlib.sha256(compressed).hexdigest(),archive_bytes=len(compressed)))
                del profile,raw,compressed,trace_summary
                path.unlink()
            while request.status=='decode':command(dict(op='step',args=dict(key=key,cancelled=False)))
            decode=time.perf_counter()-first
            recorder.enabled=False
            profile_result=recorder.flush() if events else None
            result=dict(fixture=fixture,depth=depth,prefill_s=prefill,decode_s=decode,decode_tok_s=127/decode,
                        rounds=request.rounds,drafted=request.drafted,accepted=request.accepted,
                        accepted_fraction=request.accepted/request.drafted if request.drafted else None,
                        output_per_round=127/request.rounds,new_captures=adapter.captures-captures,
                        output_sha256=digest(request.output),profile=profile_result,trace=trace)
            check(request.output==references[fixture],f'case-{sequence}-f{fixture}-k{depth}-exact',exact_tokens=True)
            command(dict(op='drop',args=dict(key=key)))
            return result
        depths=[0,1,2,3,4,6,8]
        report['warmups']=[];report['runs']=[];report['profiles']=[]
        for depth in depths:
            for fixture in range(2):
                report['warmups'].append(run_case(fixture,depth));save('draft-warmup')
        for repeat in range(3):
            # Rotate depth order and alternate fixture order to reduce monotonic
            # warming/clock bias; every exact sequence stays the same.
            order=depths[repeat:]+depths[:repeat]
            if repeat%2:order=list(reversed(order))
            for depth in order:
                for fixture in ([0,1] if repeat%2==0 else [1,0]):
                    result=run_case(fixture,depth);result['repeat']=repeat
                    check(result['new_captures']==0,f'timed-case-{sequence}-no-capture')
                    report['runs'].append(result);save('draft-timed')
        report['medians']={str(f):{str(k):statistics.median(x['decode_tok_s'] for x in report['runs'] if x['fixture']==f and x['depth']==k) for k in depths} for f in range(2)}
        if a.rank==0:
            chosen={str(f):max(depths,key=lambda k:report['medians'][str(f)][str(k)]) for f in range(2)}
            store.set('tfp19_profile_selected',json.dumps(chosen))
        chosen=json.loads(store.get('tfp19_profile_selected'));report['selected_depths']=chosen
        save('draft-sweep-complete')
        for fixture in range(2):
            for depth in sorted({4,chosen[str(fixture)]}):
                result=run_case(fixture,depth,events=True);report['profiles'].append(result);save('event-profile-complete')
            result=run_case(fixture,4,trace=True);report.setdefault('traced_cases',[]).append(result);save('kernel-profile-complete')
        command(dict(op='close',args={}))
        check(controller.replica.closed and not controller.replica.core.pool.leases and adapter.closed and not adapter.entries,
              'ordered-graph-and-request-cleanup')
        active_component=None
        counts=[controller.completed,leader.decisions,adapter.captures,adapter.replays]
        local=torch.tensor(counts,dtype=torch.int64,device='cuda');all_counts=torch.empty(24,dtype=torch.int64,device='cuda')
        dist.all_gather_into_tensor(all_counts,local)
        check(torch.equal(all_counts.view(6,4),local.expand(6,4)),'all-rank-command-sampling-graph-counts')
        report.update(commands=counts[0],samples=counts[1],captures=counts[2],replays=counts[3],peak_graph_growth_bytes=adapter.peak_retained_growth,
                      cross_rank_reference_sample_checks=backend.samples)
        torch.cuda.synchronize();dist.barrier();dist.destroy_process_group()
        report.update(passed=True,communicator_destroyed=True,finished_at=time.time());save('complete')
    except Exception as exc:
        report['error'] = type(exc).__name__+': '+str(exc)
        save('failed')
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()
        try:
            if active_component is not None:
                active_component.close_graphs()
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()


if __name__ == '__main__':
    main()
