#!/usr/bin/env python3
"""Original-weight BF16 projection tile screening and full TP6 qualification.

Microbenchmarks select a candidate only when every rank is exactly equal to
the unchanged reference kernel. Then full target/MTP tensors, resident804K
cache and direct-controller MTP4 requests qualify the selected immutable plan.
No weights change; component timing is not a serving-throughput claim.
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
    torch.manual_seed(530620)
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
    active_components = []
    try:
        store = dist.TCPStore(os.environ['MASTER_ADDR'], a.port, 6, a.rank == 0, timedelta(seconds=240))
        dist.init_process_group('nccl', store=store, rank=a.rank, world_size=6, timeout=timedelta(seconds=240))
        reader = RankPieces(a.model, a.rank)
        plan = weight_plan(reader)
        wp = workspace_plan(reader.config, a.rank, 3072, capacity, logit_rows=17, expert_chunk_rows=1024,
                            bulk_min_rows=256, attention_part_rows=128, skip_empty_attention=True)
        rp = request_plan(3072, 17)
        gp=graph_reserve(12,5,256*2**20)
        gp=dict(per_backend=gp,total=2*gp['total'])
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
        from glm53_tp6_projection_screen import screen, choose, GENERATIONS
        from tensorfold.families.glm_moe_dsa.projection_plan import LinearTile,ProjectionPlan,REFERENCE_PLAN
        from dataclasses import asdict
        local_screen=screen(weights,report,save)
        store.set(f'tfp20_screen_{a.rank}',json.dumps(local_screen))
        gathered=[json.loads(store.get(f'tfp20_screen_{rank}')) for rank in range(6)]
        selected=choose(gathered)
        candidate=ProjectionPlan(**{k:LinearTile(**v) for k,v in selected.items()})
        report.update(projection_screen=local_screen,selected_plan=asdict(candidate))
        check(all(r['rank']==i for i,r in enumerate(gathered)),'six-rank-screen-agreement')
        models={'reference':FullModel(weights,reduction),'candidate':FullModel(weights,reduction,projections=candidate)}
        model=models['reference']
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
        import statistics
        def exact(value,reference,name):
            check(torch.equal(value,reference),name,exact_tensor=True,
                  max_abs=float((value.float()-reference.float()).abs().max()))
        ids=(torch.arange(3072,device='cuda',dtype=torch.int64)*137+1000)%154880
        pos=torch.arange(3072,device='cuda');base=torch.zeros_like(pos)
        full_reference={};report['bulk_runs']=[]
        def target(m,tokens=ids,positions=pos,bases=base):
            return m.target_forward(tokens,positions,bases,positions,caches[:78],table,arena,
                                    scope=object(),visible_tokens=len(tokens))
        for name,m in models.items():
            save(name+'-full-model-qualify')
            full=target(m).clone()
            logits=m.logits(full[-1:].contiguous(),arena).clone()
            changed=target(m,(ids[:257]+37)%154880,pos[:257],base[:257]).clone()
            short=target(m,ids[:17],pos[:17],base[:17]).clone()
            draft=m.mtp_forward((ids[:257]+19)%154880,full[:257].contiguous(),pos[:257],base[:257],pos[:257],
                                caches[78],table,arena,scope=object(),visible_tokens=257).clone()
            draft_logits=m.logits(draft[-1:].contiguous(),arena).clone()
            values=dict(target3072=full,logits=logits,changed257=changed,short17=short,mtp257=draft,mtp_logits=draft_logits)
            if name=='reference':full_reference={k:v.clone() for k,v in values.items()}
            for key,value in values.items():exact(value,full_reference[key],name+'-'+key)
            del values,full,logits,changed,short,draft,draft_logits
            target(m);torch.cuda.synchronize();dist.barrier()
        # Alternate order; compare all timed outputs too, outside timed region.
        for repeat in range(3):
            for name in (['reference','candidate'] if repeat%2==0 else ['candidate','reference']):
                torch.cuda.synchronize();dist.barrier();begin=time.perf_counter()
                output=target(models[name]);torch.cuda.synchronize()
                elapsed=time.perf_counter()-begin
                local=torch.tensor([elapsed],dtype=torch.float64,device='cuda')
                dist.all_reduce(local,op=dist.ReduceOp.MAX)
                exact(output,full_reference['target3072'],f'{name}-timed-bulk-{repeat}')
                report['bulk_runs'].append(dict(mode=name,repeat=repeat,seconds=local.item()))
                save('bulk-timed')
        report['bulk_medians']={name:statistics.median(r['seconds'] for r in report['bulk_runs'] if r['mode']==name) for name in models}
        del full_reference,ids,pos,base,output
        torch.cuda.synchronize();torch.cuda.empty_cache()
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
            save(f'independent-serial-{i}');references.append(serial(prompt,128))
        report['fixtures']=[dict(text=t,prompt_tokens=len(p),prompt_sha256=digest(p),output_sha256=digest(ref)) for t,p,ref in zip(texts,prompts,references)]
        adapters={};controllers={};leaders={}
        for name,m in models.items():
            leader=RankZeroSampler(dist.group.WORLD,'cuda:0',17)
            adapter=GraphBackend(m,caches,table,arena,group=dist.group.WORLD,sampler=leader,
                                 max_graphs=12,max_rows=5,capture_bytes=256*2**20)
            active_components.append(adapter)
            controller=RequestController(Replica(RequestEngine(adapter)),store,a.rank,
                                         generation=GENERATIONS[name],timeout=timedelta(seconds=240))
            adapters[name]=adapter;controllers[name]=controller;leaders[name]=leader
        def command(name,packet):
            controller=controllers[name]
            if a.rank==0:return controller.dispatch(packet)
            controller.follow_once()
        sequence=0
        def run_case(name,fixture):
            nonlocal sequence
            sequence+=1;key=f'case-{sequence}';adapter=adapters[name];controller=controllers[name]
            captures=adapter.captures
            command(name,start_command(key,prompts[fixture],max_tokens=128,draft_tokens=4,ignore_eos=True))
            request=controller.replica.core.requests[key]
            begin=time.perf_counter()
            while request.status=='prefill':command(name,dict(op='step',args=dict(key=key,cancelled=False)))
            first=time.perf_counter();prefill=first-begin
            while request.status=='decode':command(name,dict(op='step',args=dict(key=key,cancelled=False)))
            decode=time.perf_counter()-first
            result=dict(mode=name,fixture=fixture,prefill_s=prefill,decode_s=decode,decode_tok_s=127/decode,
                        rounds=request.rounds,drafted=request.drafted,accepted=request.accepted,
                        new_captures=adapter.captures-captures,output_sha256=digest(request.output))
            check(request.output==references[fixture],f'case-{sequence}-{name}-f{fixture}-exact',exact_tokens=True)
            command(name,dict(op='drop',args=dict(key=key)))
            return result
        report['warmups']=[];report['runs']=[]
        for name in models:
            for fixture in range(2):
                report['warmups'].append(run_case(name,fixture));save('request-warmup')
        for repeat in range(3):
            for name in (['reference','candidate'] if repeat%2==0 else ['candidate','reference']):
                for fixture in ([0,1] if repeat%2==0 else [1,0]):
                    result=run_case(name,fixture);result['repeat']=repeat
                    check(result['new_captures']==0,f'timed-case-{sequence}-no-capture')
                    report['runs'].append(result);save('request-timed')
        report['medians']={str(f):{name:statistics.median(x['decode_tok_s'] for x in report['runs'] if x['fixture']==f and x['mode']==name) for name in models} for f in range(2)}
        report['execution_counts']={}
        for name,adapter in adapters.items():
            command(name,dict(op='close',args={}))
            controller=controllers[name]
            check(controller.replica.closed and not controller.replica.core.pool.leases and adapter.closed and not adapter.entries,
                  name+'-ordered-graph-and-request-cleanup')
            counts=[controller.completed,leaders[name].decisions,adapter.captures,adapter.replays]
            local=torch.tensor(counts,dtype=torch.int64,device='cuda');all_counts=torch.empty(24,dtype=torch.int64,device='cuda')
            dist.all_gather_into_tensor(all_counts,local)
            check(torch.equal(all_counts.view(6,4),local.expand(6,4)),name+'-all-rank-command-sampling-graph-counts')
            report['execution_counts'][name]=dict(commands=counts[0],samples=counts[1],captures=counts[2],replays=counts[3],peak_graph_growth_bytes=adapter.peak_retained_growth)
        active_components.clear()
        report['cross_rank_reference_sample_checks']=backend.samples
        torch.cuda.synchronize();dist.barrier();dist.destroy_process_group()
        report.update(passed=True,communicator_destroyed=True,finished_at=time.time());save('complete')
    except Exception as exc:
        report['error'] = type(exc).__name__+': '+str(exc)
        save('failed')
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()
        try:
            for component in active_components:component.close_graphs()
        finally:
            if dist.is_initialized():dist.destroy_process_group()


if __name__ == '__main__':
    main()
