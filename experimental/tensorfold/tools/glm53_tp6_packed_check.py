#!/usr/bin/env python3
"""Original-weight packed TP6 request execution and matched HTTP C1/C4 gate.

Uses original3.25bpw weights, resident804K cache, bounded independent CUDA graph
pools and the existing6-rank group. Checks changed inputs/physical offsets and
mixed request lengths and logical2048->4096 boundaries, then real HTTP C1/C4/SSE against
independent serial targets. Short-context diagnostic; not production promotion.
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
    from tensorfold.families.glm_moe_dsa.packed_backend import packed_reserve
    from tensorfold.families.glm_moe_dsa.projection_plan import LinearTile,ProjectionPlan
    from tensorfold.families.glm_moe_dsa.request_ops import Call
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    torch.manual_seed(530621)
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
        gp=graph_reserve(64,32)
        extra=packed_reserve(3072)
        report['packed_temporary_reserve']=extra
        assert plan['total']+wp['total']+rp['total']+gp['total']+extra+4*2**30 < 104*2**30
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
        model = FullModel(weights, reduction, projections=ProjectionPlan(LinearTile(16,64,3),LinearTile(64,128,2)))
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
        texts = [
            'Write a Python function that returns the sum of the squares of a list of integers.\n',
            'Explain in simple terms why leaves change color in autumn.\n',
            'The first five prime numbers are',
            'A careful programmer checks edge cases because',
        ]
        prompts = [tokenizer.encode(t, add_special_tokens=False).ids for t in texts]
        report['fixtures'] = [dict(text=t, tokens=p) for t, p in zip(texts, prompts)]
        samplings = [None, Sampling(5316, .8, 20, .95), Sampling(619, .9, 0, .9, .03), None]
        def serial(prompt, count, sampling):
            # Independent loop: no RequestEngine, MTP state or speculative commits.
            pool = CachePool(capacity)
            extent = pool.allocate(len(prompt)+count)
            hidden = backend.target(prompt, 0, extent)
            result = backend.sample(hidden[-1:], [len(prompt)], sampling)
            for i in range(1, count):
                hidden = backend.target([result[-1]], len(prompt)+i-1, extent)
                result.extend(backend.sample(hidden, [len(prompt)+i], sampling))
            backend.synchronize()
            pool.release(extent)
            return result
        expected = []
        for i, (prompt, sampling) in enumerate(zip(prompts, samplings)):
            save(f'serial-reference-{i}')
            expected.append(serial(prompt, 12, sampling))
        from tensorfold.families.glm_moe_dsa.graphs import GraphBackend
        from tensorfold.families.glm_moe_dsa.request import Extent
        from tensorfold.families.glm_moe_dsa.control import Replica,RequestController,start_command
        from tensorfold.families.glm_moe_dsa.control_sampling import RankZeroSampler
        from tensorfold.families.glm_moe_dsa.scheduler import RequestScheduler,ServingEngine
        from tensorfold.cuda.chat_template import ChatTemplate
        from tensorfold.cuda.server import App,Server,make_handler
        import urllib.request,urllib.error,statistics
        # One existing arena, distinct physical leases and logical contexts.
        graphed=GraphBackend(model,caches,table,arena,group=dist.group.WORLD,max_graphs=64,max_rows=32)
        active_component=graphed
        extents=[Extent(i*200000,4096,i+1) for i in range(4)]
        lengths=[17,33,2049,65];last_hidden=[]
        for i,(extent,n) in enumerate(zip(extents,lengths)):
            prefix=[(1000+i*31+j*17)%154880 for j in range(n)]
            hidden=backend.target(prefix,0,extent).clone()
            backend.mtp(prefix[1:],hidden[:-1],0,extent)
            last_hidden.append(hidden[-1:].clone())
        calls=[Call('target',([3500+i*31+j for j in range(5)],n,e)) for i,(n,e) in enumerate(zip(lengths,extents))]
        expected_hidden=[backend.target(*c.args).clone() for c in calls]
        packed=graphed.batch(calls)
        for i,(got,want) in enumerate(zip(packed,expected_hidden)):
            check(torch.equal(got,want),f'packed-target20-request{i}',exact_tensor=True)
        # Change order and physical bases without changing total captured rows.
        changed=[Call('target',([4800+i*19+j for j in range(5)],lengths[i],extents[i])) for i in (3,1,0,2)]
        wants=[backend.target(*c.args).clone() for c in changed]
        for i,(got,want) in enumerate(zip(graphed.batch(changed),wants)):
            check(torch.equal(got,want),f'changed-packed-target20-request{i}',exact_tensor=True)
        # Restore those target rows before deriving the next canonical MTP input.
        packed=graphed.batch(calls)
        mtps=[Call('mtp',([c.args[0][0]],h,n-1,e)) for c,h,n,e in zip(calls,last_hidden,lengths,extents)]
        wants=[backend.mtp(*c.args).clone() for c in mtps]
        for i,(got,want) in enumerate(zip(graphed.batch(mtps),wants)):
            check(torch.equal(got,want),f'packed-mtp4-request{i}',exact_tensor=True)
        canonical=[Call('mtp',(c.args[0][1:4],h[:3],n,e)) for c,h,n,e in zip(calls,packed,lengths,extents)]
        wants=[backend.mtp(*c.args).clone() for c in canonical]
        for i,(got,want) in enumerate(zip(graphed.batch(canonical),wants)):
            check(torch.equal(got,want),f'packed-canonical-mtp12-request{i}',exact_tensor=True)
        expected_logits=[model.logits(h[:2].contiguous(),arena).clone() for h in packed]
        observed=[]
        graphed.sampler=lambda logits,positions,sampling: observed.append(logits.clone()) or [0]*len(positions)
        graphed.batch([Call('sample',(h[:2],[n+1,n+2],None)) for h,n in zip(packed,lengths)])
        for i,(got,want) in enumerate(zip(observed,expected_logits)):
            check(torch.equal(got,want),f'packed-head8-request{i}',exact_tensor=True)
        report['component_graph_state']=graphed.control_state()
        graphed.close_graphs();active_component=None
        del graphed,hidden,last_hidden,expected_hidden,expected_logits,observed,packed,wants,calls,changed,mtps,canonical
        torch.cuda.synchronize();torch.cuda.empty_cache();dist.barrier()
        # Mixed sampling/depths, cancellation, changed membership and retained
        # continuation over the actual six-rank command bus.
        follow_prompt=prompts[0]+expected[0][:-1]+[5,6]
        follow_expected=serial(follow_prompt,8,None)
        leader=RankZeroSampler(dist.group.WORLD,'cuda:0',17)
        adapter=GraphBackend(model,caches,table,arena,group=dist.group.WORLD,sampler=leader,max_graphs=64,max_rows=32)
        active_component=adapter
        controller=RequestController(Replica(RequestEngine(adapter)),store,a.rank,generation='tfp21_packed_mixed_qualification',timeout=timedelta(seconds=240))
        def command(packet):
            if a.rank==0:return controller.dispatch(packet)
            controller.follow_once()
        for i in range(4):command(start_command(str(i),prompts[i],max_tokens=12,sampling=samplings[i],draft_tokens=[0,1,4,8][i],ignore_eos=True))
        command(dict(op='step_many',args=dict(keys=['0','1','2','3'],cancelled=[False,False,False,True])))
        check(controller.replica.core.requests['3'].status=='cancelled' and not controller.replica.core.requests['3'].output,'cancelled-member-not-committed')
        command(dict(op='drop',args=dict(key='3')))
        command(start_command('3b',prompts[3],max_tokens=12,sampling=samplings[3],draft_tokens=8,ignore_eos=True))
        while True:
            keys=[k for k,r in controller.replica.core.requests.items() if r.status in ('prefill','decode')]
            if not keys:break
            command(dict(op='step_many',args=dict(keys=list(reversed(keys)),cancelled=[False]*len(keys))))
        for i,key in enumerate(['0','1','2','3b']):
            check(controller.replica.core.requests[key].output==expected[i],f'mixed-packed-request-{i}-serial',exact_tokens=True)
        for key in ('1','2','3b'):command(dict(op='drop',args=dict(key=key)))
        kept=len(controller.replica.core.requests['0'].tokens)
        command(start_command('continued',follow_prompt,max_tokens=8,draft_tokens=4,ignore_eos=True,resume='0'))
        while controller.replica.core.requests['continued'].status in ('prefill','decode'):
            command(dict(op='step_many',args=dict(keys=['continued'],cancelled=[False])))
        check(controller.replica.core.requests['continued'].output==follow_expected and controller.replica.core.requests['continued'].reused_tokens==kept,'packed-retained-prefix-serial',exact_tokens=True)
        report['mixed_batch_counts']=adapter.batch_counts
        check(all(v['multi_request_groups']>0 for v in adapter.batch_counts.values()),'target-mtp-head-actually-packed')
        command(dict(op='close',args={}))
        check(controller.replica.closed and not controller.replica.core.pool.leases and adapter.closed,'mixed-packed-clean-close')
        active_component=None
        del controller,adapter,leader
        torch.cuda.synchronize();torch.cuda.empty_cache();dist.barrier()
        template = ChatTemplate(a.model)
        chat_texts = [
            'Write a complete Python LRU cache with get and put methods, type annotations, clear documentation, and unit tests covering eviction and updating existing keys. Explain the time complexity.',
            'Explain how a transformer language model processes a prompt and generates an answer. Cover attention, key-value caching, speculative decoding, and the tradeoffs of splitting a model across several GPUs. Use clear paragraphs and concrete examples.',
        ]
        bodies = [dict(model='glm-5.3',messages=[dict(role='user',content=text)],temperature=0,
                       max_tokens=128,ignore_eos=True,return_token_ids=True,
                       chat_template_kwargs=dict(enable_thinking=False)) for text in chat_texts]
        chat_prompts = [tokenizer.encode(template.render(b['messages'],tools=None,enable_thinking=False),
                                        add_special_tokens=False).ids for b in bodies]
        references = []
        for i,prompt in enumerate(chat_prompts):
            save(f'chat-independent-serial-{i}')
            references.append(serial(prompt,128,None))
        report['chat_fixtures'] = [dict(body=b,prompt_tokens=len(p),prompt_sha256=digest(p),
                                      output_sha256=digest(ref)) for b,p,ref in zip(bodies,chat_prompts,references)]
        report['modes'] = {}
        for mode in ('scalar','packed'):
            torch.cuda.synchronize();dist.barrier()
            controllers,adapters,samplers = [],[],[]
            def factory():
                torch.cuda.set_device(0)
                sampler = RankZeroSampler(dist.group.WORLD,'cuda:0',17)
                klass = GraphBackend
                extra = dict(group=dist.group.WORLD,max_graphs=64,max_rows=32)
                adapter = klass(model,caches,table,arena,sampler=sampler,**extra)
                controller = RequestController(Replica(RequestEngine(adapter)),store,a.rank,
                    generation='original_weights_tfp21_http_'+mode,timeout=timedelta(seconds=240),
                    idle_timeout=timedelta(seconds=900))
                controllers.append(controller);adapters.append(adapter);samplers.append(sampler)
                return controller
            save('http-mode-'+mode)
            data = dict(c1=[],c4=[],c4_warmups=[])
            if a.rank==0:
                scheduler = RequestScheduler(factory,packed=mode=='packed')
                engine = ServingEngine(scheduler)
                server = thread = None
                try:
                    app = App(engine,a.model,'glm-5.3',default_thinking=False,max_tokens=128,context_window=360000)
                    server = Server(('127.0.0.1',0),make_handler(app))
                    thread = threading.Thread(target=server.serve_forever,daemon=True);thread.start()
                    endpoint = 'http://127.0.0.1:'+str(server.server_address[1])
                    def post(body):
                        request = urllib.request.Request(endpoint+'/v1/chat/completions',data=json.dumps(body).encode(),
                            headers={'Content-Type':'application/json'})
                        begin=time.perf_counter()
                        with urllib.request.urlopen(request,timeout=180) as response:
                            payload=json.load(response)
                        elapsed=time.perf_counter()-begin
                        assert 'error' not in payload,payload.get('error')
                        return payload,elapsed
                    def validate(payload,i):
                        return (payload['tensorfold']['token_ids']==references[i]
                            and payload['usage']['completion_tokens']==128
                            and payload['usage']['prompt_tokens']==len(chat_prompts[i]))
                    expected_content = {}
                    for i,body in enumerate(bodies):
                        payload,_=post(body)
                        expected_content[i] = payload['choices'][0]['message'].get('content') or ''
                        check(validate(payload,i),f'{mode}-http-warmup-{i}-serial-parity',exact_tokens=True)
                    for repeat in range(3):
                        for i,body in enumerate(bodies):
                            captures=getattr(adapters[0],'captures',0)
                            payload,elapsed=post(body)
                            check(validate(payload,i),f'{mode}-http-c1-{i}-{repeat}-serial-parity',exact_tokens=True)
                            stats=payload['tensorfold']
                            data['c1'].append(dict(fixture=i,repeat=repeat,output_tokens=128,elapsed_s=elapsed,
                                decode_tok_s=127/stats['decode_s'],prefill_s=stats['prefill_s'],decode_s=stats['decode_s'],
                                new_captures=getattr(adapters[0],'captures',0)-captures,output_sha256=digest(stats['token_ids'])))
                    barrier=threading.Barrier(4)
                    def client(i):
                        barrier.wait(10)
                        payload,elapsed=post(bodies[i%2])
                        assert validate(payload,i%2),'Concurrent HTTP output differs from independent serial target'
                        return dict(fixture=i%2,elapsed_s=elapsed,output_sha256=digest(payload['tensorfold']['token_ids']))
                    for repeat in range(5):
                        # Two warmups, then three timed rounds; all capture
                        # counts are reported, including any dynamic new shape.
                        captures=adapters[0].captures
                        begin=time.perf_counter()
                        with ThreadPoolExecutor(max_workers=4) as clients:
                            concurrent=list(clients.map(client,range(4)))
                        elapsed=time.perf_counter()-begin
                        item=dict(clients=concurrent,wall_s=elapsed,output_tokens=512,
                                  aggregate_tok_s=512/elapsed,new_captures=adapters[0].captures-captures)
                        data['c4_warmups' if repeat<2 else 'c4'].append(item)
                        check(True,f'{mode}-http-four-client-{repeat}-serial-parity',exact_sequences=4)
                    # Real SSE route: collect final token IDs and visible content.
                    request=urllib.request.Request(endpoint+'/v1/chat/completions',data=json.dumps({**bodies[0],'stream':True}).encode(),
                                                   headers={'Content-Type':'application/json'})
                    events=[];done=False
                    with urllib.request.urlopen(request,timeout=180) as response:
                        for line in response:
                            if not line.startswith(b'data: '):continue
                            raw=line[6:].strip()
                            if raw==b'[DONE]':done=True;break
                            event=json.loads(raw);assert 'error' not in event,event;events.append(event)
                    terminal=next(e for e in reversed(events) if 'tensorfold' in e)
                    content=''.join(c.get('delta',{}).get('content') or '' for e in events for c in e.get('choices',[]))
                    check(done and validate(terminal,0) and content==expected_content[0],
                          f'{mode}-http-stream-serial-parity',exact_tokens=True,stream_content_parity=True)
                    before=controllers[0].completed
                    rejected=False
                    try:post({**bodies[0],'max_tokens':360001})
                    except urllib.error.HTTPError as exc:
                        rejected=exc.code==400
                        exc.close()
                    check(rejected and controllers[0].completed==before,f'{mode}-http-context-rejection')
                    # Existing core/controller cancellation and retained-prefix
                    # mechanisms are exercised again with graph execution enabled.
                    follow_prompt=chat_prompts[0]+references[0]+tokenizer.encode('\nContinue with edge cases.\n',add_special_tokens=False).ids
                    follow=[]
                    stats=engine.generate(follow_prompt,8,None,lambda t:follow.extend(t),stop_eos=False)
                    check(stats['cached']==len(chat_prompts[0])+127,f'{mode}-retained-chat-prefix',cached=stats['cached'])
                    data['followup_tokens']=follow
                    cancelled=engine.generate(chat_prompts[1]*8,12,None,lambda _:True)
                    check(cancelled['reason']=='cancelled',f'{mode}-callback-cancellation')
                finally:
                    try:
                        if server is not None:
                            if thread is not None and thread.is_alive():server.shutdown()
                            server.server_close()
                        if thread is not None:thread.join(10)
                    finally:
                        engine.close()
            else:
                factory().follow()
            controller,adapter=controllers[0],adapters[0]
            check(controller.replica.closed and not controller.replica.core.pool.leases,
                  mode+'-ordered-clean-shutdown')
            counts=[controller.completed,samplers[0].decisions,getattr(adapter,'captures',0),getattr(adapter,'replays',0)]
            local=torch.tensor(counts,dtype=torch.int64,device='cuda');all_counts=torch.empty(24,dtype=torch.int64,device='cuda')
            dist.all_gather_into_tensor(all_counts,local)
            check(torch.equal(all_counts.view(6,4),local.expand(6,4)),mode+'-all-rank-execution-counts')
            data.update(batch_counts=adapter.batch_counts,commands=counts[0],samples=counts[1],captures=counts[2],replays=counts[3],
                        peak_graph_growth_bytes=getattr(adapter,'peak_retained_growth',0))
            report['modes'][mode]=data
            save(mode+'-complete')
        if a.rank==0:
            check(report['modes']['scalar']['followup_tokens']==report['modes']['packed']['followup_tokens'],
                  'retained-chat-followup-scalar-packed-parity',exact_tokens=True)
            for mode,data in report['modes'].items():
                data['median_decode_tok_s']={str(i):statistics.median(x['decode_tok_s'] for x in data['c1'] if x['fixture']==i)
                                             for i in range(2)}
                data['median_c4_aggregate_tok_s']=statistics.median(x['aggregate_tok_s'] for x in data['c4'])
                check(all(x['new_captures']==0 for x in data['c1']),mode+'-timed-c1-no-new-captures')
        if a.rank==0:
            check(all(v['multi_request_groups']>0 for v in report['modes']['packed']['batch_counts'].values()),'http-target-mtp-head-actually-packed')
        report['fixture_output_sha256']=[digest(t) for t in expected]
        report['cross_rank_reference_sample_checks']=backend.samples
        save('numerical_checks_complete')
        torch.cuda.synchronize()
        dist.barrier()
        dist.destroy_process_group()
        report.update(passed=True, communicator_destroyed=True, finished_at=time.time())
        save('complete')
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
