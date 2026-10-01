#!/usr/bin/env python3
"""Original-weight TP6 decode graphs and local HTTP timing/serial-output gate.

Uses original3.25bpw weights, resident804K cache, bounded independent CUDA graph
pools and the existing6-rank group. Checks changed inputs/physical offsets and
logical2048->4096 boundaries, then real chat-template HTTP C1/C4 and SSE against
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
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    torch.manual_seed(530618)
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
        gp=graph_reserve()
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
        from tensorfold.families.glm_moe_dsa.control import Replica,RequestController
        from tensorfold.families.glm_moe_dsa.control_sampling import RankZeroSampler
        from tensorfold.families.glm_moe_dsa.scheduler import RequestScheduler,ServingEngine
        from tensorfold.cuda.chat_template import ChatTemplate
        from tensorfold.cuda.server import App,Server,make_handler
        import urllib.request,urllib.error,statistics
        # Compare real graph replays with the eager full model, using different
        # physical bases and the 2048->4096 logical indexer-bound transition.
        graphed = GraphBackend(model,caches,table,arena,group=dist.group.WORLD)
        active_component = graphed
        for base,length,starts in [(0,64,[31,32]),(440000,2055,[2047,2048])]:
            extent = Extent(base,360000,1+base)
            prefix = [(1000+17*i)%154880 for i in range(length)]
            save(f'graph-component-prefix-{length}')
            hidden = backend.target(prefix,0,extent).clone()
            backend.mtp(prefix[1:],hidden[:-1],0,extent)
            for start in starts:
                for n in range(1,6):
                    tokens = [(3500+31*i+start)%154880 for i in range(n)]
                    expected_hidden = backend.target(tokens,start,extent).clone()
                    actual_hidden = graphed.target(tokens,start,extent).clone()
                    check(torch.equal(actual_hidden,expected_hidden),f'target-graph-b{base}-p{start}-r{n}',exact_tensor=True)
                    expected_logits = model.logits(expected_hidden,arena).clone()
                    observed = []
                    graphed.sampler = lambda logits,positions,sampling: observed.append(logits.clone()) or [0]*len(positions)
                    graphed.sample(actual_hidden,list(range(start+1,start+n+1)),None)
                    check(torch.equal(observed[0],expected_logits),f'head-graph-b{base}-p{start}-r{n}',exact_tensor=True)
                next_token = [prefix[start+1]]
                expected_mtp = backend.mtp(next_token,hidden[start:start+1],start,extent).clone()
                actual_mtp = graphed.mtp(next_token,hidden[start:start+1],start,extent).clone()
                check(torch.equal(actual_mtp,expected_mtp),f'mtp-graph-b{base}-p{start}',exact_tensor=True)
        report['component_graph_state'] = graphed.control_state()
        report['component_peak_graph_growth_bytes'] = graphed.peak_retained_growth
        check(graphed.captures>0 and graphed.replays>graphed.captures and len(graphed.entries)<=16,
              'bounded-graph-reuse',captures=graphed.captures,replays=graphed.replays,evictions=graphed.evictions)
        graphed.close_graphs()
        check(graphed.closed and not graphed.entries,'component-graphs-released')
        active_component = None
        del graphed,hidden,expected_hidden,actual_hidden,expected_logits,observed,expected_mtp,actual_mtp
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
        for mode in ('eager','graphs'):
            torch.cuda.synchronize();dist.barrier()
            controllers,adapters,samplers = [],[],[]
            def factory():
                torch.cuda.set_device(0)
                sampler = RankZeroSampler(dist.group.WORLD,'cuda:0',17)
                klass = GraphBackend if mode=='graphs' else FullModelBackend
                extra = dict(group=dist.group.WORLD) if mode=='graphs' else {}
                adapter = klass(model,caches,table,arena,sampler=sampler,**extra)
                controller = RequestController(Replica(RequestEngine(adapter)),store,a.rank,
                    generation='original_weights_tfp18_http_'+mode,timeout=timedelta(seconds=240),
                    idle_timeout=timedelta(seconds=900))
                controllers.append(controller);adapters.append(adapter);samplers.append(sampler)
                return controller
            save('http-mode-'+mode)
            data = dict(c1=[],c4=None)
            if a.rank==0:
                scheduler = RequestScheduler(factory)
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
                    begin=time.perf_counter()
                    with ThreadPoolExecutor(max_workers=4) as clients:
                        concurrent=list(clients.map(client,range(4)))
                    elapsed=time.perf_counter()-begin
                    data['c4']=dict(clients=concurrent,wall_s=elapsed,output_tokens=512,aggregate_tok_s=512/elapsed)
                    check(True,f'{mode}-http-four-client-serial-parity',exact_sequences=4)
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
            data.update(commands=counts[0],samples=counts[1],captures=counts[2],replays=counts[3],
                        peak_graph_growth_bytes=getattr(adapter,'peak_retained_growth',0))
            report['modes'][mode]=data
            save(mode+'-complete')
        if a.rank==0:
            check(report['modes']['eager']['followup_tokens']==report['modes']['graphs']['followup_tokens'],
                  'retained-chat-followup-eager-graph-parity',exact_tokens=True)
            for mode,data in report['modes'].items():
                data['median_decode_tok_s']={str(i):statistics.median(x['decode_tok_s'] for x in data['c1'] if x['fixture']==i)
                                             for i in range(2)}
                check(all(x['new_captures']==0 for x in data['c1']),mode+'-timed-c1-no-new-captures')
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
