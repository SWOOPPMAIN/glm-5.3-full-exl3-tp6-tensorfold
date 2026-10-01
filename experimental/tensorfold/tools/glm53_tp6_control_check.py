#!/usr/bin/env python3
"""Original-weight TP6 transport, leader sampling and concurrent scheduler gate.

Independent serial target output versus real concurrent clients over TCPStore
commands and one existing NCCL group. Uses original 3.25bpw weights and resident
804K cache. Eager transport qualification; no production serving-speed claim.
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
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    torch.manual_seed(530617)
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
    try:
        store = dist.TCPStore(os.environ['MASTER_ADDR'], a.port, 6, a.rank == 0, timedelta(seconds=240))
        dist.init_process_group('nccl', store=store, rank=a.rank, world_size=6, timeout=timedelta(seconds=240))
        reader = RankPieces(a.model, a.rank)
        plan = weight_plan(reader)
        wp = workspace_plan(reader.config, a.rank, 3072, capacity, logit_rows=17, expert_chunk_rows=1024,
                            bulk_min_rows=256, attention_part_rows=128, skip_empty_attention=True)
        rp = request_plan(3072, 17)
        assert plan['total']+wp['total']+rp['total']+4*2**30 < 104*2**30
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
        from tensorfold.families.glm_moe_dsa.control import Replica, RequestController
        from tensorfold.families.glm_moe_dsa.control_sampling import RankZeroSampler
        from tensorfold.families.glm_moe_dsa.scheduler import RequestScheduler, ServingEngine
        follow_prompt = prompts[0]+expected[0]+tokenizer.encode('\nNow add an example.\n', add_special_tokens=False).ids
        follow_expected = serial(follow_prompt, 8, samplings[1])
        torch.cuda.synchronize()
        dist.barrier()
        controllers, samplers = [], []
        def factory():
            # Created in the actual model worker, after main-thread load/reference
            # writes finished. CUDA default-stream ownership stays explicit.
            torch.cuda.set_device(0)
            sampler = RankZeroSampler(dist.group.WORLD, 'cuda:0', 17)
            adapter = FullModelBackend(model, caches, table, arena, sampler=sampler)
            core = RequestEngine(adapter)
            controller = RequestController(Replica(core), store, a.rank,
                generation='original_weights_tfp17_control_v1', timeout=timedelta(seconds=240),
                idle_timeout=timedelta(seconds=900))
            controllers.append(controller)
            samplers.append(sampler)
            return controller
        save('distributed-request-clients')
        if a.rank == 0:
            scheduler = RequestScheduler(factory)
            engine = ServingEngine(scheduler)
            start = threading.Barrier(4)
            def client(i):
                tokens = []
                start.wait(10)
                stats = engine.generate(prompts[i], 12, samplings[i], lambda t: tokens.extend(t), stop_eos=False)
                return dict(tokens=tokens, stats=stats)
            try:
                begin = time.perf_counter()
                with ThreadPoolExecutor(max_workers=4) as clients:
                    got = list(clients.map(client, range(4)))
                report['diagnostic_concurrent_seconds'] = time.perf_counter()-begin
                for i, item in enumerate(got):
                    check(item['tokens'] == expected[i], f'concurrent-client-serial-parity-{i}',
                          exact_tokens=True, expected_tokens=expected[i], actual_tokens=item['tokens'], stats=item['stats'])
                completed = controllers[0].completed
                time.sleep(.1)
                check(controllers[0].completed == completed, 'idle-no-gpu-command')
                rejected = False
                try:
                    engine.generate([154880], 3)
                except ValueError:
                    rejected = True
                check(rejected and controllers[0].completed == completed, 'invalid-client-no-rank-wakeup')
                follow = []
                stats = engine.generate(follow_prompt, 8, samplings[1], lambda t: follow.extend(t), stop_eos=False)
                check(follow == follow_expected, 'retained-followup-serial-parity', exact_tokens=True,
                      expected_tokens=follow_expected, actual_tokens=follow, stats=stats)
                check(stats['cached'] == len(prompts[0])+11, 'retained-prefix-reused', cached=stats['cached'])
                # CPU callback observes disconnect while one long prefill is live;
                # cancellation is then frozen in the next all-rank step command.
                cancelled = []
                def disconnect(tokens):
                    cancelled.extend(tokens)
                    return True
                stats = engine.generate(prompts[1]*40, 12, samplings[1], disconnect)
                check(stats['reason'] == 'cancelled', 'client-cancellation-coordinated', stats=stats,
                      callback_token_count=len(cancelled))
                after = []
                engine.generate(prompts[2], 12, samplings[2], lambda t: after.extend(t), stop_eos=False)
                check(after == expected[2], 'healthy-client-after-cancellation', exact_tokens=True,
                      expected_tokens=expected[2], actual_tokens=after)
                report['leader_client_checks'] = len(report['cases'])
            finally:
                engine.close()
        else:
            factory().follow()
        controller = controllers[0]
        core = controller.replica.core
        check(controller.replica.closed and not core.requests and not core.pool.leases
              and core.pool.free == [(0, capacity)], 'close-reclaims-all-extents')
        check(controller.broken is None and controller.pending is None, 'transport-healthy-terminal')
        report['commands_completed'] = controller.completed
        report['leader_sampling_decisions'] = samplers[0].decisions
        counts = torch.tensor([controller.completed, samplers[0].decisions], dtype=torch.int64, device='cuda')
        gathered = torch.empty((12,), dtype=torch.int64, device='cuda')
        dist.all_gather_into_tensor(gathered, counts)
        check(torch.equal(gathered.view(6, 2), counts.expand(6, 2)), 'all-rank-command-and-sample-counts')
        report['cross_rank_reference_sample_checks'] = backend.samples
        report['fixture_output_sha256'] = [digest(t) for t in expected]
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
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
