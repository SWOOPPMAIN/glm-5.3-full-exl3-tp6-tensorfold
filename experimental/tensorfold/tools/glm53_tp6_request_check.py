#!/usr/bin/env python3
"""Original-weight six-rank request parity over an admitted 804K resident cache.

Compare the new eager request engine to independent serial target generation;
exercise recursive MTP, keyed sampling, four interleaved requests, cancellation,
and retained-prefix continuation. This is an initial short-request numerical
qualification, not API serving, decoded-speed claims or long-context quality.
"""
import argparse, faulthandler, hashlib, json, os, time
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
    torch.manual_seed(530616)
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
        dist.init_process_group('nccl', init_method=f'tcp://{os.environ["MASTER_ADDR"]}:{a.port}',
                                rank=a.rank, world_size=6, timeout=timedelta(seconds=240))
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
        engine = RequestEngine(backend)
        requests = [engine.start(f'r{i}', prompt, max_tokens=12, sampling=sampling, draft_tokens=4, ignore_eos=True)
                    for i, (prompt, sampling) in enumerate(zip(prompts, samplings))]
        outputs = [[] for _ in requests]
        save('four-request-interleaving')
        for step in range(64):
            for i in (3, 1, 0, 2):
                request = requests[i]
                if request.status in ('prefill', 'decode'):
                    outputs[i].extend(engine.step(request).tokens)
            save(f'interleave-round-{step}')
            if all(r.status == 'finished' for r in requests):
                break
        for i, request in enumerate(requests):
            check(outputs[i] == expected[i], f'serial-parity-request-{i}', exact_tokens=True,
                  expected_tokens=expected[i], actual_tokens=outputs[i], rounds=request.rounds,
                  drafted=request.drafted, accepted=request.accepted)
            check(request.tokens == prompts[i]+outputs[i][:-1], f'committed-prefix-{i}')
            check(request.mtp_end+len(request.pending_hidden) == len(request.tokens), f'mtp-pending-bound-{i}')
        # Release other conversations before expanding the retained first prefix.
        for request in requests[1:]:
            engine.drop(request)
        old = requests[0]
        prompt = [*old.tokens, old.pending, *tokenizer.encode('\nNow add an example.\n', add_special_tokens=False).ids]
        new = engine.start('followup', prompt, max_tokens=8, sampling=samplings[1], resume=old, ignore_eos=True)
        follow = []
        while new.status in ('prefill', 'decode'):
            follow.extend(engine.step(new).tokens)
        check(new.reused_tokens > 0 and old.status == 'transferred', 'followup-retains-prefix', reused=new.reused_tokens)
        engine.drop(new)
        # Serial reference runs only after the kept lease was released.
        reference = serial(prompt, 8, samplings[1])
        check(follow == reference, 'followup-serial-parity', exact_tokens=True, expected_tokens=reference, actual_tokens=follow)
        # Exact-prompt replay: no new target prefill should be required.
        first = engine.start('first', prompts[0], max_tokens=1, ignore_eos=True)
        first_output = engine.step(first).tokens
        replay = engine.start('replay', prompts[0], max_tokens=1, ignore_eos=True, resume=first)
        check(engine.step(replay).tokens == first_output and replay.reused_tokens == len(prompts[0]), 'exact-prompt-replay')
        engine.drop(replay)
        # Cancellation before a forward must produce nothing and release safely.
        cancelled = engine.start('cancelled', prompts[2], max_tokens=8)
        event = engine.step(cancelled, cancelled=lambda: True)
        check(event.finished and not event.tokens and not cancelled.tokens, 'cancelled-request-no-output')
        engine.drop(cancelled)
        check(not engine.pool.leases and engine.pool.free == [(0, capacity)], 'all-extents-reclaimed')
        report['cross_rank_sample_checks'] = backend.samples
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
