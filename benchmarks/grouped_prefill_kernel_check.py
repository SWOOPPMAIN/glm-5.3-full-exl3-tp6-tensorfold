#!/usr/bin/env python3
"""Isolated real-weight E3/B12X comparison; never run in a serving container.

Preparation only until explicitly executed in a bounded, separate container
on an idle GPU. No distributed initialization or NCCL. Imports/--help are CPU
only. Retains original weights; writes phase receipts before CUDA operations.
An outer controller must enforce the wall-clock deadline and idle-fleet policy.
"""
import argparse
import ast
import faulthandler
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace as NS
from unittest.mock import patch

SOURCE_HASHES = {
    'vllm/model_executor/layers/quantization/exl3.py':
        '5fafc04fac6618ea0d5a04b9ad06002ee1a7657399f7066b4e727649f5179817',
    'vllm/amos_exl3_tp6.py':
        '3756c20c7d4aaffac30e22f218621f96264c14f8a23c9f5a1bb3a276a5e77693',
    'b12x/moe/_shared/kernels/w4a16/prepare.py':
        '7de6f672235fda8e9d1e1f8ea676c2026201dc65f5743a02d366e19c59a8e7fb',
    'b12x/moe/_shared/kernels/w4a16/mixed_trellis.py':
        '4d5140de84e3dde5875ff75ad95c9a62b6041b5dde2e69f0fb96562319aabd78',
    'b12x/moe/_shared/kernels/w4a16/kernel.py':
        '524af13b672674cd9ce1fd543163bc0b65fa5a6ac62057307da0cd6c3cda3fb8',
}
INTEGRATED_EXL3_SHA = 'c5dd0c5d4025cd04832db5c2288a5d9aef1742358716ba53e60076b011ee22ed'
INTEGRATED_HELPER_SHA = '77a9e99f23177c798da2b769f992a246ce4aabd9c615df7cc8cc206d97234d73'


def require(value, message):
    if not value:
        raise RuntimeError(message)


def check_sources(package, cubin_sha, integration=False):
    hashes = dict(SOURCE_HASHES)
    if integration:
        hashes['vllm/model_executor/layers/quantization/exl3.py'] = INTEGRATED_EXL3_SHA
        hashes['vllm/amos_grouped_prefill.py'] = INTEGRATED_HELPER_SHA
    for name, expected in hashes.items():
        top, relative = name.split('/', 1)
        root = Path(importlib.util.find_spec(top).origin).parent
        require(hashlib.sha256((root/relative).read_bytes()).hexdigest() == expected,
                f'Native source mismatch: {name}')
    manifest = json.loads((package/'manifest.json').read_text())
    for name, expected in manifest['sources'].items():
        require(hashlib.sha256((package/name).read_bytes()).hexdigest() == expected,
                f'Staged source mismatch: {name}')
    binary = package/'amos_e3/grouped_fragments.cubin'
    require(hashlib.sha256(binary.read_bytes()).hexdigest() == cubin_sha, 'cubin mismatch')
    if integration:
        root = Path(importlib.util.find_spec('vllm').origin).parent
        for name in manifest['sources']:
            if name.startswith('amos_e3/'):
                require((root/name).read_bytes() == (package/name).read_bytes(),
                        'Installed E3 differs from pinned package: '+name)
        require((root/'amos_e3/grouped_fragments.cubin').read_bytes() == binary.read_bytes(),
                'Installed E3 binary differs from pinned package')
    return hashes


def native_reference(source):
    """Compile only the exact original native method in the installed namespace.

    This prevents the patched candidate from becoming its own reference. All
    native helper methods and B12X calls are unchanged in the integration image.
    """
    from vllm.model_executor.layers.quantization import exl3
    raw = source.read_bytes()
    require(hashlib.sha256(raw).hexdigest() ==
            SOURCE_HASHES['vllm/model_executor/layers/quantization/exl3.py'],
            'Original native reference source mismatch')
    tree = ast.parse(raw)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Exl3MoEMethod')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_apply_mixed_rank_sliced')
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), fn], type_ignores=[])
    namespace = dict(vars(exl3))
    exec(compile(ast.fix_missing_locations(module), str(source), 'exec'), namespace)
    return namespace[fn.name]


def load_layer(model, rank, layer_index, batch_capacity=1536):
    import torch
    from safetensors import safe_open
    from vllm.amos_exl3_tp6 import pieces
    from vllm.model_executor.layers.quantization.exl3 import (
        Exl3MoEMethod, Exl3MoEParameter, _exl3_moe_weight_loader)
    placement = json.loads((model/'TP6_PLACEMENT.json').read_text())
    require(placement['rank'] == rank, 'wrong model shard')
    require(placement['source_revision'] == '6d6bd738c0c1635513e0bd0fdf0302049bd820a9',
            'wrong model revision')
    pairs = pieces(rank)
    bits = json.loads((model/'tier_bitmap.json').read_text())[str(layer_index)]['k']
    layer = NS(local_num_experts=len(pairs), layer_name=f'model.layers.{layer_index}.mlp.experts',
               exl3_hidden_size=6144, hidden_size=6144, exl3_intermediate_size_per_partition=512,
               exl3_layer_bitrates=tuple(bits[e] for e, _ in pairs), exl3_tp6_pieces=pairs,
               exl3_params_dtype=torch.bfloat16, exl3_max_num_batched_tokens=batch_capacity,
               exl3_is_draft=layer_index == 78, activation=NS(value='silu'))
    # Only constructor metadata is substituted. Actual native loader and kernels run.
    for group, shards in (('w13', ('w1', 'w3')), ('w2', ('w2',))):
        for field in ('trellis', 'suh', 'svh', 'mcg', 'mul1'):
            with patch('vllm.model_executor.parameter.get_tensor_model_parallel_rank', return_value=rank), \
                 patch('vllm.model_executor.parameter.get_tensor_model_parallel_world_size', return_value=6):
                parameter = Exl3MoEParameter(weight_loader=_exl3_moe_weight_loader,
                    num_experts=len(pairs), shard_ids=shards, preallocate=field in ('suh', 'svh'))
            parameter.data = parameter.data.cuda()
            setattr(layer, f'{group}_{field}', parameter)
    host_clones = 0
    with safe_open(model/f'model-layer-{layer_index:03d}.safetensors', framework='pt', device='cpu') as source:
        for local, (expert, source_rank) in enumerate(pairs):
            for projection, group, shard in (('gate_proj', 'w13', 'w1'),
                                             ('up_proj', 'w13', 'w3'), ('down_proj', 'w2', 'w2')):
                for field in ('trellis', 'suh', 'svh', 'mcg'):
                    value = source.get_tensor(f'{layer.layer_name}.{expert}.{projection}.rank{source_rank}.{field}')
                    if field == 'mcg':
                        require(int(value.item()) & 0xffffffff == 3417055213, 'unexpected codebook')
                    # Direct file-backed CUDA copies stalled in the Spark OS
                    # diagnostic. Keep exact bytes in ordinary host memory.
                    anonymous = value.clone()
                    require(torch.equal(anonymous, value), 'host weight clone changed values')
                    value = anonymous
                    host_clones += 1
                    getattr(layer, f'{group}_{field}').load_exl3_weight(value.cuda(), expert_id=local, shard_id=shard)
    layer.exl3_host_clone_count = host_clones
    method = Exl3MoEMethod.__new__(Exl3MoEMethod)
    method.quant_config = NS()
    method._prepare_mixed_rank_sliced_weights(layer)
    mixed = layer.exl3_mixed_trellis
    for decode, prefill in zip(mixed['tiers'], mixed['prefill_tiers'], strict=True):
        require(decode.w13.data_ptr() == prefill.w13.data_ptr() and
                decode.w2.data_ptr() == prefill.w2.data_ptr(), 'native weight copy detected')
    return method, layer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--rank', type=int, choices=range(6), required=True)
    parser.add_argument('--layer', type=int, choices=range(3, 79), default=3)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--cubin-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--integration', action='store_true',
                        help='Exercise installed native dispatch against the original frozen method')
    parser.add_argument('--native-source', type=Path)
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--benchmark-repeats', type=int, default=12, choices=range(6, 65))
    parser.add_argument('--candidate-cubin', type=Path,
                        help='Isolated forward-kernel test in the unchanged installed runtime')
    parser.add_argument('--candidate-cubin-sha256')
    parser.add_argument('--candidate-package', type=Path,
                        help='Isolated source package override after verifying installed runtime pins')
    parser.add_argument('--require-exact-prefill', action='store_true')
    parser.add_argument('--input-recording', type=Path,
                        help='Repeat historical real MoE inputs to test full capacity with reconstructed routes')
    args = parser.parse_args()
    require(args.execute, 'GPU execution requires explicit --execute in an isolated container')
    require(not args.output.exists(), 'output already exists')
    limit = Path('/sys/fs/cgroup/memory.max').read_text().strip()
    require(limit != 'max' and int(limit) <= 4*2**30, 'requires <=4GiB container memory limit')
    available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:'))) * 1024
    require(available >= 12*2**30, 'requires >=12GiB host memory available')
    require(not args.integration or args.native_source is not None, 'Integration needs original native source')
    hashes = check_sources(args.package, args.cubin_sha256, args.integration)
    require(not args.candidate_package or (args.integration and not args.candidate_cubin
            and args.candidate_cubin_sha256), 'Package override needs integration and one candidate binary hash')
    candidate_manifest_sha = None
    if args.candidate_package:
        manifest_path = args.candidate_package/'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        candidate_manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        for name, expected in manifest['sources'].items():
            require(hashlib.sha256((args.candidate_package/name).read_bytes()).hexdigest() == expected,
                    'Candidate source mismatch: '+name)
        require(hashlib.sha256((args.candidate_package/'amos_e3/grouped_fragments.cubin').read_bytes()).hexdigest()
                == args.candidate_cubin_sha256, 'Candidate package binary mismatch')
    os.environ.update(AMOS_EXL3_TP6_PIECES='1', VLLM_EXL3_PREFILL_BLOCK_M='32')
    if args.integration:
        os.environ['AMOS_TP6_E3_PREFILL'] = '1'
    import torch
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    require(torch.cuda.get_device_capability() == (12, 1), 'requires SM121')
    torch.cuda.set_per_process_memory_fraction(2*2**30/torch.cuda.get_device_properties(0).total_memory)
    if args.integration:
        from vllm.amos_e3 import runtime
    else:
        sys.path.insert(0, str(args.package))
        from amos_e3 import runtime
    if args.candidate_package:
        directory = args.candidate_package/'amos_e3'
        spec = importlib.util.spec_from_file_location('amos_candidate_e3', directory/'__init__.py',
                                                      submodule_search_locations=[str(directory)])
        package = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = package
        spec.loader.exec_module(package)
        runtime = importlib.import_module('amos_candidate_e3.runtime')
        import vllm.amos_e3
        vllm.amos_e3.runtime = runtime
        sys.modules['vllm.amos_e3.runtime'] = runtime
    if args.candidate_cubin:
        require(hashlib.sha256(args.candidate_cubin.read_bytes()).hexdigest() ==
                args.candidate_cubin_sha256, 'Candidate cubin hash mismatch')
        runtime._MODULES[0] = runtime.DeviceModule(args.candidate_cubin)
    result = dict(passed=False, phase='initializing', rank=args.rank, layer=args.layer,
                  cases=[], source_hashes=hashes, cubin_sha256=args.cubin_sha256,
                  integration=args.integration, timings=[],
                  candidate_cubin_sha256=args.candidate_cubin_sha256,
                  candidate_manifest_sha256=candidate_manifest_sha,
                  torch_limit_gib=2, distributed_initialized=False,
                  scope='isolated single-layer numerics and stream ownership; no full-model quality or TP6 liveness claim')
    def save(phase):
        result.update(phase=phase, updated=time.time())
        tmp = args.output.with_suffix('.tmp')
        tmp.write_text(json.dumps(result, indent=2)+'\n')
        tmp.replace(args.output)
        faulthandler.cancel_dump_traceback_later()
        faulthandler.dump_traceback_later(240, exit=True)
    def compare(actual, reference, label, exact=False):
        exact = exact or args.require_exact_prefill
        torch.cuda.synchronize()
        a, b = actual.float(), reference.float()
        require(bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()), label+' nonfinite')
        if exact:
            require(torch.equal(actual, reference), label+' failed required exact parity')
        if not bool(torch.count_nonzero(b)):
            relative, cosine = 0.0, 1.0
            require(torch.equal(actual, reference), label+' stale scratch')
        else:
            relative = float(((a-b).square().mean()/b.square().mean()).sqrt())
            cosine = float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0))
            require(relative < .01 and cosine > .99995, label+' exploratory numerical gate failed')
        result['cases'].append(dict(case=label, relative_rms=relative, cosine=cosine,
                                   exact_required=exact, passed=True))
        save('completed_'+label)
    save('loading_real_expert_pieces')
    method, layer = load_layer(args.model, args.rank, args.layer)
    if args.integration:
        original = native_reference(args.native_source)
        native = lambda x, w, ids: original(method, layer, x, w, ids)
        candidate = lambda x, w, ids: method._apply_mixed_rank_sliced(layer, x, w, ids)
    else:
        native = lambda x, w, ids: method._apply_mixed_rank_sliced(layer, x, w, ids)
        candidate = lambda x, w, ids: runtime.apply(layer, x, w, ids, stream_scratch=True)
    torch.manual_seed(20261001)
    x = torch.randn((1536, 6144), device='cuda', dtype=torch.bfloat16) * .1
    ids = torch.randint(256, (1536, 8), device='cuda')
    weights = torch.rand((1536, 8), device='cuda')
    if args.input_recording:
        from safetensors import safe_open
        from safetensors.torch import load_file
        metadata = json.loads(args.input_recording.with_suffix('.json').read_text())
        require(hashlib.sha256(args.input_recording.read_bytes()).hexdigest() == metadata['sha256'],
                'Recording hash mismatch')
        desc = metadata['tensors']['input']
        require(metadata['prefix'] == f'model.layers.{args.layer}.mlp.shared_experts.gate_up_proj'
                and desc['dtype'] == 'torch.bfloat16' and desc['shape'][1] == 6144,
                'Recording must be the selected layer common BF16 input')
        config = json.loads((args.model/'config.json').read_text())
        config = config.get('text_config', config)
        require((config['n_group'], config['topk_group'], config['scoring_func'],
                 config['norm_topk_prob'], config['routed_scaling_factor'],
                 config['num_experts_per_tok']) == (1, 1, 'sigmoid', True, 2.5, 8),
                'Unrecognized full-GLM router')
        recorded = load_file(str(args.input_recording))['input'].view(torch.bfloat16).reshape(desc['shape'])
        x = recorded.repeat(((1536+recorded.shape[0]-1)//recorded.shape[0], 1))[:1536].contiguous().cuda()
        with safe_open(args.model/f'model-layer-{args.layer:03d}.safetensors', framework='pt', device='cpu') as f:
            gate = f.get_tensor(f'model.layers.{args.layer}.mlp.gate.weight').float().cuda()
            bias = f.get_tensor(f'model.layers.{args.layer}.mlp.gate.e_score_correction_bias').float().cuda()
        scores = torch.nn.functional.linear(x.float(), gate).sigmoid()
        ids = (scores + bias).topk(8, dim=-1).indices
        weights = scores.gather(1, ids)
        result.update(input_sha256=metadata['sha256'], input_scope=
            'Historical real activations repeated to1536 rows; routes reconstructed with FP32 linear/sigmoid/top8; not a live route capture')
        del recorded, gate, bias, scores
    weights = (2.5*weights/weights.sum(1, keepdim=True)).contiguous()
    save('native_short_decode_before_cold_prefill')
    if args.integration:
        # A serving boot profiles prefill before any native decode. Verify the
        # actual route-pack warmup contract that the first integration missed.
        save('cold_prefill_before_native_warmup')
        actual = candidate(x, weights, ids).clone()
        from vllm.model_executor.layers.quantization.exl3 import warmup_exl3_mixed_trellis_route_pack
        result['native_route_pack_warmups'] = warmup_exl3_mixed_trellis_route_pack(
            NS(modules=lambda: [layer]))
        require(result['native_route_pack_warmups'] > 0, 'native warmup was bypassed')
        compare(actual, native(x, weights, ids), 'cold_prefill_before_native_warmup')
        del actual
    native(x[:5], weights[:5], ids[:5])
    torch.cuda.synchronize()
    # Follow the startup test with the short-decode -> prefill transition.
    for rows in (1536, 33, 63, 64, 65, 256):
        save(f'cold_or_eager_candidate_{rows}')
        actual = candidate(x[:rows], weights[:rows], ids[:rows]).clone()
        torch.cuda.synchronize()
        reference = native(x[:rows], weights[:rows], ids[:rows]).clone()
        compare(actual, reference, f'eager_{rows}')
    mapping = layer.exl3_mixed_trellis['global_to_combined'].cpu().tolist()
    mixed = layer.exl3_mixed_trellis
    hot_cases = []
    offset = 0
    for bits, tier_ids in zip(mixed['tier_bits'], mixed['tier_ids'], strict=True):
        expert = next(e for e, v in enumerate(mapping) if v == offset)
        hot_cases.append((f'duplicate_hot_k{bits}', expert))
        offset += len(tier_ids)
    hot_cases.append(('absent_after_nonzero', next(e for e,v in enumerate(mapping) if v < 0)))
    for label, expert in hot_cases:
        save(label)
        probe_ids = torch.full_like(ids[:65], expert)
        compare(candidate(x[:65], weights[:65], probe_ids), native(x[:65], weights[:65], probe_ids), label)
    for scale in (10, 40):
        save(f'activation_scale_{scale}')
        scaled = (x[:65]*scale).contiguous()
        compare(candidate(scaled, weights[:65], ids[:65]),
                native(scaled, weights[:65], ids[:65]), f'activation_scale_{scale}')
    if args.integration:
        for rows in (1, 5, 20, 32):
            actual = candidate(x[:rows], weights[:rows], ids[:rows]).clone()
            compare(actual, native(x[:rows], weights[:rows], ids[:rows]), f'native_decode_{rows}', exact=True)
    if args.benchmark:
        # One eager E3 arena is live here; release the flush buffer before the
        # two-stream phase. Do not raise the 2GiB cap to accommodate benchmarks.
        save('paired_eager_benchmark')
        flush = torch.empty(256*1024**2, dtype=torch.uint8, device='cuda')
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for rows in (65, 256, 1024, 1536):
            values = (x[:rows], weights[:rows], ids[:rows])
            compare(candidate(*values), native(*values), f'benchmark_accuracy_{rows}')
            samples = {'native': [], 'e3': []}
            for repeat in range(args.benchmark_repeats):
                order = [('native', native), ('e3', candidate)]
                if repeat % 2:
                    order.reverse()
                for name, function in order:
                    flush.zero_()
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    begin.record()
                    measured = function(*values)
                    end.record()
                    end.synchronize()
                    samples[name].append(dict(repeat=repeat, gpu_ms=begin.elapsed_time(end),
                                              wall_ms=(time.perf_counter()-start)*1000))
            medians = {name: {key: statistics.median(v[key] for v in values)
                             for key in ('gpu_ms', 'wall_ms')}
                       for name, values in samples.items()}
            result['timings'].append(dict(rows=rows, samples=samples, medians=medians,
                gpu_speedup=medians['native']['gpu_ms']/medians['e3']['gpu_ms'],
                wall_speedup=medians['native']['wall_ms']/medians['e3']['wall_ms'],
                scope='isolated eager layer; fixed shared inputs, alternating order,256MiB L2 flush excluded; not full-model throughput'))
            save(f'completed_benchmark_{rows}')
        del flush, measured
    # Eager work is finished and no graph owns an arena yet. Reclaim that arena
    # before allocating two stream-owned arenas, preserving the 2GiB test cap.
    save('release_completed_eager_scratch')
    torch.cuda.synchronize()
    runtime._SCRATCH.clear()
    torch.cuda.empty_cache()
    # Binding is ready before cross-stream use; each stream owns its scratch arena.
    save('warming_two_streams')
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            candidate(x[:65], weights[:65], ids[:65])
    torch.cuda.synchronize()
    save('captured_changing_inputs')
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=streams[0]):
        captured = candidate(x[:65], weights[:65], ids[:65])
    for iteration in range(8):
        x[:65].normal_(std=.1)
        ids[:65].random_(0, 256)
        graph.replay()
        compare(captured, native(x[:65], weights[:65], ids[:65]), f'graph_{iteration}')
    save('two_stream_scratch_isolation')
    # Different inputs/routes are essential: identical work can hide shared-arena races.
    other_x = (-x[:65] * .375).contiguous()
    other_ids = torch.full_like(ids[:65], next(e for e,v in enumerate(mapping) if v < 0))
    inputs = [(x[:65], weights[:65], ids[:65]), (other_x, weights[:65], other_ids)]
    references = [native(*values).clone() for values in inputs]
    torch.cuda.synchronize()
    outputs = []
    for stream, values in zip(streams, inputs, strict=True):
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            outputs.append(candidate(*values).clone())
    torch.cuda.synchronize()
    for i, (actual, reference) in enumerate(zip(outputs, references, strict=True)):
        compare(actual, reference, f'two_stream_distinct_inputs_{i}')
    require(not torch.distributed.is_initialized(), 'unexpected distributed initialization')
    result.update(passed=True, peak_cuda_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                  scratch_arenas=len(runtime._SCRATCH))
    save('complete')
    faulthandler.cancel_dump_traceback_later()
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
