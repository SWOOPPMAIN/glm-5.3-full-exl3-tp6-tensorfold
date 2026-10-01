#!/usr/bin/env python3
"""Bounded real-fragment GPU qualification; no distributed initialization.

The controller owns admission holds, verifies an idle serving fleet and records
the created container before launch. One invocation has a 240-second deadline,
4 GiB cgroup and 2 GiB Torch cap. Compiled extension/source hashes are mandatory.
"""
import argparse
import faulthandler
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--layer', type=int, required=True)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--adapter', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--native-source-pins', type=Path)
    p.add_argument('--native-helper', type=Path)
    a = p.parse_args()
    assert not a.output.exists()
    limit = Path('/sys/fs/cgroup/memory.max').read_text().strip()
    assert limit != 'max' and int(limit) <= 4 * 2**30
    available = int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines()
                         if s.startswith('MemAvailable:'))) * 1024
    assert available >= 12 * 2**30
    manifest = json.loads(a.manifest.read_text())
    for name, expected in manifest.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
    assert str(a.binary) in manifest and str(a.adapter) in manifest and str(Path(__file__)) in manifest
    import numpy as np
    import torch
    from tensorfold.cuda.exl3 import experts, format as reference
    from tensorfold.families.glm_moe_dsa.checkpoint import RankPieces
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    assert torch.cuda.get_device_capability() == (12, 1)
    torch.cuda.set_per_process_memory_fraction(2*2**30/torch.cuda.get_device_properties(0).total_memory)
    spec = importlib.util.spec_from_file_location('tensorfold_exl3_experts_v1', a.binary)
    ext = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ext)
    experts._ext = lambda: ext  # exact recorded binary, no JIT during the GPU window
    spec = importlib.util.spec_from_file_location('glm53_tp6_adapter_under_test', a.adapter)
    adapter = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = adapter
    spec.loader.exec_module(adapter)
    report = dict(passed=False, phase='initializing', layer=a.layer, cases=[],
                  source_manifest=manifest, started_at=time.time(),
                  scope='single rank real expert fragments; no attention, NCCL or full-model quality/speed claim')

    def save(phase):
        report.update(phase=phase, updated_at=time.time(),
                      peak_cuda_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        tmp = a.output.with_suffix('.tmp')
        tmp.write_text(json.dumps(report, indent=2)+'\n')
        tmp.replace(a.output)
        print(json.dumps(dict(phase=phase)), flush=True)

    def compare(got, expected, label, exact, characterize=False):
        torch.cuda.synchronize()
        x, y = got.float(), expected.float()
        finite = bool(torch.isfinite(x).all()) and bool(torch.isfinite(y).all())
        same = torch.equal(x, y)
        denom = float(y.square().mean())
        relative = float((x-y).square().mean().sqrt()) / math.sqrt(denom) if denom else float('inf') if not same else 0.
        cosine = float(torch.nn.functional.cosine_similarity(x.flatten(), y.flatten(), dim=0)) if denom else float(same)
        passed = finite and (same if exact else relative < .01 and cosine > .99995)
        report['cases'].append(dict(case=label, exact_required=exact, characterize_only=characterize, bit_equal=same,
                                    relative_rms=relative, cosine=cosine, passed=passed))
        save(label)
        assert passed or characterize, (label, relative, cosine)

    faulthandler.dump_traceback_later(235, exit=True)
    try:
        save('loading_original_fragments')
        reader = RankPieces(a.model, 0)
        parts = reader.fragments(a.layer)
        layer = adapter.RoutedLayer.load(reader, a.layer)
        scratch = layer.scratch(128)
        absent = [e for e, v in enumerate(reader.routing_map(parts)) if v == len(parts)]
        assert len(absent) >= 8
        g = torch.Generator().manual_seed(531006 + a.layer)
        x = (torch.randn((128, 6144), generator=g)*.1).bfloat16().cuda()
        ids = torch.stack([torch.randperm(256, generator=g)[:8] for _ in range(128)]).cuda()
        probabilities = torch.rand((128, 8), generator=g)
        probabilities = (probabilities/probabilities.sum(1, keepdim=True)).cuda()

        def run(xx, ii, ww, mode, arena=scratch):
            return layer.routed(xx, ii, ww, arena, act_mode=mode)

        # Independent NumPy decoder across every value of six original matrices.
        # Then float64 linear algebra with the declared activation rounding.
        for bits in (3, 4):
            part = next(v for v in parts if v.bits == bits)
            matrices = []
            for projection in ('gate_proj', 'up_proj', 'down_proj'):
                prefix = f'{part.prefix}.{projection}.rank{part.original_rank}'
                t, suh, svh = [reader.read_tensor(prefix+'.'+f) for f in ('trellis', 'suh', 'svh')]
                decoded = reference.unpack(t.numpy(), bits, 'mcg')
                actual = experts.dequant(t.cuda(), 'mcg').cpu().numpy()
                assert np.array_equal(actual.view(np.uint16), decoded.view(np.uint16))
                report['cases'].append(dict(case=f'K{bits}_{projection}_all_values_decode',
                                            values=decoded.size, bit_equal=True, passed=True))
                matrices.append((decoded.astype(np.float64), suh.numpy().astype(np.float64),
                                 svh.numpy().astype(np.float64)))
                del actual, decoded, t
            one_ids = torch.tensor([[part.expert, *absent[:7]]], device='cuda', dtype=torch.int64)
            one_w = torch.full((1, 8), .125, device='cuda')
            xx = x[:1].cpu().float().numpy().astype(np.float64)

            def linear(v, m):
                wq, suh, svh = m
                rotated = reference.rotate(v*suh, -1).astype(np.float16).astype(np.float64)
                return reference.rotate(rotated@wq, -1)*svh

            def bf16(v):
                return torch.from_numpy(v).bfloat16().float().numpy().astype(np.float64)

            gg, uu = linear(xx, matrices[0]), linear(xx, matrices[1])
            for mode in (experts.ACT_BF16, experts.ACT_F32):
                if mode == experts.ACT_BF16:
                    gb, ub = bf16(gg), bf16(uu)
                    act = bf16(bf16(gb/(1+np.exp(-gb)))*ub)
                else:
                    act = gg/(1+np.exp(-gg))*uu
                expected = torch.from_numpy(linear(act, matrices[2])*.125).cuda()
                compare(run(x[:1], one_ids, one_w, mode), expected,
                        f'K{bits}_float64_reference_mode{mode}', exact=False)
            del matrices, expected, gg, uu

        native_inputs = []
        tensorfold_outputs = {}
        if a.native_source_pins:
            assert a.native_helper and str(a.native_helper) in manifest and str(a.native_source_pins) in manifest
            for scale in (1, 10, 40):
                for rows in (1, 5, 20, 65, 128):
                    native_inputs.append((f'rows{rows}_scale{scale}',
                                          (x[:rows]*scale).contiguous(), ids[:rows], probabilities[:rows]))
        # The model batch stays 3072. Include both sides of chunk boundaries and
        # a non-multiple tail; compare rows with separate one-row evaluation.
        batch_x = (torch.randn((3072, 6144), generator=g)*.1).bfloat16().cuda()
        batch_ids = torch.stack([torch.randperm(256, generator=g)[:8] for _ in range(3072)]).cuda()
        batch_w = torch.rand((3072, 8), generator=g)
        batch_w = (batch_w/batch_w.sum(1, keepdim=True)).cuda()
        batch_scratch = layer.scratch(3072)
        assert batch_scratch.kernel.rows == 128
        for mode in (experts.ACT_BF16, experts.ACT_F32):
            save(f'window_invariance_mode{mode}')
            full = run(x, ids, probabilities, mode).clone()
            for rows in (1, 5, 20, 32, 65, 128):
                for start in (0, 128-rows):
                    compare(run(x[start:start+rows], ids[start:start+rows], probabilities[start:start+rows], mode),
                            full[start:start+rows], f'mode{mode}_window_{rows}_at{start}', exact=True)
            # All absent routes after populated scratch, including a NaN poison.
            other = torch.tensor([absent[:8]]*5, device='cuda', dtype=torch.int64)
            scratch.kernel.y.fill_(float('nan'))
            compare(run(x[:5], other, probabilities[:5], mode), torch.zeros_like(x[:5]).float(),
                    f'mode{mode}_absent_after_poison', exact=True)
            # Capture each short verification shape; inputs AND routes change.
            for rows in (1, 5, 20):
                arena = layer.scratch(rows)
                xx, ii, ww = x[:rows].clone(), ids[:rows].clone(), probabilities[:rows].clone()
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    run(xx, ii, ww, mode, arena)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    captured = run(xx, ii, ww, mode, arena)
                for iteration in range(4):
                    xx.copy_(x[iteration:iteration+rows]*(iteration+1))
                    ii.copy_(ids[iteration:iteration+rows] if iteration%2 == 0 else other[:1].expand(rows, -1))
                    ww.copy_(probabilities[iteration:iteration+rows])
                    graph.replay()
                    compare(captured, run(xx, ii, ww, mode),
                            f'mode{mode}_graph_{rows}_change{iteration}', exact=True)
                del graph, arena, captured, xx, ii, ww
            # Distinct scratch per stream must also be independent.
            streams = [torch.cuda.Stream(), torch.cuda.Stream()]
            arenas = [layer.scratch(5), layer.scratch(5)]
            inputs = [(x[:5], ids[:5], probabilities[:5]), (-x[5:10], other, probabilities[5:10])]
            expected = [run(*v, mode).clone() for v in inputs]
            outputs = []
            for stream, arena, values in zip(streams, arenas, inputs):
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    outputs.append(run(*values, mode, arena).clone())
            torch.cuda.synchronize()
            for n in range(2):
                compare(outputs[n], expected[n], f'mode{mode}_distinct_stream{n}', exact=True)
            del arenas, outputs, expected, full
            for batch_rows in (129, 1537, 3072):
                result = run(batch_x[:batch_rows], batch_ids[:batch_rows], batch_w[:batch_rows], mode,
                             batch_scratch)
                for row in (0, 127, 128, batch_rows-1):
                    compare(result[row:row+1], run(batch_x[row:row+1], batch_ids[row:row+1], batch_w[row:row+1], mode),
                            f'mode{mode}_batch{batch_rows}_serial_row{row}', exact=True)
            for label, xx, ii, ww in native_inputs:
                tensorfold_outputs[(mode, label)] = run(xx, ii, ww, mode).bfloat16().cpu()
        del batch_x, batch_ids, batch_w, batch_scratch, result
        if a.native_source_pins:
            # Sequential arms keep both backends within the same 2 GiB cap.
            del run, layer, scratch, arena
            gc.collect()
            torch.cuda.empty_cache()
            save('native_source_verification')
            pins = json.loads(a.native_source_pins.read_text())
            for name, expected in pins.items():
                top, relative = name.split('/', 1)
                package = Path(importlib.util.find_spec(top).origin).parent
                assert hashlib.sha256((package/relative).read_bytes()).hexdigest() == expected, name
            os.environ.update(AMOS_EXL3_TP6_PIECES='1', AMOS_TP6_E3_PREFILL='1',
                              VLLM_EXL3_PREFILL_BLOCK_M='32')
            spec = importlib.util.spec_from_file_location('native_layer_helper', a.native_helper)
            helper = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(helper)
            save('native_loading')
            method, native_layer = helper.load_layer(a.model, 0, a.layer, batch_capacity=128)
            for label, xx, ii, ww in native_inputs:
                expected = method._apply_mixed_rank_sliced(native_layer, xx, ww, ii).clone()
                for mode in (experts.ACT_BF16, experts.ACT_F32):
                    compare(tensorfold_outputs[(mode, label)].cuda(), expected,
                            f'native_mode{mode}_{label}', exact=False, characterize=True)
            report['native_characterization'] = {
                f'mode{mode}': dict(
                    max_relative_rms=max(c['relative_rms'] for c in report['cases']
                                         if c['case'].startswith(f'native_mode{mode}_')),
                    all_exploratory_gates_passed=all(c['passed'] for c in report['cases']
                                                    if c['case'].startswith(f'native_mode{mode}_')))
                for mode in (experts.ACT_BF16, experts.ACT_F32)}
            report['native_scope'] = 'Current P24 unscaled local routed BF16 output; unique synthetic global top8; no router/shared/scaling/collective qualification'
        assert not torch.distributed.is_initialized()
        report.update(passed=True, distributed_initialized=False, finished_at=time.time())
        save('complete')
    except Exception as exc:
        report.update(error=f'{type(exc).__name__}: {exc}', finished_at=time.time())
        save('failed')
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()


if __name__ == '__main__':
    main()
