#!/usr/bin/env python3
"""Bounded exact-oracle checks for logical top-k finalization and physical output."""
import argparse
import json
from pathlib import Path
import statistics

import torch
from attention_capacity_check import compare
from canonical_index_topk import wrap_tiled_topk
from b12x.attention.dsa_indexer.tiled_topk import run_tiled_topk


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--benchmark', action='store_true')
    p.add_argument('--verify-oracle', action='store_true')
    a = p.parse_args()
    assert not a.output.exists()
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(2 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    fast = wrap_tiled_topk(run_tiled_topk)
    assert not (a.benchmark and a.verify_oracle), 'Bound oracle checks separately from timing'

    def run(**kwargs):
        values, ids = fast(**kwargs)
        if a.verify_oracle:
            from canonical_index_oracle import oracle_tiled_topk, matches
            ref_values, ref_ids = oracle_tiled_topk(**kwargs)
            torch._assert_async(matches(values, ids, ref_values, ref_ids).all(), 'Independent canonical top-k oracle mismatch')
        return values, ids
    report = dict(phase='running', no_distributed_initialization=True, checks=[],
                  score_comparison='FP32 bit patterns, including negative-infinity padding',
                  independent_oracle_enabled=a.verify_oracle)
    rows, k, bq, bk, chunk = 33, 2048, 32, 256, 32768

    def record(row):
        report['checks'].append(row)
        report['peak_cuda_allocated_gib'] = torch.cuda.max_memory_allocated()/2**30
        a.output.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(row), flush=True)

    with torch.inference_mode():
        for context in (8192, 32768, 65536):
            torch.manual_seed(23530 + context)
            lengths_cpu = torch.tensor(([0, 1, 63, 2047, 2048, 2049, context//2,
                                         context-1, context]*4)[:rows], dtype=torch.int32)
            lengths = lengths_cpu.cuda()
            # A non-monotone page permutation exposes sorting physical IDs too early.
            pages_cpu = torch.randperm(context//64, dtype=torch.int32)[None, :]
            pages = pages_cpu.cuda().expand(rows, -1)
            for distribution in ('random', 'all_tied', 'boundary_ties'):
                if distribution == 'random':
                    cpu = torch.randn((rows, context))
                elif distribution == 'all_tied':
                    cpu = torch.full((rows, context), 0.25)
                else:
                    cpu = torch.randint(-2, 3, (rows, context)).float() / 8
                valid = torch.arange(context)[None, :] < lengths_cpu[:, None]
                order = cpu.masked_fill(~valid, -torch.inf).argsort(dim=1, descending=True, stable=True)[:, :k]
                oracle_ids = order.masked_fill(torch.arange(k)[None, :] >= lengths_cpu[:, None], context).sort(1).values
                oracle_valid = oracle_ids < context
                oracle_values = cpu.gather(1, oracle_ids.clamp(max=context-1)).masked_fill(~oracle_valid, -torch.inf).cuda()
                logical = oracle_ids.to(torch.int32).masked_fill(~oracle_valid, -1).cuda()
                physical = (pages_cpu[0, oracle_ids.clamp(max=context-1)//64]*64+oracle_ids%64).to(torch.int32).masked_fill(~oracle_valid, -1).cuda()
                source = cpu.cuda()
                for physical_output in (False, True):
                    cv = ci = None
                    tiles = []
                    for start in range(0, context, chunk):
                        width = min(chunk, context-start)
                        padded = torch.full((64, width), -torch.inf, device='cuda')
                        padded[:rows].copy_(source[:, start:start+width])
                        tiled = padded.view(2, bq, width//bk, bk).permute(0, 2, 1, 3).contiguous().view(-1)
                        tiles.append(tiled)
                        cv, ci = run(tile_logits=tiled, k_start=None, lengths=lengths,
                                     topk=k, block_q=bq, block_k=bk, num_k_tiles=width//bk,
                                     input_index_offset=start, output_index_offset=start,
                                     input_extent=width, zero_row_start=True, is_first=start == 0,
                                     carry_values=cv, carry_indices=ci,
                                     output_page_table=pages if physical_output and start+width == context else None)
                    torch.cuda.synchronize()
                    checks = dict(ids=compare(physical if physical_output else logical, ci),
                                  scores=compare(oracle_values.view(torch.int32), cv.view(torch.int32)))
                    record(dict(context=context, distribution=distribution, physical_output=physical_output,
                                **checks))
                    assert checks['ids']['exact'] and checks['scores']['exact'], 'Independent stable top-k oracle mismatch'
                # Capture a warmed single-chunk path, then change scores and causal lengths
                # in-place to ensure graph replay does not use stale host decisions.
                if context == 8192 and distribution == 'random':
                    ov, oi = torch.empty_like(cv), torch.empty_like(ci)
                    kwargs = dict(tile_logits=tiles[0], k_start=None, lengths=lengths, topk=k,
                                  block_q=bq, block_k=bk, num_k_tiles=context//bk, input_extent=context,
                                  zero_row_start=True, output_values=ov, output_indices=oi,
                                  output_page_table=pages)
                    for _ in range(2):
                        run(**kwargs)
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        run(**kwargs)
                    tiles[0].fill_(0.25)
                    lengths.copy_(torch.full_like(lengths, context))
                    graph.replay()
                    torch.cuda.synchronize()
                    want_ids = (pages[0, torch.arange(k, device='cuda')//64]*64 + torch.arange(k, device='cuda')%64).expand(rows, -1)
                    gc = dict(ids=compare(want_ids, oi), scores=compare(torch.full_like(ov, 0.25).view(torch.int32), ov.view(torch.int32)))
                    record(dict(context=context, graph_changed_input=True, **gc))
                    assert gc['ids']['exact'] and gc['scores']['exact']
                    lengths.copy_(lengths_cpu)
        if a.verify_oracle:
            from canonical_index_oracle import matches
            bad_ids = ci.clone()
            bad_ids[8, 0] = -7
            bad_values = cv.clone()
            bad_values[8, 0] += 1
            assert not bool(matches(cv, bad_ids, cv, ci).all())
            assert not bool(matches(bad_values, ci, cv, ci).all())
            report['oracle_negative_controls'] = dict(wrong_id_rejected=True, wrong_score_rejected=True)
        if a.benchmark:
            report['selector_timing'] = []
            for bench_rows, context in ((3072, 8192), (3072, 32768), (512, 32768)):
                tiled = torch.randn(((bench_rows//bq)*(context//bk)*bq*bk,), device='cuda')
                lens = torch.arange(context-bench_rows+1, context+1, dtype=torch.int32, device='cuda')
                bench_pages = torch.randperm(context//64, dtype=torch.int32, device='cuda')[None, :].expand(bench_rows, -1)
                vals = torch.empty((bench_rows, k), device='cuda')
                inds = torch.empty((bench_rows, k), device='cuda', dtype=torch.int32)
                kwargs = dict(tile_logits=tiled, k_start=None, lengths=lens, topk=k, block_q=bq,
                              block_k=bk, num_k_tiles=context//bk, input_extent=context,
                              zero_row_start=True, output_values=vals, output_indices=inds,
                              output_page_table=bench_pages)
                timing = dict(rows=bench_rows, context=context, scope='Selector only; unchanged synthetic scores; no full-model speed claim')
                for name, fn in (('native', run_tiled_topk), ('canonical', run)):
                    for _ in range(2):
                        fn(**kwargs)
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        for _ in range(5):
                            fn(**kwargs)
                    graph.replay()
                    torch.cuda.synchronize()
                    samples = []
                    for _ in range(3):
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start.record()
                        for _ in range(10):
                            graph.replay()
                        end.record()
                        end.synchronize()
                        samples.append(start.elapsed_time(end)/50)
                    timing[name+'_ms'] = statistics.median(samples)
                    del graph
                timing['overhead_ms'] = timing['canonical_ms']-timing['native_ms']
                report['selector_timing'].append(timing)
                print(json.dumps(timing), flush=True)
                a.output.write_text(json.dumps(report, indent=2)+'\n')
        report.update(phase='complete', all_exact=True,
                      peak_cuda_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        a.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
