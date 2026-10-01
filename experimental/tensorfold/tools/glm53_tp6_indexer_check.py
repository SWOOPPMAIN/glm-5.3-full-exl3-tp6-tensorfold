#!/usr/bin/env python3
"""Bounded isolated full GLM indexer qualification with real original weights."""
import argparse
import faulthandler
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'module', 'manifest', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--layer', type=int, choices=(0, 78), required=True)
    a = p.parse_args()
    assert not a.output.exists()
    limit = Path('/sys/fs/cgroup/memory.max').read_text().strip()
    assert limit != 'max' and int(limit) <= 4*2**30
    available = int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines()
                         if s.startswith('MemAvailable:')))*1024
    assert available >= 12*2**30
    manifest = json.loads(a.manifest.read_text())
    for name, expected in manifest.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
    assert str(a.module) in manifest and str(Path(__file__)) in manifest
    import torch
    from tensorfold.families.glm_moe_dsa.checkpoint import RankPieces
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(2*2**30/torch.cuda.get_device_properties(0).total_memory)
    assert torch.cuda.get_device_capability() == (12, 1)

    def module(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    dense = module('tensorfold.families.glm_moe_dsa.dense', a.module.with_name('dense.py'))
    attn = module('tensorfold.families.glm_moe_dsa.attention', a.module.with_name('attention.py'))
    idx = module('tensorfold.families.glm_moe_dsa.indexer', a.module)
    report = dict(passed=False, phase='initializing', layer=a.layer, cases=[], started_at=time.time(),
        source_manifest=manifest, scope='Original BF16 indexer weights with synthetic hidden states; complete indexer forward, fixed BF16 projections, norm/RoPE/FP8 cache/scoring/canonical top2048; no full attention-layer or engine qualification')

    def save(phase):
        report.update(phase=phase, updated_at=time.time(), peak_cuda_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        tmp = a.output.with_suffix('.tmp')
        tmp.write_text(json.dumps(report, indent=2)+'\n'); tmp.replace(a.output)
        print(json.dumps(dict(phase=phase)), flush=True)

    def compare(got, expected, label, exact=False, tolerance=1e-5):
        torch.cuda.synchronize()
        x, y = got.float(), expected.float()
        same = torch.equal(x, y)
        finite = bool(torch.isfinite(x).all()) and bool(torch.isfinite(y).all())
        denom = float(y.square().mean())
        relative = float((x-y).square().mean().sqrt())/math.sqrt(denom) if denom else (0. if same else float('inf'))
        passed = finite and (same if exact else relative < tolerance)
        report['cases'].append(dict(case=label, exact_required=exact, bit_equal=same,
                                    relative_rms=relative, passed=passed))
        save(label)
        assert passed, (label, relative)

    def oracle(scores, positions, bases, cap):
        s = scores.cpu()
        ids = torch.argsort(s, dim=-1, descending=True, stable=True)[:, :2048].sort(-1).values
        n = torch.minimum(torch.minimum(positions.cpu()+1, cap-bases.cpu()),
                          torch.tensor(s.shape[1])).clamp_min(0)
        ids[ids >= n[:, None]] = -1
        return ids.int().cuda(), n.clamp_max(2048).int().cuda()

    def select_oracle(s, pos, base, cap, scratch, label):
        t, c = scratch.tokens[:len(pos)], scratch.counts[:len(pos)]
        idx.select_scores(s, pos, base, cap, t, c, scratch)
        et, ec = oracle(s, pos, base, cap)
        compare(t, et, label+'_ids', exact=True)
        compare(c, ec, label+'_counts', exact=True)

    faulthandler.dump_traceback_later(235, exit=True)
    try:
        save('prepare_original_weights')
        torch.manual_seed(530603+a.layer)
        w = idx.IndexerWeights(RankPieces(a.model, 0), a.layer)
        table = attn.rope_table(1048576, 'cuda')
        positions = torch.tensor([0, 1, 127, 511, 2047, 2048, 32767, 131071, 359999, 1048575,
                                  4095, 0, 2048, 512, 131072, 360000], dtype=torch.int64, device='cuda')
        rows = len(positions)
        hidden = torch.randn((rows, 6144), dtype=torch.bfloat16, device='cuda')
        qr = torch.randn((rows, 2048), dtype=torch.bfloat16, device='cuda')
        q = torch.nn.functional.linear(qr, w.wq).reshape(rows, 32, 128)
        kw = torch.nn.functional.linear(hidden, w.wk_weights)
        compare(q[0].reshape(-1), (w.wq.double()@qr[0].double()).bfloat16(), 'original_q_projection', tolerance=.005)
        k = torch.empty((rows, 128), dtype=torch.bfloat16, device='cuda')
        idx.normalize_keys(kw[:, :128], w.norm_weight, w.norm_bias, k)
        from vllm.amos_stable_attention_norm import _layer as native_norm
        import tensorfold.families.glm5_next.cuda.sparse as radix
        report['reused_radix_source'] = dict(path=radix.__file__, sha256=hashlib.sha256(Path(radix.__file__).read_bytes()).hexdigest())
        compare(k, native_norm(kw[:, :128], w.norm_weight, w.norm_bias), 'native_key_norm', exact=True)
        x = kw[:, :128].double(); centered = x-x.mean(-1, keepdim=True)
        expected = centered*torch.rsqrt(centered.square().mean(-1, keepdim=True)+1e-6)*w.norm_weight.double()+w.norm_bias.double()
        compare(k, expected.bfloat16(), 'key_norm_float64', tolerance=.001)
        qo, ko = torch.empty_like(q), torch.empty_like(k)
        # Include zero and very small inputs before RoPE rounding.
        q[11].zero_(); k[11].zero_()
        q[12].mul_(1e-12); k[12].mul_(1e-12)
        attn.apply_rope(q, k, positions, table, qo, ko, real_heads=32, indexer=True)
        q8 = torch.empty_like(q, dtype=torch.float8_e4m3fn)
        weights = kw[:, 128:]
        wo = torch.empty((rows, 32), dtype=torch.float32, device='cuda')
        idx.quantize_queries(qo, weights, q8, wo)
        from vllm.model_executor.layers.sparse_attn_indexer import fused_indexer_q_rope_quant
        nq, nw = fused_indexer_q_rope_quant(positions, q, table, weights, 128**-.5, 32**-.5, False)
        compare(q8, nq, 'native_query_fp8', exact=True)
        compare(wo, nw, 'native_query_weights', exact=True)
        from vllm import _custom_ops as native_ops
        native_binary = Path(importlib.util.find_spec('vllm._C_stable_libtorch').origin)
        report['native_binary'] = dict(path=str(native_binary), sha256=hashlib.sha256(native_binary.read_bytes()).hexdigest())
        native_cache = torch.zeros((1, 64, 132), dtype=torch.uint8, device='cuda')
        slots = torch.arange(rows, dtype=torch.int64, device='cuda')
        slots[-1] = -1
        ck = torch.zeros((64, 128), dtype=torch.float8_e4m3fn, device='cuda')
        cs = torch.zeros(64, dtype=torch.float32, device='cuda')
        idx.write_keys(ko, slots, ck, cs)
        native_ops.indexer_k_quant_and_cache(ko, native_cache, slots, 128, 'ue8m0')
        flat = native_cache.view(-1)
        compare(ck, flat[:8192].view(torch.float8_e4m3fn).reshape(64,128), 'native_key_fp8', exact=True)
        compare(cs, flat[8192:].view(torch.float32), 'native_key_scales_zero_tiny_padded', exact=True)
        # Native and independent FP64 scoring on real-weight queries.
        cap = 8192
        cache = torch.randn((cap, 128), dtype=torch.bfloat16, device='cuda').to(torch.float8_e4m3fn)
        scales = torch.full((cap,), 2**-5, dtype=torch.float32, device='cuda')
        pos = torch.tensor([0, 511, 2047, 2048, 4095, 4095, 4095, 2048, 511, 8191, 0, 4095, 2048, 4095, 8191, 4095],
                           dtype=torch.int64, device='cuda')
        base = torch.zeros_like(pos); base[4:6] = 4096
        s = idx.IndexerScratch(rows, 8192, 'cuda')
        tokens, counts = idx.select_tokens(q8, wo, cache, scales, pos, base, s)
        saved_tokens, saved_counts = tokens.clone(), counts.clone()
        score_saved = s.scores.clone()
        from b12x.attention.dsa_indexer import logits_contiguous, ContiguousMetadata
        native = logits_contiguous(q_fp8=q8, weights=wo, kv_fp8=(cache, scales),
            metadata=ContiguousMetadata(k_start=base.int(), k_end=(base+pos+1).int()), preinitialize_invalid_logits=True)
        for row in range(rows):
            n, b = int(pos[row])+1, int(base[row])
            prod = q8[row].double()@cache[b:b+n].double().T
            ref = (prod.relu()*wo[row].double()[:,None]).sum(0)*scales[b:b+n].double()
            compare(score_saved[row,:n], ref, f'fp64_score_row{row}')
            compare(score_saved[row,:n], native[row,b:b+n], f'native_score_row{row}')
        et, ec = oracle(score_saved, pos, base, cap)
        compare(tokens, et, 'scored_selection_oracle', exact=True)
        compare(counts, ec, 'scored_count_oracle', exact=True)
        logical_native = torch.full_like(score_saved, -float('inf'))
        for row in range(rows):
            n,b = int(pos[row])+1,int(base[row])
            logical_native[row,:n] = native[row,b:b+n]
        nt,nc = oracle(logical_native,pos,base,cap)
        compare(saved_tokens, nt, 'native_scored_canonical_ids', exact=True)
        compare(saved_counts, nc, 'native_scored_counts', exact=True)
        for row in (0, 3, 4, 9, 11, 15):
            t, c = idx.select_tokens(q8[row:row+1], wo[row:row+1], cache, scales, pos[row:row+1], base[row:row+1], s)
            compare(t, saved_tokens[row:row+1], f'serial_row{row}', exact=True)
        # Selector edge values and large tied boundaries, independent stable CPU sort.
        adversarial = torch.randint(-4, 5, (4, 8192), device='cuda').float()
        adversarial[0].zero_(); adversarial[1,::2] = -0.; adversarial[1,1::2] = 0.
        adversarial[2,:2048] = 3.; adversarial[2,2048:4096] = torch.nextafter(torch.tensor(3.,device='cuda'),torch.tensor(4.,device='cuda'))
        ap = torch.tensor([8191, 8191, 8191, -1], dtype=torch.int64, device='cuda')
        adversarial[3].fill_(-float('inf'))
        select_oracle(adversarial, ap, torch.zeros_like(ap), 8192, s, 'signed_zero_adjacent_float_ties_empty')
        # Changed inputs and request positions under CUDA graph replay.
        stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream): idx.select_tokens(q8, wo, cache, scales, pos, base, s)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream): idx.select_tokens(q8, wo, cache, scales, pos, base, s)
        for change in range(3):
            q8.copy_(q8.float().roll(1, 0).to(q8.dtype)); wo.copy_(wo.roll(1,0))
            pos.copy_(pos.roll(1,0)); base.copy_(base.roll(1,0))
            graph.replay()
            gt, gc = s.tokens.clone(), s.counts.clone()
            idx.select_tokens(q8, wo, cache, scales, pos, base, s)
            compare(s.tokens, gt, f'graph_change{change}_tokens', exact=True)
            compare(s.counts, gc, f'graph_change{change}_counts', exact=True)
        del graph
        # Independent workspaces on distinct streams.
        sb = idx.IndexerScratch(rows,8192,'cuda'); stream_b = torch.cuda.Stream()
        wb = (-wo*.5).contiguous()
        idx.select_tokens(q8,wb,cache,scales,pos,base,sb); bt,bc=sb.tokens.clone(),sb.counts.clone()
        idx.select_tokens(q8,wo,cache,scales,pos,base,s); at,ac=s.tokens.clone(),s.counts.clone()
        stream.wait_stream(torch.cuda.current_stream()); stream_b.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream): idx.select_tokens(q8,wo,cache,scales,pos,base,s)
        with torch.cuda.stream(stream_b): idx.select_tokens(q8,wb,cache,scales,pos,base,sb)
        torch.cuda.synchronize()
        compare(s.tokens,at,'stream_a',exact=True); compare(sb.tokens,bt,'stream_b',exact=True)
        # Long-context score/selector; still a small bounded component cache.
        longcap = 360020+4096
        kl = torch.randn((longcap,128),dtype=torch.bfloat16,device='cuda').to(torch.float8_e4m3fn)
        sl = torch.full((longcap,),2**-5,dtype=torch.float32,device='cuda')
        lp = torch.tensor([359999,360000,4095,0],dtype=torch.int64,device='cuda')
        lb = torch.tensor([0,0,360020,0],dtype=torch.int64,device='cuda')
        ls = idx.IndexerScratch(4,360001,'cuda')
        idx.select_tokens(q8[:4],wo[:4],kl,sl,lp,lb,ls)
        et,ec=oracle(ls.scores,lp,lb,longcap)
        compare(ls.tokens,et,'360K_selected_ids',exact=True); compare(ls.counts,ec,'360K_counts',exact=True)
        # Long all-equal tie boundary spans many selector chunks.
        ls.scores.fill_(2.)
        for r in range(4): ls.scores[r,int(lp[r])+1:] = -float('inf')
        select_oracle(ls.scores,lp,lb,longcap,ls,'360K_equal_scores')
        # Model3072 rows with mixed sparse/dense positions, requests and query values.
        bq=q8.repeat(3072//rows,1,1); bw=wo.repeat(3072//rows,1)
        bp=pos.repeat(3072//rows); bb=base.repeat(3072//rows)
        bs=idx.IndexerScratch(3072,8192,'cuda')
        idx.select_tokens(bq,bw,cache,scales,bp,bb,bs)
        for r in (0,15,16,127,128,1535,1536,3071):
            compare(bs.tokens[r:r+1],at[r%rows:r%rows+1],f'batch3072_row{r}',exact=True)
            compare(bs.counts[r:r+1],ac[r%rows:r%rows+1],f'batch3072_count{r}',exact=True)
        save('complete_indexer_forward')
        layer = idx.Indexer(w)
        nr = 32
        fh=torch.randn((nr,6144),dtype=torch.bfloat16,device='cuda')
        fq=torch.randn((nr,2048),dtype=torch.bfloat16,device='cuda')
        fp=torch.cat((torch.arange(4096,4112),torch.arange(2048,2064))).long().cuda()
        fb=torch.cat((torch.zeros(16),torch.full((16,),8192))).long().cuda()
        fslots=(fp+fb).contiguous()
        fk=torch.randn((16384,128),dtype=torch.bfloat16,device='cuda').to(torch.float8_e4m3fn)
        fks=torch.full((16384,),2**-5,dtype=torch.float32,device='cuda')
        fs=idx.IndexerForwardScratch(nr,8192,'cuda')
        ft,fc=layer.forward(fh,fq,fp,fb,fslots,fk,fks,table,fs)
        ft,fc=ft.clone(),fc.clone()
        qoriginal=fs.q.clone(); kworiginal=fs.kw.clone()
        # Independent full64-bit matmul on sampled rows, then fixed-row invariance.
        for row in (0,15,16,31):
            expected=(w.wq.double()@fq[row].double()).reshape(32,128)
            compare(qoriginal[row],expected,f'fixed_dense_q_fp64_row{row}',tolerance=.004)
            expected=w.wk_weights.double()@fh[row].double()
            compare(kworiginal[row],expected,f'fixed_dense_kw_fp64_row{row}',tolerance=.004)
            t,c=layer.forward(fh[row:row+1],fq[row:row+1],fp[row:row+1],fb[row:row+1],
                              fslots[row:row+1],fk,fks,table,fs)
            compare(fs.q[:1],qoriginal[row:row+1],f'full_serial_q_row{row}',exact=True)
            compare(fs.kw[:1],kworiginal[row:row+1],f'full_serial_kw_row{row}',exact=True)
            compare(t,ft[row:row+1],f'full_serial_tokens_row{row}',exact=True)
            compare(c,fc[row:row+1],f'full_serial_counts_row{row}',exact=True)
        # Cross the16-row projection tile boundary without changing the results.
        t,c=layer.forward(fh[:17],fq[:17],fp[:17],fb[:17],fslots[:17],fk,fks,table,fs)
        compare(t,ft[:17],'full_verify17_tokens',exact=True)
        compare(fs.q[:17],qoriginal[:17],'full_verify17_projection',exact=True)
        # Complete projection/norm/RoPE/quantize/write/score/select graph.
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream): layer.forward(fh,fq,fp,fb,fslots,fk,fks,table,fs)
        torch.cuda.current_stream().wait_stream(stream)
        full_graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(full_graph,stream=stream): layer.forward(fh,fq,fp,fb,fslots,fk,fks,table,fs)
        for change in range(3):
            fh.copy_((fh.roll(1,0)*.75).contiguous());fq.copy_((fq.roll(2,0)*.9).contiguous())
            fp.add_(1);fslots.copy_(fp+fb)
            full_graph.replay()
            gt,gc=fs.tokens.clone(),fs.counts.clone();gq=fs.q8.float().clone();gk=fk.float().clone();gks=fks.clone()
            layer.forward(fh,fq,fp,fb,fslots,fk,fks,table,fs)
            compare(fs.tokens,gt,f'full_graph{change}_tokens',exact=True)
            compare(fs.counts,gc,f'full_graph{change}_counts',exact=True)
            compare(fs.q8,gq,f'full_graph{change}_q8',exact=True)
            compare(fk,gk,f'full_graph{change}_cache',exact=True)
            compare(fks,gks,f'full_graph{change}_scales',exact=True)
        del full_graph
        ft,fc=fs.tokens.clone(),fs.counts.clone()
        bh,bq=fh.repeat(96,1),fq.repeat(96,1)
        bp,bb=fp.repeat(96),fb.repeat(96)
        bslots=torch.full_like(bp,-1) # cache already contains these query keys
        bfs=idx.IndexerForwardScratch(3072,8192,'cuda')
        layer.forward(bh,bq,bp,bb,bslots,fk,fks,table,bfs)
        for row in (0,15,16,31,32,1535,1536,3071):
            compare(bfs.tokens[row:row+1],ft[row%nr:row%nr+1],f'full_batch3072_row{row}',exact=True)
            compare(bfs.counts[row:row+1],fc[row%nr:row%nr+1],f'full_batch3072_count{row}',exact=True)
        assert not torch.distributed.is_initialized()
        report.update(passed=True,finished_at=time.time(),distributed_initialized=False)
        save('complete')
    except Exception as exc:
        report.update(error=f'{type(exc).__name__}: {exc}'); save('failed'); raise
    finally:
        faulthandler.cancel_dump_traceback_later()


if __name__ == '__main__': main()
