#!/usr/bin/env python3
"""Bounded single-GPU full-GLM MLA test; no distributed initialization or serving.

Outer controller must hold/drain admission, require13GiB host headroom, and set
4GiB cgroup/240s timeout. Real BF16 kv_b weights; synthetic queries/cache values.
"""
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
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--layer', type=int, choices=(0, 78), required=True)
    p.add_argument('--module', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
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
    spec = importlib.util.spec_from_file_location('full_glm_attention_under_test', a.module)
    attn = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = attn
    spec.loader.exec_module(attn)
    report = dict(passed=False, phase='initializing', layer=a.layer, cases=[], started_at=time.time(),
        source_manifest=manifest, scope='Real BF16 kv_b heads plus synthetic cache/query inputs; RoPE, latent projection and sparse attention components only; no indexer selection, compact cache, TP collective or full model qualification')

    def save(phase):
        report.update(phase=phase, updated_at=time.time(), peak_cuda_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        tmp = a.output.with_suffix('.tmp')
        tmp.write_text(json.dumps(report, indent=2)+'\n'); tmp.replace(a.output)
        print(json.dumps(dict(phase=phase)), flush=True)

    def compare(got, expected, label, exact=False, tolerance=.006):
        torch.cuda.synchronize()
        x, y = got.float(), expected.float()
        finite = bool(torch.isfinite(x).all()) and bool(torch.isfinite(y).all())
        same = torch.equal(x, y)
        denom = float(y.square().mean())
        relative = float((x-y).square().mean().sqrt())/math.sqrt(denom) if denom else (0. if same else float('inf'))
        cosine = float(torch.nn.functional.cosine_similarity(x.flatten(), y.flatten(), dim=0)) if denom else float(same)
        passed = finite and (same if exact else relative < tolerance and cosine > .99995)
        report['cases'].append(dict(case=label, exact_required=exact, bit_equal=same, relative_rms=relative,
                                    cosine=cosine, passed=passed))
        save(label)
        assert passed, (label, relative, cosine)

    def rope_ref(x, pos, table, offset):
        y = x.clone()
        pair = x[..., offset:offset+64].double().reshape(x.shape[0], -1, 32, 2)
        cs = table[pos].double()
        c, s = cs[:, None, :32], cs[:, None, 32:]
        out = torch.stack((pair[..., 0]*c-pair[..., 1]*s, pair[..., 1]*c+pair[..., 0]*s), -1)
        y[..., offset:offset+64] = out.reshape(x.shape[0], *x.shape[1:-1], 64).bfloat16()
        return y

    def attention_ref(qa, qr, lc, kc, tok, cnt, pos, base, real):
        out = torch.zeros_like(qa, dtype=torch.float64)
        for row in range(qa.shape[0]):
            ids = tok[row, :int(cnt[row])].long()
            ids = ids[(ids >= 0) & (ids <= pos[row])]
            slots = ids+base[row]
            slots = slots[(slots >= 0) & (slots < lc.shape[0])]
            if slots.numel() == 0:
                continue
            scores = (qa[row, :real].double()@lc[slots].double().T+
                      qr[row, :real].double()@kc[slots].double().T)/16
            out[row, :real] = scores.softmax(-1)@lc[slots].double()
        return out

    faulthandler.dump_traceback_later(235, exit=True)
    try:
        save('prepare_rope_and_real_heads')
        reader = RankPieces(a.model, 0)
        torch.manual_seed(530602+a.layer)
        table = attn.rope_table(1048576, 'cuda')
        positions = torch.tensor([0, 1, 31, 511, 512, 2047, 2048, 32767, 32768, 131071,
                                  131072, 359999, 360000, 1048575], dtype=torch.int64, device='cuda')
        rows = len(positions)
        from vllm import _custom_ops as native_ops
        native_binary = Path(importlib.util.find_spec('vllm._C_stable_libtorch').origin)
        report['native_rope_binary'] = dict(path=str(native_binary),
            sha256=hashlib.sha256(native_binary.read_bytes()).hexdigest())
        for indexer, heads, dim, kd, offset in ((False, 11, 256, 64, 192), (True, 32, 128, 128, 0)):
            q = torch.randn((rows, heads, dim), dtype=torch.bfloat16, device='cuda')
            k = torch.randn((rows, kd), dtype=torch.bfloat16, device='cuda')
            qo, ko = torch.empty_like(q), torch.empty_like(k)
            attn.apply_rope(q, k, positions, table, qo, ko, real_heads=heads, indexer=indexer)
            compare(qo, rope_ref(q, positions, table, offset), f'rope_q_indexer{indexer}', exact=True)
            compare(ko, rope_ref(k, positions, table, 0), f'rope_k_indexer{indexer}', exact=True)
            nq, nk = q[..., offset:offset+64].contiguous(), k[:, :64].unsqueeze(1).contiguous()
            native_ops.rotary_embedding(positions, nq, nk, 64, table, False)
            compare(qo[..., offset:offset+64], nq, f'native_rope_q_indexer{indexer}', exact=True)
            compare(ko[:, :64], nk[:, 0], f'native_rope_k_indexer{indexer}', exact=True)
        del q, k, qo, ko
        save('prepare_sparse_cache')
        extent = 360020
        cap = extent+4096
        lc = torch.randn((cap, 512), dtype=torch.bfloat16, device='cuda')
        kc = torch.randn((cap, 64), dtype=torch.bfloat16, device='cuda')
        positions = torch.tensor([0, 511, 512, 2047, 2048, 32767, 32768, 131071, 131072,
                                  359999, 360000, 4095, 0, 2048, 512, 359999, 2047, 131071, 4095, 2048],
                                 dtype=torch.int64, device='cuda')
        rows = len(positions)
        bases = torch.zeros(rows, dtype=torch.int64, device='cuda'); bases[11] = extent; bases[18] = extent
        counts = torch.tensor([min(int(p)+1, 2048) for p in positions.cpu()], dtype=torch.int32, device='cuda')
        tokens = torch.full((rows, 2048), -1, dtype=torch.int32, device='cuda')
        for row in range(rows):
            n, pos = int(counts[row]), int(positions[row])
            tokens[row, :n] = torch.linspace(0, pos, n, device='cuda').round().int()
        # Mixed empty/negative/future keys, including fully absent rows.
        counts[12] = 0
        tokens[13, -1] = positions[13]+1
        tokens[14].fill_(-1)
        q = torch.randn((rows, 11, 256), dtype=torch.bfloat16, device='cuda')
        kr = torch.randn((rows, 64), dtype=torch.bfloat16, device='cuda')
        qr_full, kr_rotated = torch.empty_like(q), torch.empty_like(kr)
        for rank, real in ((0, 11), (5, 9)):
            save(f'prepare_rank{rank}')
            wk, wv = attn.load_head_weights(reader, a.layer, rank)
            if real < 11:
                q[:, real:] = float('nan')
                wk[real:] = float('nan'); wv[real:] = float('nan')
            attn.apply_rope(q, kr, positions, table, qr_full, kr_rotated, real_heads=real)
            qa = torch.empty((rows, 11, 512), dtype=torch.bfloat16, device='cuda')
            expected = torch.zeros_like(qa, dtype=torch.float64)
            expected[:, :real] = torch.einsum('rhd,hdl->rhl', qr_full[:, :real, :192].double(), wk[:real].double())
            attn.absorb_query(qr_full, wk, qa, real_heads=real)
            compare(qa, expected, f'rank{rank}_real_kv_b_absorb')
            qr = qr_full[..., 192:].contiguous()
            s = attn.AttentionScratch(rows, 'cuda', real_heads=real)
            out = torch.full_like(qa, float('nan'))
            attn.attend(qa, qr, lc, kc, tokens, counts, positions, bases, out, s)
            expected = attention_ref(qa, qr, lc, kc, tokens, counts, positions, bases, real)
            compare(out, expected, f'rank{rank}_float64_attention')
            for row in range(rows):
                compare(out[row:row+1], expected[row:row+1],
                        f'rank{rank}_position{int(positions[row])}_row{row}_reference')
            assert not torch.count_nonzero(out[12]) and not torch.count_nonzero(out[14])
            if real < 11:
                assert not torch.count_nonzero(out[:, real:])
            reference = out.clone()
            for row in (0, 2, 4, 8, 10, 11, 12, 13, 14, 19):
                attn.attend(qa[row:row+1], qr[row:row+1], lc, kc, tokens[row:row+1], counts[row:row+1],
                            positions[row:row+1], bases[row:row+1], out[row:row+1], s)
                compare(out[row:row+1], reference[row:row+1], f'rank{rank}_serial{row}', exact=True)
            expanded = torch.empty((rows, 11, 256), dtype=torch.bfloat16, device='cuda')
            attn.expand_value(out, wv, expanded, real_heads=real)
            expected = torch.zeros_like(expanded, dtype=torch.float64)
            expected[:, :real] = torch.einsum('rhl,hvl->rhv', out[:, :real].double(), wv[:real].double())
            compare(expanded, expected, f'rank{rank}_real_kv_b_expand')
            # Changing positions/counts/request extent under a captured graph.
            graph_out = torch.empty_like(out)
            stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                attn.attend(qa, qr, lc, kc, tokens, counts, positions, bases, graph_out, s)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                attn.attend(qa, qr, lc, kc, tokens, counts, positions, bases, graph_out, s)
            saved = (qa.clone(), tokens.clone(), counts.clone(), positions.clone(), bases.clone())
            for change in range(4):
                qa.copy_(saved[0]*(change+1)/4)
                tokens.copy_(saved[1].roll(change, 0)); counts.copy_(saved[2].roll(change, 0))
                positions.copy_(saved[3].roll(change, 0)); bases.copy_(saved[4].roll(change, 0))
                graph.replay()
                attn.attend(qa, qr, lc, kc, tokens, counts, positions, bases, out, s)
                compare(graph_out, out, f'rank{rank}_graph_change{change}', exact=True)
            for target, source in zip((qa, tokens, counts, positions, bases), saved): target.copy_(source)
            # Distinct scratch buffers on two simultaneous CUDA streams.
            stream_b = torch.cuda.Stream()
            s_b = attn.AttentionScratch(rows, 'cuda', real_heads=real)
            qa_b = (-qa*.75).contiguous()
            out_b = torch.empty_like(out)
            expected_b = torch.empty_like(out)
            attn.attend(qa_b, qr, lc, kc, tokens, counts, positions, bases, expected_b, s_b)
            attn.attend(qa, qr, lc, kc, tokens, counts, positions, bases, out, s)
            expected_a = out.clone()
            stream.wait_stream(torch.cuda.current_stream()); stream_b.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                attn.attend(qa, qr, lc, kc, tokens, counts, positions, bases, out, s)
            with torch.cuda.stream(stream_b):
                attn.attend(qa_b, qr, lc, kc, tokens, counts, positions, bases, out_b, s_b)
            torch.cuda.synchronize()
            compare(out, expected_a, f'rank{rank}_stream_a', exact=True)
            compare(out_b, expected_b, f'rank{rank}_stream_b', exact=True)
            # 3072 configured model rows remain supported with128-row partials.
            copies = (3072+rows-1)//rows
            batch = [v.repeat((copies,)+(1,)*(v.ndim-1))[:3072].contiguous()
                     for v in (qa, qr, tokens, counts, positions, bases)]
            batch_out = torch.empty((3072, 11, 512), dtype=torch.bfloat16, device='cuda')
            batch_s = attn.AttentionScratch(3072, 'cuda', real_heads=real)
            attn.attend(batch[0], batch[1], lc, kc, *batch[2:], batch_out, batch_s)
            for row in (0, 127, 128, 1535, 1536, 3071):
                compare(batch_out[row:row+1], expected_a[row%rows:row%rows+1],
                        f'rank{rank}_batch3072_row{row}', exact=True)
            del batch, batch_out, batch_s, s_b, qa_b, out_b, expected_a, expected_b
            del graph, saved, s, out, graph_out, qa, qr, expected, expanded, reference, wk, wv
        assert not torch.distributed.is_initialized()
        report.update(passed=True, finished_at=time.time(), distributed_initialized=False)
        save('complete')
    except Exception as exc:
        report.update(error=f'{type(exc).__name__}: {exc}')
        save('failed')
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()


if __name__ == '__main__':
    main()
