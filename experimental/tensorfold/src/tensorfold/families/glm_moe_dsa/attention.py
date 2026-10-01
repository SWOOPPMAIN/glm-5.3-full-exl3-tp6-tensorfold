"""Full GLM's RoPE MLA components for TP6; no Flash NoPE or pooled indexer.

Both BF16 reference and compact FP8 latent caches are supported. Full serving
still needs integration and a qualified request/cache admission planner.
All rows use the same reduction order for serial/draft verification consistency.
"""
import torch
import triton
import triton.language as tl

LATENT = tl.constexpr(512)
ROPE = tl.constexpr(64)
NOPE = tl.constexpr(192)
VALUE = tl.constexpr(256)
CHUNK = tl.constexpr(512)
TILE = tl.constexpr(32)


def load_head_weights(reader, layer, rank, device='cuda'):
    """Original BF16 kv_b rows, sliced into11 slots with two padded on rank5."""
    first, last, slots = reader.config.head_range(rank)
    name = f'model.layers.{layer}.self_attn.kv_b_proj.weight'
    if reader.tensor_meta(name)['shape'] != [64*448, 512] or reader.tensor_meta(name)['dtype'] != 'BF16':
        raise ValueError('Expected original full GLM BF16 kv_b rows')
    whole = reader.read_tensor(name).view(64, 448, 512)
    local = torch.zeros((slots, 448, 512), dtype=torch.bfloat16, device=device)
    local[:last-first].copy_(whole[first:last])
    return local[:, :192].contiguous(), local[:, 192:].contiguous()


def rope_table(capacity, device):
    """Full-model default RoPE: FP32 frequencies/trig, stored as BF16 pairs."""
    if type(capacity) is not int or not 1 <= capacity <= 1048576:
        raise ValueError('RoPE capacity outside full GLM model range')
    inv = 1.0 / (8000000.0 ** (torch.arange(0, 64, 2, dtype=torch.float32, device=device)/64))
    phase = torch.arange(capacity, dtype=torch.float32, device=device)[:, None]*inv[None, :]
    return torch.cat((phase.cos(), phase.sin()), dim=1).bfloat16()


@triton.jit
def _rope(Q, K, POS, TABLE, QOUT, KOUT, H: tl.constexpr, REAL: tl.constexpr,
          QD: tl.constexpr, KD: tl.constexpr, OFFSET: tl.constexpr, BLOCK: tl.constexpr):
    r, h = tl.program_id(0), tl.program_id(1)
    d = tl.arange(0, BLOCK)
    pos = tl.load(POS+r).to(tl.int64)
    if h < H:
        rotate = (d >= OFFSET) & (d < OFFSET+64)
        x = tl.load(Q+(r*H+h)*QD+d, (d < QD) & (h < REAL), 0).to(tl.float32)
        partner = tl.load(Q+(r*H+h)*QD+(d ^ 1), rotate & (h < REAL), 0).to(tl.float32)
        f = (d-OFFSET)//2
        c = tl.load(TABLE+pos*64+f, rotate, 0).to(tl.float32)
        s = tl.load(TABLE+pos*64+32+f, rotate, 0).to(tl.float32)
        y = tl.where(rotate, x*c+tl.where(d % 2 == 0, -partner, partner)*s, x)
        tl.store(QOUT+(r*H+h)*QD+d, y, d < QD)
    else:
        x = tl.load(K+r*KD+d, d < KD, 0).to(tl.float32)
        partner = tl.load(K+r*KD+(d ^ 1), d < 64, 0).to(tl.float32)
        c = tl.load(TABLE+pos*64+d//2, d < 64, 0).to(tl.float32)
        s = tl.load(TABLE+pos*64+32+d//2, d < 64, 0).to(tl.float32)
        y = tl.where(d < 64, x*c+tl.where(d % 2 == 0, -partner, partner)*s, x)
        tl.store(KOUT+r*KD+d, y, d < KD)


def apply_rope(q, k, positions, table, q_out, k_out, *, real_heads, indexer=False):
    """Interleaved RoPE, last64 of attention Q or first64 of indexer Q/K.

    Positions are validated by the request/cache planner before capture. The
    caller supplies outputs so the operation allocates nothing during replay.
    """
    rows, heads, dim = q.shape
    expected_q, expected_k = (128, 128) if indexer else (256, 64)
    if (dim != expected_q or k.shape != (rows, expected_k) or q_out.shape != q.shape
            or k_out.shape != k.shape or not 0 < real_heads <= heads
            or positions.shape != (rows,) or positions.dtype != torch.int64
            or table.ndim != 2 or table.shape[1] != 64
            or any(t.dtype != torch.bfloat16 for t in (q, k, table, q_out, k_out))
            or not all(t.is_cuda and t.is_contiguous() and t.device == q.device
                       for t in (q, k, positions, table, q_out, k_out))):
        raise ValueError('Invalid full GLM RoPE tensors')
    _rope[(rows, heads+1)](q, k, positions, table, q_out, k_out, heads, real_heads,
                          dim, expected_k, 0 if indexer else 192, triton.next_power_of_2(dim),
                          num_warps=4, enable_fp_fusion=False)


@triton.jit
def _absorb(Q, WK, OUT, H: tl.constexpr, REAL: tl.constexpr):
    r, h, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    d = tl.arange(0, 256)
    n = block*16+tl.arange(0, 16)
    q = tl.load(Q+(r*H+h)*256+d, (d < NOPE) & (h < REAL), 0).to(tl.float32)
    w = tl.load(WK+(h*NOPE+d[:, None])*LATENT+n[None, :],
                (d[:, None] < NOPE) & (h < REAL), 0).to(tl.float32)
    y = tl.sum(q[:, None]*w, axis=0)
    tl.store(OUT+(r*H+h)*LATENT+n, y)


@triton.jit
def _expand(X, WV, OUT, H: tl.constexpr, REAL: tl.constexpr):
    r, h, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    d = tl.arange(0, LATENT)
    n = block*8+tl.arange(0, 8)
    x = tl.load(X+(r*H+h)*LATENT+d, h < REAL, 0).to(tl.float32)
    w = tl.load(WV+(h*VALUE+n[:, None])*LATENT+d[None, :], h < REAL, 0).to(tl.float32)
    y = tl.sum(w*x[None, :], axis=1)
    tl.store(OUT+(r*H+h)*VALUE+n, y)


def absorb_query(q, wk, out, *, real_heads):
    r, h, d = q.shape
    _projection_check(q, wk, out, (r, h, 256), (h, 192, 512), (r, h, 512), real_heads)
    _absorb[(r, h, 32)](q, wk, out, h, real_heads, num_warps=4, enable_fp_fusion=False)
    return out


def expand_value(latent, wv, out, *, real_heads):
    r, h, d = latent.shape
    _projection_check(latent, wv, out, (r, h, 512), (h, 256, 512), (r, h, 256), real_heads)
    _expand[(r, h, 32)](latent, wv, out, h, real_heads, num_warps=4, enable_fp_fusion=False)
    return out


def _projection_check(x, w, out, xs, ws, os, real):
    if (x.shape != xs or w.shape != ws or out.shape != os or not 0 < real <= xs[1]
            or not all(t.is_cuda and t.is_contiguous() and t.dtype == torch.bfloat16 and t.device == x.device
                       for t in (x, w, out))):
        raise ValueError('Invalid full GLM latent projection tensors')


@triton.jit
def _attention_chunks(QA, QR, LC, KC, LS, TOKENS, COUNTS, POS, BASE, PO, PM, PL,
                      ROWS: tl.constexpr, H: tl.constexpr, REAL: tl.constexpr,
                      CAP: tl.constexpr, WIDTH: tl.constexpr, COMPACT: tl.constexpr):
    r, chunk = tl.program_id(0), tl.program_id(1)
    hh = tl.arange(0, 16)
    d = tl.arange(0, 512)
    dr = tl.arange(0, 64)
    count = tl.minimum(tl.load(COUNTS+r), WIDTH)
    pos = tl.load(POS+r).to(tl.int64)
    base = tl.load(BASE+r).to(tl.int64)
    q = tl.load(QA+(r*H+hh[:, None])*512+d[None, :], hh[:, None] < REAL, 0)
    qr = tl.load(QR+(r*H+hh[:, None])*64+dr[None, :], hh[:, None] < REAL, 0)
    m = tl.full((16,), float('-inf'), tl.float32)
    z = tl.zeros((16,), tl.float32)
    o = tl.zeros((16, 512), tl.float32)
    for tile in range(CHUNK//TILE):
        i = chunk*CHUNK+tile*TILE+tl.arange(0, TILE)
        tok = tl.load(TOKENS+r*WIDTH+i, (i < count) & (i < WIDTH), -1).to(tl.int64)
        slot = base+tok
        valid = (i < count) & (i < WIDTH) & (tok >= 0) & (tok <= pos) & (slot >= 0) & (slot < CAP)
        kv = tl.load(LC+slot[:, None]*512+d[None, :], valid[:, None], 0.)
        if COMPACT:
            scale = tl.load(LS+slot[:,None]*4+(d[None,:]//128),valid[:,None],0.)
            kv = (kv.to(tl.float32)*scale).to(tl.bfloat16)
        kr = tl.load(KC+slot[:, None]*64+dr[None, :], valid[:, None], 0)
        # Scale uses original Q/K width192+64, not absorbed width512+64.
        scores = (tl.dot(q, tl.trans(kv))+tl.dot(qr, tl.trans(kr)))*0.0625
        scores = tl.where(valid[None, :] & (hh[:, None] < REAL), scores, float('-inf'))
        tile_m = tl.max(scores, axis=1)
        active = tile_m != float('-inf')
        next_m = tl.where(active, tl.maximum(m, tile_m), m)
        alpha = tl.where(active, tl.where(m == float('-inf'), 0., tl.exp(m-next_m)), 1.)
        probs = tl.where(valid[None, :] & active[:, None], tl.exp(scores-next_m[:, None]), 0.)
        o = o*alpha[:, None]+tl.dot(probs.to(tl.bfloat16), kv)
        z = z*alpha+tl.sum(probs, axis=1)
        m = next_m
    target = (chunk*ROWS+r)*H+hh
    tl.store(PO+target[:, None]*512+d[None, :], o, hh[:, None] < H)
    tl.store(PM+target, m, hh < H)
    tl.store(PL+target, z, hh < H)


@triton.jit
def _attention_merge(PO, PM, PL, OUT, ROWS: tl.constexpr, H: tl.constexpr, NCH: tl.constexpr):
    r, h = tl.program_id(0), tl.program_id(1)
    d = tl.arange(0, 512)
    m, z = float('-inf'), 0.
    o = tl.zeros((512,), tl.float32)
    for c in range(NCH):
        base = (c*ROWS+r)*H+h
        cm, cz = tl.load(PM+base), tl.load(PL+base)
        co = tl.load(PO+base*512+d)
        active = cz > 0
        nm = tl.where(active, tl.maximum(m, cm), m)
        a = tl.where(active, tl.where(m == float('-inf'), 0., tl.exp(m-nm)), 1.)
        b = tl.where(active, tl.exp(cm-nm), 0.)
        o, z, m = o*a+co*b, z*a+cz*b, nm
    # Inactive/padded rows overwrite stale scratch, including all-empty routes.
    tl.store(OUT+(r*H+h)*512+d, tl.where(z > 0, o/z, 0.))


class AttentionScratch:
    def __init__(self, rows, device, real_heads=11):
        if type(rows) is not int or not 1 <= rows <= 3072 or real_heads not in (9, 11):
            raise ValueError('Expected TP6 heads and1..3072 rows')
        self.rows, self.part_rows, self.real_heads = rows, min(rows, 128), real_heads
        n = 4*self.part_rows*11
        self.po = torch.empty((n, 512), dtype=torch.float32, device=device)
        self.pm = torch.empty(n, dtype=torch.float32, device=device)
        self.pl = torch.empty_like(self.pm)


def attend(qa, q_rope, latent_cache, rope_cache, tokens, counts, positions, bases, out, scratch, *, latent_scales=None):
    """Selected token attention with per-row sequence position and cache extent.

    ``tokens`` contains up to2048 unique logical token IDs, canonically ordered;
    ``bases`` maps each row's sequence to its contiguous physical cache extent.
    Future keys, negative padding and zero-count rows cannot read stale values.
    Each stream/graph owns its scratch. Selection and cache admission are external.
    """
    rows = qa.shape[0]
    cap = latent_cache.shape[0]
    compact = latent_scales is not None
    if compact and (latent_scales.shape != (cap,4) or latent_scales.dtype != torch.float32
                    or not latent_scales.is_cuda or not latent_scales.is_contiguous() or latent_scales.device != qa.device):
        raise ValueError('Invalid compact MLA cache scales')
    if (qa.shape != (rows, 11, 512) or q_rope.shape != (rows, 11, 64)
            or latent_cache.shape != (cap, 512) or rope_cache.shape != (cap, 64)
            or tokens.shape != (rows, 2048) or tokens.dtype != torch.int32
            or counts.shape != (rows,) or counts.dtype != torch.int32
            or positions.shape != (rows,) or positions.dtype != torch.int64
            or bases.shape != (rows,) or bases.dtype != torch.int64
            or out.shape != qa.shape or not 1 <= rows <= scratch.rows
            or any(t.dtype != torch.bfloat16 for t in (qa, q_rope, rope_cache, out))
            or latent_cache.dtype != (torch.float8_e4m3fn if compact else torch.bfloat16)
            or not all(t.is_cuda and t.is_contiguous() and t.device == qa.device for t in
                       (qa, q_rope, latent_cache, rope_cache, tokens, counts, positions, bases, out,
                        scratch.po, scratch.pm, scratch.pl))):
        raise ValueError('Invalid full GLM TP6 attention tensors')
    for start in range(0, rows, scratch.part_rows):
        stop = min(rows, start+scratch.part_rows)
        n = stop-start
        _attention_chunks[(n, 4)](qa[start:stop], q_rope[start:stop], latent_cache, rope_cache,
            latent_scales if compact else latent_cache,
            tokens[start:stop], counts[start:stop], positions[start:stop], bases[start:stop],
            scratch.po, scratch.pm, scratch.pl, n, 11, scratch.real_heads, cap, 2048, compact,
            num_warps=8, num_stages=1, enable_fp_fusion=False)
        _attention_merge[(n, 11)](scratch.po, scratch.pm, scratch.pl, out[start:stop], n, 11, 4,
                                  num_warps=4, enable_fp_fusion=False)
    return out
