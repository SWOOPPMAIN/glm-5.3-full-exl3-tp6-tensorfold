"""Full GLM token indexer: original32x128 heads, RoPE, FP8 and exact top2048.

No pooled Flash tokens or gate weighting. Outputs are ascending logical token
IDs. A request planner owns contiguous cache extents, validates positions and
unique write slots, and scopes selections to one forward (MTP has its own).
Scratch is per stream and shared across layers, with bounded score row chunks.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from .indexer_plan import score_width

from tensorfold.families.glm5_next.cuda.sparse import _select_split, split_chunk, split_chunks


class IndexerWeights:
    def __init__(self, reader, layer, device='cuda'):
        prefix = f'model.layers.{layer}.self_attn.indexer.'
        shapes = {'wq_b.weight': (4096, 2048), 'wk.weight': (128, 6144),
                  'weights_proj.weight': (32, 6144), 'k_norm.weight': (128,), 'k_norm.bias': (128,)}
        values = {}
        for name, shape in shapes.items():
            meta = reader.tensor_meta(prefix+name)
            if meta['dtype'] != 'BF16' or tuple(meta['shape']) != shape:
                raise ValueError('Expected original full GLM BF16 indexer weights')
            values[name] = reader.read_tensor(prefix+name, device=device)
        self.wq = values['wq_b.weight']
        self.wk_weights = torch.cat((values['wk.weight'], values['weights_proj.weight']), dim=0)
        # Native keeps the original BF16 values in FP32 norm parameters.
        self.norm_weight = values['k_norm.weight'].float()
        self.norm_bias = values['k_norm.bias'].float()


@triton.jit
def _norm(X, W, B, Y, R, STRIDE: tl.constexpr):
    r = tl.program_id(0)*8+tl.arange(0, 8)[:, None]
    d = tl.arange(0, 128)[None, :]
    x = tl.load(X+r*STRIDE+d, r < R, 0).to(tl.float32)
    mean = tl.sum(tl.where(r < R, x, 0), 1)[:, None]/128
    centered = x-mean
    var = tl.sum(tl.where(r < R, centered*centered, 0), 1)[:, None]/128
    y = centered*libdevice.rsqrt(var+1e-6)*tl.load(W+d)+tl.load(B+d)
    tl.store(Y+r*128+d, y, r < R)


def normalize_keys(k, weight, bias, out):
    rows = k.shape[0]
    if (k.shape != (rows, 128) or out.shape != k.shape or k.stride(1) != 1
            or k.dtype != torch.bfloat16 or out.dtype != torch.bfloat16
            or weight.shape != (128,) or bias.shape != (128,)
            or weight.dtype != torch.float32 or bias.dtype != torch.float32
            or not all(t.is_cuda and t.device == k.device for t in (k, weight, bias, out))
            or not all(t.is_contiguous() for t in (weight, bias, out))):
        raise ValueError('Expected BF16 keys and FP32 indexer norm parameters')
    _norm[(triton.cdiv(rows, 8),)](k, weight, bias, out, rows, k.stride(0), num_warps=2, num_stages=1)
    return out


@triton.jit
def _quant_q(Q, W, Q8, WO, R, WS: tl.constexpr, softmax_scale, head_scale):
    r, h = tl.program_id(0), tl.program_id(1)
    d = tl.arange(0, 128)
    q = tl.load(Q+(r*32+h)*128+d).to(tl.float32)
    scale = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.max(tl.abs(q), 0), 1e-10)*(1./448.))))
    tl.store(Q8+(r*32+h)*128+d, tl.minimum(tl.maximum(q/scale, -448.), 448.))
    w = tl.load(W+r*WS+h).to(tl.float32)
    tl.store(WO+r*32+h, w*scale*softmax_scale*head_scale)


def quantize_queries(rotated_q, weights, q8, scaled_weights):
    """Quantize already BF16-rounded RoPE queries; head scales are UE8M0."""
    r = rotated_q.shape[0]
    if (rotated_q.shape != (r, 32, 128) or q8.shape != rotated_q.shape
            or rotated_q.dtype != torch.bfloat16 or q8.dtype != torch.float8_e4m3fn
            or weights.shape != (r, 32) or scaled_weights.shape != weights.shape
            or weights.dtype != torch.bfloat16 or scaled_weights.dtype != torch.float32
            or weights.stride(1) != 1
            or not all(t.is_contiguous() for t in (rotated_q, q8, scaled_weights))
            or not all(t.is_cuda and t.device == rotated_q.device for t in (rotated_q, weights, q8, scaled_weights))):
        raise ValueError('Invalid full GLM query quantization tensors')
    _quant_q[(r, 32)](rotated_q, weights, q8, scaled_weights, r, weights.stride(0), 128**-.5, 32**-.5,
                       num_warps=1)
    return q8, scaled_weights


@triton.jit
def _write_k(K, SLOTS, CACHE, SCALES, CAP: tl.constexpr):
    r = tl.program_id(0)
    slot = tl.load(SLOTS+r).to(tl.int64)
    if slot >= 0 and slot < CAP:
        d = tl.arange(0, 128)
        k = tl.load(K+r*128+d).to(tl.float32)
        # Native indexer cache uses1e-4 before division; Q uses1e-10.
        scale = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.max(tl.abs(k), 0), 1e-4)/448.)))
        tl.store(CACHE+slot*128+d, tl.minimum(tl.maximum(k/scale, -448.), 448.))
        tl.store(SCALES+slot, scale)


def write_keys(rotated_k, slots, cache, scales):
    rows, capacity = rotated_k.shape[0], cache.shape[0]
    if (rotated_k.shape != (rows, 128) or cache.shape != (capacity, 128)
            or slots.shape != (rows,) or slots.dtype != torch.int64
            or scales.shape != (capacity,) or scales.dtype != torch.float32
            or rotated_k.dtype != torch.bfloat16 or cache.dtype != torch.float8_e4m3fn
            or not all(t.is_cuda and t.is_contiguous() and t.device == rotated_k.device
                       for t in (rotated_k, slots, cache, scales))):
        raise ValueError('Invalid full GLM key cache tensors')
    _write_k[(rows,)](rotated_k, slots, cache, scales, capacity, num_warps=1)


@triton.jit
def _score(Q, W, K, KS, POS, BASE, OUT, CAP: tl.constexpr, WIDTH: tl.constexpr, TILE: tl.constexpr):
    r, c = tl.program_id(0), tl.program_id(1)
    j = c*TILE+tl.arange(0, TILE)
    d, h = tl.arange(0, 128), tl.arange(0, 32)
    pos, base = tl.load(POS+r).to(tl.int64), tl.load(BASE+r).to(tl.int64)
    slot = base+j
    valid = (j < WIDTH) & (j <= pos) & (slot >= 0) & (slot < CAP)
    q = tl.load(Q+(r*32+h[:, None])*128+d[None, :]).to(tl.bfloat16)
    k = tl.load(K+slot[:, None]*128+d[None, :], valid[:, None], 0.).to(tl.bfloat16)
    products = tl.dot(q, tl.trans(k))
    w = tl.load(W+r*32+h)
    score = tl.sum(tl.maximum(products, 0)*w[:, None], 0)*tl.load(KS+slot, valid, 0)
    tl.store(OUT+r*WIDTH+j, tl.where(valid, score, float('-inf')), j < WIDTH)


@triton.jit
def _finish_indices(IDS, POS, BASE, TOK, COUNT, CAP: tl.constexpr, WIDTH: tl.constexpr):
    r = tl.program_id(0)
    d = tl.arange(0, 2048)
    pos, base = tl.load(POS+r).to(tl.int64), tl.load(BASE+r).to(tl.int64)
    n = tl.maximum(0, tl.minimum(tl.minimum(pos+1, CAP-base), WIDTH))
    ids = tl.load(IDS+r*2048+d)
    tl.store(TOK+r*2048+d, tl.where((ids < n) & (base >= 0), ids, -1).to(tl.int32))
    tl.store(COUNT+r, tl.where(base >= 0, tl.minimum(n, 2048), 0).to(tl.int32))


@triton.jit
def _all_visible(POS, BASE, TOK, COUNT, CAP: tl.constexpr):
    # With at most2048 visible keys, every visible ID belongs to the top2048,
    # independently of its score. Preserve canonical ascending ID order.
    r=tl.program_id(0);d=tl.arange(0,2048)
    pos=tl.load(POS+r).to(tl.int64);base=tl.load(BASE+r).to(tl.int64)
    n=tl.where(base>=0,tl.maximum(0,tl.minimum(tl.minimum(pos+1,CAP-base),2048)),0)
    tl.store(TOK+r*2048+d,tl.where(d<n,d,-1).to(tl.int32))
    tl.store(COUNT+r,n.to(tl.int32))


class IndexerScratch:
    def __init__(self, rows, capacity, device, part_rows=16):
        if (type(rows) is not int or not 1 <= rows <= 3072 or type(capacity) is not int
                or not 1 <= capacity <= 1048576 or type(part_rows) is not int or not 1 <= part_rows <= 32):
            raise ValueError('Invalid full GLM indexer scratch capacity')
        self.rows, self.capacity = rows, capacity
        self.part_rows, self.width = min(rows, part_rows), max(2048, capacity)
        self.cp, self.chs = split_chunks(self.width), split_chunk(self.width)
        n = self.part_rows
        self.scores = torch.empty((n, self.width), dtype=torch.float32, device=device)
        self.tot = torch.empty((n, 1024), dtype=torch.int32, device=device)
        self.hist = torch.empty((n, self.cp, 256), dtype=torch.int32, device=device)
        self.gt = torch.empty((n, self.cp), dtype=torch.int32, device=device)
        self.ids = torch.empty((n, 2048), dtype=torch.int64, device=device)
        self.tokens = torch.empty((rows, 2048), dtype=torch.int32, device=device)
        self.counts = torch.empty(rows, dtype=torch.int32, device=device)


def select_scores(scores, positions, bases, cache_capacity, tokens, counts, scratch):
    """Select exactly highest scores; ties favor lower IDs, output in ID order.

    Input scores must be finite for visible IDs, -inf elsewhere. This low-level
    entry supports tests and scoring fusion; the normal entry is select_tokens.
    """
    rows, width = scores.shape
    if (not 2048 <= width <= scratch.width or not 1 <= rows <= scratch.part_rows or scores.dtype != torch.float32
            or not scores.is_contiguous() or tokens.shape != (rows, 2048) or counts.shape != (rows,)):
        raise ValueError('Invalid indexer selection scratch or scores')
    cp,chs=split_chunks(width),split_chunk(width)
    scratch.tot.zero_()
    for step in range(5):
        _select_split[(rows, cp)](scores, width, width, positions, counts,
            scratch.tot, scratch.hist, scratch.gt, scratch.ids, tokens, counts,
            W=2048, K=2048, PL=1, CHS=chs, CP=cp, STEP=step,
            SEG=False, VIS=False, TOKS=False, num_warps=4)
    _finish_indices[(rows,)](scratch.ids, positions, bases, tokens, counts, cache_capacity, width, num_warps=4)


def select_tokens(q8, weights, cache, scales, positions, bases, scratch, *, visible_tokens=None):
    """Return tokens/counts owned by scratch; copy if needed beyond next call.

    Query rows may refer to distinct request extents. The planner validates each
    extent and restricts position to its length. visible_tokens is a host-verified
    upper bound on max(positions)+1, including every graph replay. No GPU sync
    or distributed communication is introduced; omitted bounds scan capacity.
    """
    rows, cap = q8.shape[0], cache.shape[0]
    if (q8.shape != (rows, 32, 128) or weights.shape != (rows, 32) or cache.shape != (cap, 128)
            or scales.shape != (cap,) or positions.shape != (rows,) or bases.shape != (rows,)
            or positions.dtype != torch.int64 or bases.dtype != torch.int64
            or q8.dtype != torch.float8_e4m3fn or cache.dtype != q8.dtype
            or weights.dtype != torch.float32 or scales.dtype != torch.float32
            or not 1 <= rows <= scratch.rows
            or not all(t.is_cuda and t.is_contiguous() and t.device == q8.device for t in
                (q8, weights, cache, scales, positions, bases, scratch.scores, scratch.tokens))):
        raise ValueError('Invalid full GLM token indexer tensors')
    width=score_width(scratch.capacity,visible_tokens)
    if visible_tokens is not None and visible_tokens<=2048:
        _all_visible[(rows,)](positions,bases,scratch.tokens,scratch.counts,cap,num_warps=4)
        return scratch.tokens[:rows],scratch.counts[:rows]
    for start in range(0, rows, scratch.part_rows):
        end = min(rows, start+scratch.part_rows)
        n = end-start
        scores=scratch.scores.view(-1)[:n*width].view(n,width)
        _score[(n, triton.cdiv(width, 64))](q8[start:end], weights[start:end], cache, scales,
            positions[start:end], bases[start:end], scores, cap, width, 64, num_warps=4,
            enable_fp_fusion=False)
        select_scores(scores, positions[start:end], bases[start:end], cap,
                      scratch.tokens[start:end], scratch.counts[start:end], scratch)
    return scratch.tokens[:rows], scratch.counts[:rows]


class IndexerForwardScratch(IndexerScratch):
    """Complete indexer projection buffers, reused across full-indexer layers."""
    def __init__(self, rows, capacity, device, part_rows=16):
        super().__init__(rows, capacity, device, part_rows)
        self.q = torch.empty((rows,32,128),dtype=torch.bfloat16,device=device)
        self.q_rotated = torch.empty_like(self.q)
        self.q8 = torch.empty_like(self.q,dtype=torch.float8_e4m3fn)
        self.kw = torch.empty((rows,160),dtype=torch.bfloat16,device=device)
        self.k = torch.empty((rows,128),dtype=torch.bfloat16,device=device)
        self.k_rotated = torch.empty_like(self.k)
        self.scaled_weights = torch.empty((rows,32),dtype=torch.float32,device=device)


class Indexer:
    """Full-indexer forward, given hidden rows and already normalized q-LoRA.

    Each owning layer has original weights and its own FP8 cache. Shared layers
    reuse this forward's selection, not cache keys from another layer. MTP owns
    a separate indexer/cache. Slot admission and speculative commit are external.
    ``slots`` must be unique for active rows; -1 means no cache write.
    """
    def __init__(self, weights):
        self.weights = weights

    def forward(self, hidden, q_lora, positions, bases, slots, cache, scales, table, scratch, *, visible_tokens=None):
        from .attention import apply_rope
        from .dense import linear
        rows = hidden.shape[0]
        if (hidden.shape != (rows,6144) or q_lora.shape != (rows,2048)
                or not 1 <= rows <= scratch.rows):
            raise ValueError('Invalid full GLM indexer input geometry')
        w = self.weights
        q,kw = scratch.q[:rows],scratch.kw[:rows]
        k,kr = scratch.k[:rows],scratch.k_rotated[:rows]
        qr,q8,wo = scratch.q_rotated[:rows],scratch.q8[:rows],scratch.scaled_weights[:rows]
        linear(q_lora,w.wq,q.view(rows,4096))
        linear(hidden,w.wk_weights,kw)
        normalize_keys(kw[:,:128],w.norm_weight,w.norm_bias,k)
        apply_rope(q,k,positions,table,qr,kr,real_heads=32,indexer=True)
        quantize_queries(qr,kw[:,128:],q8,wo)
        write_keys(kr,slots,cache,scales)
        return select_tokens(q8,wo,cache,scales,positions,bases,scratch,visible_tokens=visible_tokens)
