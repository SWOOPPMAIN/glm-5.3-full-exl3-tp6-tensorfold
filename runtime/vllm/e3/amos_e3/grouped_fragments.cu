// Experimental GLM6 mixed K3/K4 fragment port, 2026-09-12.
// Derived from MiaAI-Lab GLM-5.3-Flash-EXL3-2x-DGX-Sparks,
// commit 9348755653f6f8cda5d56562c05462724c40fcbd, overlay/exl3_fat_moe.cu.
// Mia source: AGPL-3.0, see LICENSE.mia. Modified code is experimental;
// NOT GPU-validated, NOT enabled in serving. Individual 512-wide rotations
// and packed fragment bytes retained; no ordinary TP6 repartitioning.
// Portable device-only build: no torch host extension / no target GPU needed.

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "vendor/util.cuh"
#include "vendor/ptx.cuh"
#include "vendor/quant/exl3_dq.cuh"
#include "vendor/quant/hadamard_inner.cuh"

// Grouped fat-expert kernels. See exl3_fat_moe.cuh for the contract.
//
// Mainloop: TILE_M rows x 128 columns per CTA, K advanced 32 per pipeline
// stage with cp.async multi-buffering. B tiles (16x16 K4/MCG trellis tiles,
// 64 int16 words each) are staged through shared memory and dequantized in
// registers by the warp that owns that 16-column block, once per 16 K, and
// reused across all M blocks of the tile. The gate/up kernel runs two B
// streams (gate, up) for the same 128 intermediate columns so the SwiGLU and
// the down-projection input Hadamard fuse into its epilogue.


constexpr int FM_THREADS = 256;                 // 8 warps
constexpr int FM_WARPS = FM_THREADS / 32;
constexpr int FM_TILE_N = 128;                  // one 16-col block per warp
constexpr int FM_TILE_K = 32;                   // per pipeline stage
constexpr int FM_STAGES = 4;
constexpr int FM_B_STAGE_WORDS = 2 * FM_WARPS * 64; // maximum K4 shared allocation
constexpr float FM_HAD_SCALE = 0.088388347648f; // 1/sqrt(128)

constexpr int FM_MB_GATEUP = 4;                 // 64-row tiles, 2 B streams
constexpr int FM_MB_DOWN = 4;                   // 64-row tiles, 2 B streams (256 cols)

template <int MB, int NS>
constexpr int fm_smem_bytes()
{
    constexpr int a_stage = 2 * MB * 16 * FM_TILE_K * 2; // gate/up have distinct input rotations
    constexpr int b_stage = NS * FM_B_STAGE_WORDS * 2;
    constexpr int pipe = FM_STAGES * (a_stage + b_stage);
    constexpr int epi = 16 * NS * FM_TILE_N * 4;
    return pipe > epi ? pipe : epi;
}

// 128-element Hadamard over one row held as float4 per lane, then optional
// per-column scale. Same arithmetic as fat_had_ff_128 / had_ff_r_128_inner.
__device__ __forceinline__ void fm_had_row(float4& v, int lane)
{
    float s0 = v.x + v.y;
    float d0 = v.x - v.y;
    float s1 = v.z + v.w;
    float d1 = v.z - v.w;
    v.x = s0 + s1;
    v.y = d0 + d1;
    v.z = s0 - s1;
    v.w = d0 - d1;
    shuffle_had_f2x32(v.x, v.y, lane);
    shuffle_had_f2x32(v.z, v.w, lane);
    v.x *= FM_HAD_SCALE;
    v.y *= FM_HAD_SCALE;
    v.z *= FM_HAD_SCALE;
    v.w *= FM_HAD_SCALE;
}

// Match the live B12X fp16 intermediate storage boundaries, without spilling
// whole intermediates to global memory. Tensor-core accumulation still FP32.
__device__ __forceinline__ void fm_round_half4(float4& v)
{
    v.x = __half2float(__float2half_rn(v.x));
    v.y = __half2float(__float2half_rn(v.y));
    v.z = __half2float(__float2half_rn(v.z));
    v.w = __half2float(__float2half_rn(v.w));
}

__device__ __forceinline__ float4 fm_load_half4(const half* p)
{
    half4 h = *reinterpret_cast<const half4*>(p);
    return make_float4(__low2float(h.x), __high2float(h.x), __low2float(h.y), __high2float(h.y));
}

__device__ __forceinline__ void fm_mul_half4(float4& v, const half* p)
{
    float4 s = fm_load_half4(p);
    v.x *= s.x; v.y *= s.y; v.z *= s.z; v.w *= s.w;
}

__device__ __forceinline__ void fm_store_half4(half* p, const float4& v)
{
    half4 h(__floats2half2_rn(v.x, v.y), __floats2half2_rn(v.z, v.w));
    *reinterpret_cast<half4*>(p) = h;
}

// XOR swizzle of the 16-byte chunk column inside a 32-wide (64 B) A row so
// ldmatrix phases (8 consecutive rows, one chunk) hit 8 distinct bank groups.
__device__ __forceinline__ int fm_swz(int row, int chunk)
{
    return chunk ^ ((row >> 1) & 3);
}

// ---------------------------------------------------------------------------
// Gather + input Hadamard: h13[row] = had128(x[token[row]] * suh[expert[row]])
// ---------------------------------------------------------------------------

extern "C" __global__ __launch_bounds__(FM_THREADS)
void glm6_e3_gather(
    const __nv_bfloat16* __restrict__ x,
    const int64_t* __restrict__ row_token,
    const int* __restrict__ row_expert,
    const half* const* __restrict__ gate_suh_ptrs,
    const half* const* __restrict__ up_suh_ptrs,
    half* __restrict__ h13g,
    half* __restrict__ h13u,
    const int* __restrict__ num_rows_ptr,
    int size_k)
{
    const int num_rows = *num_rows_ptr;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int blk = blockIdx.y;
    for (int row = blockIdx.x * FM_WARPS + warp; row < num_rows; row += gridDim.x * FM_WARPS)
    {
        const int64_t token = row_token[row];
        const int expert = row_expert[row];
        const __nv_bfloat16* src = x + token * (int64_t)size_k + blk * 128 + lane * 4;
        #pragma unroll
        for (int p=0; p<2; ++p)
        {
            const half* scale = (p == 0 ? gate_suh_ptrs[expert] : up_suh_ptrs[expert]) + blk*128 + lane*4;
            // Live B12X _run_input_rotation: BF16->fp16, fp16 input-scale
            // multiplication, FP32 Hadamard, fp16 store. Gate/up remain distinct.
            float4 v = make_float4(
                __half2float(__hmul(__float2half_rn(__bfloat162float(src[0])), scale[0])),
                __half2float(__hmul(__float2half_rn(__bfloat162float(src[1])), scale[1])),
                __half2float(__hmul(__float2half_rn(__bfloat162float(src[2])), scale[2])),
                __half2float(__hmul(__float2half_rn(__bfloat162float(src[3])), scale[3])));
            fm_had_row(v, lane);
            half* dst = (p == 0 ? h13g : h13u) + (int64_t)row*size_k + blk*128 + lane*4;
            fm_store_half4(dst, v);
        }
    }
}

// ---------------------------------------------------------------------------
// Shared mainloop: acc[NS][MB][2] += A(tile rows, K) @ B_ns(K, 128 cols)
// ---------------------------------------------------------------------------

// NB_STRIDE: 16-col block offset between streams (0 = different matrices
// over the same columns, 8 = adjacent 128-col halves of one matrix).
template <int MB, int NS, int NB_STRIDE, int BITS, bool DUAL_A>
__device__ __forceinline__ void fm_mainloop(
    const half* const (&a)[NS],          // separate gate/up input buffers
    int size_k,
    int row0,
    int rows,
    const uint16_t* const (&packed)[NS], // per-stream trellis, (K/16, tiles_n, 64)
    int tiles_n,
    int n_block0,                        // first 16-col block of this tile
    half* sh_a,                          // FM_STAGES * MB*16*32 halves
    uint16_t* sh_b,                      // FM_STAGES * NS * FM_B_STAGE_WORDS
    FragC (&acc)[NS][MB][2])
{
    constexpr int TILE_M = MB * 16;
    constexpr int AS = DUAL_A ? NS : 1;
    constexpr int FM_PACKED_WORDS = 16 * BITS;
    constexpr int FM_B_STAGE_WORDS = 2 * FM_WARPS * FM_PACKED_WORDS;
    constexpr int A_ONE = TILE_M * FM_TILE_K;
    constexpr int A_STAGE = AS * A_ONE;
    constexpr int A_CHUNKS = AS * TILE_M * 4;                 // 16 B chunks per stage
    constexpr int A_ITERS = (A_CHUNKS + FM_THREADS - 1) / FM_THREADS;
    constexpr int B_STAGE = NS * FM_B_STAGE_WORDS;       // int16 words
    constexpr int B_TILE_CHUNKS = FM_PACKED_WORDS / 8;
    constexpr int B_CHUNKS = NS * 2 * FM_WARPS * B_TILE_CHUNKS;      // 16 B chunks per stage
    constexpr int B_ITERS = (B_CHUNKS + FM_THREADS - 1) / FM_THREADS;

    const int t = threadIdx.x;
    const int warp = t >> 5;
    const int lane = t & 31;
    const int k_tiles = size_k / FM_TILE_K;

    #pragma unroll
    for (int s = 0; s < NS; ++s)
        #pragma unroll
        for (int mb = 0; mb < MB; ++mb)
        {
            acc[s][mb][0] = {};
            acc[s][mb][1] = {};
        }

    // Match native FC1's four K32 stripes within each K128 tile.
    FragC stripes[DUAL_A ? 4 : 1][NS][MB][2] = {};

    auto load_stage = [&](int stage, int kt)
    {
        half* sa = sh_a + stage * A_STAGE;
        #pragma unroll
        for (int i = 0; i < A_ITERS; ++i)
        {
            int c = i * FM_THREADS + t;
            if (c < A_CHUNKS)
            {
                int ai = c / (TILE_M * 4);
                int row = (c / 4) % TILE_M;
                int chunk = c & 3;
                int src_row = row < rows ? row : rows - 1;
                const half* src = a[ai] + (int64_t) (row0 + src_row) * size_k + kt * FM_TILE_K + chunk * 8;
                half* dst = sa + ai * A_ONE + row * FM_TILE_K + fm_swz(row, chunk) * 8;
                cp_async(dst, src);
            }
        }
        uint16_t* sb = sh_b + stage * B_STAGE;
        #pragma unroll
        for (int i = 0; i < B_ITERS; ++i)
        {
            int c = i * FM_THREADS + t;
            if (c < B_CHUNKS)
            {
                int s = c / (2 * FM_WARPS * B_TILE_CHUNKS);
                int r = c % (2 * FM_WARPS * B_TILE_CHUNKS);
                int j = r / (FM_WARPS * B_TILE_CHUNKS);          // k16 sub-step
                int nb = (r / B_TILE_CHUNKS) % FM_WARPS;         // 16-col block (= warp)
                int q = r % B_TILE_CHUNKS;                       // 16 B chunk of the 128 B tile
                const uint16_t* src = packed[s]
                    + ((int64_t) (kt * 2 + j) * tiles_n + n_block0 + s * NB_STRIDE + nb) * FM_PACKED_WORDS + q * 8;
                uint16_t* dst = sb + (s * 2 + j) * (FM_WARPS * FM_PACKED_WORDS) + nb * FM_PACKED_WORDS + q * 8;
                cp_async(dst, src);
            }
        }
    };

    #pragma unroll
    for (int s = 0; s < FM_STAGES - 1; ++s)
    {
        if (s < k_tiles) load_stage(s, s);
        cp_async_fence();
    }

    const int a_row = (lane & 7) + 8 * ((lane >> 3) & 1);
    const int a_chunk_hi = lane >> 4;

    for (int kt = 0; kt < k_tiles; ++kt)
    {
        cp_async_wait<FM_STAGES - 2>();
        __syncthreads();
        int nk = kt + FM_STAGES - 1;
        if (nk < k_tiles) load_stage(nk % FM_STAGES, nk);
        cp_async_fence();

        const int stage = kt % FM_STAGES;
        const half* sa = sh_a + stage * A_STAGE;
        const uint16_t* sb = sh_b + stage * B_STAGE;

        #pragma unroll
        for (int j = 0; j < 2; ++j)
        {
            FragB fb[NS][2];
            #pragma unroll
            for (int s = 0; s < NS; ++s)
            {
                const uint32_t* wb = reinterpret_cast<const uint32_t*>(
                    sb + (s * 2 + j) * (FM_WARPS * FM_PACKED_WORDS) + warp * FM_PACKED_WORDS);
                dq_dispatch<BITS, 1>(wb, lane << 3, fb[s][0], fb[s][1]);
            }
            #pragma unroll
            for (int mb = 0; mb < MB; ++mb)
            {
                FragA fa;
                int row = mb * 16 + a_row;
                int chunk = j * 2 + a_chunk_hi;
                #pragma unroll
                for (int s = 0; s < NS; ++s)
                {
                    ldsm4(fa, sa + (DUAL_A ? s * A_ONE : 0) + row * FM_TILE_K + fm_swz(row, chunk) * 8);
                    if constexpr (DUAL_A) {
                        #pragma unroll
                        for (int stripe = 0; stripe < 4; ++stripe) {
                            if ((kt & 3) == stripe) {
                                ptx_mma_m16n8k16(fa, fb[s][0], stripes[stripe][s][mb][0]);
                                ptx_mma_m16n8k16(fa, fb[s][1], stripes[stripe][s][mb][1]);
                            }
                        }
                    } else {
                        ptx_mma_m16n8k16(fa, fb[s][0], acc[s][mb][0]);
                        ptx_mma_m16n8k16(fa, fb[s][1], acc[s][mb][1]);
                    }
                }
            }
        }
    }
    if constexpr (DUAL_A) {
        #pragma unroll
        for (int s = 0; s < NS; ++s)
            #pragma unroll
            for (int mb = 0; mb < MB; ++mb)
                #pragma unroll
                for (int n = 0; n < 2; ++n)
                    #pragma unroll
                    for (int v = 0; v < 4; ++v)
                        acc[s][mb][n][v] = __fadd_rn(stripes[0][s][mb][n][v],
                            __fadd_rn(__fadd_rn(stripes[1][s][mb][n][v], stripes[3][s][mb][n][v]),
                                      stripes[2][s][mb][n][v]));
    }
    cp_async_wait<0>();
    __syncthreads();
}

// Stage one 16-row block of accumulators into sh_c[16][NS * 128] (fp32).
template <int NS>
__device__ __forceinline__ void fm_stage_acc(
    float* sh_c, FragC (&acc0)[2], FragC (&acc1)[2], int warp, int lane)
{
    constexpr int W = NS * FM_TILE_N;
    int r0 = lane >> 2;
    int col = (lane & 3) * 2 + warp * 16;
    {
        float* d0 = sh_c + r0 * W + col;
        float* d1 = sh_c + (r0 + 8) * W + col;
        d0[0] = acc0[0][0]; d0[1] = acc0[0][1]; d0[8] = acc0[1][0]; d0[9] = acc0[1][1];
        d1[0] = acc0[0][2]; d1[1] = acc0[0][3]; d1[8] = acc0[1][2]; d1[9] = acc0[1][3];
    }
    if constexpr (NS == 2)
    {
        float* d0 = sh_c + r0 * W + FM_TILE_N + col;
        float* d1 = sh_c + (r0 + 8) * W + FM_TILE_N + col;
        d0[0] = acc1[0][0]; d0[1] = acc1[0][1]; d0[8] = acc1[1][0]; d0[9] = acc1[1][1];
        d1[0] = acc1[0][2]; d1[1] = acc1[0][3]; d1[8] = acc1[1][2]; d1[9] = acc1[1][3];
    }
}

// ---------------------------------------------------------------------------
// Grouped gate/up GEMM, FP16 output for the exact native activation
// ---------------------------------------------------------------------------

extern "C" __global__ __launch_bounds__(FM_THREADS, 1)
void glm6_e3_gateup(
    const half* __restrict__ h13,
    const half* __restrict__ h13u,
    const uint16_t* const* __restrict__ gate_ptrs,
    const uint16_t* const* __restrict__ up_ptrs,
    const half* const* __restrict__ gate_svh_ptrs,
    const half* const* __restrict__ up_svh_ptrs,
    const half* const* __restrict__ down_suh_ptrs,
    half* __restrict__ h2,
    const int* __restrict__ seg_expert,
    const int* __restrict__ seg_row0,
    const int* __restrict__ seg_rows,
    const int* __restrict__ num_segs_ptr,
    const int* __restrict__ expert_bits,
    int size_k,
    int size_n,
    float act_limit)
{
    constexpr int MB = FM_MB_GATEUP;
    constexpr int NS = 2;
    extern __shared__ __align__(16) unsigned char fm_smem[];
    half* sh_a = reinterpret_cast<half*>(fm_smem);
    uint16_t* sh_b = reinterpret_cast<uint16_t*>(sh_a + 2 * FM_STAGES * MB * 16 * FM_TILE_K);
    float* sh_c = reinterpret_cast<float*>(fm_smem);

    const int num_segs = *num_segs_ptr;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int tiles_n = size_n / 16;
    const int n_base = blockIdx.x * FM_TILE_N;

    for (int seg = blockIdx.y; seg < num_segs; seg += gridDim.y)
    {
        const int e = seg_expert[seg];
        const int row0 = seg_row0[seg];
        const int rows = seg_rows[seg];
        const uint16_t* const packed[NS] = { gate_ptrs[e], up_ptrs[e] };
        const half* svh_g = gate_svh_ptrs[e] + n_base;
        const half* svh_u = up_svh_ptrs[e] + n_base;
        const half* suh_d = down_suh_ptrs[e] + n_base;

        FragC acc[NS][MB][2];
        const half* const a[NS] = {h13, h13u};
        if (expert_bits[e] == 3)
            fm_mainloop<MB, NS, 0, 3, true>(a, size_k, row0, rows, packed, tiles_n, n_base / 16, sh_a, sh_b, acc);
        else
            fm_mainloop<MB, NS, 0, 4, true>(a, size_k, row0, rows, packed, tiles_n, n_base / 16, sh_a, sh_b, acc);

        #pragma unroll
        for (int mb = 0; mb < MB; ++mb)
        {
            int rows_mb = rows - mb * 16;
            if (rows_mb <= 0) break;
            fm_stage_acc<NS>(sh_c, acc[0][mb], acc[1][mb], warp, lane);
            __syncthreads();
            #pragma unroll
            for (int rr = 0; rr < 2; ++rr)
            {
                int r = warp + rr * FM_WARPS;
                if (r < rows_mb)
                {
                    const float* src = sh_c + r * (NS * FM_TILE_N) + lane * 4;
                    float4 g = *reinterpret_cast<const float4*>(src);
                    float4 u = *reinterpret_cast<const float4*>(src + FM_TILE_N);
                    half* dst = h2 + (int64_t)(row0 + mb*16 + r)*2*size_n + n_base + lane*4;
                    fm_store_half4(dst, g);
                    fm_store_half4(dst + size_n, u);
                }
            }
            __syncthreads();
        }
    }
}

// ---------------------------------------------------------------------------
// Grouped down GEMM, FP16 router-order output for the exact native sum
// ---------------------------------------------------------------------------

extern "C" __global__ __launch_bounds__(FM_THREADS, 2)
void glm6_e3_down(
    const half* __restrict__ h2,
    const uint16_t* const* __restrict__ down_ptrs,
    const half* const* __restrict__ down_svh_ptrs,
    half* __restrict__ out,
    const int64_t* __restrict__ row_route,
    const float* __restrict__ row_weight,
    const int* __restrict__ seg_expert,
    const int* __restrict__ seg_row0,
    const int* __restrict__ seg_rows,
    const int* __restrict__ num_segs_ptr,
    const int* __restrict__ expert_bits,
    int size_k,
    int size_n)
{
    constexpr int MB = FM_MB_DOWN;
    constexpr int NS = 2;                       // two adjacent 128-col halves
    constexpr int TILE_N2 = NS * FM_TILE_N;
    extern __shared__ __align__(16) unsigned char fm_smem[];
    half* sh_a = reinterpret_cast<half*>(fm_smem);
    uint16_t* sh_b = reinterpret_cast<uint16_t*>(sh_a + 2 * FM_STAGES * MB * 16 * FM_TILE_K);
    float* sh_c = reinterpret_cast<float*>(fm_smem);

    const int num_segs = *num_segs_ptr;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int tiles_n = size_n / 16;
    const int n_base = blockIdx.x * TILE_N2;

    for (int seg = blockIdx.y; seg < num_segs; seg += gridDim.y)
    {
        const int e = seg_expert[seg];
        const int row0 = seg_row0[seg];
        const int rows = seg_rows[seg];
        const uint16_t* const packed[NS] = { down_ptrs[e], down_ptrs[e] };
        const half* svh = down_svh_ptrs[e] + n_base;

        FragC acc[NS][MB][2];
        const half* const a[NS] = {h2, h2};
        if (expert_bits[e] == 3)
            fm_mainloop<MB, NS, FM_TILE_N / 16, 3, false>(a, size_k, row0, rows, packed, tiles_n, n_base / 16, sh_a, sh_b, acc);
        else
            fm_mainloop<MB, NS, FM_TILE_N / 16, 4, false>(a, size_k, row0, rows, packed, tiles_n, n_base / 16, sh_a, sh_b, acc);

        #pragma unroll
        for (int mb = 0; mb < MB; ++mb)
        {
            int rows_mb = rows - mb * 16;
            if (rows_mb <= 0) break;
            fm_stage_acc<NS>(sh_c, acc[0][mb], acc[1][mb], warp, lane);
            __syncthreads();
            #pragma unroll
            for (int rr = 0; rr < 2; ++rr)
            {
                int r = warp + rr * FM_WARPS;
                if (r < rows_mb)
                {
                    const float* srow = sh_c + r * TILE_N2;
                    int frow = row0 + mb * 16 + r;
                    half* dst = out + row_route[frow] * (int64_t) size_n + n_base + lane * 4;
                    #pragma unroll
                    for (int s = 0; s < NS; ++s)
                    {
                        float4 v = *reinterpret_cast<const float4*>(srow + s * FM_TILE_N + lane * 4);
                        fm_store_half4(dst + s * FM_TILE_N, v);
                    }
                }
            }
            __syncthreads();
        }
    }
}

