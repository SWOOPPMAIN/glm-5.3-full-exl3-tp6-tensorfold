"""Native arithmetic epilogues for the grouped EXL3 matrix kernels.

Reuse B12X's activation methods and compiled ordered top-k sum. Scratch belongs
to E3's current-stream arena; the native runtime supplies immutable weight and
launch bindings only. The source-pinned installer must qualify this adaptation.
"""
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_ptr
from cuda.bindings import driver as cuda
from b12x.moe._shared.kernels.w4a16.kernel import W4A16FusedMoeKernel

_ACTIVATION_CACHE = {}


class NativeActivation:
    _cast_elem = W4A16FusedMoeKernel._cast_elem
    _had128_quad = W4A16FusedMoeKernel._had128_quad
    _sigmoid_f32 = W4A16FusedMoeKernel._sigmoid_f32
    _silu_f32 = W4A16FusedMoeKernel._silu_f32
    _run_activation_compact = W4A16FusedMoeKernel._run_activation_compact

    def __init__(self):
        self.is_fp16 = True
        self.fast_math = True
        self.cta_threads = 256
        self.intermediate_size = 512
        self.top_k = 8
        self.direct_topk_routes = True
        self.use_expert_map = False
        self.fc1_cols = 1024
        self.moe_block_size = 64
        self.activation_is_situ = False

    @cute.kernel
    def kernel(self, fc1: cute.Tensor, out: cute.Tensor, scales: cute.Tensor,
               local_ids: cute.Tensor, ne: cutlass.Int32, rows: cutlass.Int32):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        grid, _, _ = cute.arch.grid_dim()
        self._run_activation_compact(fc1, out, scales, local_ids, local_ids,
            local_ids, local_ids, ne, ne, tid, bid, grid, rows)

    @cute.jit
    def __call__(self, fc1: cute.Pointer, out: cute.Pointer, scales: cute.Pointer,
                 local_ids: cute.Pointer, ne: cutlass.Int32, rows: cutlass.Int32,
                 stream: cuda.CUstream):
        source = cute.make_tensor(fc1, cute.make_layout((rows*8*1024,), stride=(1,)))
        target = cute.make_tensor(out, cute.make_layout((rows*8*512,), stride=(1,)))
        rotations = cute.make_tensor(scales, cute.make_layout((ne*1536,), stride=(1,)))
        experts = cute.make_tensor(local_ids, cute.make_layout((rows*8,), stride=(1,)))
        self.kernel(source, target, rotations, experts, ne, rows).launch(
            grid=[128, 1, 1], block=[256, 1, 1], stream=stream)


def pointer(tensor, dtype, alignment=16):
    return make_ptr(dtype, tensor.data_ptr(), cute.AddressSpace.gmem,
                    assumed_align=alignment)


def activate(layer, fc1, output, sorted_experts, rows):
    import torch
    scales = layer.exl3_mixed_trellis['rotations'].intermediate
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    args = [pointer(fc1, cutlass.Float16), pointer(output, cutlass.Float16),
            pointer(scales, cutlass.Float16), pointer(sorted_experts, cutlass.Int32, 4),
            cutlass.Int32(scales.shape[0]), cutlass.Int32(rows), stream]
    key = fc1.device.index
    if key not in _ACTIVATION_CACHE:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('Native E3 activation must be prepared before graph capture')
        _ACTIVATION_CACHE[key] = cute.compile(NativeActivation(), *args)
    _ACTIVATION_CACHE[key](*args)


def ordered_sum(layer, fc2, output, weights, ids):
    import torch
    mixed = layer.exl3_mixed_trellis
    state = mixed['runtime']['prefill']
    launch = state['launch']
    binding = mixed['runtime_bindings'][id(launch)]
    launch.topk_sum.compiled(
        pointer(fc2, cutlass.Float16), pointer(output, cutlass.Float32),
        pointer(weights, cutlass.Float32, 4),
        pointer(ids, cutlass.Int32 if ids.dtype == torch.int32 else cutlass.Int64,
                4 if ids.dtype == torch.int32 else 8),
        binding.global_to_combined_ptr, binding.down_svh_ptr,
        cutlass.Int32(launch.topk_sum.num_experts),
        cutlass.Int32(launch.topk_sum.route_num_experts), int(ids.shape[0]),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream))
