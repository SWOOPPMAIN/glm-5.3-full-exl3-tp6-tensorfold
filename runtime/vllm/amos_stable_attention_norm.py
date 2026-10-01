"""Fixed reduction arithmetic for the pinned full-GLM attention norms.

The compiler changes reduction strategy with the maximum batch-size hint.
These opaque operations keep a fixed per-row reduction tree at every request
size, preserving the qualified fused BF16 output rounding and original weights.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def _rms_kernel(X, W, Y, rows, stride: tl.constexpr, D: tl.constexpr,
                EPS: tl.constexpr, XB: tl.constexpr):
    row = tl.program_id(0) * XB + tl.arange(0, XB)[:, None]
    if D == 2048:
        col_base = tl.arange(0, 1024)[None, :]
        partial = tl.full([XB, 1024], 0, tl.float32)
        for offset in tl.range(0, D, 1024):
            col = offset + col_base
            x = tl.load(X + row*stride + col, row < rows, other=0).to(tl.float32)
            partial = tl.where(row < rows, partial + x*x, partial)
        total = tl.sum(partial, 1)[:, None]
        for offset in tl.range(0, D, 1024):
            col = offset + col_base
            x = tl.load(X + row*stride + col, row < rows, other=0).to(tl.float32)
            w = tl.load(W + col).to(tl.float32)
            value = (x * libdevice.rsqrt(total / D + EPS)) * w
            tl.store(Y + row*D + col, value, row < rows)
    else:
        col = tl.arange(0, D)[None, :]
        x = tl.load(X + row*stride + col, row < rows, other=0).to(tl.float32)
        w = tl.load(W + col).to(tl.float32)
        total = tl.sum(tl.where(row < rows, x*x, 0), 1)[:, None]
        value = (x * libdevice.rsqrt(total / D + EPS)) * w
        tl.store(Y + row*D + col, value, row < rows)


@triton.jit
def _layer_kernel(X, W, B, Y, rows, stride: tl.constexpr, XB: tl.constexpr):
    row = tl.program_id(0) * XB + tl.arange(0, XB)[:, None]
    col = tl.arange(0, 128)[None, :]
    x = tl.load(X + row*stride + col, row < rows, other=0).to(tl.float32)
    w = tl.load(W + col)
    b = tl.load(B + col)
    mean = tl.sum(tl.where(row < rows, x, 0), 1)[:, None] / 128
    centered = x - mean
    variance = tl.sum(tl.where(row < rows, centered*centered, 0), 1)[:, None] / 128
    value = (centered * libdevice.rsqrt(variance + 1e-6)) * w + b
    tl.store(Y + row*128 + col, value, row < rows)


def _output(x):
    return torch.empty(x.shape, dtype=x.dtype, device=x.device)


def _rms(x: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor:
    if (x.ndim != 2 or x.shape[1] not in (512, 2048) or x.stride(1) != 1
            or x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16
            or weight.shape != (x.shape[1],) or not weight.is_contiguous()
            or not x.is_cuda or weight.device != x.device or epsilon != 1e-5):
        raise ValueError('Stable attention RMS requires the pinned BF16 512/2048 geometry')
    result = _output(x)
    if x.shape[0]:
        _rms_kernel[(triton.cdiv(x.shape[0], 2),)](
            x, weight, result, x.shape[0], x.stride(0), x.shape[1], epsilon, 2,
            num_warps=8, num_stages=1)
    return result


def _layer(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    if (x.ndim != 2 or x.shape[1] != 128 or x.stride(1) != 1
            or x.dtype != torch.bfloat16 or weight.dtype != torch.float32
            or bias.dtype != torch.float32 or weight.shape != (128,) or bias.shape != (128,)
            or not weight.is_contiguous() or not bias.is_contiguous()
            or not x.is_cuda or weight.device != x.device or bias.device != x.device):
        raise ValueError('Stable indexer LayerNorm requires BF16 rows and FP32 parameters')
    result = _output(x)
    if x.shape[0]:
        _layer_kernel[(triton.cdiv(x.shape[0], 8),)](
            x, weight, bias, result, x.shape[0], x.stride(0), 8,
            num_warps=2, num_stages=1)
    return result


def _rms_fake(x: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor:
    return _output(x)


def _layer_fake(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return _output(x)


direct_register_custom_op('amos_tp6_attention_rms', _rms, fake_impl=_rms_fake)
direct_register_custom_op('amos_tp6_indexer_norm', _layer, fake_impl=_layer_fake)


class AttentionRMSNorm(torch.nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        if hidden_size not in (512, 2048) or eps != 1e-5:
            raise ValueError('Unexpected full-GLM attention normalization geometry')
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.variance_size = None
        self.hidden_size = hidden_size

    def forward(self, x):
        return torch.ops.vllm.amos_tp6_attention_rms(x, self.weight, self.variance_epsilon)


class IndexerLayerNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        if dim != 128 or eps != 1e-6:
            raise ValueError('Unexpected full-GLM indexer normalization geometry')
        self.dim = dim
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.bias = torch.nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x):
        return torch.ops.vllm.amos_tp6_indexer_norm(x, self.weight, self.bias)
