"""Original TP6 expert fragments, with an explicit per-stream scratch owner.

This component returns an unscaled FP32 local sum. Its caller must decide the
BF16 cast, routed scaling and collective order from the full-model contract.
It is not a registered full-model engine.
"""
from dataclasses import dataclass

import torch

from tensorfold.cuda.exl3 import experts as kernel

MAX_BATCH_ROWS = 3072
EXPERT_CHUNK_ROWS = 128
EXPERT_CHUNK_OPTIONS = (128,256,512,1024)


def validate_chunk_rows(rows):
    # Top8 grouping uses32B per input row plus128B static shared memory.
    # Keep every candidate below the kernel's48KiB default shared-memory limit.
    if type(rows) is not int or rows not in EXPERT_CHUNK_OPTIONS:
        raise ValueError('Expert chunks must be128,256,512 or1024 rows')
    return rows


@dataclass
class RoutedLayer:
    weights: kernel.Exl3RoutedExperts
    global_to_local: torch.Tensor

    @classmethod
    def load(cls, reader, layer, device='cuda'):
        weights, mapping = reader.prepare_experts(layer, device=device)
        return cls(weights, torch.tensor(mapping, dtype=torch.int32, device=device))

    def scratch(self, rows, *, chunk_rows=EXPERT_CHUNK_ROWS):
        return RoutedScratch(self, rows, chunk_rows=chunk_rows)

    def routed(self, x, global_ids, probabilities, scratch, *, act_mode):
        """Sum top-8 original fragments, with zero contribution from other ranks.

        IDs must come from the validated top-k router: in [0, 256), unique within
        each row. No device-to-host data checks occur in this graph-safe path.
        ``scratch`` belongs to this layer and one stream/graph at a time.
        """
        rows = x.shape[0]
        if (scratch.layer is not self or not 1 <= rows <= scratch.rows
                or x.shape != (rows, self.weights.dims)
                or x.dtype not in (torch.bfloat16, torch.float16)
                or global_ids.shape != (rows, 8) or global_ids.dtype != torch.int64
                or probabilities.shape != (rows, 8) or probabilities.dtype != torch.float32
                or not all(t.is_cuda and t.is_contiguous() and t.device == self.global_to_local.device
                           for t in (x, global_ids, probabilities))):
            raise ValueError('Expected contiguous CUDA inputs, valid TP6 scratch and top-8 int64 IDs')
        if act_mode not in (kernel.ACT_BF16, kernel.ACT_F32):
            raise ValueError('Choose and qualify the SwiGLU arithmetic explicitly')
        # Keep the configured 3072-token model batch while bounding the expert
        # workspace. The general grouping kernel uses R*8*4 shared-memory bytes;
        # a single 3072-row launch exceeds its default shared-memory allowance.
        # Kernels use a fixed reduction order per row, independent of this split.
        for start in range(0, rows, scratch.chunk_rows):
            stop = min(start + scratch.chunk_rows, rows)
            count = stop - start
            picks = scratch.local_ids[:count]
            torch.index_select(self.global_to_local, 0, global_ids[start:stop].view(-1), out=picks.view(-1))
            # General TensorFold leaves absent slots to the caller (for shared
            # experts). Our absent slots mean another rank owns that fragment.
            # Clear a previous token's result before reusing its scratch slot.
            scratch.kernel.y[:count * 8].zero_()
            kernel.routed(x[start:stop], picks, probabilities[start:stop], self.weights, scratch.kernel,
                          scratch.output[start:stop], count, act_mode=act_mode)
        return scratch.output[:rows]


class RoutedScratch:
    def __init__(self, layer, rows, *, chunk_rows=EXPERT_CHUNK_ROWS):
        if type(rows) is not int or not 1 <= rows <= MAX_BATCH_ROWS:
            raise ValueError(f'Scratch capacity must be in 1..{MAX_BATCH_ROWS}')
        self.layer, self.rows = layer, rows
        self.chunk_rows = min(rows, validate_chunk_rows(chunk_rows))
        device = layer.global_to_local.device
        self.kernel = kernel.Scratch(layer.weights, self.chunk_rows, 8, device=device)
        self.local_ids = torch.empty((self.chunk_rows, 8), dtype=torch.int32, device=device)
        self.output = torch.empty((rows, layer.weights.dims), dtype=torch.float32, device=device)
