# Original BF16 projection tiles

TFP20 follows the dense-projection cost identified in [TFP19](DECODE_PROFILING.md).
It preserves original mixed K3/K4 3.25 bpw expert weights and BF16 dense weights,
all 79 target/MTP layers, the existing six-rank group and resident 804K cache.
[Measurements and screening rejections](../../results/tfp20-tensorfold.json).

## Qualified configuration

The unchanged reference uses M16 × N64 output tiles, K64 reduction tiles,
four warps and two pipeline stages. The selected plan uses:

| Path | Output tile M × N | Stages |
| --- | --- | ---: |
| Fewer than 256 rows | 16 × 64 | 3 |
| At least 256 rows | 64 × 128 | 2 |

K64 traversal, four warps, FP32 accumulation and disabled FP fusion remain
fixed. No weight conversion or quantization is added. The plan is immutable
per model and included in six-rank control-state agreement. Graph backends
reject plan drift before reuse. The default retains the reference tile;
select the measured configuration explicitly:

```python
from tensorfold.families.glm_moe_dsa.projection_plan import LinearTile, ProjectionPlan
from tensorfold.families.glm_moe_dsa.model import FullModel

plan = ProjectionPlan(
    decode=LinearTile(16, 64, 3),
    prefill=LinearTile(64, 128, 2),
    bulk_min_rows=256,
)
model = FullModel(weights, reduction, projections=plan)
```

This assumes already admitted weights, communicator, cache and workspace.
It is not a complete serving launcher.

## Measurements

Three warm, alternating repeats per mode. Full-prefill timing uses the slowest
rank's elapsed time for each pass; direct-controller rates use rank zero and
exclude prefill, first token and HTTP delivery.

| Workload | Reference | Candidate | Ratio |
| --- | ---: | ---: | ---: |
| Synthetic 3,072-token full target pass | 8.841 s | 8.157 s | 1.084× input throughput |
| Code, MTP4, 128 outputs | 24.29 tok/s | 24.36 tok/s | 1.003× |
| Prose, MTP4, 128 outputs | 24.43 tok/s | 25.78 tok/s | 1.055× |

Code decode is effectively unchanged; its three runs vary substantially. The
prose gain is promising on this fixture but does not establish a general decode
speedup. Both serial reference output hashes also match TFP19.

Synthetic prefill corresponds to 347.5 →
376.6 input tok/s. It uses deterministic synthetic token IDs
and warm kernels. This is not the uncached authentic-prompt prefill measurement
in the serving table. Decode uses the original chat template, thinking disabled,
greedy generation and independent serial target references. Each timed answer
matches those references exactly; no new graph capture occurs during timing.

## Screening and correctness

`tools/glm53_tp6_projection_screen.py` screens five small-row tiles and five
bulk-row tiles on eleven original-weight projection geometries, including the
rank-five output-projection shape and padded vocabulary. Ranks four/five lack
shared experts, so their shared-geometry component checks use original dense
weight subsets with zero timing weight. Full-model passes use the actual shards.

Every eligible tile must preserve all tested BF16 outputs and changed-input
FP32 outputs on every rank. Static compiler resource failures or arithmetic
differences reject the tile; a CUDA execution error fails the experiment.
Sampled CPU FP64 products characterize the reference's BF16 rounding and do not
replace exact candidate/reference comparison. The measured screen performed
**2970 comparisons**, with **0 tile rejections** across ranks.

Five CUDA-event measurements per timed geometry use a 64 MiB write sweep
between samples. This reduces hot-cache bias without proving a full L2 flush.
A weighted shape-cost heuristic chooses a single common candidate from the
slowest rank's score. Its score is not a model-throughput prediction; the
separate full-model/request measurements above determine practical benefit.

Full target tensors at 3,072, changed 257 and 17 rows, MTP at 257 rows, and
selected logits must match the unchanged model exactly. All timed 3,072-row
passes are also compared outside the timed interval. Two bounded graph backends
share weights/cache/workspace serially, with 12 entries, five target/head rows,
one MTP row and a 256 MiB capture-growth allowance per backend.

**306 strict GPU checks** passed, including **108 tensor**
and **96 sequence comparisons**. **103 CPU checks** passed.
All six workers exited zero with no OOM and no guard trips. Peak PyTorch
allocation was 100.20 GiB; minimum host available memory
was 10.69 GiB.

## Scope and remaining work

Use the [drained-fleet admission contract](README.md#gpu-qualification-contract)
with both graph allowances included. The selected tile is qualified only for
these tested shapes/workloads; broader API, packed batching and real long-context
quality/performance still need gates. TensorFold is not the live serving backend.
The same current P24 configuration was reopened and application acceptance passed.

The tile/pipeline approach is informed by the official
[Triton matrix-multiplication tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html).
The probe and immutable plan are Swoopp additions to the existing local BF16
kernel; see [credits](../../CREDITS.md) for framework and model origins.
