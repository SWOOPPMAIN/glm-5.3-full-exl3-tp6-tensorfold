# Decode profiling and draft depth

TFP19 uses the original mixed K3/K4 3.25 bpw weights, all 79 layers, a fully
resident 804K cache, and the existing six-rank NCCL group. It does not change
production settings. [Complete measurements](../../results/tfp19-tensorfold.json).

## Repeatable sweep

`tools/glm53_tp6_profile_check.py` runs the real request controller directly.
The checkpoint chat template renders the same short code/prose fixtures used
in TFP18, with thinking disabled and greedy generation of 128 output tokens.
Independent serial target sequences gate every candidate's output.

Each depth/fixture pair warms once, followed by three uninstrumented repeats.
Depth order rotates and fixture order alternates. Timed runs must create no
new CUDA graphs. Reported rates are 127 tokens divided by decode time after
the first token; prefill and HTTP delivery are excluded.

| Draft depth | Code median tok/s | Prose median tok/s |
| --- | ---: | ---: |
| 0 | 8.42 | 9.34 |
| 1 | 16.08 | 16.37 |
| 2 | 21.40 | 19.10 |
| 3 | 23.69 | 22.75 |
| 4 | 24.72 | 24.98 |
| 6 | 24.28 | 25.65 |
| 8 | 24.31 | 24.60 |

Depth four remains the reference: it leads code. Depth six gains only about
2.7% on this prose fixture. These short, fixed workloads do not establish a
universal optimal depth or a production speedup. They differ from P24's
512-output-token serving benchmark.

## What the profiles show

Separate instrumented requests record host ranges and CUDA event intervals.
Two further runs trace three warm depth-four decode rounds each, on all six
ranks, using the [PyTorch profiler](https://docs.pytorch.org/docs/2.10/profiler.html).
Shapes, stacks and memory tracking are disabled. Trace timings do not select
the throughput winners.

- Dense BF16 `_linear` projections account for **38–44%** of rank-zero kernel
  time in these traces, the largest compute category.
- Grouped EXL3 experts account for about **20–23%** on rank zero.
- NCCL accounts for **20–30%** on rank zero, including waits for peers. Ranks
  four/five do less shared-expert dense work and spend longer in NCCL; this
  supports a compute-imbalance hypothesis, not a pure link-speed diagnosis.
- Request coordination and validation account for about **7%** of rank-zero
  command time in the separate event-profile runs.
- Large host time inside sampling mostly waits for preceding target/head GPU
  work. It should not be interpreted as equivalent sampling-compute cost.

Kernel sums may overlap across streams; summaries also provide per-device
interval unions. CUDA event intervals include launch, idle and wait time.
Host command/apply ranges nest. Neither instrumented timing is an independent
uninstrumented throughput result.

The next target is the original BF16 projection kernel: tune tiles and memory
access while preserving weights and qualifying its reduction/output behavior.
Packed concurrent target passes and broader API/long-context checks remain pending.

## Evidence and execution

All **630 GPU checks** passed, including **366 exact sequence comparisons**
(61 requests per rank), and **95 CPU checks** passed. All workers exited zero,
no OOMs occurred, and guards stayed clear. Peak PyTorch allocation was 99.90 GiB;
minimum host available memory was 11.10 GiB. Twelve compressed traces are
retained with hashes in private operational evidence; the sharing repository
contains selected summaries.

Use the [same drained-fleet admission contract](README.md#gpu-qualification-contract)
as prior probes. The probe uses a 24-entry graph limit, nine target/head rows,
and the existing one-row MTP graph path. Add `graph_reserve(24, 9)` to admission.
A portable launcher and complete backend image are still pending. P24 remains live.
