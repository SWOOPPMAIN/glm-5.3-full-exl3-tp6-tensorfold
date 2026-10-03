# E3 prefill: actual TP6 routes

October 3, 2026. **Row32 is qualified and selected in full-model serving.**
The original 3.25 bpw weights, 3072-token budget, adaptive MTP, 512-row native/E3
boundary, dual-port RoCEnante and 2 MiB crossover remain selected. TensorFold
development is excluded.

## Findings

Two distinct synthetic code/prose prompts supplied the first 3072-token chunk.
We recorded all 75 routed layers on all six ranks: **900 layer records**, plus
36 input/output pairs from layers 3, 40 and 77. IDs and FP32 routing weights
matched across ranks. The local mappings matched the original four-piece
ownership rule `(4 * expert + source_piece) % 6`; weights were not repartitioned.

| Geometry, across all layers/ranks | Code | Prose |
| --- | ---: | ---: |
| Unused capacity in 64-row control tiles | 26.46% | 26.64% |
| Estimated padded-row reduction with 32-row tiles | 14.05% | 14.13% |
| Estimated segment-count increase with 32-row tiles | 71.91% | 71.74% |
| Mean per-layer maximum/mean padded rank work | 1.044 | 1.044 |

These are route geometry measurements, **not throughput gains**. Smaller tiles
perform less unused row arithmetic but fetch/dequantize packed weights more
often. The worst observed maximum/mean padded rank work was 1.115 for code and
1.105 for prose. This is a workload-count comparison, not measured rank waiting.

The CPU compiler also found a concrete tradeoff:

| Gate/up kernel | Registers/thread | Stack frame | Reported spill stores / loads |
| --- | ---: | ---: | ---: |
| 64-row control | 255 | 176 bytes | 752 / 764 bytes |
| Experimental 32-row | 225 | 32 bytes | 0 / 0 bytes |

The rebuilt 64-row kernel's machine-code sections match the installed binary.
The CUDA candidate changes only the two `FM_MB_*` constants from 4 to 2; its
Python route planner and shared-memory launch size change to match. Static
compiler spill reports alone do not establish runtime traffic or speed.

## Exact-input GPU comparison

All six serving ranks were drained and stopped before isolated tests. Each probe
waited for its exact container's fresh 8 GiB / 2-second memory guard before CUDA
initialization. Containers had a 4 GiB memory limit, a 2 GiB Torch budget and a
280-second process deadline. Peak Torch allocation was **1.435 GiB**.

Across layers 3/40/77, six ranks and code/prose captures, **468 bitwise output
comparisons passed**, including CUDA graphs replayed with changing inputs and
routes. Smaller row counts use prefixes of the captured 3072-row operands.
Both kernels matched the original serving-local outputs.

| Rows | Row32 / row64 speed ratio, geometric mean | Equivalent latency reduction | Faster measured cells |
| --- | ---: | ---: | ---: |
| 513 | 1.376× | 27.3% | 36 / 36 |
| 768 | 1.302× | 23.2% | 36 / 36 |
| 1024 | 1.265× | 21.0% | 36 / 36 |
| 1536 | 1.220× | 18.0% | 36 / 36 |
| 3072 | **1.135×** | **11.9%** | **36 / 36** |

There were eight samples per arm/cell, alternating execution order, for 2,880
timed calls across 180 cells. A 64 MiB cache flush preceded each call and was
excluded from timing. Times include routing and native epilogues. Both arms
used one shared stream-local scratch arena. These are isolated component
results with one layer resident; **they are not full-model tok/s gains**.

After the component comparison, the row64 image was explicitly resumed with fresh functional,
native, Pi-router, Code and Chat acceptance. Its existing exact-image short/long
numerical qualification was reused, not reported as rerun. Final available RAM
was at least 14.07 GiB. This component-stage closeout is historical; the full-model
comparison and selected row32 image below supersede it.

[All samples and comparison checks](../../results/e3-row32-component.json) ·
[Serving closeout](../../results/e3-row32-outcome.json)

[Route results and qualification](../../results/e3-prefill-diagnostic.json) ·
[Compiler results and source hashes](../../results/e3-row32-compile.json)

## Full-model comparison and selected serving

Four alternating visits (row64, row32, row64, row32) used the same six containers,
weights, precision, 3072 budget, 512-row native boundary, original adaptive MTP and
communication settings. Each workload has three measured samples per policy,
spread across two visits. Kernels and exact prompts were warmed separately.

| Measurement | Row64 control | Selected row32 | Change |
| --- | ---: | ---: | ---: |
| Cold 8K prefill | 944.6 tok/s | **981.8 tok/s** | **+3.94%** |
| Cold 32K prefill | 934.8 tok/s | **965.5 tok/s** | **+3.28%** |
| Cold 8K / 32K TTFT | 8.673 / 35.053 s | **8.344 / 33.938 s** | Lower |
| Prose, 512 output tokens | 35.36 tok/s | 36.83 tok/s | +4.17% |
| Code, 512 output tokens | 47.74 tok/s | 48.65 tok/s | +1.91% |
| Cached short C4, aggregate | 80.07 tok/s | 78.12 tok/s | −2.44% |
| Cached 8K C4, aggregate | 69.47 tok/s | 70.53 tok/s | +1.53% |
| Cached 32K C4, aggregate | 68.73 tok/s | 66.95 tok/s | −2.59% |

Cold-prefill ranges do not overlap: 8K control **942.1–946.7** versus row32
**979.7–982.7** tok/s; 32K control **933.4–935.4** versus row32 **964.1–965.8**.
These are observed ranges, not confidence intervals. The 12-cell cached matrix
improves 1.84% by geometric mean, with workload regressions shown above. Generation
and MTP acceptance vary between repeats; the decode kernels are unchanged, so those
differences are not an isolated E3 decode-speed claim. The highest single code
sample was 54.9 tok/s; use the 48.65 median, not that sample, as the representative result.

Both policies passed the 4096-position short reference, 512-position tails at
8K/32K/128K and eight functional checks before timing. Their reported numerical
summaries were identical, with 100% top-1 agreement against the frozen reference.
The 128K coarsened diagnostic covers about 46% common reference mass; it is not
comprehensive 360K quality evaluation. No preemption occurred in the 72 cached
measurements. There were also 12 conventional decode and 12 cold-prefill measurements.

Row32 was explicitly selected on all six ranks. Native API, Pi-router, Code and
Chat acceptance passed after reopening. All six GPUs remained exclusively owned;
final available memory was at least **10.33 GiB**, with the exact-container guards
clear. No 12-hour soak was run.

[Results, quality and decision](../../results/e3-row32-serving.json) ·
[Every timed sample](../../results/e3-row32-serving-samples.json)

Recalculate the published timing comparison:

```sh
python3 benchmarks/analyze_e3_public_samples.py --samples results/e3-row32-serving-samples.json
```

### Image and persistent policy

Selected image ID:
`sha256:b4988201229054893df527528c38e8091d4d5392d44c314406e66493ec9300da`.
This is a local Docker image ID, not a pullable registry digest.

Every rank requires this persistent file at `/root/.cache/amos-e3-rows.json`:

```json
{"revision":"e33-selected32","rows":32}
```

Install it in the rank's mounted cache **before startup**. Missing/invalid controls
fail visibly. Change policies only while the six-rank service is idle and drained.
Each rank latches the choice at target layer 3 for the whole prefill chunk; row32
and row64 share one scratch arena. Verify all six `.applied.json` acknowledgments
against the exact control digest after a real uncached prompt above 512 rows.
The image retains the 512-row native boundary; lowering it needs a separate
actual-route native/row32 comparison and full-model qualification.

[The builder](../../runtime/vllm/build_e3_rows.py) appends one layer to the exact
124-layer diagnostic base. Inputs are that retained base archive, the E32 probe
manifest and tested `row32/runtime.py`/binary, and the row32 CUDA source produced
by the patch below. It installs both kernels and the required policy wrapper in
both vLLM source roots, preserving image configuration and all weights. Its original
`amos_grouped_prefill.py` input remains in this repository; the builder changes its
import to the policy wrapper. This is not a clean-machine build.

CPU control checks:

```sh
PYTHONPATH=runtime/vllm python3 benchmarks/test_e3_rows_policy.py
```

## Native/E3 crossover comparison (E34)

The follow-up component comparison is complete. **The serving boundary is still
512 rows.** The next full-model candidate is native through 32 rows and row32 E3
above 32; it has not been promoted or measured in full-model serving.

All six serving ranks were stopped for these probes. Native, row64 E3 and row32
E3 used the same original weights and captured code/prose operands, including
their shorter prefixes, on ranks 0–5 and layers 3/40/77. Native retained its real
3072-capacity prefill plan. Row32 and row64 shared one E3 scratch arena.

**1,296 bitwise comparisons passed**, including changing-input graph replays at
256 and 513 rows. All three arms passed before timing. Eight rotated/reversed
repetitions per arm/cell yielded **7,776 timed calls across 324 cells**. The
64 MiB cache flush was outside timing; routing and epilogues were included.

| Input rows | Row32 / native speed ratio, geometric mean | Faster cell medians |
| --- | ---: | ---: |
| 33 | 1.307× | 36 / 36 |
| 65 | 1.280× | 36 / 36 |
| 128 | 1.272× | 36 / 36 |
| 256 | 1.241× | 36 / 36 |
| 384 | 1.240× | 36 / 36 |
| 512 | 1.239× | 36 / 36 |
| 513 | 1.236× | 36 / 36 |
| 768 | 1.258× | 36 / 36 |
| 1024 | 1.258× | 36 / 36 |

Row32 was faster in all 324 medians; observed sample ranges were disjoint in
313 cells. These ranges are not confidence intervals. Comparing the slowest
rank median per layer/prompt/row count also favored row32 in all 54 comparisons;
the smallest such ratio was 1.169×. Ranks were not synchronized for each timed
call, so these maxima approximate component critical-path costs, not distributed
latency. Three layers and two synthetic captures do not cover every workload.

The joint native/E3 comparison used a predeclared 3 GiB Torch budget, a 4 GiB
container limit and a 280-second deadline. The native and E3 scratch buffers
must coexist for alternating measurements; this differs from E32's two-E3-arm
2 GiB budget. Peak Torch allocation was **2.115 GiB**. Each exact-container guard
was fresh and clear with at least 12 GiB available before release; the unchanged
8 GiB / 2-second floor and pressure protections remained active. Rank 0/layer 3
passed the joint-memory screen before the other probes were released.

The selected `b498…` row32 image was explicitly resumed after all probes stopped.
Fresh functional, native, Pi-router, Code and Chat checks passed, along with all
six policy acknowledgments and exclusive GPU ownership. The existing numerical
qualification for that exact image, tuning and row policy was reused, not rerun.
Minimum available RAM at closeout was **13.33 GiB**; this snapshot is not a new
memory-footprint claim. Serving weights, precision, capacity and dispatch are unchanged.

[Results and closeout](../../results/e3-crossover-component.json) ·
[Every timing sample and scalar quality check](../../results/e3-crossover-samples.json)

Recompute all medians, ratios and comparison counts without GPUs:

```sh
python3 benchmarks/analyze_e3_crossover_samples.py --samples results/e3-crossover-samples.json
```

The public check validates exported comparison results; it cannot independently
recompute model outputs without the private captured tensors. To repeat the GPU
probe, stage [e3_crossover_check.py](../../benchmarks/e3_crossover_check.py) and
the existing loader helper with the [exact staged manifest](../../results/e3-crossover-manifest.json).
It records installed source/binary pins and private capture hashes. Use the same exclusive stopped-
fleet and exact-container guard procedure described above. This probe uses the
installed native and both installed E3 kernels; preserve all source/binary pins
and original capture hashes. Do not run it alongside serving.

The next full-model test must use one boundary rule for both generation and
teacher scoring, preserve native decode through 32 rows, pass the existing
short/long gates, and compare short prefills and chunk tails with repeated
matched workloads. **These isolated ratios are not an additional 24–31% serving
throughput gain.** The full-model measurements above remain the latest serving results.

## Historical diagnostic capture (E31)

A source-only diagnostic image is qualified:
`sha256:340a9bae07ab134120249cf0224108fda7b3706720724bd1b2fe705245e7ce20`.
It calls a bounded capture hook after the unchanged E3 result is computed, then
returns that same result. All capture controls are removed in serving.

The 4096-position short check, 512-position tails at 8K/32K/128K, eight functional
checks and native/Pi-router/Code/Chat acceptance passed. Top-1 agreement was 100%
on those numerical checks. The 128K coarsened KL is a lower-bound diagnostic with
about 46% common reference mass; this is not a broad 360K quality claim.
Final observed available RAM was at least 10.46 GiB, with all exact-container
8 GiB / 2-second guards clear.

The first code request hit a client helper's one-token streaming assertion.
The model stayed healthy; all 75 captures on every rank were already present
and were hash-verified without repeating that request. Its original usage was
not retained. Prose used a nonstreaming completion with verified usage.

The diagnostic capture itself did not measure serving speed. The E33 comparison
above supplies the latest serving measurements.

### Reproduce the component candidate

Apply [the exact tested delta](../../benchmarks/e3-row32.patch) to a separate copy
of `runtime/vllm/e3`, preserving the vendor headers and licenses, then use its
`build.sh` with CUDA 13.0 / `sm_121a`. Keep generated binaries out of Git. The
recipe uses the retained image and is not a clean-machine serving build.

Stage [the probe](../../benchmarks/e3_rows_check.py), its
[real-weight loader helper](../../benchmarks/grouped_prefill_kernel_check.py),
the modified runtime/binary under `row32/`, and a manifest containing the exact
image ID, probe-file hashes, imported runtime hashes and capture-file hashes.
The helper's historical standalone validation entry point has older pins;
this probe imports only its loader and uses its own manifest checks.

After stopping all six serving ranks and verifying every GPU is free, start
one bounded probe per rank for a single layer. Mount that rank's original
shards at `/model`, private captures at `/captures`, and its unique test directory
at `/probe`. Bind the unchanged memory guard to the recorded container ID, verify
it is fresh, armed and clear with at least 12 GiB available, then create
`/probe/go`. Never reuse the directory, release it before the guard, or run this
alongside serving. Inspect terminal states before the next layer or a deliberate
same-image serving resume.

Collect all 18 result files and run
[`analyze_e3_rows.py`](../../benchmarks/analyze_e3_rows.py). It checks the exact
comparison counts, alternating sample order and one-arena contract before
summarizing ratios. Full-model qualification and application acceptance remain
separate requirements for promotion.

## Reproduction

This needs the retained serving image and original TP6 shards. The repository
still does not provide a portable image release or a verified clean build.

1. Verify the deployed source hashes, weights, tuning and all six GPU owners.
   Freeze two synthetic prompts whose first 3072 tokens differ, using the exact
   chat renderer; record full-prompt and prefix hashes. Do not capture application
   conversations.
2. Build the diagnostic with
   [`build_e3_capture.py`](../../runtime/vllm/build_e3_capture.py). Its input is the
   original runtime from repository commit `7ee9297bdf574e9e34c6050a161c5a247531dad7`,
   SHA256 `2651478eee38d054cd163df0bd01e219515cd244fcacb9e354b1395e55dbc00b`,
   and the exact MTP-control base archive. A 126-layer append hit Docker's depth
   limit; use [`squash_patch_layers.py`](../../runtime/vllm/squash_patch_layers.py)
   to merge only the last three regular-file source layers, preserving the first
   123 layers and runtime configuration. Verify every merged file after loading.
3. Own and drain the six-node maintenance window. Keep fresh exact-container
   guards and pressure protections. Launch only this forward diagnostic image;
   qualify short/long references and functional behavior before capture.
4. On every rank, arm `/root/.cache/amos-e3-capture-control.json` only for that
   owned window. Schema 1 requires revision `e3<number>-<label>`, `rows: 3072`,
   `input_layers: [3,40,77]`, `max_bytes_per_rank` at most 335544320,
   and numeric `armed_at`/`expires_at` no more than 180 seconds apart. The hook
   skips graph capture and records only the first qualifying call per layer.
5. Submit one cold synthetic prompt with one output token. Require API completion
   and usage; visible text is unnecessary for a one-token reasoning response.
   Remove the exact owned controls in cleanup. Verify no diagnostic error marker,
   all 75 manifests per rank, file hashes, byte limits and the three input layers.
6. Repeat with a new revision and the other prompt. Keep tensor files private.
   Combine the verified manifests and run
   [`analyze_e3_routes.py`](../../benchmarks/analyze_e3_routes.py). It validates
   rank agreement, physical ownership and route-count conservation before
   calculating padding. No GPU is needed for analysis.
7. Remove controls, explicitly select the qualified image, reopen promptly and
   run actual native/router/Code/Chat acceptance. Never run standalone GPU probes
   alongside serving. A future component comparison requires its own drained,
   exclusively owned window.

CPU boundary checks: `python3 benchmarks/test_e3_capture.py`.

## Attribution

The E3 kernels derive from MiaAI-Lab's EXL3 fat-expert work, retaining the bundled
AGPL notice. ExLlamaV3 supplies the trellis/Hadamard building blocks; b12x supplies
the native activation and router-order epilogues; vLLM supplies serving. The
full-model TP6 recipe builds on Adapt and Kindling's work. The local capture,
ownership audit and row-tile experiment preserve these sources and their
per-component licenses; see [credits](../../CREDITS.md).
