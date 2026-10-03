# E3 prefill: actual TP6 routes

October 3, 2026. **Component comparison passed; full-model row32 qualification remains.**
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
| Unused capacity in current 64-row tiles | 26.46% | 26.64% |
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
| Current 64-row | 255 | 176 bytes | 752 / 764 bytes |
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

The current row64 serving image was explicitly resumed, with fresh functional,
native, Pi-router, Code and Chat acceptance. Its existing exact-image short/long
numerical qualification was reused, not reported as rerun. Final available RAM
was at least 14.07 GiB. Row32 is **not promoted**. A comparison image has been
staged for full-model numerical gates and balanced serving measurements.

[All samples and comparison checks](../../results/e3-row32-component.json) ·
[Serving closeout](../../results/e3-row32-outcome.json)

[Route results and qualification](../../results/e3-prefill-diagnostic.json) ·
[Compiler results and source hashes](../../results/e3-row32-compile.json)

## What is serving

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

No new serving speed is reported here. The communication comparison remains
the latest throughput measurement.

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
