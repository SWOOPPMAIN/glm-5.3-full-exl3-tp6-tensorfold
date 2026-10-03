# E3 prefill: actual TP6 routes

October 3, 2026. **Diagnostic complete; kernel optimization remains in progress.**
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
The 32-row candidate changes only the two `FM_MB_*` constants from 4 to 2. It is
**compiled only, not GPU-tested or promoted**. Static compiler spill reports do
not establish runtime traffic or speed. Next: exact captured-input comparisons,
then full-model quality gates and repeated serving timings if the component wins.

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
