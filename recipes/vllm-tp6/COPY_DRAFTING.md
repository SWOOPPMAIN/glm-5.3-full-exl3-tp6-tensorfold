# Target-verified copy drafting — completed

**Retain original adaptive MTP.** GPU ngram passed our bounded correctness gates
but substantially reduced general generation and four-request throughput. The
selected forward image includes the tested compatibility fixes; its active profile
is `mtp4`. Full GLM-5.3, original 3.25 bpw EXL3, target/draft MXFP8, TP6,
360K configured context, four admitted requests and the 3072-token budget remain.

## Matched measurements

Three samples per cell and arm: one MTP control repeat, three GPU-copy repeats,
then two MTP control repeats. Frozen synthetic prompts, low reasoning, temperature
zero, seed 17, identical output budgets and separately recorded kernel warmups.
The copy matrix uses 256 output tokens per request and warmed prefixes. C4 mixes
exact repetition, edited code, prose and another repetition, with four requests
observed running. Throughput includes first-token delay and all proposal/verification
overhead. This differs from the earlier E3 short code/prose/tool workload.

| Workload | MTP output tok/s | GPU copy output tok/s | Copy change |
| --- | ---: | ---: | ---: |
| edited-code-2048 | 38.45 | 32.96 | -14.3% |
| edited-code-8192 | 33.78 | 26.50 | -21.5% |
| mixed-c4-2048 | 71.38 | 34.54 | -51.6% |
| mixed-c4-8192 | 70.33 | 29.36 | -58.3% |
| prose-2048 | 33.09 | 22.92 | -30.7% |
| prose-8192 | 29.96 | 22.59 | -24.6% |
| repeat-code-2048 | 39.10 | 31.18 | -20.3% |
| repeat-code-8192 | 51.76 | 55.91 | +8.0% |

Standard generation uses 512 output tokens and estimates decode from streaming.
Cold prefill uses distinct cache salts and divides prompt tokens by TTFT.

| Measurement | MTP | GPU copy | Copy change |
| --- | ---: | ---: | ---: |
| Prose output tok/s | 35.53 | 24.05 | -32.3% |
| Code output tok/s | 48.31 | 24.10 | -50.1% |
| Cold 8K input tok/s | 978.93 | 1004.21 | +2.6% |
| Cold 32K input tok/s | 961.05 | 983.62 | +2.3% |

Copy's matrix geometric-mean ratio is **0.705** versus MTP.
The measurements do not justify a hybrid implementation. The 8K exact-repeat gain is narrow (+8.0%, about 55.9 versus 51.8 output tok/s); 2K repetition, edited code, prose and both C4 mixes regress. Per-request routing and shared proposal scheduling would add unmeasured costs and compatibility work. A future hybrid needs a new selector/cost hypothesis and repeated mixed-workload evidence.

[Summary, ranges and quality](../../results/copy-drafting-serving.json) ·
[Every timed sample](../../results/copy-drafting-serving-samples.json) ·
[Synthetic fixtures](../../results/copy-drafting-fixtures.json)

## Correctness and scope

All three visits passed the frozen 4,096-position short reference and 512-position
tails at 8K/32K/128K, followed by eight basic API/tool checks. The numerical gate
requires coarsened KL ≤0.001, top-1 agreement ≥0.995 and NLL increase ≤0.01.
The 128K reference covers about 46% common probability mass; this is not full
360K quality evaluation. Exact-copy and rename-edit AST checks passed, as did
unequal output lengths, four-stream cancellation and a healthy follow-up.
No preemptions occurred in the timed matrix. Native, Pi-router and actual Code/Chat
acceptance passed on the selected MTP image, including tools and memory integration.

Both pinned ngram proposers use vLLM's native target rejection sampler; synthetic
acceptance stays disabled. GPU ngram allows async scheduling; CPU ngram disables
it in this fork and was checked on CPU only. Standalone copy explicitly disables
MTP-only policy and draft-EH hooks while retaining target shared384/projections.
The runtime trial used four speculative slots and a five-token lookup key.

**Counter limitation:** GPU-worker trimming does not update the engine-core slot
list used by the draft counter. Reported slots include invalid/no-match positions;
slots not accepted are not an exact count of valid copied proposals rejected by
the target. Samples retain raw counters, accepted tokens per wall/server-decode
time, TTFT and stream gaps. Isolated GPU proposer cost was not measured.

The first MTP visit used the history-fix image. Copy and final MTP used an additional
ngram-only configuration repair. Target kernels, weights, precision and the MTP
validation branch are unchanged; both images passed full numerical checks.
All samples and control drift are retained. Three samples and one copy boot do
not establish a many-boot confidence interval or long-duration reliability.

## Compatibility fixes and attribution

The initial copy launch failed on all six ranks before loading weights: validating
the target alias as a separate draft model checked 64 heads against TP6 before
normal target padding to 72. The conditional repair verifies ngram's target aliases
and leaves full target padding/validation and model-backed draft validation intact.
All 40 extracted-validator CPU cases passed, followed by actual six-rank startup
and the runtime gates above. [Patch/checker](../../benchmarks/ngram_config_fix.py) ·
[CPU receipt](../../results/copy-drafting-config-fix-cpu.json).

The history-scatter repair gives padded writes unique modulo destinations while
masked positions preserve prior values. All 2,048 history rows in 512 batches
passed an independent CPU append oracle at capacities 8/32/128/360000; the original
failed 158 batches. Matching kernels and the target verifier were unchanged.
[Patch/checker](../../benchmarks/ngram_scatter_fix.py) ·
[Receipt](../../results/copy-drafting-scatter-fix-cpu.json).

vLLM supplies the proposers, async scheduling and rejection sampler. Swoopp supplied
the narrow fixes, TP6 integration, tests and comparison. The underlying model,
quantization, E3 and collective contributors retain [their credits](../../CREDITS.md).
[Pinned source review](../../results/copy-drafting-source-review.json) ·
[CPU proposer checks](../../results/copy-drafting-cpu.json) ·
[Tensor semantics](../../results/copy-drafting-tensor-cpu.json).

## Reproduce

Recompute the published comparison without a GPU:

```bash
python3 benchmarks/analyze_copy_public_samples.py \
  results/copy-drafting-serving-samples.json \
  --summary results/copy-drafting-serving.json
```

The source-pinned checkers accept `--source ORIGINAL --candidate NEW --output RECEIPT`.
Use the retained original `ngram_proposer_gpu.py` for the scatter checker and
`config/speculative.py` for the configuration checker. The overlay builders
[`build_ngram_history.py`](../../runtime/vllm/build_ngram_history.py) and
[`build_ngram_config.py`](../../runtime/vllm/build_ngram_config.py) take
`--base-archive --source --checks --output --receipt`; run with `PYTHONPATH=benchmarks`
from the repository root. Apply history then configuration. They require retained
image layers and exact source hashes, and emit delta archives. They do not provide
a clean-machine build or a complete distributable image.

For a new runtime comparison, reserve all six GPUs, drain applications, own the
maintenance hold and arm fresh exact-container 8 GiB / 2-second guards including
pressure protections. Use the [launch contract](README.md), selected E3 controls
and `tuning.json` for MTP; `copy-tuning.json` explicitly disables seven MTP-only
controls for `ngram-gpu4`. Run numerical and functional gates before timing.

```bash
python3 benchmarks/copy_matrix.py \
  --base "$GLM_API" --key-file "$GLM_KEY_FILE" \
  --image "$QUALIFIED_IMAGE_ID" --profile mtp4 \
  --tuning-file recipes/vllm-tp6/tuning.json \
  --fixtures results/copy-drafting-fixtures.json \
  --label mtp-control-a --repetitions 1 --output mtp-control-a.json
```

Then run three repeats with the copy profile/tuning and two with MTP, using the
same fixtures and new output paths. The public client checks rendered token/hash
identity and measures the matrix; image/profile are operator assertions. It does
not launch containers, enforce site guards, run the numerical/application gates or
choose a serving policy. The executed private controller additionally verified all
six launch configurations and guards. Public-driver packaging checks passed; a
clean-machine replay remains unverified. Standard cold/decode results use
[`performance.py`](../../benchmarks/performance.py). Reopen qualified serving promptly.
