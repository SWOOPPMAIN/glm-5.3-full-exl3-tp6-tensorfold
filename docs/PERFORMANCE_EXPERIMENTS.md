# Ranked performance experiments — October 3, 2026

**Active vLLM optimization scope. TensorFold work is excluded.**
Prompt-cache replay results are recorded in [cache reuse](../results/cache-reuse.json)
and its [reproduction protocol](../benchmarks/CACHE_REUSE.md).
Smaller/adaptive prefill budgets were rejected at the numerical gate; see
[the experiment](../recipes/vllm-tp6/PREFILL_BUDGETS.md). MTP retuning also
finished without a promoted policy. [Dual-port RoCEnante](../recipes/vllm-tp6/COMMUNICATION.md)
is selected after a 72-measurement comparison: +2.37% across the cached matrix,
with per-workload regressions and unchanged prefill. Six-rank size profiling and
the bounded crossover screen are complete: retain 2 MiB; 128 KiB loses C4 speed
and 16 MiB fails fidelity. [E3 route capture](../recipes/vllm-tp6/E3_PREFILL.md)
is complete: about 26.5% of current tile capacity is unused padding. A 32-row
candidate eliminates compiler-reported gate/up spills but has not been GPU-tested.
E3 kernel comparisons and copy drafting remain.
Keep full GLM, TP6, the original 3.25 bpw experts and the current dense precision.
Start from [P27 measurements](../results/p27-serving.json), with production serving
remaining the priority. No numerical speedup promise is supported for this list.

| Priority | Experiment | Primary benefit | Concrete first comparison |
| --- | --- | --- | --- |
| 1 | Prompt reuse across real agent turns | Faster follow-up first token; less repeated prefill | Replay three alternating synthetic code/tool conversations at 8K/32K/128K; log rendered-prefix hashes, cached tokens, eviction and TTFT |
| 2 | Decode-aware prefill budgets | Shorter streaming stalls under mixed traffic | Three decoders plus a fresh 32K prompt; compare current 3072 budget with 1536/768 and a bounded adaptive budget |
| 3 | Recalibrate current adaptive MTP | More accepted output per target pass | Current policy versus depth 1/2/4 by code/prose/tools, C1/C4 and short/long context; count draft costs and accepted tokens |
| 4 | Six-rank communication crossover | Less waiting between compute steps | Use observed P27 message sizes; compare small-message RoCE thresholds, larger-message NCCL and per-rank skew on the already-enabled dual rails |
| 5 | E3 prefill layout and dispatch | Higher genuinely uncached prefill throughput | Native/E3 crossover and row tiles on real mixed-K/routed inputs, then cold 8K/32K; preserve activation/reduction order |
| 6 | Target-verified copy/ngram drafting | Faster repeated code and boilerplate | Standalone proposal method versus current MTP on exact-repeat, edited-repeat and prose controls; count verification waste |

## Why this order

**Prompt reuse:** the first completed replay found no avoidable loss relative to
the native MTP cache allowance across three alternating 8K/32K/128K histories.
Keep the existing policy; additional cache-policy changes are not justified by
these measurements. Raw cold-prefill tok/s does not describe an agent that repeatedly
resends almost the same history. First establish whether the current serving path
actually retains those shared tokens; do not assume caching is absent. Keep prompt
rendering stable where semantics allow, and test branch/fork reuse and eviction.
Changing retained reasoning changes the prompt, so treat that as a separately
qualified template choice. Do not silently change the model's reasoning policy.
[vLLM prefix-cache documentation](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)
describes the underlying reuse mechanism.

**Mixed scheduling:** compare median and p95 token gaps, new-request TTFT and total
completed output. Reducing prompt work per round can help ongoing replies while
slowing an arriving long prompt. This is a latency/throughput tradeoff, not a free
cold-prefill gain. [vLLM tuning documentation](https://docs.vllm.ai/en/latest/configuration/optimization/)
explains the token-budget tradeoff; compatibility must be checked against our pinned fork.
The October 3 fixed/adaptive 1536/768 screens all failed strict fidelity at 8K.
Even mixed 3072 differs from the isolated oracle. Retain 3072; numerical
consistency across chunk/batch shapes is a prerequisite to another budget trial.

**MTP:** adaptive depth and request/phase policy are already installed. Retune using
actual workload classes, especially tools and long-context cache pressure, rather
than presenting MTP4 or adaptive drafting as a new feature. Optimize accepted tokens
per total target-plus-draft time, including graph capture and rejection work.
The [270 measurements and 72 calibration cells](../recipes/vllm-tp6/MTP_TUNING.md)
are complete. Fixed depth 3's first-pass advantage shrank to 0.9% with repeats;
no global fixed-depth policy was promoted. The fitted windows 8/16/32 were within
0.3% overall of original in a bracketed screen, with all three C4 averages lower.
Retain original adaptive MTP; this rejects promotion, not every possible retuning.

**TP6 communication:** two NCCL rails, performance-core affinity, custom small-message
RoCE and decode projection sharding were already present. Six-rank profiling showed
that decode's custom RoCE path used one HCA while NCCL prefill used both. The qualified
dual-HCA custom path is now selected. Actual histograms matched across all six
ranks. The 128 KiB crossover passed existing numerical gates but lost 10.1% in
the bounded C4 screen; 16 MiB failed fidelity before timing. Retain 2 MiB.
Preserve deterministic
sum order and validate with all six ranks; two-node bandwidth alone is insufficient.
The previous generic NCCL protocol screen found no gain, so avoid repeating that sweep
without a new measured bottleneck. [Kindling's TP6 recipe](https://github.com/kindlingai/glm-5.3-full-exl3-tp6)
is a reference for the existing collective split.

**E3:** this targets the vLLM prefill path, not a rerun of the negative TensorFold
expert-launch screen. Measure rank imbalance, padding, routed token counts and
complete MoE latency; a faster isolated GEMM can lose after routing and reductions.
Use current graph and memory constraints rather than blindly raising the batch budget.
The two actual-route captures cover all 75 layers on six ranks. A 32-row layout
would reduce padded rows about 14% while increasing segment count about 72%.
Its CPU compile removes gate/up spills; compare exact captured inputs before
claiming a gain. The qualified diagnostic hook is inactive in serving.

**Copy drafting:** no new drafter weights are required. Full target verification is
mandatory. First validate compatibility with the pinned TP6 scheduler; combining it
with existing MTP is additional implementation work. Flash's DFlash2 checkpoint is
not established as a compatible full-GLM drafter.
[vLLM speculative-decoding reference](https://docs.vllm.ai/en/latest/features/speculative_decoding/).

## New upstream ideas worth inspecting

[Mia v1.4 changelog, pinned review](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/blob/cf28cc4f8038be322cdeda220c6f1c8ace8f27d1/CHANGELOG.md)
adds prompt-state retention fixes, bounded interleaving of prompt work with decode,
and experimental TP-N/three-Spark handling. These are useful design references;
they are not drop-in support for our full mixed-K model on six ranks. Separate its
stream presentation buffer from actual compute improvement: smoothing display alone
does not increase tokens generated. Its startup connection and rank-consistency fixes
are robustness ideas, not demonstrated TP6 speedups.

[TensorFold 0.6.1](https://github.com/ashhart/TensorFold/releases/tag/v0.6.1)
includes CUDA prompt batching and family-specific improvements. Our port still needs
strict numerical repair before a production comparison. Its published gains on other
models/hardware cannot be added to our baseline, and NVFP4 paths do not directly apply
to the retained EXL3 checkpoint.

The [Kindling review at `3b9c548`](https://github.com/kindlingai/glm-5.3-full-exl3-tp6/commit/3b9c548c22487537c1af1323af30fa9b0daef345)
adds an opt-in D13 adaptive-MTP launcher and alternating benchmark/hang records;
the scheduler and graph-overlay code did not change in that merge. Our phase-aware
policy already implements this class of adaptation, so their fixed-k4-relative
gains are not additional gains on our baseline. Its `kring` CUPTI tool addresses
an observed hang; do not count instrumentation as a throughput optimization. Kindling
Spark OS 0.9.5 changes Wi-Fi/setup/console behavior; upgrading it is not a measured
inference speed improvement.

## Work to deprioritize with present evidence

- Early shared-expert overlap (P25): with the real router, component changes ranged
  from **1.46% slower to 0.59% faster**. No consistent useful gain; no full-model promotion.
- Repeating TFP56's expert launch grid: best 3072-row local improvement was **0.16%**.
- Timing rejected TFP57 latent layouts: numerical checks fail before timing eligibility.
- Dense GPTQ, lower-bpw weights or a Flash drafter: outside this weight/precision-preserving plan.
- Treating a TensorFold rebase or OS update as an automatic speedup.

See [local negative evidence](../results/tensorfold-latest.json).

## Common comparison and acceptance

Use the same pinned weights, prompts, chat template, output budgets and inference
settings. Keep code/prose/tools separate; measure C1 and C4, cold and reused prompts,
TTFT, median/p95 token gaps, accepted draft rate, total throughput and minimum host RAM.
Warm kernels separately and use balanced repeated control/candidate measurements
inside one owned experiment. A benchmark control is not a production rollback.

For GPU work, drain the fleet and run exactly one backend with fresh exact-container
8 GiB / 2-second guards. No canary may overlap the six serving workers. Reject numerical
failures before speed selection, then check original-reference short/long tails and
actual native/router/client behavior. No 12-hour soak is part of this plan. Do not
change serving defaults until a repeatable benefit and the existing quality gates pass.
