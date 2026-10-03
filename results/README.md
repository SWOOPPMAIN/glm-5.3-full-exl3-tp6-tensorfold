# Results — current baseline and historical experiments

## Current serving: vLLM P27

Latest October 3 image adds bounded MTP controls with **original adaptive MTP
still selected**. [First-pass results](mtp-first-pass.json) cover 60 cells with
one visit each; no experimental policy has been promoted. Original-reference
short/long and native/Pi/Code/Chat checks passed on the selected original policy.
[Conditions, limitations and replay recipe](../recipes/vllm-tp6/MTP_TUNING.md).

October 3 update: the current image adds scheduler-budget controls while retaining
3072 and the P27 inference settings. All four smaller/adaptive screens failed
strict fidelity. [Results](prefill-budgets.json) and
[recipe, mixed-load limitation and packaging repair](../recipes/vllm-tp6/PREFILL_BUDGETS.md).
Native/Pi/Code/Chat acceptance passed; no new speedup is claimed.

[Samples and quality checks](p27-serving.json) · [hardening closeout](production-hardening.json)

October 2 baseline: **35.13 prose / 47.35 code output tok/s**, medians of three
warm requests; **76.52 aggregate C4 output tok/s**, median of three rounds.
Cold 8K/32K prefill: **938.34 / 931.56 input tok/s**; TTFT **8.730 / 35.173 s**,
two repeats per context. Cold prefill means input tokens / TTFT, including the
first generation step and delivery. The hardening pass did not rerun speed tests.

Teacher-forced short reference comparison: 4,096 positions, exact top-1 agreement
and zero measured coarsened KL. The 512-position tails at 8K/32K/128K also match
that frozen reference exactly. This is not comprehensive 360K task evaluation.
128K speed was not remeasured in this pass; the P24 timing below is historical.
The current KV allocation remains 24 GiB per rank; historical P24 slot counts
must not be confused with TensorFold's separate 804K experimental allocation.

Latest local TensorFold benchmark, failed fidelity gate and rejected kernel
candidates are in [the status report](TENSORFOLD_STATUS.md). Historical results
below retain their original scope and source snapshots.

## Historical serving: vLLM P24

[Summary](p24-serving.json) · [individual warm decode samples](p24-decode-samples.json)

| Workload | Result |
| --- | ---: |
| Code, 512 output tokens, three warm requests | 46.77 tok/s median; 44.25–47.35 range |
| Prose, 512 output tokens, three warm requests | 33.69 tok/s median; 33.54–34.17 range |
| Four concurrent requests, one run | 72.36 output tok/s combined |
| Cold 8192 / 32768 / 131072-token prompt TTFT | 9.669 / 39.010 / 157.338 s |

Decode is estimated from visible streamed output; MTP can emit multiple
tokens per event. Effective prefill is input tokens divided by TTFT, not
isolated GPU kernel throughput. Warm decode was recorded at 10:47 UTC.
Cold requests use unique cache salts. Short-prefill comparisons use five
requests per exact shape across adjacent boots; long-prefill controls are
earlier runs. Four-request throughput is one run per boot, not a stable ceiling.

P24 improves 128–512-token prompt latency by 6–17% versus its preceding
control. Long prefill is essentially unchanged; C4 is 1.9% below that control.
Do not describe P24 as a uniform throughput gain.

The deployed profile configures 360K context and 24 GiB KV allocation, with
**470,847 shared cache tokens**, confirmed from the current boot log. An
earlier status summary conflated this with TensorFold's 804K test allocation.
The configured context is not evidence of comprehensive 360K reasoning quality.

## TensorFold: TFP14

[Six-rank result summary](tfp14-tensorfold.json)

| Synthetic workload | Control | Row-sharded reductions |
| --- | ---: | ---: |
| Full target, 3072 rows | 11065.54 ms | 9617.51 ms |
| MTP, 3072 rows | 180.62 ms | 155.61 ms |
| Short 17-row captured target pass | 181.66 ms | 180.67 ms |

Timings are the maximum of six per-rank medians. Full passes use five repeats
on one loaded model, control before candidate. The short graph uses eleven.
The target gain is **15.1% throughput / 13.1% lower latency**. Short graph
results establish no meaningful decoded-token speed gain.

43 CPU checks and 618 exact GPU comparisons passed. All six probes exited
successfully with guards clear. Maximum PyTorch allocation was 100.18 GiB;
minimum observed host-available memory was 11.37 GiB. Cgroup memory alone
understates unified GPU use.

These are synthetic model-forward measurements with all 804K cache slots
resident, not user-request generation or authentic 360K prompt benchmarks.
They must not be compared directly with vLLM's input/output token rates.
The later attention experiment is recorded below.

## TensorFold: TFP15 attention tuning

[Six-rank summary](tfp15-tensorfold.json). Fixed-loop 128-row control: **9.551 s**;
best `skip128`: **8.805 s** (+8.5% throughput) for the same synthetic
3,072-row full target. All **1,098 GPU checks** passed: 546 exact
comparisons and 552 independent FP64 tolerance checks.
CPU checks: 45. Peak PyTorch allocation: 100.27 GiB; minimum
host-available memory: 10.67 GiB. All six guards clear.

Three timed repeats per full-pass mode, control first; no interleaved rerun.
The candidate is not serving. The same caution about synthetic timing and
authentic long-context quality applies.

## TensorFold: TFP16 request execution

[Six-rank summary](tfp16-tensorfold.json). All 17 checks on each rank passed
(102 total), including five exact output-sequence comparisons per rank
(30 total) against independent serial target generation.
Four short requests used greedy, keyed top-k and
top-p/min-p sampling, followed by retained-prefix continuation and exact-prompt
replay. Every sample asserted agreement across ranks.

The four requests accepted 26 of 55 MTP proposals (47.3%), yielding 44 output
tokens across 18 target verification passes (2.44 per pass), excluding the
initial prompt-head samples. These short-fixture counts are not wall-clock speed.

64 CPU checks passed. Peak PyTorch allocation: 99.92 GiB;
minimum host-available memory: 11.06 GiB. All guards clear.
GPU cancellation coverage is before a forward; deeper cancellation paths and
all four rejection positions are covered by CPU state-machine tests.

This was eager execution with raw-text prompts, not an API or speed benchmark.
The 804K cache was resident; authentic long-context quality remains unqualified.

## TensorFold: TFP17 distributed requests

[Six-rank summary](tfp17-tensorfold.json). All 28 checks passed, including
6 client output comparisons against independent serial generation. Four clients
used the actual six-rank command channel, leader sampling and eager scheduler.
Retained follow-up, invalid input, idle wake, callback cancellation, a healthy
request after cancellation and full shutdown were checked.

Each rank completed 43 identical commands and
106 leader-owned sampling decisions.
82 CPU checks passed. Peak PyTorch allocation: 99.92 GiB;
minimum host-available memory: 11.08 GiB. Guards stayed clear.

This qualifies distributed request mechanics using short raw-text fixtures.
It does not establish serving throughput, long-context quality or HTTP/chat API
compatibility. The production model at that experiment was vLLM P24; current serving is P27.

## TensorFold: TFP18 decode graphs and local HTTP

[Six-rank summary](tfp18-tensorfold.json), [recipe and conditions](../recipes/tensorfold-tp6/DECODE_GRAPHS.md).
329 GPU checks passed, including 264 exact tensor checks and 27 sequence comparisons.
91 CPU checks passed. The actual HTTP handler used the checkpoint chat template,
four clients, SSE text/token/usage checks, retained history and cancellation.

With 128 output tokens, three warm C1 repeats measured code 22.78 → 24.48 tok/s
and prose 23.75 → 23.64 tok/s. One C4 run per mode measured 22.34 → 23.11 tok/s
combined, including prefill. All outputs matched independent serial decoding.
These runs compare TensorFold eager and graphs, not the separate P24 workload.
They do not qualify broad API behavior, long-context quality or production deployment.

## TensorFold: TFP19 draft depth and kernel profiling

[Summary](tfp19-tensorfold.json), [method and interpretation](../recipes/tensorfold-tp6/DECODE_PROFILING.md).
630 GPU checks and 95 CPU checks passed. MTP4 led the short direct-controller
code fixture at 24.72 tok/s; MTP6 led prose at 25.65 tok/s versus 24.98 for MTP4.
All 42 timed requests used warm graphs; each answer matched independent serial
generation. Separate three-round traces identified dense BF16 projections as
the largest compute category. This is not a serving promotion or broad quality gate.

## TensorFold: TFP20 BF16 projection tiles

[Summary](tfp20-tensorfold.json), [recipe and limits](../recipes/tensorfold-tp6/PROJECTIONS.md).
306 strict GPU checks and 103 CPU checks passed. Separate component screening
performed 2970 comparisons and rejected 0 tiles across ranks.
Synthetic full3,072-token prefill: 8.841 → 8.157 seconds.
Short MTP4 direct-controller code/prose: 24.36 / 25.78 tok/s.
Original weights preserved; broader API/long-context and serving promotion remain pending.

## TensorFold: TFP21 packed requests

[Summary](tfp21-tensorfold.json), [recipe and limits](../recipes/tensorfold-tp6/PACKED_REQUESTS.md).
230 GPU checks and 119 CPU tests passed. Matched short HTTP C4 aggregate
throughput: 23.61 → 51.77 output tok/s, including prefill and delivery.
Original weights and TFP20 projection plan preserved; complete API/long-context
qualification and serving promotion remain pending.

## Historical TensorFold: TFP22 first incomplete scheduling run

[Recorded evidence](tfp22-tensorfold.json). 349 checks passed before a
host-memory guard stop during the final automatic HTTP concurrency measurements.
The 125 CPU tests and 8,218-token scheduling parity passed; the complete GPU
qualification was incomplete in that attempt. The bundled source remains TFP21;
later local qualification and current blockers are in [the status report](TENSORFOLD_STATUS.md).

## Current vLLM prompt-cache replay

[Summary](cache-reuse.json), [72 per-request samples](cache-reuse-samples.json),
and [reproduction protocol](../benchmarks/CACHE_REUSE.md). Three alternating
code/tool conversations retain their prefixes at 8K, 32K and 128K; no avoidable
misses were observed relative to the native MTP block allowance. The inference
configuration was unchanged. These short-answer fixtures do not remeasure
code/prose generation or four-request throughput.
