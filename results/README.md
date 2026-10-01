# Results — October 1, 2026

## Serving: vLLM P24

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
compatibility. The production model remains vLLM P24.

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
