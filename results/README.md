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
The attention candidate is still unmeasured on GPU.
