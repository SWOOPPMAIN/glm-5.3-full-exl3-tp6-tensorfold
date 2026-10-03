# Full GLM-5.3 EXL3 · TP6 · TensorFold research

Full GLM-5.3 on **six NVIDIA DGX Sparks**, preserving the original
**3.25 bpw mixed K3/K4 EXL3 weights**. Serving uses **vLLM P27**.
TensorFold is an experimental research port and is not serving production.

**Status — October 3, 2026 UTC:** production hardening is complete within the
requested brief-check scope. All six controlled reboots, native API and
Code/Chat acceptance passed. OS, network and memory settings persist across
reboot. No 12-hour soak was run. [Operating notes](recipes/vllm-tp6/OPERATIONS.md).

## Latest serving measurements

| Measurement | vLLM P27 |
| --- | ---: |
| Prose generation, one request | **36.4 output tok/s median** |
| Code generation, one request | **47.9 output tok/s median** |
| Four concurrent requests, cached short mixed workload | **78.6 output tok/s combined median** |
| Cold prefill, 8K / 32K | **977 / 963 input tok/s** |
| Time to first token, 8K / 32K | **8.38 / 34.04 s** |
| Configured context / admitted requests | **360,000 tokens / 4** |
| KV allocation | **24 GiB per rank** |

October 3, three samples per workload across four alternating boundary-policy
visits in the same six containers. **Row32 E3 now handles prefill above 32 rows:**
65–512-token cold prompts have **4.4–7.7% lower one-token latency**, with separated
observed ranges. Long-prefill medians change −0.5% / −0.2%; prose +2.6%, code
−3.2%, with overlapping ranges. The cached matrix changes +0.9% overall.
These generation differences are not isolated E3 decode-kernel gains.
The earlier row64/row32 study's 3.9% / 3.3% cold-prefill gains remain documented.
Prefill is prompt tokens divided by TTFT. C4 uses 256 output tokens per request;
single-request generation uses 512. All requests share the cache.
[Every sample, quality checks and replay](recipes/vllm-tp6/E3_PREFILL.md).

New cache replay: repeated 8K / 32K / 128K code-tool histories reached first
tokens in **0.60 / 0.64 / 0.87 seconds**, versus **8.65 / 35.28 / 142.39 seconds**
cold. All 72 synthetic checks passed. This validates existing caching; it is
not a new runtime speedup. [Protocol and results](benchmarks/CACHE_REUSE.md).

Prefill-budget update: fixed/adaptive 1536 and 768-token policies failed our
numerical gate. Serving retains **3072** on a newly qualified scheduler-control
image; no speedup is claimed. [Results and recipe](recipes/vllm-tp6/PREFILL_BUDGETS.md).

MTP tuning: **270 workload measurements** and **72 cost-calibration cells** are
complete. Recalibrated policies showed no useful overall gain in the bounded
screen; the original adaptive policy remains selected. [Results and replay](recipes/vllm-tp6/MTP_TUNING.md).

The latest local TensorFold HTTP measurements remain around 19–21 output
tok/s for one request and 368–388 cold-prefill tok/s. Its strict numerical
fidelity gate still fails. The bundled source is the historical TFP21 snapshot;
[later results through TFP57](results/TENSORFOLD_STATUS.md) are reported separately.

## Recipes

1. [Prepare and verify original weights](recipes/weights/README.md)
2. [Serve with the qualified vLLM TP6 image](recipes/vllm-tp6/README.md)
3. [Operate the six-node deployment](recipes/vllm-tp6/OPERATIONS.md)
4. [Benchmark a candidate](benchmarks/README.md)
5. [Explore the historical TensorFold port](recipes/tensorfold-tp6/README.md)

The serving recipe requires the retained P27 image. A portable image release
and a verified clean-machine build remain unfinished. Weights and compiled
artifacts are not included in this repository.

## Next experiments

[Ranked experiment plan](docs/PERFORMANCE_EXPERIMENTS.md): prompt reuse,
mixed prefill/decode scheduling, workload-aware MTP, six-rank communication,
E3 prefill kernels, and target-verified copy/ngram drafting. Prompt reuse has
been measured; smaller prefill budgets and MTP retuning were rejected for promotion.
Dual-port communication and its bounded crossover screen are complete; retain
the 2 MiB cutoff. [E3 row32](recipes/vllm-tp6/E3_PREFILL.md) passed component and
full-model checks and is selected for repeatable cold-prefill gains. The native/E3
component comparison passed 1,296 exact checks; the full-model boundary
comparison is now complete. [Copy/ngram drafting](recipes/vllm-tp6/COPY_DRAFTING.md)
remains: source/CPU checks, a history-scatter fix and an offline launcher are
prepared; no serving comparison or promotion yet.
[Latest upstream review](docs/UPSTREAM_REVIEW_20261003.md).
TensorFold development is excluded from this optimization goal.

## Credits

Built on **Z.ai**, **davidsyoung**, **Turboderp / ExLlamaV3**, **vLLM**,
**local-inference-lab / b12x**, **Ash Hart / TensorFold**, **MiaAI-Lab**,
**Adapt AI Systems**, **Kindling**, **Matt Mastracci**, and the wider Spark community.

See [CREDITS.md](CREDITS.md), [third-party notices](THIRD_PARTY_NOTICES.md),
[source provenance](provenance/README.md), and [the roadmap](docs/ROADMAP.md).
