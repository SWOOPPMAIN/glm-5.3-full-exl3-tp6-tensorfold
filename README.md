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
| Prose generation, one request | **35.1 output tok/s median** |
| Code generation, one request | **47.4 output tok/s median** |
| Four concurrent requests | **76.5 output tok/s combined median** |
| Cold prefill, 8K / 32K | **938 / 932 input tok/s** |
| Time to first token, 8K / 32K | **8.73 / 35.17 s** |
| Configured context / admitted requests | **360,000 tokens / 4** |
| KV allocation | **24 GiB per rank** |

These October 2 measurements precede the hardening closeout; no new speedup
is claimed from that work. Prefill is prompt tokens divided by time to first
token. Concurrent requests share the cache; four full-length contexts are
not promised. [Samples, quality checks and limitations](results/README.md).

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
and E3 prefill kernels. These are proposed tests, not measured gains.
TensorFold remains outside the production completion requirements.

## Credits

Built on **Z.ai**, **davidsyoung**, **Turboderp / ExLlamaV3**, **vLLM**,
**local-inference-lab / b12x**, **Ash Hart / TensorFold**, **MiaAI-Lab**,
**Adapt AI Systems**, **Kindling**, **Matt Mastracci**, and the wider Spark community.

See [CREDITS.md](CREDITS.md), [third-party notices](THIRD_PARTY_NOTICES.md),
[source provenance](provenance/README.md), and [the roadmap](docs/ROADMAP.md).
