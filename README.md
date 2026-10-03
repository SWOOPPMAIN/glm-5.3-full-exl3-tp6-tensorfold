# Full GLM-5.3 EXL3 · TP6 · TensorFold research

Full GLM-5.3 on **six NVIDIA DGX Sparks**, preserving the original
**3.25 bpw mixed K3/K4 EXL3 weights**. Serving uses **vLLM P27**.
TensorFold remains deferred research and is not serving production.

**October 3, 2026:** all six ordered optimization experiments are complete.
The selected forward image passed numerical, native API, router and actual
Code/Chat checks. Original adaptive MTP remains selected; standalone copy
drafting passed correctness but lost general throughput. No 12-hour soak.

## Latest serving measurements

| Measurement | Selected MTP profile |
| --- | ---: |
| Prose / code generation, one request | **35.5 / 48.3 output tok/s** |
| Cold prefill, 8K / 32K | **979 / 961 input tok/s** |
| Time to first token, 8K / 32K | **8.37 / 34.09 s** |
| Configured context / admitted requests | **360,000 tokens / 4** |
| KV allocation | **24 GiB per rank** |

Three samples per workload, with MTP controls bracketing GPU-copy testing.
Single-request generation uses 512 output tokens; prefill is prompt tokens / TTFT.
Four-request results depend on the workload: the earlier E3 short mixed workload
measured **78.6 combined output tok/s**; the latest copy/edit/prose matrix and every
sample are in [the completed comparison](recipes/vllm-tp6/COPY_DRAFTING.md).
These visits are not a new MTP-kernel speedup or a many-boot confidence interval.

## Optimization outcomes

- **Prompt reuse:** all 72 checks passed. Repeated 8K/32K/128K code-tool histories
  reached first tokens in **0.60/0.64/0.87 s**. Existing caching retained.
- **Prefill budget:** smaller/adaptive policies failed fidelity; retain **3072**.
- **MTP:** 270 measurements plus 72 cost cells; retain original adaptive policy.
- **Communication:** dual-HCA RoCEnante selected; retain **2 MiB** crossover.
- **E3:** row32 improves matched cold prefill **3.9%/3.3%** at 8K/32K.
  Native through 32 rows / E3 above 32 cuts 65–512-token latency **4.4–7.7%**.
- **Copy drafting:** correctness passed; broad speed regressions reject promotion.

[Results and limitations](results/README.md) ·
[Experiment outcomes](docs/PERFORMANCE_EXPERIMENTS.md) ·
[Upstream ideas reviewed](docs/UPSTREAM_REVIEW_20261003.md)

## Recipes

1. [Prepare and verify original weights](recipes/weights/README.md)
2. [Serve with the qualified vLLM TP6 image](recipes/vllm-tp6/README.md)
3. [Operate the six-node deployment](recipes/vllm-tp6/OPERATIONS.md)
4. [Benchmark a candidate](benchmarks/README.md)
5. [Explore historical TensorFold research](recipes/tensorfold-tp6/README.md)

The serving recipe requires the retained P27-derived image. A portable image
release and verified clean-machine build remain unfinished. Weights and compiled
artifacts are not included. All six controlled host reboots passed in the earlier
hardening pass; the latest comparison did not repeat that OS test.

TensorFold's later local HTTP results remain about 19–21 output tok/s and 368–388
cold-prefill tok/s, with strict fidelity failing. The bundled source is historical
TFP21; [later results through TFP57](results/TENSORFOLD_STATUS.md) are separate.

## Credits

Built on **Z.ai**, **davidsyoung**, **Turboderp / ExLlamaV3**, **vLLM**,
**local-inference-lab / b12x**, **Ash Hart / TensorFold**, **MiaAI-Lab**,
**Adapt AI Systems**, **Kindling**, **Matt Mastracci**, and the wider Spark community.

[Credits](CREDITS.md) · [Third-party notices](THIRD_PARTY_NOTICES.md) ·
[Source provenance](provenance/README.md) · [Roadmap](docs/ROADMAP.md)
