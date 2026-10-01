# Full GLM-5.3 EXL3 · TP6 · TensorFold

Full GLM-5.3 on **six NVIDIA DGX Sparks**, keeping the original **3.25 bpw
mixed K3/K4 EXL3 weights**. This is Swoopp's private working repository for the
serving recipe, measured optimizations, and ongoing TensorFold port.

**Status — October 1, 2026:** vLLM serves the model today. TensorFold can run
the full target/MTP model with decode graphs on all six ranks. Short chat-template
HTTP requests, four concurrent clients and streamed replies pass exact-output
checks. Full API qualification and deployment remain in development.

## Performance

Measured on our six-Spark vLLM deployment, current P24 profile:

| Measurement | Result |
| --- | ---: |
| Code generation, one request | **46.8 tok/s median**; 47.3 best of three |
| Prose generation, one request | **33.7 tok/s median**; 34.2 best of three |
| Four concurrent requests | **72.4 output tok/s combined** |
| Uncached prefill, 8K / 32K / 128K | **847 / 840 / 833 input tok/s** |
| Time to first token, 8K / 32K / 128K | **9.67 / 39.01 / 157.34 s** |
| Configured context / shared KV cache | **360,000 / 470,847 tokens** |

Prefill here means prompt tokens divided by time to first token, including
the first generation step and delivery. These are workload measurements,
not a hardware ceiling. See [results and methodology](results/README.md).

The latest TensorFold attention experiment improved a synthetic 3,072-token
full-model pass from **9.55 to 8.80 seconds** (+8.5% throughput),
with **1,098 GPU checks passing** (546 exact comparisons and
552 FP64 reference checks). The resident test cache holds
**804,000 tokens**. This is a separate test configuration; the improvement is
not deployed to vLLM. [Detailed result](results/tfp15-tensorfold.json).

The request core also passed **102 checks across six ranks**. Four interleaved
requests and a retained-prefix follow-up matched serial generation exactly
on every rank. [Scope and limits](results/tfp16-tensorfold.json). The distributed controller
and scheduler also pass serial parity with concurrent clients, prefix reuse and
cancellation. [TFP17](results/tfp17-tensorfold.json). Decode graphs and local HTTP
then passed **329 GPU checks**; short code generation improved **22.8 → 24.5 tok/s**
versus TensorFold eager execution, while prose was unchanged. These 128-token
diagnostic runs are separate from the P24 measurements above.
[TFP18 results and limits](results/tfp18-tensorfold.json).

## Recipes

1. [Prepare and verify the original weights](recipes/weights/README.md)
2. [Serve with the qualified vLLM TP6 image](recipes/vllm-tp6/README.md)
3. [Work on the TensorFold TP6 port](recipes/tensorfold-tp6/README.md)
4. [Benchmark a candidate](benchmarks/README.md)

The serving recipe currently requires our retained P24 image. A portable
image release and a complete clean-machine build are not included yet.
The repository includes source, configuration, and measured results; model
weights and compiled artifacts are downloaded or built separately.

## What's here

| Directory | Contents |
| --- | --- |
| `weights/` | Pinned checkpoint manifest, lossless resharing and verification |
| `runtime/vllm/` | Serving overlays, node entrypoint, memory guard, E3 sources |
| `experimental/tensorfold/` | Source snapshot through decode graph and short HTTP qualification |
| `experiments/` | Completed attention experiment and next development steps |
| `results/` | Serving measurements and separate TensorFold experiment results |
| `provenance/` | Source revisions, import hashes, and local change records |

## Credits

Built on **Z.ai**, **davidsyoung**, **Turboderp / ExLlamaV3**, **vLLM**,
**local-inference-lab / b12x**, **Ash Hart / TensorFold**, and **MiaAI-Lab**.
**Adapt AI Systems**, **Kindling**, **Matt Mastracci**, and the wider Spark
community supplied recipes, kernels, and optimization ideas that informed this work.

See [CREDITS.md](CREDITS.md) for specific contributions and source links,
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for licenses, and
[the roadmap](docs/ROADMAP.md) for what remains.
