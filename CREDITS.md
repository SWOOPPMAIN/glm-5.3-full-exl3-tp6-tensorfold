# Credits and lineage

This is an integration and optimization project. The model, quantization,
inference foundations, and much of the kernel work come from the projects below.
Links identify the source of a contribution; they do not imply endorsement.

## Model and serving foundations

| Contributor / project | Contribution |
| --- | --- |
| [Z.ai / GLM](https://huggingface.co/zai-org/GLM-5.3) | Original full GLM-5.3 model and native MTP architecture. Model terms remain separate from repository code. |
| [davidsyoung](https://huggingface.co/davidsyoung/GLM-5.3-EXL3-TR3-3.25bpw) | The original mixed K3/K4, 3.25 bpw TR3 checkpoint used here. We redistribute its TP4 fragments across six ranks without requantizing them. |
| [Turboderp / ExLlamaV3](https://github.com/turboderp-org/exllamav3) | EXL3 format, trellis/codebook conventions, quantization machinery, and kernel headers. |
| [brandonmmusic-max/exllamav3](https://github.com/brandonmmusic-max/exllamav3) | ExLlamaV3 fork pinned by the earlier mixed-K SM121 image build. |
| [vLLM](https://github.com/vllm-project/vllm) and [local-inference-lab/vllm](https://github.com/local-inference-lab/vllm) | Serving engine, model implementation, speculative decoding, and GB10/EXL3 integration. |
| [local-inference-lab/b12x](https://github.com/local-inference-lab/b12x) and [blackwell-llm-docker](https://github.com/local-inference-lab/blackwell-llm-docker) | Blackwell kernels, mixed-K runtime releases, build integration, and RoCE collectives. |
| [original-el8](https://github.com/original-el8), [Luke Alonso](https://github.com/lukealonso), and b12x contributors | RoCEnante collective implementation and vLLM adapter; exact imported revisions are in the retained [provenance](runtime/vllm/rocenante/PROVENANCE.md). |
| [NVIDIA](https://github.com/NVIDIA), [PyTorch](https://github.com/pytorch/pytorch), [Triton](https://github.com/triton-lang/triton), and [Hugging Face](https://github.com/huggingface) | Hardware/runtime libraries, tensor and kernel tooling, checkpoint hosting and formats. |

## Recipes, kernels, and TensorFold

| Contributor / project | How the work is used |
| --- | --- |
| [Adapt AI Systems — full GLM TP6 recipe](https://github.com/adapt-ai-systems/spark-recipes/tree/main/glm/53-full-exl3-tp6) | Initial reference for full-model EXL3 TP6 with MTP and a large context. Our measurements and fragment ownership are independently recorded. |
| [d3y4n — full GLM on four Sparks](https://github.com/d3y4n/glm-5.3-4x-dgx-spark) | Earlier full-model mixed-K SM121 image/recipe lineage (`76fe48b`) from which our serving image evolved. Its original TP4/DFlash2 profile is not the current TP6/MTP profile. |
| [Kindling — full EXL3 TP6](https://github.com/kindlingai/glm-5.3-full-exl3-tp6) | E3 grouped-prefill source lineage, TP6 implementation reference, and performance experiments. Imported E3 base: `0ecf21ebcccd924f7f7c47eda0333282d8e177f5`. |
| [Matt Mastracci](https://github.com/mmastrac) / [Kindling Flash](https://github.com/kindlingai/glm-5.3-flash-gx10) | Adaptive drafting and prefill research. Swoopp's request/phase MTP policy independently adapts the idea discussed in Kindling's D13 experiment. |
| [MiaAI-Lab — Spark EXL3](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks) | ARM/SM121 image lineage and the original E3 grouped expert kernel, subsequently extended by Kindling and adapted here. E3 remains AGPL-3.0. |
| [Ash Hart / TensorFold](https://github.com/ashhart/TensorFold) and its contributors | Base framework, CUDA infrastructure, EXL3 primitives, and serving architecture underlying our experimental full-model family. |
| [MiaAI-Lab — TensorFold Spark recipe](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold) | Pinned patched TensorFold baseline: 52 patches from `ed026ef92d1650120dada1294a112acb6c8f2f48` on TensorFold v0.5.0. Later upstream work was reviewed separately; it is not all merged here. |
| [Jay Leaton](https://github.com/jayleaton/glm53-tensorfold-spark) | Tool/reasoning handling, L2 prefetch, and EXL3 decode-load work credited by the imported Mia patches. Original detailed notices are retained below. |
| [drowzeys](https://huggingface.co/drowzeys) | Full-GLM Spark recipes, GB10 runtime investigations, and lower-bit quantization experiments informed our review. We keep davidsyoung's original 3.25 bpw weights. |

The imported TensorFold tree also contains earlier contributions from MLX,
mlx-lm, mlx-vlm, their individual contributors, Qwen, Z Lab, DeepSeek, and
others. Their detailed attribution is preserved in the upstream
[third-party notices](experimental/tensorfold/THIRD_PARTY_NOTICES.md),
[Mia credits](experimental/tensorfold/LICENSES/MiaAI-Lab-TensorFold-CREDITS.txt),
and [Mia notice](experimental/tensorfold/LICENSES/MiaAI-Lab-TensorFold-NOTICE.txt).
These files describe the imported tree, including functionality outside the full TP6 path.

## Other research reviewed

We also reviewed work from [knapcio](https://github.com/knapcio),
[tonyd2wild](https://github.com/tonyd2wild),
[jnardiello](https://github.com/jnardiello), and
[ciprianveg/gb10-vllm](https://github.com/ciprianveg/gb10-vllm).
These are research references, not a claim that each project's code is installed.

## Swoopp's changes

- Lossless six-rank fragment placement, verification, and the full-model TP6 integration.
- Workload/phase-aware MTP scheduling, projection work, shared-expert fixes,
  deterministic arithmetic repairs, and native/E3 prefill dispatch in vLLM.
- The `glm_moe_dsa` TensorFold family: original checkpoint reader, full target
  and MTP assembly, bounded memory, sparse attention/indexing, and TP6 reductions.
- Six-rank numerical and performance qualification, host protection, and application integration.
- This repository's recipes, measurement summaries, source inventory, and documentation.

Source imports and portability changes are listed in [provenance/imports.json](provenance/imports.json).
Please report missing or incorrect attribution through an issue.
