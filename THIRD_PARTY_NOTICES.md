# Third-party notices

The root license covers new Swoopp packaging and documentation. It does not
replace licenses on imported or derived code, model weights, or runtime dependencies.

| Path / dependency | License and provenance |
| --- | --- |
| `experimental/tensorfold/` | TensorFold v0.5.0 MIT code plus Mia Apache-2.0 patches and the full TP6 port. Preserve its `LICENSE`, `LICENSES/`, `THIRD_PARTY_NOTICES.md`, and patch manifest. |
| `runtime/vllm/rocenante/` | b12x and vLLM Apache-2.0 code. Original license, source hashes, and change notes are retained in that directory. |
| `runtime/vllm/e3/amos_e3/`, excluding `vendor/` | Mia E3 → Kindling mixed-K/fragment extensions → Swoopp native activation and ordered reduction changes. **AGPL-3.0**, with the full license in `LICENSE.mia` and `LICENSES/AGPL-3.0.txt`. |
| `runtime/vllm/e3/amos_e3/vendor/` | ExLlamaV3 headers, MIT; retain `LICENSE.exllamav3`. |
| vLLM-derived overlays and patch context in `runtime/vllm/` | Retain existing source headers; upstream vLLM/b12x portions remain Apache-2.0. License text: `LICENSES/Apache-2.0.txt`. |
| Kindling-derived portions | Preserve `LICENSES/Kindling-MIT.txt`; E3-derived portions retain AGPL-3.0. |
| Model checkpoint and copied model metadata | Governed by the checkpoint's GLM-5.3 terms; consult the [pinned checkpoint](https://huggingface.co/davidsyoung/GLM-5.3-EXL3-TR3-3.25bpw/tree/6d6bd738c0c1635513e0bd0fdf0302049bd820a9). No weight tensors are included here. |

The base image and libraries it contains have their own notices and terms,
including NVIDIA software, PyTorch, Triton, NCCL, CUTLASS, FlashInfer, and
Hugging Face libraries. This repository does not redistribute a container image.

Changes to imported code are identified in [the source inventory](provenance/imports.json)
and [credits](CREDITS.md). TensorFold GPU probes use `MASTER_ADDR` instead of a
private fabric address. The benchmark client defaults to loopback. Kernel
source in the qualified TensorFold snapshot is unchanged from its recorded revision.
