# Roadmap

## Current serving: completed

- Full GLM-5.3 EXL3 TP6 on vLLM P27, original mixed K3/K4 3.25 bpw weights.
- Verified short and long-tail agreement with the frozen serving reference.
- Native API, router and Code/Chat acceptance.
- All six controlled reboots; persistent OS default, dual-fabric network and memory settings.
- Brief health observation and exclusive GPU ownership checks. No long soak required or run.

[Current measurements](../results/README.md) · [operations](../recipes/vllm-tp6/OPERATIONS.md)

## Optional performance work

The [ranked experiment plan](PERFORMANCE_EXPERIMENTS.md) starts with prompt reuse,
mixed prefill/decode scheduling and retuning the current MTP policy. Six-rank
communication and E3 prefill follow. These are proposed tests, not promised gains.
Preserve current weights/precision and qualify each change before promotion.

## TensorFold: deferred research

The later local full-model port has HTTP and graph execution, but strict fidelity
still fails. It is not a prerequisite for current serving. [Latest status](../results/TENSORFOLD_STATUS.md).
The bundled executable source remains the historical TFP21 export; a vetted export
of the later port and its diagnostic tooling is separate work.

## Packaging still needed

- Verified clean-machine P27 image build and distributable registry artifact.
- Portable fleet launcher and site configuration independent of our private controller.
- Reproducible OS site overlay, with networking selected for the operator's topology.

These release gaps are separate from the completed local deployment hardening.
