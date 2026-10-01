# TensorFold full GLM-5.3 TP6 port

**Experimental.** This is the full `glm_moe_dsa` model, with original mixed
K3/K4 3.25 bpw expert fragments. Full-model target/MTP forward execution works;
`tensorfold serve` does **not** register or serve this family yet.

## Source

[`experimental/tensorfold/`](../../experimental/tensorfold/README.md) is a
source snapshot of our qualified port revision
`b049970f9a8e65c802a8d8d00ccd0118f62456a7`.
The complete framework source is retained to preserve imports and upstream
notices; the new family is under `src/tensorfold/families/glm_moe_dsa/`.

Its base is TensorFold v0.5.0 commit
`9cd52ab4daba68ddd09be89be8f23ad43175e821`, with 52 patches from the pinned
Mia recipe. [The patch manifest](../../experimental/tensorfold/mia-recipe-provenance.json)
records their hashes. This is not a claim to include current upstream main.

## CPU checks

In an existing Python environment with PyTorch, Triton and NumPy available:

```bash
cd experimental/tensorfold
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_glm53_full_tp6_*.py'
```

These exercise shard geometry, exact arithmetic references, workspace sizing,
input contracts, and request state transitions against an independent serial
oracle. They do not substitute for six-GPU qualification.

## GPU qualification contract

The qualified runs loaded all 79 target/MTP layers on each assigned rank and
touched the entire 804,000-token cache. Earlier experiments cover hidden
states, logits and changed-input graphs; TFP16 covers eager request execution
against serial target generation.

- `tools/glm53_tp6_profile_check.py`: draft-depth sweep and bounded kernel traces.
- `tools/glm53_tp6_graph_check.py`: decode graph and real local HTTP gate.
- `tools/glm53_tp6_control_check.py`: distributed controller/scheduler gate.
- `tools/glm53_tp6_request_check.py`: preceding request-core parity gate.
- `tools/glm53_tp6_attention_tuning_check.py`: attention comparison and graphs.
- `tools/glm53_tp6_bulk_reduction_check.py`: the preceding reduction experiment.
- `tools/glm53_tp6_reduction_check.py`: independent rank-order reduction oracle.
- `tools/glm53_tp6_compile_experts.py`: bounded CPU-only extension build.
- `src/tensorfold/families/glm_moe_dsa/experts-sm121.json`: qualified binary
  hash and seven source hashes; inference refuses an unrecognized binary.

The binary is not included. Rebuilding requires the matched ARM64/CUDA/PyTorch
toolchain; a different binary must be requalified before updating its pin.
The loader expects the admitted artifact at
`/opt/amos-tensorfold/compiled/tensorfold_exl3_experts_v1.so`.

GPU probes require an **empty, drained fleet**, one coordinated six-rank NCCL
group, rank-specific original shards, source manifests, and exact-container
guard admission files. Supply the head fabric IP through `MASTER_ADDR`.
Admission requires at least 110 GiB available per stopped host, a 108 GiB
container limit and the existing 8 GiB host guards. The probe also bounds
PyTorch allocations to 104 GiB. Do not run these alongside the serving model.

Our site controller is not exported; prepare its portable replacement before
attempting this experiment on a new fleet. The source and past results are
available for review without launching it.

## Latest result and next work

The five attention modes passed 1,098 GPU checks, including original
full-model target/MTP outputs and changed-input graphs. `skip128` led the
synthetic target pass: **9.55 → 8.80 seconds** (+8.5% throughput).
See [the evidence](../../results/tfp15-tensorfold.json). Defaults remain unchanged;
use explicit candidate settings recorded there.

The eager request core then passed 102 checks across six ranks, including
30 exact sequence comparisons with an independent serial
loop. See [TFP16](../../results/tfp16-tensorfold.json) and the
[request execution contract](REQUEST_ENGINE.md).

The distributed controller, leader-only sampler and concurrent-client scheduler
passed 28 checks across six ranks, including 6 client output comparisons
against independent serial decoding. See [TFP17](../../results/tfp17-tensorfold.json).

Decode graphs and the actual App/HTTP path now pass 329 GPU checks, including
264 exact tensor checks and 27 sequence comparisons. The code fixture gained
7.5% over eager execution; prose was unchanged. See the
[graph recipe and measurement limits](DECODE_GRAPHS.md).

Packed GPU batches, broader API behavior and authentic long-context quality
remain unfinished. The scheduler serializes model passes; background priority
applies while waiting, without active preemption.

TFP19 then passed 630 checks while sweeping draft depths 0–8 and profiling
warm decode. Dense BF16 projections lead compute time; draft depth alone offers
only a small improvement on the prose fixture. [Method and next target](DECODE_PROFILING.md).
