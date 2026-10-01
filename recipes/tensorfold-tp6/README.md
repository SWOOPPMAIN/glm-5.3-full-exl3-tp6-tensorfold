# TensorFold full GLM-5.3 TP6 port

**Experimental.** This is the full `glm_moe_dsa` model, with original mixed
K3/K4 3.25 bpw expert fragments. Full-model target/MTP forward execution works;
`tensorfold serve` does **not** register or serve this family yet.

## Source

[`experimental/tensorfold/`](../../experimental/tensorfold/README.md) is a
source snapshot of our qualified port revision
`e69eb5f82c0467e2b07fc41d0ee8429ae7342dbe`.
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
and input contracts. They do not substitute for six-GPU qualification.

## GPU qualification contract

The qualified run loaded all 79 target/MTP layers on each assigned rank,
touched the entire 804,000-token cache, and compared target hidden states,
logits, MTP, uneven input lengths, and changed-input CUDA graphs.

- `tools/glm53_tp6_bulk_reduction_check.py`: the full six-rank experiment.
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

## Latest result and next experiment

At 3,072 synthetic input tokens, 1,024-row expert chunks and a fully resident
804K cache, row-sharded reductions improved the full pass from 11.07 to 9.62 s.
All 618 GPU comparisons were exact. See [the evidence](../../results/tfp14-tensorfold.json).

[Attention tuning](../../experiments/README.md) is preserved separately as an
unapplied patch. Request scheduling, recursive MTP acceptance, authentic
long-context quality, and application integration remain on the roadmap.
