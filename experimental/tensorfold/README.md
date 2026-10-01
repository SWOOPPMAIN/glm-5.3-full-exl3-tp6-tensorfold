# TensorFold source snapshot for full GLM TP6

This directory preserves the complete source dependencies of the experimental
full-model port. Start with [the TP6 recipe](../../recipes/tensorfold-tp6/README.md).

- Base: [Ash Hart / TensorFold](https://github.com/ashhart/TensorFold), v0.5.0.
- Patched baseline: [MiaAI-Lab](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold).
- Swoopp port: `src/tensorfold/families/glm_moe_dsa/`.
- Snapshot: `e69eb5f82c0467e2b07fc41d0ee8429ae7342dbe`.

Source under `src/` matches the qualified revision byte for byte. GPU probe
scripts replace one private head address with `MASTER_ADDR`; those portability
edits have not been rerun on six GPUs. Original framework and Mia license
notices remain in this directory. Only the full-TP6 CPU tests and tools are
included; unrelated upstream test suites and documentation are not exported.

Full GLM TP6 family registration is disabled pending engine integration.
