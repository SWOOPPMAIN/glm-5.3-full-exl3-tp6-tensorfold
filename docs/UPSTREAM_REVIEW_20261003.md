# Upstream review — October 3, 2026

Read-only review of 14 repository heads. These are candidates and references,
not changes installed on the six-Spark deployment. Preserve full GLM-5.3,
original 3.25 bpw EXL3, current dense precision, TP6 and 360K context.
TensorFold development remains excluded.

## Candidates worth investigating after the ordered experiments

| Source | Finding | Relevance and qualification still needed |
| --- | --- | --- |
| [knapcio full GLM](https://github.com/knapcio/GLM-5.3-4x-DGX-Spark-TP4/tree/be5c80a6bc85b97bb4b0908cf128ff3add74d05a) | Skip indexer query/logits when context is at most 2,048 tokens and every token is selected | Potential short-request decode benefit. Source explicitly supports TP4; our TP6 graph and indexer paths need a port and numerical checks. No long-context prefill gain established. |
| Same recipe | Stop MTP proposals using the current draft's cumulative confidence | A different hypothesis from our historical acceptance/cost controller and completed window/depth sweep. Preserve target verification; measure proposal overhead and identical workloads. |
| Same recipe | Descriptor-scoped padding and discard of consumed attention temporaries from L2 | Audit our actual graph and attention traffic first. Their padding change removes overhead from their own remap, which our path may not contain. Their split-attention kernel is not our installed backend. |
| [Christopher Owen's memory saver](https://github.com/christopherowen/dgx-spark-memory-saver/tree/6f97319e7c00e5422b0afe4cabbd348211bfcb34) | Pack GPU leaf page tables into shared CPU pages | Upstream tested our exact kernel/driver versions. Its profiled backing storage fell by about 3.03 GiB versus stock 64 KiB UVM; throughput was comparable. All six local nodes lack the packing control parameter. Investigate a UVM-only build for our existing OS, followed by drained activation, allocation/readback and serving checks. No local gain measured. |
| [Mia's vLLM scheduler](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/blob/6278ecb01034cea9ef6de0f851d09fccafe3e835/CHANGELOG.md) | Let runnable prefill progress when a waiting request cannot obtain KV | Useful robustness reference. TP2 progress tests passed, while timing and temperature-zero identity comparisons were inconclusive. Check whether our scheduler has the same failure; preserve the qualified 3072 budget. |

knapcio's recipe uses Int4-Int8Mix, four nodes and about 66K admitted context.
Its benchmark rates do not establish an improvement over our EXL3 TP6 serving.

## Other reviewed changes

- **Kindling / Adapt full TP6:** both remain at `3b9c548`, already reviewed.
  D13 launcher integration does not add a new scheduler implementation to our
  existing phase-aware controller. Adapt's recipe index last changed September 29.
- **Kindling OS:** 0.9.5 concerns Wi-Fi/setup/console behavior. The separate
  [Talos dgx1022.5 release](https://github.com/kindlingai/talos-dgx-kernel/commit/6bb35acecef0cebc454ecd776f3c5b445a267fbf)
  adopts Christopher Owen's UVM packing patch; a Talos migration is unnecessary
  to investigate the allocator idea.
- **Mia TensorFold recipe v1.5:** eight-stream serving, scaled memory reserve and
  serial cancellation fixes. Reported aggregate gains apply to Flash/DFlash2 on
  two or three Sparks; single-stream speed is unchanged. Recorded only.
- **Jnardiello:** [E37/E38 reports](https://github.com/jnardiello/GLM-5.3-Flash-FP8-4-DGX-Spark-Switchless/tree/53359a299fe13bd079094a48be795a6400d0a2df/docs/benchmarks/experiments)
  reject slower butterfly collectives and withhold an attention-preparation
  prototype. Its confidence-based verification is a useful design reference;
  output-head quantization is outside our fixed-precision plan.
- **Tony / Kindling Flash:** recent launcher, KV-layout and attribution updates
  remain Flash-specific. Distinguish main-branch commit dates from repository
  push timestamps and avoid counting mirrored changes twice.
- **Ciprianveg:** remote DSpark reports 39.5 tok/s local versus 38.5 over RDMA
  on its measured TP4 setup. No compatible full-GLM remote drafter or gain for
  our exclusively reserved six GPUs is established.

[Pinned heads, source hashes and decisions](../results/upstream-review-20261003.json).
Credit remains with each linked author and their cited upstream contributors;
this review imports no implementation from these new candidates.
