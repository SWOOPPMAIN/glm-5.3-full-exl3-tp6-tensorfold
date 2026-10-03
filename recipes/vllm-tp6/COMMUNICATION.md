# TP6 communication: dual-port RoCEnante

**Selected October 3:** use both validated fabric HCAs for the existing b12x
RoCEnante collectives. The pinned image, weights, dense precision, MTP policy,
3072-token prefill budget and 360K configured context remain unchanged.

[Comparison and quality](../../results/communication-dual-roce.json) ·
[Individual samples](../../results/communication-samples.json) ·
[Six-rank profile](../../results/communication-profile.json)

## Measured result

Two alternating visits per configuration, three samples per workload in total:
single-port ×2, dual-port ×2, fresh single-port ×1, fresh dual-port ×1.
The cached matrix contains 72 measurements: code/prose/tools, short/8K/32K,
nine single-request cells and three four-request mixed cells per configuration.
Separate 512-token generation and cold-prefill tests also have three samples
per configuration. Profiling was inactive during speed measurements.

| Workload, median | Single port | Dual port | Change |
| --- | ---: | ---: | ---: |
| Code, 512 output tokens | 46.71 tok/s | 47.79 tok/s | +2.3% |
| Prose, 512 output tokens | 35.91 tok/s | 35.18 tok/s | −2.0% |
| Cached mixed C4, short | 75.13 tok/s | 78.28 tok/s | +4.2% |
| Cached mixed C4, 8K | 68.61 tok/s | 70.65 tok/s | +3.0% |
| Cached mixed C4, 32K | 66.65 tok/s | 69.03 tok/s | +3.6% |
| Cold prefill, 8K | 941.65 tok/s | 939.28 tok/s | −0.3% |
| Cold prefill, 32K | 931.88 tok/s | 931.12 tok/s | −0.1% |

The equal-cell geometric mean improves **2.37%** across the cached matrix.
Cached single-request 8K code is 2.3% lower and 32K prose is 4.0% lower.
Most median p95 stream gaps improve. This is a modest overall/concurrency
tradeoff, not a universal speedup. Three samples do not establish statistical
significance. Adaptive draft depth and host state also affect these results.
The original first visit followed profiling; the additional fresh-container
visit helps assess that difference but does not eliminate all order effects.

## Why both ports now help decode

Eighteen bounded traces covered all six ranks during C1 decode, C4 decode and
cold 32K prefill. Decode used custom RoCE reductions/gathers and no NCCL kernels;
only one HCA carried that traffic. NCCL prefill already used both ports almost
equally. Enabling a second NCCL rail again would not address that decode path.

The existing native b12x proxy supports two HCAs and assigns a peer connection
using `(local_rank + peer_rank) % hca_count`. At TP6 this gives two peers on the
first HCA and three on the second. Native API C1/C4 measurements after the change
show about **60%** of transmitted bytes on the second HCA on every rank, as
expected. These port counters establish utilization, not isolated bandwidth gains.

Kernel durations include synchronization and rank skew; they are not all removable
wire time. The traces did not record exact operand-size histograms. Actual size
profiling and crossover trials remain unfinished in the communication experiment.

## Configuration and reproduction

Apply [tuning.json](tuning.json) using the existing
[launch contract](README.md#launch-contract). The transport additions are:

```text
AMOS_TP6_ROCE_RAILS=2
VLLM_ROCE_ALLREDUCE_MAX_SIZE=2097152
B12X_ROCE_HCA=<first-validated-HCA>,<second-validated-HCA>
B12X_ROCE_GID_INDEX=<validated-local-index>
```

`AMOS_TP6_ROCE_RAILS` records controller intent; it does not discover interfaces.
Your controller must pass the comma-separated HCA pair and explicit GID index.
The current b12x API accepts one GID index per node: **both selected HCAs on that
node must expose its intended IPv4 RoCEv2 fabric address at that same index**.
Indices may differ between nodes. Validate link state, MTU 9000, fabric addresses
and both GID mappings before stopping a running fleet. Refuse an incompatible
mapping. Do not change the management route to make the check pass.

NCCL keeps its existing dual-rail address-family/subnet policy. The custom
all-reduce cutoff remains 2 MiB; the all-gather shard cutoff remains 16 MiB.
No collective kernel or reduction order was changed in this trial.

1. Own a drained six-node window with fresh exact-container 8 GiB / 2-second
   memory guards and pressure protections. Retain the original adaptive MTP
   and 3072 prefill control files.
2. Record the image, actual rank environment and fabric mappings. Qualify the
   candidate against the original short and 8K/32K/128K references before speed
   selection. The final identical-configuration restart reused those numerical
   results and ran fresh functional checks; it did not rerun the numerical suite.
3. Use the [MTP matrix client](MTP_TUNING.md) with `--scope matrix`. For each
   transport, use two repetitions on its first visit and one on its second.
   Keep the public frozen fixtures, output length, template and policy identical.
   Warm each exact prompt separately. The client requires external ownership
   and guard enforcement; it does not manage the cluster.
4. Use [the performance client](../../benchmarks/README.md) for separate 512-token
   generation and uniquely salted cold 8K/32K prompts, with the same visit order.
5. Pool individual samples and take each cell's median. Divide candidate by
   control medians, then take the geometric mean of the twelve matrix ratios.
   Do not average visit medians or combine input/output token rates.
6. Select explicitly, update the supervisor under the owned hold, and run native,
   Pi-router and Code/Chat checks. All passed here without replacing the final
   six candidate containers. Final sampled minimum available RAM was 13.53 GiB.

Short-reference top-1 agreement was 100%; mean coarsened KL was approximately
5.1e-10. All three long-context tail gates passed. Coarsened KL is a lower bound,
and the 128K common reference mass is limited; this is not comprehensive 360K
task-quality validation. No 12-hour soak was performed.

## Attribution

The collective implementation is **local-inference-lab / b12x RoCEnante**;
its dual-HCA support was already present. See the pinned
[source lock](../../runtime/vllm/rocenante/source-lock.json),
[b12x PR 295](https://github.com/local-inference-lab/b12x/pull/295), and
[vLLM integration PR 597](https://github.com/local-inference-lab/vllm/pull/597).
[Kindling's full TP6 recipe](https://github.com/kindlingai/glm-5.3-full-exl3-tp6)
provided the serving integration reference. Our contribution here is validated
six-node mapping, guarded orchestration, workload measurements and selection.
