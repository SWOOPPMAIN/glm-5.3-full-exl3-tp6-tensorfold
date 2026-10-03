# MTP tuning: first comparison, October 3

**In progress. The original adaptive MTP policy remains selected.**
The new image adds bounded experiment controls without changing the selected
inference policy. This is not a promoted throughput improvement.

## Measurements and limits

[Results](../../results/mtp-first-pass.json) cover 60 workload cells: original
adaptive MTP and fixed depths 1–4, each at short, 8K and 32K context, with single
code/prose/tool requests and a four-request code/prose/tool/code mix. Each cell
has **one visit**, with 256 output tokens per request. Warmups are separate.

Output tok/s below includes cached prefill and delivery; C4 is aggregate:

| Policy | Short code C1 | Short prose C1 | Short tools C1 | Short C4 | 8K C4 | 32K C4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original adaptive | 42.09 | 32.43 | 44.15 | 73.77 | 67.33 | 65.56 |
| Fixed 1 | 35.30 | 31.97 | 34.93 | 75.24 | 66.66 | 64.85 |
| Fixed 2 | 41.08 | 32.38 | 41.66 | 76.34 | 72.23 | 70.09 |
| Fixed 3 | 44.09 | 32.33 | 45.29 | 76.12 | 71.04 | 64.94 |
| Fixed 4 | 43.84 | 28.46 | 44.55 | 71.77 | 65.05 | 61.34 |

Equal-cell geometric mean throughput ratios versus original were 0.910, 1.000,
1.025 and 0.970 for depths 1–4. These are exploratory single-visit comparisons.
Depth 3 merits repeated single-request testing; depth 2 merits repeated long C4
testing. Neither supports a global fixed-depth selection yet.

These workloads differ from the historical 512-token serving benchmark. Do not
compare their rates as a deployment regression or new peak. Tool argument deltas
count toward streaming timing, and the output cap can truncate a call. Tool
argument text/hashes were not retained in the first pass; this is not a complete
tool-output equivalence test. Generated continuations can differ across policies.

Receipts retain native acceptance counters and accepted tokens per wall second.
Summed request-decode time divided by draft rounds is an estimated step cost;
concurrent runs include declining residency and do not isolate constant-batch
GPU time. Judge candidate policies on end-to-end measurements as well as costs.

## Runtime and qualification

The wrapper changes only bounded depth/cost/probe settings in the existing
request/phase controller. It preserves native acceptance accounting, target
verification, token history, prefill, weights and precision. It latches a policy
between live request groups, never halfway through an existing group.

Qualification completed 38 CPU checks, exact comparisons at 4,096 short-reference
positions and 512-position tails at each of 8K/32K/128K, and 40 functional checks
across the five policies. Native, Pi-router and Code/Chat acceptance passed after
selecting the original policy. Exact-container guards remained clear. These
checks do not qualify an unmeasured recalibrated policy for production.

## Required persistent controls

Rank 0 must have `/root/.cache/amos-tp6-mtp-tuning.json` before startup:

```json
{"revision":"mtp1-selected-original","mode":"original","depth":0,"window":16,"cost_ms":null}
```

Also retain the baseline 3072 [prefill control](PREFILL_BUDGETS.md). Both files
must survive cache cleanup and container replacement. Missing or invalid controls
fail explicitly; there is no fallback. Verify the corresponding `.applied.json`
values and content hash after applying a policy and warming a request.

Fixed experiments use `mode: "fixed"`, integer `depth` 1–4, `window: 16` and
`cost_ms: null`, with a unique revision. Cost experiments use `mode: "costs"`,
`depth: 0`, a window of 4/8/16/32 and a four-by-four cost table keyed by concurrency
`"1"` through `"4"`, with depth 1–4 costs in milliseconds. Costs must be finite
and strictly between 1 and 10000. The window controls full-depth probe frequency;
the native controller already chooses a depth each batch. Cost-mode performance
has not yet been measured in this first pass.

## Reproduce the workload comparison

Use the [frozen synthetic fixtures](../../benchmarks/mtp-fixtures.json) with the
[API replay client](../../benchmarks/mtp_matrix.py). It verifies exact rendered
token hashes, actual peak concurrency, output counts, prefix queries and zero
preemptions. Each visit warms the exact prompts before timing them.

The operator must own a drained six-node window, check exclusive GPU ownership
and fresh host guards, verify the actual image, and apply/acknowledge the selected
policy. The public client does not inspect hosts or change the backend; its
image/policy metadata are operator assertions. The private fleet controller and
original numerical oracles are not included in this repository.

```bash
python3 benchmarks/mtp_matrix.py \
  --base http://127.0.0.1:8953 \
  --key-file /path/to/private-api-key \
  --image sha256:2e947348fd26b8d58e4535e0321126935069a5e521784fa573f19fb560975d9d \
  --policy-file local/selected-mtp-policy.json \
  --output-dir local/benchmarks --name visit-original \
  --scope matrix --repetitions 1
```

Drain before each policy change. Rotate/reverse policy order across repeated
visits and use the [analysis helper](../../benchmarks/analyze_mtp_matrix.py) to
compare complete receipts. It reports measurements without selecting a policy.

## Incremental image assembly

Sources: [wrapper](../../runtime/vllm/amos_mtp_tuning.py),
[hash-checked installer](../../runtime/vllm/patch_mtp_tuning.py),
[builder](../../runtime/vllm/build_mtp_tuning.py) and
[CPU checks](../../runtime/vllm/test_mtp_tuning.py).
The builder requires the exact retained 124-layer budget-control image archive;
it adds one source-only layer in both Python roots and enables the control flag.

```bash
python3 runtime/vllm/build_mtp_tuning.py \
  --base-archive /path/to/retained-budget-control.delta.tar \
  --output /path/to/mtp-tuning.delta.tar \
  --receipt /path/to/mtp-tuning-build.json
```

The Docker delta requires the exact lower image chain already installed. Verify
the image and patched file hashes on every rank. This is not a registry release
or clean-machine build. The resulting image has 125 layers; another layer needs
the bounded [upper-layer packaging repair](PREFILL_BUDGETS.md) first.

## Remaining work

Complete two balanced repeat visits, measure C2/C3 costs, fit a candidate cost/
probe policy and compare it repeatedly against original. Require original
short/long quality and actual native/router/client acceptance before promoting
any changed policy. TensorFold work is outside this experiment.
