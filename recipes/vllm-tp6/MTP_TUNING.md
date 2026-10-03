# MTP tuning: repeated comparisons and calibration, October 3

**Bounded experiment complete. Retain the original adaptive MTP policy.**
The new image adds bounded experiment controls without changing the selected
inference policy. This is not a promoted throughput improvement.

## Measurements and limits

[Repeated results](../../results/mtp-repeated-matrix.json) cover 60 workload cells: original
adaptive MTP and fixed depths 1–4, each at short, 8K and 32K context, with single
code/prose/tool requests and a four-request code/prose/tool/code mix. Each cell
has **three visits**, for 180 measurements, with 256 output tokens per request.
Warmups are separate. Policy orders were original/4/2/1/3, 3/1/2/4/original and
2/original/4/3/1. The [first-pass snapshot](../../results/mtp-first-pass.json)
is retained as historical evidence.

Median output tok/s below includes cached prefill and delivery; C4 is aggregate:

| Policy | Short code C1 | Short prose C1 | Short tools C1 | Short C4 | 8K C4 | 32K C4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original adaptive | 42.09 | 32.97 | 44.15 | 75.60 | 67.33 | 66.53 |
| Fixed 1 | 34.60 | 31.97 | 34.65 | 75.13 | 66.09 | 64.13 |
| Fixed 2 | 41.05 | 32.61 | 41.48 | 77.13 | 71.16 | 68.29 |
| Fixed 3 | 42.01 | 32.33 | 45.29 | 76.12 | 69.56 | 65.81 |
| Fixed 4 | 43.84 | 30.29 | 44.55 | 71.77 | 65.05 | 63.73 |

Equal-cell geometric mean throughput ratios versus original are 0.897, 0.994,
1.009 and 0.971 for depths 1–4. Depth 3's first-pass 2.5% advantage shrank to
0.9% with repeats. Depth 2 helps the long concurrent mixes but loses on some
single-request workloads. No global fixed-depth policy has been promoted.

These workloads differ from the historical 512-token serving benchmark. Do not
compare their rates as a deployment regression or new peak. Tool argument deltas
count toward streaming timing, and the output cap can truncate a call. Tool
argument text/hashes were not retained in the first pass; this is not a complete
tool-output equivalence test. Generated continuations can differ across policies
and across identical prompts within a batch.

Receipts retain native acceptance counters and accepted tokens per wall second.
Summed request-decode time divided by draft rounds is an estimated step cost;
concurrent runs include declining residency and do not isolate constant-batch
GPU time. Judge candidate policies on end-to-end measurements as well as costs.

## Cost calibration

[Summary](../../results/mtp-calibration.json),
[selected counter observations](../../results/mtp-calibration-observations.json)
and [fitted table with per-cell ranges](../../results/mtp-calibrated-cost-fit.json).
Three balanced visits measured 72 code/prose calibration cells at 8K context,
depths 1–4 and concurrency 2/3/4, covering **6,130 simultaneous batch steps**.
Minimum observed available RAM was 10.22 GiB; all overlap gates passed.

Mixed requests finish at different times, so their full-request cost estimates
blend several batch sizes. An initial attempt to require identical outputs from
cloned prompts failed. The final method measures native draft-counter growth
strictly inside the interval when every client is decoding. Text identity is
recorded, but is not used as a proxy for concurrency or as a quality claim.

The interval excludes the first second after the last client's first token and
the final half-second before the earliest completion. It requires constant native
concurrency, at least ten distinct counter observations, complete fixed-depth
batches, at least two seconds of overlap, regression R² ≥ 0.98, agreement with
the endpoint estimate within 5%, and measured HTTP timing uncertainty ≤ 1% of
the interval. Counter delivery and polling still add uncertainty; these are
serving-step estimates, not isolated GPU kernel times.

The fitted costs below are **unqualified candidate data**, in milliseconds:

| Concurrency | Depth 1 | Depth 2 | Depth 3 | Depth 4 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 54.89 | 63.73 | 73.38 | 82.87 |
| 2 | 69.49 | 85.05 | 100.21 | 112.81 |
| 3 | 83.05 | 102.80 | 121.69 | 138.29 |
| 4 | 95.61 | 119.61 | 141.11 | 157.87 |

C1 uses request-decode time per draft round from matching 8K code/prose matrix
cells. C2–C4 use the counter slopes. Each value averages the code/prose medians
of three visits. Homogeneous calibration prompts can have different expert
routing from mixed traffic. The fitted policies therefore require actual mixed
workload comparisons and the existing numerical/application gates before promotion.
The candidates use probe windows 8, 16 and 32; their completed screen is below.

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
the native controller already chooses a depth each batch. The fitted cost-mode
candidates were rejected for promotion; this recipe retains original for serving.

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

Use `--scope candidate_matrix` for the expanded 18-cell comparison: it adds
mixed C2/C3 at short, 8K and 32K to the original 12 cells. The candidate screen
brackets windows 16/32/8 with original-policy visits. A promising screen needs repeats
before promotion; this screen found no useful overall gain.

To reproduce one fixed-depth calibration visit, apply its control externally,
then run the [cost client](../../benchmarks/mtp_costs.py):

```bash
python3 benchmarks/mtp_costs.py \
  --base http://127.0.0.1:8953 --key-file /path/to/private-api-key \
  --image sha256:2e947348fd26b8d58e4535e0321126935069a5e521784fa573f19fb560975d9d \
  --policy-file local/fixed-depth3.json --output-dir local/benchmarks \
  --name cost-round0-depth3 --order-offset 0
```

Repeat the depth orders 3/1/4/2, 2/4/1/3 and 1/3/2/4, using order offsets 0, 1 and 2.
Each visit contains six cells (code/prose at C2/C3/C4). The client verifies the
frozen rendered prompts and records failed cells before stopping. Its image,
policy and host-guard qualifications remain the operator's responsibility.
[The fitter](../../benchmarks/fit_mtp_costs.py) accepts fixed-depth matrix and cost
receipts via `--input ... --output ...`; it refuses incomplete coverage or fewer
than three samples per cell and writes a fit report containing candidate controls
without applying them.

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

## Fitted-policy screen and decision

[Screen results](../../results/mtp-candidate-screen.json) and
[documented outcome](../../results/mtp-outcome.json). One 18-cell visit for each
candidate was bracketed by original-policy visits, adding 90 measurements.
Cells cover code/prose/tools at short/8K/32K and mixed concurrency 2/3/4.
The table gives geometric-mean throughput change against the two original visits:

| Fitted probe window | All 18 cells | C1 | C2 | C3 | C4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8 | +0.19% | +1.62% | −0.80% | +0.06% | −2.88% |
| 16 | −0.24% | +0.37% | +0.19% | −0.12% | −2.59% |
| 32 | −0.27% | −0.18% | +1.89% | −0.33% | −2.60% |

**Retain original.** None demonstrates a useful general gain that justifies
production qualification. Each fitted policy passed eight functional checks;
full numerical qualification was not run for these unselected candidates.
Original policy reopened on the same image and containers, with native/Pi-router/
Code/Chat acceptance passed and a final minimum available RAM of 10.43 GiB.

These are one-visit candidate screens, not statistically established regressions
or proof that all adaptive retuning is exhausted. Three-visit fixed-depth results,
measured cost calibration and this screen support stopping this bounded pass.
A future new hypothesis still needs repeated performance and original-reference
short/long numerical gates before promotion. Communication is next; TensorFold
work is excluded.
