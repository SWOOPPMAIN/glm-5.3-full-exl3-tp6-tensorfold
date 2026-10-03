# Prefill budget experiment — October 3, 2026

**Decision: retain 3072.** Fixed and adaptive smaller budgets failed the
existing numerical gate. No candidate speed winner was selected.
[Measurements and receipt hashes](../../results/prefill-budgets.json).

## What was tested

The scheduler's per-step token allowance can change independently of its
3072-token kernel allocation. The new hook supports fixed 1536/768 budgets
and adaptive variants that apply the smaller cap when decoders are active.
It latches a validated control revision between live batches and acknowledges
its hash. Target sampling, rejection, weights, precision and context stay intact.

The starting mixed workload had three 1024-token decoders, followed by a cold
32748-token code/tool history once all three streams emitted. Three repeats
measured median arrival TTFT **35.75 s**, aggregate output **37.93 tok/s**,
and maximum stream-event gaps around **3.39–3.42 s**. This differs from the
four-short-request 76.5 tok/s workload. MTP can emit several tokens per event.

Quality screening uses the last 512 teacher-forced positions of a frozen 8K
prompt, compared with the original isolated reference. Mixed screens run the
same prompt while three requests each generate 1024 tokens. Each screen below
ran once, with no preemption; failed candidates did not proceed to timing.

| Policy | Traffic | Coarsened KL | Top-1 agreement | NLL delta | Gate |
| --- | --- | ---: | ---: | ---: | --- |
| Current 3072 | Three decoders + 8K | 0.003896 | 99.805% | 0.003312 | Fail |
| Fixed 1536 | Isolated 8K | 0.003684 | 99.609% | 0.006664 | Fail |
| Fixed 768 | Isolated 8K | 0.005934 | 99.219% | 0.010674 | Fail |
| Adaptive 1536 | Three decoders + 8K | 0.004089 | 99.805% | 0.002768 | Fail |
| Adaptive 768 | Three decoders + 8K | 0.005117 | 99.609% | 0.005982 | Fail |

Required limits: KL ≤ 0.001, top-1 agreement ≥ 99.5%, NLL delta ≤ 0.01.
Even the 3072 mixed control differs from the isolated reference. This exposes
batch/chunk sensitivity; it does not prove a regression introduced by the new
budget hook. The precise kernel cause has not been isolated. Do not relax the
gate or describe these results as a task-quality ranking.

At **isolated 3072**, the new image matched the original 4096-position short
reference and all 512-position tails at 8K/32K/128K exactly: top-1 100%, measured
coarsened KL and NLL delta zero. Native API, Pi-router and Code/Chat acceptance
passed after reopening. The same six candidate containers remained running
with fresh, clear 8 GiB / 2-second guards. No older image was restored.

Coarsened KL is a lower bound, not full-vocabulary KL. Reported reference mass
at 128K is only about 46%; these gates do not establish comprehensive 360K quality.
Exact replay requires the retained frozen references identified by SHA256 in
the result file; those reference artifacts are not included in this repository.

## Current control file

The current image requires this file on rank 0's persistent cache mount:
`/root/.cache/amos-tp6-prefill-budget.json` inside the container.

```json
{"revision":"budget1-selected3072","mode":"baseline","budget":3072}
```

Keep the host cache directory mounted at `/root/.cache`; preserve the control
file across restarts and cache cleanup. A missing or invalid file stops scheduling;
there is no automatic fallback. The scheduler writes the accepted revision and
hash to `amos-tp6-prefill-budget.applied.json` in the same directory.
Only change controls in an owned, drained experiment and verify acknowledgment.
The 1536/768 and adaptive modes remain **unqualified for serving**.

## Reproducing the patch and packaging repair

Source: [budget hook](../../runtime/vllm/amos_prefill_budget.py),
[pinned installer](../../runtime/vllm/patch_prefill_budget.py),
[Dockerfile](../../runtime/vllm/Dockerfile.prefill-budget),
[CPU checks](../../runtime/vllm/test_prefill_budget.py).
The installer requires the exact P27 scheduler source hash and patches both
installed source trees. This remains an incremental recipe requiring the retained
base image; it does not supply a clean-machine image build.

The original base has 125 layers. Adding two patch layers built successfully
but Docker's classic layer store refused to load the 127-layer result.
[Moby's layer limit](https://raw.githubusercontent.com/moby/moby/v28.5.1/layer/layer_store.go)
explains that failure. We merged the last four small source-patch layers into
one, retaining 123 lower layers and the final runtime configuration: 124 total.
All six nodes verified the patched files in CPU-only containers before GPU launch.

The [bounded merger](../../runtime/vllm/squash_patch_layers.py) accepts ordered
Docker delta archives containing the needed layers:

```bash
python3 runtime/vllm/squash_patch_layers.py \
  --archive /path/to/retained-p27-tail.delta.tar \
  --archive /path/to/budget-patch.delta.tar \
  --keep-layers 123 --tag local/glm53:budget-control \
  --output /path/to/new-budget.delta.tar \
  --receipt /path/to/new-budget-provenance.json
```

It checks source/config/layer hashes and merged contents/metadata. It only
supports regular files and directories; it rejects links, whiteouts, special
files and type replacements. It is not a general OCI merger. See the
[OCI layer specification](https://github.com/opencontainers/image-spec/blob/main/layer.md).
Loading a delta requires its exact lower chain already present in Docker's classic
store. Check the resulting image ID and files on every node before deployment.

No lower-budget throughput or streaming-gap improvement is claimed. Further
budget work needs numerical consistency across batch and chunk shapes first.
