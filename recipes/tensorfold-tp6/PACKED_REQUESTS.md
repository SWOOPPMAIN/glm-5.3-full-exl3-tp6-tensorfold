# Packed requests on six ranks

TFP21 combines compatible operations from up to four requests into one target,
MTP or vocabulary pass. It retains the original 3.25 bpw weights, BF16 dense
weights, [TFP20 projection plan](PROJECTIONS.md), resident 804K cache and one
model/workspace. [Measurements](../../results/tfp21-tensorfold.json).

## What changed

The scalar and packed paths drive the same request algorithm. It can suspend
at target, draft or sampling operations. The packed driver groups matching
operations within the existing row limits, executes them together, and returns
owned hidden-state copies to each request before reusing the shared arena.

Each row carries its own logical token position and physical cache base.
Overlapping request leases are rejected before model execution. The head can
process several requests together while each keeps its own seed, position and
sampling policy. Rank zero still broadcasts each request's sampling decisions.

One prepared `step_many` command advances a group. Every rank agrees on the
request state before execution and on resulting state before clients receive
output. Cancellation flags are frozen at the command boundary. A partial
failure latches the controller and retains leases; it never retries the group.

Graph inputs now accept different cache bases per row. The separate
`mtp_batch` graph path admits multiple rows, including canonical MTP commits.
The scalar MTP path still graphs one row. Both modes in this test use a bounded
64-entry graph cache with up to 32 target/packed-MTP rows. Head batches remain
bounded by the existing 17-row vocabulary workspace.

## Matched HTTP measurements

Original checkpoint chat template, thinking disabled, greedy generation,
128 output tokens per request. Each mode runs C1 warmups and three timed
repeats per fixture, then two C4 warmups and three timed four-client rounds.
Every complete answer must equal independent serial target generation.

| Measurement, output tok/s | Scalar | Packed |
| --- | ---: | ---: |
| Code C1 | 25.40 | 23.06 |
| Prose C1 | 24.96 | 24.85 |
| Four clients combined | 23.61 | 51.77 |

Single-request code regressed in this always-packed experiment; prose was
effectively unchanged. The next dispatch policy should use scalar commands
for one active request and packed commands for several, with transition tests
before promotion.

C4 throughput changes by **119.2%**. It is 512 output tokens divided
by total four-client wall time, including prompt work, HTTP delivery and any
reported graph capture. C1 uses 127 tokens divided by decode time after the
first token; it excludes prefill. All timed C1 graphs were warm. Timed C4 new
capture counts were **[0, 0, 0]** for scalar and
**[0, 0, 0]** for packed mode.

Modes run consecutively in one guarded load. These short fixtures do not
establish a hardware ceiling, a confidence interval or long-context speed.
Gains include multirow canonical-MTP graphs as well as combining clients.
They are not a matched comparison with P24's 512-output-token serving workload.

## Using the path

The existing admitted model should use the TFP20 projection plan. Construct
its backend and controller in the owning worker, then explicitly enable packing:

```python
# Existing admitted six-rank group, model, cache, arena and CPU store required.
def factory():
    torch.cuda.set_device(0)
    sampler = RankZeroSampler(group, 'cuda:0', 17)
    backend = GraphBackend(model, caches, table, arena, group=group,
                           sampler=sampler, max_graphs=64, max_rows=32)
    core = RequestEngine(backend)
    return RequestController(Replica(core), store, rank,
                             generation=shared_unique_generation)

if rank == 0:
    engine = ServingEngine(RequestScheduler(factory, packed=True))
else:
    factory().follow()
```

The snippet assumes the named classes are imported from the full-TP6 modules;
it is not a clean-machine deployment command. See the probe for full ownership,
startup and teardown ordering. Family registration and backend packaging remain
pending, and P24 remains the live service.

## Qualification and admission

The GPU probe checks packed tensors against separate original-weight forwards
at different physical offsets and logical contexts, including a 2,048-token
indexer boundary. It covers 20 target rows, four recursive-MTP rows, 12 canonical
MTP rows and eight head rows, with changed request ordering and membership.

Mixed sampling/depths, a cancelled member, replacement ownership and retained
continuation run through the six-rank command bus against serial references.
The actual HTTP path checks C1/C4, SSE content and usage, context rejection,
callback cancellation and retained-prefix continuation.

**230 GPU checks**, including **120 exact tensor** and
**89 exact sequence comparisons**, passed. **119 CPU tests**
cover the shared request algorithm, packed metadata, six-store coordination,
concurrent clients, cancellation, retained ownership and failure latching.

All six workers exited zero, with no OOM or guard trips. Peak PyTorch allocation
was **99.93 GiB**; minimum host available memory was
**10.36 GiB**. Add `packed_reserve(3072)` and
`graph_reserve(64, 32)` to the existing weight, workspace, request and runtime
admission budget. Follow the [drained-fleet contract](README.md#gpu-qualification-contract).

Before promotion, long-prefill scheduling needs a shared per-round token budget
so several large prompt chunks cannot delay cancellation or existing decodes.
Broader tools/reasoning/stop/disconnect behavior, authentic long-context quality
and performance, packaging and application deployment remain open gates.
