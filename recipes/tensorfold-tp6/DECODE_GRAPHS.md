# Full TP6 decode graphs

`GraphBackend` uses the same admitted model, cache, workspace and existing
six-rank NCCL group as `FullModelBackend`. Construct it in the scheduler worker
with `group=dist.group.WORLD`; each follower constructs its own matching adapter.
This is a development integration recipe, not a complete fleet launcher.

## Capture contract

- Target rows 1–5, canonical MTP row 1, and vocabulary rows 1–5 use graphs.
  Larger target/MTP calls retain eager execution.
- Each graph owns fixed token, position, physical base/slot and hidden inputs.
  Logical visible bounds determine reuse; physical cache bases do not.
- At most 16 entries are retained. Each uses an independent graph memory pool,
  allowing varying serial target/MTP/head order. Eviction synchronizes and resets
  the old graph before replacing it.
- Add `graph_reserve()` to model, cache, workspace, request and runtime admission.
  Its default 512 MiB allowance bounds retained positive PyTorch allocator growth;
  it is not a prediction or measurement of total CUDA/NCCL driver memory.
- All ranks agree on graph admission and include deterministic graph state in
  request-command agreement. A capture/replay failure latches the backend;
  it never automatically retries a partial operation.
- `Replica.close` releases graphs on their owning worker before group destruction.
  Model outputs remain borrowed workspace views; retain them before another pass.

See [PyTorch graph memory management](https://docs.pytorch.org/docs/2.10/notes/cuda.html#graph-memory-management)
for address lifetime and graph-pool requirements.

## TFP18 measurement

Original mixed K3/K4 3.25 bpw weights; all 79 layers; fully resident 804K test
cache. The checkpoint chat template renders code and prose prompts with thinking
disabled and greedy decoding. Each answer contains 128 generated tokens.

| Local HTTP diagnostic | Eager | Graphs |
| --- | ---: | ---: |
| Code, C1 median of three | 22.78 tok/s | 24.48 tok/s |
| Prose, C1 median of three | 23.75 tok/s | 23.64 tok/s |
| Four clients, combined | 22.34 tok/s | 23.11 tok/s |

C1 uses 127 tokens divided by engine decode time after the first token. Timed
runs created no new graphs. C4 uses 512 tokens divided by total four-client wall
time, including prefill; it has one measurement per mode. Every timed output
matched an independent serial target sequence. This fixed eager-then-graph order
and short fixture set establish a modest diagnostic gain, not a general speedup.
They are not the same workload as the 512-token P24 serving measurements.

All 329 GPU checks passed: 264 exact tensor checks, 27 sequence comparisons,
plus graph reuse/cleanup, request validation, prefix reuse, callback cancellation
and rank agreement. There were 91 CPU checks. Each rank executed 508 commands
and 2,305 leader sampling decisions per HTTP mode; graphs captured eight entries
and replayed 4,209 times. Peak PyTorch allocation was 99.93 GiB; minimum host
available memory was 10.82 GiB. All workers exited successfully and guards stayed clear.

See [the complete summary](../../results/tfp18-tensorfold.json). The 804K resident
cache does not establish long-context quality. Packed batching, broader API
regressions, production routing and a portable image remain pending. P24 remains
the serving backend.

The subsequent [draft-depth/profile experiment](DECODE_PROFILING.md) uses up to
24 graph entries and nine target/head rows for depth-eight verification.
It preserves the one-row MTP graph path and the underlying model kernels.
