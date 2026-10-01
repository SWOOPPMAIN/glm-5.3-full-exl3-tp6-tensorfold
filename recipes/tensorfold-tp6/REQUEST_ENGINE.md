# Full-model request execution

This is an experimental execution core. `tensorfold serve` still does not
register full GLM TP6. An admitted model, one six-rank communicator, 79 layer
caches, RoPE table and one shared workspace must already exist.

## Files

Under `experimental/tensorfold/src/tensorfold/families/glm_moe_dsa/`:

- `request_backend.py`: connects the qualified full model to eager request steps.
- `graphs.py` / `graph_plan.py`: bounded target/MTP/head capture and replay
  using fixed inputs and independent graph pools; [details](DECODE_GRAPHS.md).
- `request.py`: chunked prefill, recursive MTP, target verification, cache leases
  and retained-prefix reuse.
- `control.py`: prepare/execute/result agreement over the existing TCPStore.
- `control_sampling.py`: rank-zero token decisions on the existing NCCL group.
- `scheduler.py`: bounded client queues, one model worker, cancellation and
  deliberate eviction/reuse of terminal prefixes; `ServingEngine` matches the
  CUDA App call interface and passes the short HTTP checks in TFP18.
- `memory.py`: `request_plan` reserves additional request buffers and sampler
  temporaries, above model/cache/workspace storage and the runtime reserve.

## Execution rules

One worker serializes model passes. Every rank must receive identical start,
step, resume, cancel and drop commands. `RequestController` now distributes
these through the existing TCPStore, with six-rank prepare/result agreement.
Rank zero broadcasts each sampled token decision before EOS or draft branching.
Idle followers block on CPU doorbell keys. Failed exchanges latch; there is no
automatic command retry or worker replacement.

Construct the backend, core and rank-zero controller inside the scheduler
worker factory. On followers, construct them in the owning thread and call
`controller.follow()`. Client callbacks run in client threads, and cancellation
is frozen at each distributed step boundary. A long prefill/verify step may
finish before a disconnect takes effect. Waiting/output queues are bounded;
a slow consumer cancels only its own request.

The target cache records input tokens. MTP position `p` combines target hidden
state `p` with token `p+1`; further draft steps consume normalized MTP hidden
states. Target sampling uses the seed and absolute output position. A draft
is accepted only when it matches that target sample.

A verification round commits only the matching input prefix. Rejected cache
rows remain outside the visible prefix and are overwritten before reuse.
Accepted target hidden rows are retained for canonical MTP absorption before
the next draft. Borrowed model output buffers are copied before another pass.

## Capacity and conversation reuse

Each request reserves a contiguous extent for prompt plus output allowance.
At most four request leases are retained, including finished, cancelled and
failed conversations. The caller must drop a lease or resume its prefix before
admitting another. Dropping synchronizes outstanding writes before reuse.

A finished conversation can transfer its entire committed target prefix to a
follow-up request. The last emitted token has not yet run through the target;
the follow-up processes it when it appears in the new prompt. Exact-prompt
replay retains the final hidden row and samples it under the new settings.

Growth requires adjacent free space. A nonmatching prompt or failed growth
leaves the old conversation intact; no silent eviction or cache copying occurs.
The scheduler explicitly evicts terminal leases to make room, preferring the
longest matching finished prefix. Active leases are never evicted. If active
requests consume the needed capacity, admission waits while those requests
continue. Invalid requests are rejected before any retained state is evicted.

## Evidence and remaining work

[TFP16](../../results/tfp16-tensorfold.json) compares short raw-text requests
with independent serial generation, using original weights and the fully
resident 804K cache. It includes four interleaved requests and prefix reuse.
CPU tests additionally cover every four-draft rejection position and cancellation
after target, during sampling and mid-draft.

TFP16 alone does not qualify API behavior, long-context reasoning or speed.
TFP18 adds short checkpoint chat-template HTTP, streamed text/token/usage parity
and warm timing; tools, reasoning, stop/history, disconnect handling, continuous
batches and long-context behavior still need broader qualification.
The serving deployment remains vLLM P24.

[TFP17](../../results/tfp17-tensorfold.json) exercises this controller and scheduler
with four concurrent real client threads on all six GPUs, plus retained follow-up,
invalid input, idle wake, callback cancellation and ordered shutdown. CPU
regressions also cover eight queued clients, slow consumers and rank faults.
