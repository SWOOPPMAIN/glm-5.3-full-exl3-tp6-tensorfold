# Full-model request execution

This is an experimental execution core. `tensorfold serve` still does not
register full GLM TP6. An admitted model, one six-rank communicator, 79 layer
caches, RoPE table and one shared workspace must already exist.

## Files

Under `experimental/tensorfold/src/tensorfold/families/glm_moe_dsa/`:

- `request_backend.py`: connects the qualified full model to eager request steps.
- `request.py`: chunked prefill, recursive MTP, target verification, cache leases
  and retained-prefix reuse.
- `memory.py`: `request_plan` reserves additional request buffers and sampler
  temporaries, above model/cache/workspace storage and the runtime reserve.

## Execution rules

One worker serializes model passes. Every rank must receive identical start,
step, resume, cancel and drop commands. The six-rank command bus and HTTP
scheduler are not yet provided. Per-rank independent cancellation callbacks
would break collective ordering.

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
An API scheduler must choose terminal-prefix eviction deliberately.

## Evidence and remaining work

[TFP16](../../results/tfp16-tensorfold.json) compares short raw-text requests
with independent serial generation, using original weights and the fully
resident 804K cache. It includes four interleaved requests and prefix reuse.
CPU tests additionally cover every four-draft rejection position and cancellation
after target, during sampling and mid-draft.

This does not qualify chat templates, long-context reasoning, API cancellation,
streaming/tool/reasoning formatting, continuous batches, or generation speed.
The serving deployment remains vLLM P24.
