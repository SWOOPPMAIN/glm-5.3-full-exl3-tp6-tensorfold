# Reproduce the P27 prompt-cache replay

This tests the existing vLLM prefix cache with three synthetic code/tool histories.
It changes no model settings, weights, reasoning policy, or serving containers.
Run generation only in an owned, drained six-GPU window with fresh exact-container
8 GiB / 2-second guards and the existing pressure protections. Reopen serving
and check the actual applications after each context-size window.

## Frozen fixtures

Generate each size against the same pinned P27 tokenizer and chat template:

```bash
python3 benchmarks/cache_fixtures.py \
  --base http://127.0.0.1:8953 --key-file /path/to/private-api-key \
  --target 8192 --output local/benchmarks/cache-8k.json
```

Repeat with targets `32768` and `131072`. The generator contains the exact seven
fixture definitions used by the measured local harness. Compare its printed
fixture hash with [the result summary](../results/cache-reuse.json).
It uses CPU chat rendering only. The site-specific maintenance/guard operator
is not published; this generator does not reserve or drain your fleet.

Each fixture contains three payloads (`alpha`, `bravo`, `charlie`): a fixed
system instruction and tool schema, a long generated Python module, a prior
assistant tool call with retained reasoning, its result, and a JSON-answer request.
Requests use temperature 0, seed 17, low reasoning, streaming usage and a
256-token output ceiling. These are short-answer residency tests, not code
quality or generation-throughput benchmarks.

## Request order

1. Send each base payload once: A, B, C. Give each conversation its own fresh
   `cache_salt`, then keep that salt throughout its warm tests. Record three
   genuinely cold controls without shared-prefix hits between the seed requests.
2. Replay A, B, C twice, unchanged. This tests alternating residency without
   inserting cold copies that would artificially evict the warm histories.
3. For each conversation, append the seed response's content and reasoning as
   an assistant message, then a user confirmation of the same JSON result.
4. Extend that appended history with another `read_fixture_result` assistant
   call and tool result, then the original JSON-answer question.
5. Make two forks from step 3 by changing only its final user question to
   `Branch 0: confirm the original fixture result.` and `Branch 1: ...`, followed
   by the original JSON-answer question.
6. Make a sibling from the base's first two messages (system plus long source),
   followed by `For this independent review, call read_fixture_result for module
   NAME.` Verify the generated tool name and module argument.

There are 24 requests per context size: three cold seeds and 21 reused requests.
Keep the generated assistant reasoning when reconstructing histories. Do not
silently toggle `clear_thinking` or change tools, message order, or templates.

## Measurements and checks

Before each generation, use `/v1/chat/completions/render` to obtain the exact
rendered token IDs and hash them. This pinned fork's `/tokenize` endpoint omits
historical `reasoning_content`; its count was five tokens short for our seed
fixture. Using that count would misdiagnose a serving/cache problem.

Record native `/metrics` before and after each request; allow asynchronous stats
to settle. Require cache-query growth and reported prompt usage to equal the
rendered length, zero preemptions, and no other native requests. Record cached
counter growth, first visible reasoning/content/tool event, completion time,
streamed usage, answer hashes and semantic checks.

Compute the longest token prefix shared with previously completed prompts of
the same salt. For this 64-token cache block and native MTP/EAGLE matcher, a
conservative resident-prefix expectation is:

```python
max(0, 64 * (min(common_prefix_tokens, prompt_tokens - 1) // 64) - 64)
```

The extra block is the native draft-cache matching allowance; do not label it
avoidable eviction. Earlier generated tokens can also supply hits beyond this
prompt-only lower bound. Compare every warm request with that bound, and compare
identical replay answers with its cold seed.

The shared cache reports 470,847 token slots at 24 GiB/rank. Three 128K histories
fit within that capacity; this protocol makes no claim about four simultaneous
128K histories or behavior after deliberately exceeding cache capacity.

Report cold and cached TTFT separately. Dividing the full cached prompt length
by its short TTFT is not cold-prefill throughput. This measurement validates
existing automatic prefix caching; it is not a newly deployed speedup.

Underlying caching and scheduling are from [vLLM](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/).
The synthetic fixtures and replay protocol are Swoopp's. See the repository
[credits](../CREDITS.md) for the full GLM/EXL3/TP6 stack.
