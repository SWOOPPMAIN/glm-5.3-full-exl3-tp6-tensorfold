# Target-verified copy drafting

**Source review and CPU checks only; not deployed or speed-qualified.**
The target remains full GLM-5.3, original 3.25 bpw EXL3, TP6, target MXFP8,
360K configured context, four admitted requests and a 3072-token prefill budget.
No additional drafter checkpoint is required by the pinned ngram implementations.

## What the pinned source supports

The vLLM runner contains CPU `ngram` and GPU `ngram_gpu` proposers. Both use
the native target rejection sampler. Greedy verification accepts matching target
tokens until the first mismatch. For stochastic sampling, a deterministic copied
proposal has probability one; the target probability controls acceptance, with
the rejected proposal excluded from the recovery distribution. Synthetic acceptance
must remain disabled. This source review is not full-model fidelity evidence.

CPU ngram disables asynchronous scheduling in this fork. GPU ngram supports it,
so it is the preferred compatibility trial. Both reject the existing
`adaptive_speculative_tokens_window` setting. A standalone trial must explicitly
disable MTP cost/request-phase/tuning controls and draft-EH sharding, while retaining
the target's shared384 layout and decode projections. Four speculative tokens allow
reuse of the existing target graph shapes; variable-length copies and mixed batches
still need runtime verification.

## Checks completed

- The unchanged CPU proposer passed 400 randomized longest-match oracle checks
  and four-request history checks through nearly 360K tokens on the coordinator CPU.
- Its constructor's empty sampled-token lists did not trigger Numba compilation.
  A real request must warm the proposer before any timing.
- The GPU proposer's uncompiled tensor logic, run on CPU, passed 400 additional
  oracle checks. This does not establish CUDA compilation or GPU performance.
- Three scatter cases at exact context capacity overwrote the last copied-history
  token with its previous value because padded indices were clamped to the same
  position. Those cases have no remaining generation space. This is a proposal
  scratch-history issue; incorrect target output has not been demonstrated.
- Offline launch checks retained identical non-speculative arguments, preserved
  the current MTP launch, and rejected inherited MTP-only settings for copy profiles.
  The deployed launcher was not changed.

[Pinned source review](../../results/copy-drafting-source-review.json) ·
[CPU proposer checks](../../results/copy-drafting-cpu.json) ·
[Tensor semantics checks](../../results/copy-drafting-tensor-cpu.json)

## Remaining trial

1. Verify the full pinned configuration and proposal history handling, including
   terminal requests, cancellations, varying accepted lengths and C1/C4 batches.
2. Use a separate owned, drained six-GPU window with fresh exact-container
   8 GiB / 2-second guards. Keep the current target settings and selected E3 policy.
3. Qualify original short/long numerical references and functional behavior before
   timing. Target verification is mandatory; copied tokens are only proposals.
4. Compare exact-repeat code, edited-repeat code and prose controls against the
   original adaptive MTP policy, with matching prompts and output budgets.
   Report accepted copies, rejected work, proposal cost, TTFT, stream gaps,
   single-request generation and aggregate throughput separately.
5. Investigate combining copy proposals with MTP only if standalone measurements
   justify the additional implementation. Reopen qualified serving promptly.

There is no copy-drafting serving-speed claim. The proposal and verification
implementations are from vLLM; their exact source hashes are retained in the review.
See the repository [credits](../../CREDITS.md).
