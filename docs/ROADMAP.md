# Roadmap

## Working and measured

- Full GLM-5.3 EXL3, TP6, native MTP, vLLM API and Swoopp application routing.
- Original 3.25 bpw expert fragments; lossless redistribution and verified rank ownership.
- P24 serving: numerical gates, code/prose, cold 8K/32K/128K, four concurrent requests.
- TensorFold full target/MTP forward passes, original weights, bounded scratch,
  and fully resident 804K test cache on all six ranks.
- Expert chunk tuning and fixed-order bulk reductions with exact GPU comparisons.
- Attention tile skipping and row-batch comparisons, with FP64 reference checks.

## Next

1. Use the qualified attention options and detailed stage profile in the
   full-engine candidate.
2. Complete the TensorFold request engine: recursive MTP, accepted-prefix cache
   commit/reclaim, cancellation, sampling, and four-request scheduling.
3. Preserve conversation prefixes; test tool calls, reasoning, streaming, and usage.
4. Measure authentic code/prose and long prompts against the same weights and fixtures.
5. Package and qualify the backend, then integrate it through Max/Pi-router/Code.
6. Publish a clean-machine image build and portable fleet launcher when verified.

The attention options passed component and full-forward GPU checks; a higher
generation speed is still unmeasured. The original task includes end-to-end
integration; component checks alone do not complete it.
