# Roadmap

## Working and measured

- Full GLM-5.3 EXL3, TP6, native MTP, vLLM API and Swoopp application routing.
- Original 3.25 bpw expert fragments; lossless redistribution and verified rank ownership.
- P24 serving: numerical gates, code/prose, cold 8K/32K/128K, four concurrent requests.
- TensorFold full target/MTP forward passes, original weights, bounded scratch,
  and fully resident 804K test cache on all six ranks.
- Expert chunk tuning and fixed-order bulk reductions with exact GPU comparisons.
- Attention tile skipping and row-batch comparisons, with FP64 reference checks.
- Eager request core: recursive MTP, keyed target verification, four interleaved
  requests and retained-prefix continuation against serial target generation.

- Distributed TP6 request controller, leader-only sampler and bounded
  concurrent-client scheduler, qualified against original-weight serial output.

- Bounded decode graphs and actual checkpoint chat-template HTTP C1/C4/SSE,
  with exact serial output checks and separate short-context timing.

- Seven-depth exact-output sweep and bounded all-rank decode profiling;
  dense BF16 projections identified as the main compute target.

- BF16 projection tile screening and immutable per-model plans, qualified on
  all six ranks against original-kernel tensors and serial request outputs.

- Packed target/MTP/head execution across disjoint request cache leases,
  qualified against serial output and actual short HTTP C1/C4/SSE.

## Next

1. Diagnose the TFP22 transient host-memory peak and finish automatic dispatch/shared-prefill qualification.
2. Screen larger EXL3 expert chunks using TensorFold 0.6.1 shared-memory opt-in; compare matched workloads with P24.
3. Adapt Mia complete tool-call/keepalive fixes and 0.6.1 HTTP/reasoning fixes; extend conversation/fork cache reuse and API checks.
4. Measure authentic code/prose and uncached 8K/32K/128K; qualify real 360K quality.
5. Package the backend and promote measured gains through Max/Pi-router/Code.
6. Publish a verified clean-machine image build and portable fleet launcher.

Short-request parity does not establish serving throughput or long-context
quality. Full application integration remains part of the original goal.
