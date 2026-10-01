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

## Next

1. Connect the qualified request scheduler to the real App/HTTP frontend.
2. Capture decode graphs and pack concurrent requests into shared target passes.
3. Qualify chat templates, conversation history, tools, reasoning, streaming and usage.
4. Measure authentic code/prose and uncached 8K/32K/128K; qualify real 360K quality.
5. Package the backend and promote measured gains through Max/Pi-router/Code.
6. Publish a verified clean-machine image build and portable fleet launcher.

Short-request parity does not establish serving throughput or long-context
quality. Full application integration remains part of the original goal.
