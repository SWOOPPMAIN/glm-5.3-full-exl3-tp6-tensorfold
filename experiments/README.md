# Experiments

**Current status (October 3):** [Deferred local port and latest results](../results/TENSORFOLD_STATUS.md). The source and experiment notes below describe the historical TFP21/early-TFP22 export, not the current production backend.

## Completed: TFP15 attention tuning

The [original proposal](completed/tfp15-attention-tuning.patch) is retained as
a historical patch against TFP14. The current TensorFold snapshot contains the
qualified implementation; do not apply this patch to it. The
[results](../results/tfp15-tensorfold.json) identify the executed source revision.

All five attention modes passed the recorded component and full-model checks.
The selected settings remain explicit options; serving integration is still
incomplete. The full-model request core subsequently passed short-request
serial parity in [TFP16](../results/tfp16-tensorfold.json). Distributed request commands and the scheduler subsequently passed
[TFP17](../results/tfp17-tensorfold.json). Decode graphs and short App/HTTP checks subsequently passed
[TFP18](../results/tfp18-tensorfold.json). Profiling and a seven-depth sweep subsequently passed
[TFP19](../results/tfp19-tensorfold.json). BF16 projection tiles passed
[TFP20](../results/tfp20-tensorfold.json), followed by packed request execution
and matched HTTP C1/C4 in [TFP21](../results/tfp21-tensorfold.json).

Next: select scalar or packed dispatch by active client count, bound prompt work
per round, and complete broader API/long-context qualification before promotion.

## Pending: TFP22 scheduling and host-memory diagnosis

[Candidate patch](pending/tfp22-scheduling.patch) against the qualified TFP21
source; [incomplete GPU evidence](../results/tfp22-tensorfold.json).
The source snapshot remains TFP21. The candidate passed 125 CPU tests and
six-rank output parity for an 8,218-token prompt alongside decoding,
cancellation, a new arrival and retained continuation. It uses scalar execution
for one live client and shared 3072/256-row prompt budgets for grouped work.

The final automatic-mode HTTP concurrency run stopped when one host's available
memory reached 4.86 GiB for two seconds. Its exact-container guard killed that
worker; the controller stopped the remaining test peers. No worker was
OOM-killed. The transient allocation is not yet attributed, so there is no
qualified TFP22 throughput claim. The same P24 configuration reopened and
passed native and Code/Chat checks.

Next: collect bounded host/process memory samples and graph-capture observations,
repair the memory budget or allocation cause, and complete the remaining gates.
Do not lower guards to make the candidate pass.
