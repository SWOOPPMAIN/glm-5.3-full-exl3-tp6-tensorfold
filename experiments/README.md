# Experiments

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
[TFP19](../results/tfp19-tensorfold.json). BF16 projection tuning, packed batching
and broader API/long-context qualification remain next.
