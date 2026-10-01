# Experiments

## Completed: TFP15 attention tuning

The [original proposal](completed/tfp15-attention-tuning.patch) is retained as
a historical patch against TFP14. The current TensorFold snapshot contains the
qualified implementation; do not apply this patch to it. The
[results](../results/tfp15-tensorfold.json) identify the executed source revision.

All five attention modes passed the recorded component and full-model checks.
The selected settings remain explicit options; serving integration is still
incomplete. Next: use the detailed profile and complete the full-model request
engine, recursive MTP acceptance, cache ownership and client integration.
