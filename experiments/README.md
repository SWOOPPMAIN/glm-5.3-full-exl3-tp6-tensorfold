# Next experiment: attention tuning

`attention-tuning.patch` preserves the current candidate separately from the
qualified TensorFold snapshot. It adds optional skipping of empty attention
tiles, 128/256/512/1024-row scratch choices, detailed attention profiling, and
independent FP64/component/full-model checks.

**Status:** CPU checks passed in the working tree; six-GPU correctness and
performance qualification have not run. No serving speed gain is claimed.
Defaults remain the existing 128-row, fixed-loop behavior.

Apply only to a disposable copy of `experimental/tensorfold/`, from inside
that copy, using the absolute path to this patch:

```bash
git apply --check /path/to/attention-tuning.patch
git apply /path/to/attention-tuning.patch
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_glm53_full_tp6_*.py'
```

The full GPU driver requires `MASTER_ADDR` and the same guarded stopped-fleet
contract as the [TensorFold recipe](../recipes/tensorfold-tp6/README.md).
