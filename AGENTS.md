# Repository guidance

- Keep the root README short; place detailed instructions in recipes and results.
- Preserve original mixed K3/K4 3.25 bpw weights and per-component licenses.
- Separate measured serving results, synthetic TensorFold results, and unexecuted ideas.
- Record source revisions, benchmark conditions, and limitations for every claimed improvement.
- Keep credentials, private host mappings, raw application conversations, and compiled artifacts out of Git.
- Do not launch GPU probes alongside the serving model. Require a drained,
  explicitly owned window and fresh exact-container memory guards.
- Preserve forward-only iteration; never silently restore older serving configurations.
- TensorFold full-model family registration and serving integration remain incomplete.
- Update `provenance/imports.json` when refreshing imported files, preserving their origin.
