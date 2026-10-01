# Vendored collective implementation

The `roce/` runtime comes from [local-inference-lab/b12x](https://github.com/local-inference-lab/b12x),
revision `ffb7442d04a9f50b950df1fb17280acad881b7d5` (PR 295).
The vLLM adapter, worker health fragments and tests come from
[local-inference-lab/vllm](https://github.com/local-inference-lab/vllm), revision
`a7935eb1d8fa51400cd13452b1d988197deef3b8` (PR 597).

Both projects use Apache License 2.0; the license text is retained in
`LICENSE-APACHE-2.0`. Existing source notices are preserved. `source-lock.json`
records the exact imported files. `UPSTREAM.md` is the imported upstream
description; its measurements describe that upstream four-node deployment.
Local six-node correctness and fault-injection results are documented in the
parent README. The local integration changes are implemented separately in
`../patch_rocenante.py`.
