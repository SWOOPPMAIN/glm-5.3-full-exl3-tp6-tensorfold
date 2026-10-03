# TensorFold status through TFP57

**Deferred research; not serving production.**
[Selected measurements and source pins](tensorfold-latest.json).
The bundled `experimental/tensorfold/` source remains the historical TFP21 export;
this report describes later local experiments, not an updated executable snapshot.

## What progressed

Later local work added automatic request dispatch, bounded prompt scheduling,
fork/tool behavior, original-weight model integration, an HTTP serving adapter,
and bounded CUDA graphs. Execution and self-parity checks progressed through
TFP51–54. Those successes did not pass the independent production fidelity gate.

TFP54 ran eager / graphs32 / graphs64 / graphs64 / graphs32 / eager in one
controlled experiment. C1 uses 256 output tokens, C4 uses 128 per client, and
cold inputs contain 8K/32K tokens. Across those profiles: about 19–21 output
tok/s C1, 49–51 aggregate C4 and 368–388 input tok/s cold prefill. See the JSON
for samples, graph captures and evictions. This is not a matched P27 A/B.

## Why it was not promoted

The latest full fidelity run (TFP49) has **97.05%** short top-1 agreement and
**82.81%** on the 128K tail against the frozen serving reference. Required top-1
is at least **99.5%**; coarsened KL must be at most **0.001**. Short KL is about
**0.0199**, and all tested long-context KL gates fail. These are teacher-forced
reference-agreement measurements, not benchmark task-accuracy percentages.

TFP56 screened nine expert launch pairs: 5,184 exact candidate checks passed,
but no useful prefill gain emerged (best 3,072-row local proxy improvement only
0.16%). Existing launch defaults were retained.

TFP57 screened twelve new latent projection layouts. All 1,410 execution
checks completed, but **1,710 / 2,484 numerical checks failed**. Only the original
reference remained eligible; failed candidates were excluded before timing.
Changed compiler reduction order is a hypothesis requiring proof, not a
confirmed cause or a speed improvement.

A future port must repair reference agreement before further deployment work.
TensorFold is not required to keep the current vLLM service available.
