# Benchmark a serving candidate

Run against the native API in a drained measurement window. Use identical
weights, frozen fixtures, explicit tuning, and the same workload for comparisons.
This harness sends requests; it does not change or restart the service.

```bash
mkdir -p local/benchmarks
python3 benchmarks/performance.py prepare \
  --base http://127.0.0.1:8953 \
  --key-file /path/to/private-api-key \
  --fixtures local/benchmarks/fixtures.json
python3 benchmarks/performance.py run \
  --base http://127.0.0.1:8953 \
  --key-file /path/to/private-api-key \
  --fixtures local/benchmarks/fixtures.json \
  --output local/benchmarks/candidate.json --label candidate \
  --repetitions 3 --prefill-repetitions 2
```

Use an SSH tunnel when the native API binds to the remote loopback interface.
The key stays in a local private file. Output includes generated text; review
it before sharing. The ignored `local/` directory is the default place for
raw fixtures, responses, logs, and credentials.

Record image/source revisions, prompt/output lengths, MTP configuration,
prefix-cache hits, preemptions, concurrency, TTFT and output tok/s. Compare
numerical correctness before promoting a speed result. Test the real client
and router path after native API qualification.

## Prompt-cache residency

See [the synthetic Code/tool replay protocol](CACHE_REUSE.md) and
[per-request samples](../results/cache-reuse-samples.json). Keep cold prefill,
cached first-token latency, generation and aggregate throughput separate.
