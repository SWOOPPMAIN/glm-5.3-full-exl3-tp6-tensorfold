"""Measure fixed-depth draft costs while every client is actively decoding.

Uses native per-batch counters and wall time, without a profiler or backend
patch. Complete streams and full-run counters remain in the diagnostic receipt.
"""
from concurrent.futures import ThreadPoolExecutor
import statistics
import threading
import time

from mtp_matrix import stream, idle

DRAFTS = 'vllm:spec_decode_num_drafts_total'
DRAFT_TOKENS = 'vllm:spec_decode_num_draft_tokens_total'
ACCEPTED = 'vllm:spec_decode_num_accepted_tokens_total'
RUNNING = 'vllm:num_requests_running'


def measure_overlap(api, item, count, depth):
    before = idle(api)
    gate = threading.Barrier(count)
    first = [None] * count
    start = time.monotonic()

    def one(index):
        gate.wait(timeout=30)
        def on_first():
            first[index] = time.monotonic() - start
        row = stream(api, item['payload'], on_first=on_first)
        row.update(fixture=item['id'], first_from_origin=first[index],
                   finished_from_origin=time.monotonic()-start)
        for key in ('answer', 'reasoning', 'tool_calls', 'semantic_pass', 'strict_json'):
            row.pop(key, None)
        return row

    observations = []
    with ThreadPoolExecutor(max_workers=count) as pool:
        futures = [pool.submit(one, i) for i in range(count)]
        while not all(f.done() for f in futures):
            left = time.monotonic()-start
            metrics = api.metrics()
            right = time.monotonic()-start
            observations.append(dict(left=left, right=right,
                                     metrics={k:metrics[k] for k in (DRAFTS, DRAFT_TOKENS, ACCEPTED, RUNNING)}))
            time.sleep(.25)
        rows = [f.result() for f in futures]
    after = idle(api)
    delta = {k:v-before.get(k, 0) for k,v in after.items()}
    errors = []
    def check(condition, message):
        if not condition:
            errors.append(message)
    check(all(r['usage']['completion_tokens'] == 256 and r['usage']['prompt_tokens'] == item['prompt_tokens'] for r in rows), 'Token accounting differs')
    check(delta['vllm:num_preemptions_total'] == 0, 'Preemption occurred')
    check(delta['vllm:prefix_cache_queries_total'] == count*item['prompt_tokens'], 'Unexpected traffic or unsettled prefix counters')
    peak = max(o['metrics'][RUNNING] for o in observations)
    check(peak == count, 'Requested concurrency was not observed')
    # Bound the sample strictly inside client-observed simultaneous decoding.
    lower = max(first) + 1.0
    upper = min(r['finished_from_origin'] for r in rows) - .5
    steady = [o for o in observations if o['left'] >= lower and o['right'] <= upper]
    check(len(steady) >= 10, 'Insufficient observations inside steady overlap')
    fit = None
    if len(steady) >= 2:
        check(all(o['metrics'][RUNNING] == count for o in steady), 'Native concurrency changed inside overlap')
        check(len({o['metrics'][DRAFTS] for o in steady}) >= 10, 'Draft counters did not refresh frequently enough')
        first_obs, last_obs = steady[0], steady[-1]
        duration = (last_obs['left']+last_obs['right']-first_obs['left']-first_obs['right'])/2
        error_seconds = (last_obs['right']-last_obs['left']+first_obs['right']-first_obs['left'])/2
        drafts = last_obs['metrics'][DRAFTS]-first_obs['metrics'][DRAFTS]
        tokens = last_obs['metrics'][DRAFT_TOKENS]-first_obs['metrics'][DRAFT_TOKENS]
        accepted = last_obs['metrics'][ACCEPTED]-first_obs['metrics'][ACCEPTED]
        check(duration >= 2 and error_seconds/duration <= .01, 'Overlap timing too short or uncertain')
        check(drafts > 0 and drafts % count == 0, 'Draft counts do not form complete constant-size batches')
        check(tokens == drafts*depth and 0 <= accepted <= tokens, 'Fixed-depth counters disagree')
        if drafts > 0 and duration > 0:
            x = [(o['left']+o['right'])/2 for o in steady]
            y = [o['metrics'][DRAFTS] for o in steady]
            slope = statistics.linear_regression(x, y).slope
            r_squared = statistics.correlation(x, y)**2
            endpoint_step = 1000*duration*count/drafts
            regression_step = 1000*count/slope
            check(r_squared >= .98 and abs(regression_step/endpoint_step-1) <= .05,
                  'Draft-counter slope is inconsistent or nonstationary')
            fit = dict(duration_seconds=duration, timing_uncertainty_seconds=error_seconds,
                       drafts=drafts, draft_tokens=tokens, accepted=accepted,
                       batch_steps=drafts/count, step_ms=regression_step,
                       endpoint_step_ms=endpoint_step, counter_slope_r_squared=r_squared,
                       accepted_per_wall_second=accepted/duration,
                       observations=len(steady), first=first_obs, last=last_obs)
    elapsed = max(r['finished_from_origin'] for r in rows)
    full_drafts = delta[DRAFTS]
    return dict(requests=count, peak_running=peak, rows=rows, metrics_delta=delta,
                observations=observations, steady_window=fit, eligible=not errors,
                eligibility_errors=errors, elapsed_seconds=elapsed,
                aggregate_output_tok_s=sum(r['usage']['completion_tokens'] for r in rows)/elapsed,
                full_run_step_ms_estimate=1000*delta['vllm:request_decode_time_seconds_sum']/full_drafts if full_drafts else None,
                identical_clone_outputs=len({(r['answer_sha256'],r['reasoning_sha256']) for r in rows}) == 1,
                min_available_gib=None,
                method='Least-squares slope of fixed-depth native draft counters during simultaneous client decode, trimmed1s at the start and0.5s at the end. At least10 distinct counter samples, constant native concurrency, complete batches, R2>=.98, endpoint agreement within5%, and<=1% polling timing uncertainty required.',
                limitation='Wall time includes target, draft and scheduling. Unprofiled API polling is present. Matching prompts can produce different continuations; this is a timing calibration, not output-equivalence evidence.')
