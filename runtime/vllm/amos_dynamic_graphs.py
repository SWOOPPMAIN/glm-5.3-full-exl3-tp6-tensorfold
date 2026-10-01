"""Exact target CUDA graph shapes for full GLM TP6 dynamic MTP.

This module does not change sampling, weights, or the draft model's dispatcher.
The launch flag is deliberately separate from enabling adaptive scheduling.
"""
import hashlib
import json
import os
from pathlib import Path

CALIBRATION_PATH = Path('/root/.cache/amos-tp6-mtp-calibration.json')


def allow_dynamic_full_graphs(config):
    """Retain the requested FULL mode only for our exact-shape TP6 extension."""
    if os.environ.get('AMOS_TP6_DYNAMIC_GRAPHS','0') != '1':
        return False
    parallel = config.parallel_config
    if not (
        config.speculative_config.num_speculative_tokens == 4
        and parallel.tensor_parallel_size == 6
        and parallel.pipeline_parallel_size == 1
        and parallel.data_parallel_size == 1
        and not parallel.use_ubatching
        and not parallel.use_sequence_parallel_moe
        and not config.compilation_config.pass_config.enable_sp
        and config.scheduler_config.max_num_seqs == 4
        and config.lora_config is None
    ):
        raise ValueError('FULL dynamic TP6 graphs require the explicit MTP4/DP1/C4 layout')
    return True


def initialize(dispatcher, mode, query_len):
    dispatcher.amos_query_lengths = ()
    if os.environ.get('AMOS_TP6_DYNAMIC_GRAPHS', '0') != '1' or query_len == 1:
        return
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import BatchDescriptor

    config = dispatcher.vllm_config
    parallel = config.parallel_config
    if not (
        query_len == 5
        and parallel.tensor_parallel_size == 6
        and parallel.pipeline_parallel_size == 1
        and parallel.data_parallel_size == 1
        and not parallel.use_ubatching
        and not parallel.use_sequence_parallel_moe
        and not config.compilation_config.pass_config.enable_sp
        and config.scheduler_config.max_num_seqs == 4
        and config.lora_config is None
        and mode in (CUDAGraphMode.FULL_DECODE_ONLY, CUDAGraphMode.FULL_AND_PIECEWISE)
        and config.compilation_config.max_cudagraph_capture_size >= 20
    ):
        raise ValueError('Dynamic TP6 graphs require the qualified MTP4/DP1/C4 configuration: '
            + str({'mode':str(mode),'query_len':query_len,'tp':parallel.tensor_parallel_size,
                   'dp':parallel.data_parallel_size,'pp':parallel.pipeline_parallel_size,
                   'max_num_seqs':config.scheduler_config.max_num_seqs,
                   'max_capture':config.compilation_config.max_cudagraph_capture_size,
                   'sp':config.compilation_config.pass_config.enable_sp,
                   'sp_moe':parallel.use_sequence_parallel_moe,'ubatching':parallel.use_ubatching,
                   'lora':config.lora_config is not None}))
    dispatcher.amos_query_lengths = tuple(range(1, query_len + 1))
    # Include q=1 for a final decode step with no scheduled draft tokens.
    # Distinguish e.g. 4 requests x 2 tokens from 2 requests x 4 tokens.
    for length in dispatcher.amos_query_lengths:
        for requests in range(1, 5):
            dispatcher.add_cudagraph_key(CUDAGraphMode.FULL, BatchDescriptor(
                num_tokens=length * requests, num_reqs=requests, uniform=True))


def actual_query_len(runner, max_scheduled):
    lengths = getattr(runner.cudagraph_dispatcher, 'amos_query_lengths', ())
    if max_scheduled in lengths:
        return max_scheduled
    return runner.uniform_decode_query_len


def exact_descriptor(dispatcher, num_tokens, query_len, uniform, has_lora):
    """Return an exact shape, or None to use the unmodified dispatcher path."""
    lengths = getattr(dispatcher, 'amos_query_lengths', ())
    if not lengths or not uniform:
        return None
    from vllm.forward_context import BatchDescriptor

    length = query_len if query_len is not None else dispatcher.uniform_decode_query_len
    if has_lora or length not in lengths or num_tokens <= 0 or num_tokens % length:
        raise ValueError('Invalid uniform dynamic TP6 graph shape')
    requests = num_tokens // length
    if not 1 <= requests <= 4:
        raise ValueError('Dynamic TP6 graph request count is outside 1..4')
    return BatchDescriptor(num_tokens=num_tokens, num_reqs=requests, uniform=True)


def capture_query_len(runner, descriptor):
    if not getattr(runner.cudagraph_dispatcher, 'amos_query_lengths', ()) or not descriptor.uniform:
        return None
    if not descriptor.num_reqs or descriptor.num_tokens % descriptor.num_reqs:
        raise ValueError('Invalid dynamic TP6 capture descriptor')
    length = descriptor.num_tokens // descriptor.num_reqs
    if length not in runner.cudagraph_dispatcher.amos_query_lengths:
        raise ValueError('Unregistered dynamic TP6 capture query length')
    return length


def apply_calibration(scheduler):
    """A maintenance-only fixed-depth sweep, applied between requests.

    The flag is disabled for normal serving. A single CPU scheduler selects K
    through the existing dynamic schedule; worker ranks receive the same K.
    No request payload, sampling rule, or target weight changes here.
    """
    if os.environ.get('AMOS_MTP_CALIBRATION', '0') != '1' or scheduler.running:
        return
    if (scheduler.num_spec_tokens != 4 or scheduler.dynamic_sd_lookup is None
            or scheduler.acceptance_length_controller is not None
            or scheduler.scheduler_config.max_num_seqs != 4
            or os.environ.get('AMOS_TP6_DYNAMIC_GRAPHS') != '1'):
        raise ValueError('Calibration requires MTP4, dynamic lookup, C4 and all-depth graphs')
    raw = CALIBRATION_PATH.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if getattr(scheduler, 'amos_calibration_digest', None) == digest:
        return
    control = json.loads(raw)
    if not isinstance(control, dict) or set(control) != {'revision','depth'}:
        raise ValueError('Calibration control must contain only revision and depth')
    revision, depth = control['revision'], control['depth']
    if not isinstance(revision, str) or not revision or len(revision) > 80:
        raise ValueError('Invalid calibration revision')
    if type(depth) is not int or not 1 <= depth <= 4:
        raise ValueError('Calibration depth must be an integer in 1..4')
    scheduler.dynamic_sd_lookup = [depth] * 5
    scheduler.amos_calibration_digest = digest
    receipt = dict(control, sha256=digest, scheduler_pid=os.getpid())
    target = CALIBRATION_PATH.with_suffix('.applied.json')
    tmp = target.with_suffix('.tmp')
    tmp.write_text(json.dumps(receipt, sort_keys=True)+'\n')
    tmp.replace(target)
