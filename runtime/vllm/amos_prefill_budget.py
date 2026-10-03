"""Bounded CPU scheduling experiment; preserves the 3072-token kernel arena.

Unqualified until original-reference short/long tails and mixed-load tests pass.
The control file is explicit and latched only between batches of live requests.
No fallback configuration, model launch, target sampler or draft policy changes.
"""
import hashlib
import json
import os
from pathlib import Path
import re

CONTROL = Path('/root/.cache/amos-tp6-prefill-budget.json')


def validate(control):
    if not isinstance(control, dict) or set(control) != {'revision', 'mode', 'budget'}:
        raise ValueError('Budget control requires revision, mode and budget only')
    if not isinstance(control['revision'], str) or not re.fullmatch(r'[A-Za-z0-9_.-]{1,80}', control['revision']):
        raise ValueError('Invalid budget control revision')
    if control['mode'] not in ('baseline', 'fixed', 'adaptive'):
        raise ValueError('Unqualified budget mode')
    if type(control['budget']) is not int or control['budget'] not in (768, 1536, 3072):
        raise ValueError('Budget must be one of 768,1536,3072')
    if control['mode'] == 'baseline' and control['budget'] != 3072:
        raise ValueError('Baseline must preserve the current3072 budget')
    return control


def check_config(scheduler):
    config = scheduler.vllm_config
    p, s = config.parallel_config, config.scheduler_config
    if not (p.tensor_parallel_size == 6 and p.pipeline_parallel_size == 1
            and p.data_parallel_size == 1 and s.max_num_seqs == 4
            and s.max_num_batched_tokens == 3072 and scheduler.max_num_scheduled_tokens == 3072
            and config.model_config.max_model_len == 360000
            and config.model_config.hf_config.hidden_size == 6144
            and config.speculative_config.num_speculative_tokens == 4):
        raise ValueError('Budget experiment requires the qualified full GLM TP6 configuration')


def select(scheduler, path=CONTROL):
    if os.getenv('AMOS_TP6_PREFILL_BUDGET_CONTROL', '0') != '1':
        return scheduler.max_num_scheduled_tokens
    current = getattr(scheduler, 'amos_prefill_budget_control', None)
    if current is None or not scheduler.running:
        check_config(scheduler)
        raw = path.read_bytes()
        if len(raw) > 1024:
            raise ValueError('Oversized scheduler control file')
        digest = hashlib.sha256(raw).hexdigest()
        if digest != getattr(scheduler, 'amos_prefill_budget_digest', None):
            current = validate(json.loads(raw))
            # Record the exact accepted revision before scheduling any requests.
            out = path.with_suffix('.applied.json')
            tmp = out.with_suffix('.tmp')
            tmp.write_text(json.dumps(dict(current, sha256=digest, scheduler_pid=os.getpid()))+'\n')
            tmp.replace(out)
            scheduler.amos_prefill_budget_control = current
            scheduler.amos_prefill_budget_digest = digest
    if current['mode'] == 'baseline':
        return 3072
    if current['mode'] == 'fixed':
        return current['budget']
    decoders = [r for r in scheduler.running if not r.is_finished()
                and r.num_computed_tokens >= r.num_prompt_tokens]
    return current['budget'] if decoders else 3072
