#!/usr/bin/env python3
"""Avoid treating ngram's target metadata alias as an independent draft model.

The full VllmConfig still pads and verifies the target. Model-backed proposers
retain their existing virtual-TP and independent draft validation unchanged.
"""
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

SOURCE='807a2e4573a8c6002d1748cf4cf82706b564dc317058c6988c8bbc2d809e2d4e'
OLD='''        if self.draft_model_config:
            self._maybe_apply_virtual_tp_to_draft()
            self.draft_model_config.verify_with_parallel_config(
                self.draft_parallel_config
            )
'''
NEW='''        if self.draft_model_config:
            if self.method in ("ngram", "ngram_gpu"):
                # Copy proposers alias target metadata to obtain vocab size;
                # they have no independent attention model. Target virtual-TP
                # padding and validation run later in VllmConfig.__post_init__.
                if (
                    self.draft_model_config is not self.target_model_config
                    or self.draft_parallel_config is not self.target_parallel_config
                ):
                    raise ValueError("Copy proposer must alias target configuration")
            else:
                self._maybe_apply_virtual_tp_to_draft()
                self.draft_model_config.verify_with_parallel_config(
                    self.draft_parallel_config
                )
'''


def digest(raw):return hashlib.sha256(raw).hexdigest()


def patch_source(raw):
    assert digest(raw)==SOURCE
    text=raw.decode();assert text.count(OLD)==1
    result=text.replace(OLD,NEW).encode();compile(result,'speculative.py','exec')
    return result


def actual_validator(raw):
    tree=ast.parse(raw)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SpeculativeConfig')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_verify_args')
    method.decorator_list=[];method.returns=None
    scope={'math':math,'os':os}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),'actual-_verify_args','exec'),scope)
    return scope['_verify_args']


def checks(source,candidate):
    rows=[]
    for patched,raw in ((False,source),(True,candidate)):
        method=actual_validator(raw)
        for kind in ('ngram','ngram_gpu','mtp','eagle','draft_model'):
            for alias in (False,True):
                for divisible in (False,True):
                    events=[];parallel=object()
                    def verify(value):
                        events.append('draft_validation')
                        if not divisible:raise ValueError('heads64 not divisible by TP6')
                    model=SimpleNamespace(verify_with_parallel_config=verify)
                    obj=SimpleNamespace(tensor_parallel_size=None,num_speculative_tokens=4,
                        rejection_sample_method='standard',synthetic_acceptance_rates=None,synthetic_acceptance_length=None,
                        dspark_confidence_threshold=0.,dspark_budget_frac=1.,dspark_confidence_temperature=1.,
                        dspark_sps_overhead_ms=0.,dspark_sps_curve=None,method=kind,
                        dspark_capacity_verification_mode='compact',draft_model_config=model,
                        target_model_config=model if alias else object(),draft_parallel_config=parallel,
                        target_parallel_config=parallel,use_heterogeneous_vocab=False,
                        _maybe_apply_virtual_tp_to_draft=lambda:events.append('virtual_tp'),
                        verify_equal_vocab_size_if_draft_model=lambda:events.append('vocab_check'))
                    try:method(obj);passed=True
                    except ValueError:passed=False
                    expected=alias if patched and kind in ('ngram','ngram_gpu') else divisible
                    assert passed==expected
                    if patched and kind in ('ngram','ngram_gpu'):
                        assert 'draft_validation' not in events and 'virtual_tp' not in events
                    else:assert events[:2]==['virtual_tp','draft_validation']
                    rows.append(dict(patched=patched,method=kind,aliases_target=alias,
                        draft_heads_divisible=divisible,accepted=passed,events=events))
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('source','candidate','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();assert not a.candidate.exists() and not a.output.exists()
    raw=a.source.read_bytes();candidate=patch_source(raw);rows=checks(raw,candidate)
    a.candidate.parent.mkdir(parents=True,exist_ok=True);a.candidate.write_bytes(candidate)
    result=dict(phase='source_validator_checked_cpu_only',source_sha256=digest(raw),candidate_sha256=digest(candidate),
        checks=rows,passed=True,gpu_used=False,target_validation_changed=False,
        limitation='Runs the actual extracted validator with configuration doubles. Full pinned Pydantic construction, TP6 startup and target fidelity remain runtime gates.')
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('phase','source_sha256','candidate_sha256','passed')}))
