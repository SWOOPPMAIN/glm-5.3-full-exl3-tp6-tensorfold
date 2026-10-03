#!/usr/bin/env python3
"""Install the budget hook only onto the exact qualified P27 scheduler."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

RELATIVE = 'v1/core/sched/scheduler.py'
P27_SHA256 = '792c55448736f6e89ce14f4baaa5005718ec5640c57d2da1c0d10668c2925abd'


def patch(root):
    path = root/RELATIVE
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != P27_SHA256:
        raise ValueError('Scheduler differs from qualified P27; refusing an unreviewed patch')
    source = raw.decode()
    before = '        token_budget = self.max_num_scheduled_tokens\n'
    after = '        from vllm import amos_prefill_budget\n        token_budget = amos_prefill_budget.select(self)\n'
    if source.count(before) != 1:
        raise ValueError('Scheduler budget assignment is not unique')
    source = source.replace(before, after)
    compile(source, str(path), 'exec')
    path.write_text(source)
    shutil.copy2(Path(__file__).with_name('amos_prefill_budget.py'), root/'amos_prefill_budget.py')
    return {RELATIVE: hashlib.sha256(path.read_bytes()).hexdigest()}


if __name__ == '__main__':
    print(json.dumps(patch(Path(sys.argv[1]))))
