"""CPU safety invariants for a pending scheduler experiment, not GPU fidelity."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import amos_prefill_budget as policy


def scheduler():
    return NS(max_num_scheduled_tokens=3072, running=[], vllm_config=NS(
        parallel_config=NS(tensor_parallel_size=6, pipeline_parallel_size=1, data_parallel_size=1),
        scheduler_config=NS(max_num_seqs=4, max_num_batched_tokens=3072),
        model_config=NS(max_model_len=360000, hf_config=NS(hidden_size=6144)),
        speculative_config=NS(num_speculative_tokens=4)))


class BudgetTests(unittest.TestCase):
    def test_control_cannot_change_mid_request(self):
        with TemporaryDirectory() as tmp, patch.dict('os.environ', AMOS_TP6_PREFILL_BUDGET_CONTROL='1'):
            p, s = Path(tmp)/'budget.json', scheduler()
            p.write_text(json.dumps(dict(revision='control', mode='adaptive', budget=1536)))
            self.assertEqual(policy.select(s, p), 3072)
            s.running = [NS(num_computed_tokens=4096, num_prompt_tokens=4096, is_finished=lambda: False)]
            p.write_text(json.dumps(dict(revision='next', mode='fixed', budget=768)))
            self.assertEqual(policy.select(s, p), 1536)
            self.assertEqual(json.loads(p.with_suffix('.applied.json').read_text())['revision'], 'control')
            s.running.clear()
            self.assertEqual(policy.select(s, p), 768)
            self.assertEqual(json.loads(p.with_suffix('.applied.json').read_text())['revision'], 'next')

    def test_stale_finished_requests_do_not_throttle_isolated_prefill(self):
        with TemporaryDirectory() as tmp, patch.dict('os.environ', AMOS_TP6_PREFILL_BUDGET_CONTROL='1'):
            p, s = Path(tmp)/'budget.json', scheduler()
            p.write_text(json.dumps(dict(revision='adaptive', mode='adaptive', budget=768)))
            policy.select(s, p)
            s.running = [NS(num_computed_tokens=100, num_prompt_tokens=100, is_finished=lambda: True),
                         NS(num_computed_tokens=100, num_prompt_tokens=32000, is_finished=lambda: False)]
            self.assertEqual(policy.select(s, p), 3072)

    def test_invalid_control_preserves_previous_acknowledgement(self):
        with TemporaryDirectory() as tmp, patch.dict('os.environ', AMOS_TP6_PREFILL_BUDGET_CONTROL='1'):
            p, s = Path(tmp)/'budget.json', scheduler()
            p.write_text(json.dumps(dict(revision='control', mode='baseline', budget=3072)))
            policy.select(s, p)
            old = p.with_suffix('.applied.json').read_bytes()
            p.write_text(json.dumps(dict(revision='bad', mode='fixed', budget=32768)))
            with self.assertRaises(ValueError):
                policy.select(s, p)
            self.assertEqual(p.with_suffix('.applied.json').read_bytes(), old)
            self.assertEqual(s.amos_prefill_budget_control['revision'], 'control')

    def test_unqualified_configuration_and_missing_control_refuse(self):
        with TemporaryDirectory() as tmp, patch.dict('os.environ', AMOS_TP6_PREFILL_BUDGET_CONTROL='1'):
            p, s = Path(tmp)/'missing.json', scheduler()
            with self.assertRaises(FileNotFoundError):
                policy.select(s, p)
            s.vllm_config.parallel_config.tensor_parallel_size = 2
            with self.assertRaises(ValueError):
                policy.select(s, p)


if __name__ == '__main__':
    unittest.main()
