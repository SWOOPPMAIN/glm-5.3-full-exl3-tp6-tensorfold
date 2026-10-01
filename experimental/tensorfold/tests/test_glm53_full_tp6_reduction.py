"""Admission and uneven-shard geometry; GPU qualification uses six real ranks."""
import json
from pathlib import Path
import unittest

from tensorfold.families.glm_moe_dsa.config import Config
from tensorfold.families.glm_moe_dsa.memory import workspace_plan
from tensorfold.families.glm_moe_dsa.reduction_plan import reduction_plan


class ReductionPlan(unittest.TestCase):
    def test_uneven_rows_fit_six_equal_shards_and_existing_gather_storage(self):
        for rows in (1,5,6,7,17,127,128,129,255,256,257,2053,3071,3072):
            p=reduction_plan(rows,bulk_min_rows=1)
            self.assertGreaterEqual(p['padded_rows'],rows)
            self.assertLess(p['padded_rows']-rows,6)
            self.assertEqual(p['padded_rows']%6,0)
            self.assertLessEqual(p['padded_rows'],p['gather_rows'])
            # Every real row has exactly one owner; padding belongs only at end.
            owners=[list(range(r*p['shard_rows'],min((r+1)*p['shard_rows'],rows))) for r in range(6)]
            self.assertEqual([row for shard in owners for row in shard],list(range(rows)))

    def test_bulk_admission_counts_send_and_sum_buffers(self):
        cfg=Config.from_dict(json.loads(Path('tests/fixtures/glm53_tp6/config.json').read_text()))
        base=workspace_plan(cfg,0,3072,804000,expert_chunk_rows=1024)
        bulk=workspace_plan(cfg,0,3072,804000,expert_chunk_rows=1024,bulk_min_rows=256)
        self.assertEqual(bulk['collective']-base['collective'],42*2**20)
        self.assertEqual(bulk['total']-base['total'],42*2**20)
        for key in ('workspace','cache','rope'):self.assertEqual(base[key],bulk[key])
        for threshold in (None,256):
            short=reduction_plan(17,bulk_min_rows=threshold)
            self.assertEqual(short['total'],6*17*6144*2)
            self.assertEqual(short['shard_rows'],0)

    def test_invalid_sizes_and_thresholds_fail_before_allocation(self):
        for bad in (True,0,3073,1.0,'256'):
            with self.assertRaises(ValueError):reduction_plan(bad)
            with self.assertRaises(ValueError):reduction_plan(3072,bulk_min_rows=bad)


if __name__=='__main__':unittest.main()
