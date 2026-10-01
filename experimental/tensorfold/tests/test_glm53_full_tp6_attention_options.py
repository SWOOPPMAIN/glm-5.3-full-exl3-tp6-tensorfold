"""Attention scratch admission and option propagation; GPU exactness is separate."""
import json
from pathlib import Path
import unittest
from tensorfold.families.glm_moe_dsa.attention import AttentionScratch,validate_attention_options
from tensorfold.families.glm_moe_dsa.config import Config
from tensorfold.families.glm_moe_dsa.memory import workspace_plan,tensor_storage_bytes


class AttentionOptions(unittest.TestCase):
    def test_bounded_partials_account_for_all_heads_and_merge_fields(self):
        for rows in (1,17,129,2053,3072):
            for part in (128,256,512,1024):
                s=AttentionScratch(rows,'meta',9,part_rows=part,skip_empty=True)
                n=min(rows,part)
                self.assertEqual(s.part_rows,n)
                self.assertTrue(s.skip_empty)
                self.assertEqual(s.po.shape,(4*n*11,512))
                self.assertEqual(tensor_storage_bytes(s),4*n*11*(512+2)*4)
        for bad in (True,0,127,2048,128.0):
            with self.assertRaises(ValueError):validate_attention_options(bad,False)
        for bad in (None,0,1,'true'):
            with self.assertRaises(ValueError):validate_attention_options(128,bad)

    def test_model_memory_plan_passes_options_without_expanding_kv_pool(self):
        cfg=Config.from_dict(json.loads(Path('tests/fixtures/glm53_tp6/config.json').read_text()))
        old=workspace_plan(cfg,0,3072,804000,expert_chunk_rows=1024,bulk_min_rows=256)
        for part in (128,256,512,1024):
            new=workspace_plan(cfg,0,3072,804000,expert_chunk_rows=1024,bulk_min_rows=256,
                               attention_part_rows=part,skip_empty_attention=True)
            extra=4*(part-128)*11*(512+2)*4
            self.assertEqual(new['workspace']-old['workspace'],extra)
            self.assertEqual(new['total']-old['total'],extra)
            self.assertLess(extra,80*2**20)
            self.assertTrue(new['skip_empty_attention'])
            self.assertEqual(new['attention_part_rows'],part)
            for key in ('cache','collective','rope'):self.assertEqual(new[key],old[key])


if __name__=='__main__':unittest.main()
