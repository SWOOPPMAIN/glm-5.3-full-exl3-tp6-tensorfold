import json,unittest
from pathlib import Path
from types import SimpleNamespace as NS
import torch
from tensorfold.families.glm_moe_dsa.experts import RoutedLayer,validate_chunk_rows
from tensorfold.families.glm_moe_dsa.config import Config
from tensorfold.families.glm_moe_dsa.memory import workspace_plan


class ExpertChunkGeometry(unittest.TestCase):
    def test_requested_chunks_reach_grouping_arena_and_tail_window(self):
        layer=RoutedLayer(NS(dims=6144,width=512,count=171),torch.empty(256,dtype=torch.int32,device='meta'))
        for chunk in (128,256,512,1024):
            scratch=layer.scratch(3072,chunk_rows=chunk)
            self.assertEqual(scratch.chunk_rows,chunk)
            self.assertEqual(scratch.kernel.rows,chunk)
            self.assertEqual(scratch.local_ids.shape,(chunk,8))
            ids,members=scratch.kernel.window(17)
            self.assertEqual(ids.shape,(136,));self.assertEqual(members.shape,(136,17))
            short=layer.scratch(4,chunk_rows=chunk)
            self.assertEqual(short.kernel.rows,4)
            self.assertLess(chunk*8*4+128,48*1024)
        for bad in (True,0,64,1536,3072,128.0):
            with self.assertRaises(ValueError):validate_chunk_rows(bad)

    def test_memory_plan_changes_scratch_without_changing_cache_or_model_rows(self):
        cfg=Config.from_dict(json.loads(Path('tests/fixtures/glm53_tp6/config.json').read_text()))
        old=workspace_plan(cfg,0,3072,804000)
        for chunk in (256,512,1024):
            new=workspace_plan(cfg,0,3072,804000,expert_chunk_rows=chunk)
            self.assertGreater(new['workspace'],old['workspace'])
            self.assertLess(new['workspace']-old['workspace'],600*2**20)
            self.assertEqual(new['cache'],old['cache']);self.assertEqual(new['collective'],old['collective'])
            self.assertEqual(new['rows'],3072);self.assertEqual(new['expert_chunk_rows'],chunk)


if __name__=='__main__':unittest.main()
