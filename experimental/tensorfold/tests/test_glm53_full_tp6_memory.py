"""Memory accounting must count retained storage, including list-held trellises."""
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import torch

from tensorfold.families.glm_moe_dsa.config import Config
from tensorfold.families.glm_moe_dsa.memory import tensor_storage_bytes,workspace_plan


class MemoryAccounting(unittest.TestCase):
    def test_views_aliases_and_nested_trellis_list_count_once(self):
        for device in ('cpu','meta'):
            x=torch.empty(123,dtype=torch.int16,device=device)
            y=torch.empty(71,dtype=torch.float32,device=device)
            obj=NS(keep=[x,x[12:],{'other':y}],alias=x.reshape(3,41),weights=NS(weight=torch.empty(19,device=device)))
            self.assertEqual(tensor_storage_bytes(obj,skip=('weights',)),123*2+71*4)
            self.assertEqual(tensor_storage_bytes(obj,device_type='cuda'),0)
            obj.loop=obj
            self.assertEqual(tensor_storage_bytes(obj,skip=('weights',)),123*2+71*4)

    def test_actual_scratch_shapes_and_cache_payload_scale_separately(self):
        cfg=Config.from_dict(json.loads((Path(__file__).parent/'fixtures/glm53_tp6/config.json').read_text()))
        small=workspace_plan(cfg,0,17,4096)
        long=workspace_plan(cfg,0,3072,360000)
        pooled=workspace_plan(cfg,0,3072,804000)
        self.assertEqual(long['cache'],360000*(79*656+22*132))
        self.assertEqual(pooled['cache']-long['cache'],444000*54728)
        self.assertEqual(long['collective'],6*3072*6144*2)
        self.assertGreater(long['workspace'],small['workspace'])
        self.assertLess(long['workspace'],1024**3)
        self.assertEqual(long['total'],sum(long[k] for k in ('workspace','cache','collective','rope')))


if __name__=='__main__':unittest.main()
