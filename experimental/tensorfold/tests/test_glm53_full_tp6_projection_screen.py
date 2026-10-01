"""A candidate rejected by any rank must never reach full-model qualification."""
import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from glm53_tp6_projection_screen import choose,DECODE,PREFILL,tile_key,GENERATIONS

class Screening(unittest.TestCase):
    def rows(self):
        return [dict(rank=i,**{mode:{tile_key(t):dict(exact=True,score_ms=20.+j) for j,t in enumerate(tiles)}
                            for mode,tiles in [('decode',DECODE),('prefill',PREFILL)]}) for i in range(6)]
    def test_all_rank_exact_and_slowest_rank_cost(self):
        rows=self.rows();fast=tile_key(DECODE[1])
        for r in rows:r['decode'][fast]['score_ms']=2.
        self.assertEqual(choose(rows)['decode']['n'],32)
        rows[5]['decode'][fast]['exact']=False
        self.assertEqual(choose(rows)['decode']['n'],64)
        rows[5]['decode'][fast].update(exact=True,score_ms=50.)
        self.assertEqual(choose(rows)['decode']['n'],64)
    def test_actual_probe_generations_are_admitted_and_isolated(self):
        from unittest.mock import Mock
        from tensorfold.families.glm_moe_dsa.doorbell import RequestDoorbell
        store=Mock()
        bells=[RequestDoorbell(store,0,generation=GENERATIONS[name]) for name in ('reference','candidate')]
        self.assertNotEqual(bells[0]._key(1,1),bells[1]._key(1,1))
        for bell in bells:self.assertEqual(bell.publish(),1)
        self.assertEqual(store.set.call_count,10)

    def test_missing_ranks_and_nonfinite_cost_fail(self):
        rows=self.rows()
        for bad in [rows[:-1],rows[:-1]+[rows[0]]]:
            with self.assertRaises(ValueError):choose(bad)
        rows[0]['prefill'][tile_key(PREFILL[0])]['score_ms']=float('nan')
        with self.assertRaises(ValueError):choose(rows)

if __name__=='__main__':unittest.main()
