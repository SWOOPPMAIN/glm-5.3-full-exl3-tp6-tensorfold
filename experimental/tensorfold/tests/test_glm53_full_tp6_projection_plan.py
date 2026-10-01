"""Projection ownership and admission; real arithmetic is a separate GPU gate."""
from dataclasses import FrozenInstanceError
from types import SimpleNamespace as NS
from concurrent.futures import ThreadPoolExecutor
import unittest
from unittest.mock import Mock,patch

from tensorfold.families.glm_moe_dsa.projection_plan import (
    LinearTile,ProjectionPlan,REFERENCE_PLAN,current_projection_plan,projection_scope)
from tensorfold.families.glm_moe_dsa.model import FullModel
from tensorfold.families.glm_moe_dsa.request_backend import FullModelBackend


class ProjectionContracts(unittest.TestCase):
    def test_bounds_immutable_and_threshold_selection(self):
        p=ProjectionPlan(LinearTile(16,32,3),LinearTile(64,128,2))
        self.assertEqual(p.tile(255),p.decode);self.assertEqual(p.tile(256),p.prefill)
        self.assertEqual(p.tile(3072),p.prefill)
        for n in (0,3073,True):
            with self.assertRaises(ValueError):p.tile(n)
        for args in [(True,64,2),(128,64,2),(16,256,2),(16,64,4)]:
            with self.assertRaises(ValueError):LinearTile(*args)
        with self.assertRaises(ValueError):ProjectionPlan(decode=LinearTile(32,64,2))
        with self.assertRaises(ValueError):ProjectionPlan(bulk_min_rows=17)
        with self.assertRaises(FrozenInstanceError):p.decode=LinearTile()

    def test_nested_scope_and_exception_restore_the_callers_plan(self):
        outer=ProjectionPlan(decode=LinearTile(16,32,2));inner=ProjectionPlan(decode=LinearTile(16,128,2))
        self.assertEqual(current_projection_plan(),REFERENCE_PLAN)
        with projection_scope(outer):
            with self.assertRaisesRegex(RuntimeError,'sentinel'):
                with projection_scope(inner):
                    self.assertIs(current_projection_plan(),inner)
                    raise RuntimeError('sentinel')
            self.assertIs(current_projection_plan(),outer)
        self.assertEqual(current_projection_plan(),REFERENCE_PLAN)

    def test_worker_scopes_do_not_share_a_mutable_global(self):
        first=ProjectionPlan(decode=LinearTile(16,16,2))
        second=ProjectionPlan(decode=LinearTile(16,128,3))
        with projection_scope(first),ThreadPoolExecutor(max_workers=1) as worker:
            self.assertEqual(worker.submit(current_projection_plan).result(),REFERENCE_PLAN)
            def use():
                with projection_scope(second):return current_projection_plan()
            self.assertEqual(worker.submit(use).result(),second)
            self.assertIs(current_projection_plan(),first)

    def test_full_model_head_and_target_own_their_projection_scope(self):
        m=FullModel.__new__(FullModel);m._projections=ProjectionPlan(decode=LinearTile(16,32,3))
        m.weights=object();seen=[]
        def see(*a,**kw):seen.append(current_projection_plan());return 'owned'
        m.vocab=NS(embed=see,project=see);m.target=NS(forward=see)
        arena=NS(weights=m.weights,rows=5,vocab=object(),decoder=object(),target_selection=object(),hidden=[0]*5,residual=[0]*5)
        with projection_scope(ProjectionPlan(decode=LinearTile(16,128,2))):
            outer=current_projection_plan()
            self.assertEqual(m.target_forward([1,2],None,None,None,None,None,arena,scope=object()),'owned')
            self.assertEqual(m.logits([1,2],arena),'owned')
            self.assertIs(current_projection_plan(),outer)
        self.assertEqual(seen,[m.projections]*3)
        with self.assertRaises(AttributeError):m.projections=REFERENCE_PLAN

    def test_backend_rejects_policy_drift_before_a_captured_call(self):
        b=FullModelBackend.__new__(FullModelBackend);b.device='cuda:0';b.stream=object()
        b.projection_plan=REFERENCE_PLAN;b.model=NS(projections=REFERENCE_PLAN)
        old=b.control_state()
        with patch('tensorfold.families.glm_moe_dsa.request_backend.torch.cuda.current_stream',return_value=b.stream):
            b._stream()
            b.model.projections=ProjectionPlan(decode=LinearTile(16,32,2))
            with self.assertRaises(RuntimeError):b._stream()
        self.assertEqual(b.control_state(),old)
        b.projection_plan=b.model.projections
        self.assertNotEqual(b.control_state(),old)

if __name__=='__main__':unittest.main()
