"""Host metadata sent to the real backend, checked without launching CUDA."""
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock

import torch

from tensorfold.families.glm_moe_dsa.memory import request_plan
from tensorfold.families.glm_moe_dsa.request import Extent
from tensorfold.families.glm_moe_dsa.request_backend import FullModelBackend


class BackendContracts(unittest.TestCase):
    def backend(self):
        # Only the CUDA stream/device check is substituted. The actual adapter
        # constructs inputs and delegates to the model with these CPU tensors.
        b = FullModelBackend.__new__(FullModelBackend)
        b._stream = Mock()
        b.capacity, b.vocab, b.rows, b.logit_rows = 804000, 154880, 3072, 17
        b.device = torch.device('cpu')
        b.model = NS(target_forward=Mock(), mtp_forward=Mock())
        b.caches, b.table, b.workspace = tuple(object() for _ in range(79)), object(), object()
        return b

    def test_logical_positions_and_visible_bound_ignore_physical_offset(self):
        b = self.backend()
        ids, positions, bases, slots, bound = b._inputs([123, 456], 359997, Extent(440000, 360000, 1))
        self.assertEqual(ids.tolist(), [123, 456])
        self.assertEqual(positions.tolist(), [359997, 359998])
        self.assertEqual(bases.tolist(), [440000, 440000])
        self.assertEqual(slots.tolist(), [799997, 799998])
        self.assertEqual(bound, 359999)
        self.assertTrue(all(t.is_contiguous() and t.dtype == torch.int64 for t in (ids, positions, bases, slots)))

    def test_outside_extent_invalid_ids_and_oversized_rows_are_rejected(self):
        b = self.backend()
        for ids, start, extent in [([3], 5, Extent(10, 5, 1)), ([3], -1, Extent(10, 5, 1)),
                                   ([3], 0, Extent(803999, 2, 1)), ([3], 0, Extent(-1, 3, 1)),
                                   ([3]*3073, 0, Extent(0, 10000, 1)),
                                   ([154880], 0, Extent(0, 10, 1)), ([True], 0, Extent(0, 10, 1))]:
            with self.assertRaises(ValueError):
                b.target(ids, start, extent)
        b.model.target_forward.assert_not_called()

    def test_target_and_draft_use_distinct_cache_owners_and_new_scopes(self):
        b = self.backend()
        e = Extent(1234, 100, 1)
        hidden = torch.zeros((2, 6144), dtype=torch.bfloat16)
        b.target([1, 2], 13, e)
        b.mtp([2, 3], hidden, 13, e)
        target, mtp = b.model.target_forward.call_args, b.model.mtp_forward.call_args
        self.assertEqual(target.args[4], b.caches[:78])
        self.assertIs(mtp.args[1], hidden)
        self.assertIs(mtp.args[5], b.caches[78])
        self.assertEqual(target.kwargs['visible_tokens'], 15)
        self.assertEqual(mtp.kwargs['visible_tokens'], 15)
        self.assertIsNot(target.kwargs['scope'], mtp.kwargs['scope'])

    def test_request_reserve_bounds_all_four_retained_chunks_and_sampler(self):
        plan = request_plan(3072, 17)
        held = [torch.empty((3072+17, 6144), dtype=torch.bfloat16, device='meta') for _ in range(4)]
        self.assertGreater(plan['hidden_retained'], sum(t.numel()*t.element_size() for t in held))
        self.assertLess(plan['total'], 2**30)
        for args in [(1, 2), (3073, 17), (True, 1), (3072, 129)]:
            with self.assertRaises(ValueError):
                request_plan(*args)


if __name__ == '__main__':
    unittest.main()
