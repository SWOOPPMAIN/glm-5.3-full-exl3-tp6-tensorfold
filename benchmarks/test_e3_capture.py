"""CPU checks for the diagnostic's bounded, inactive-by-default behavior."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"runtime/vllm"))
import amos_e3_capture as capture


class Bounds(unittest.TestCase):
    def control(self):
        return dict(schema=1, revision='e31-prose', input_layers=[3,40,77], rows=3072,
                    max_bytes_per_rank=capture.MAX_BYTES, armed_at=100, expires_at=180)

    def test_reject_unbounded_or_unsafe_control(self):
        for changes in ({'expires_at':281}, {'expires_at':float('inf')},
                        {'rows':3073}, {'revision':'../outside'}, {'input_layers':list(range(78))},
                        {'max_bytes_per_rank':capture.MAX_BYTES+1}, {'armed_at':True}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                capture.validate(dict(self.control(), **changes))

    def test_expiry_checked_even_between_polls(self):
        with patch.object(capture, '_control', self.control()), patch.object(capture, '_checked', 20), \
                patch.object(capture.time, 'monotonic', return_value=20.1):
            for wall, active in ((99,False),(100,True),(179.9,True),(180,False)):
                with patch.object(capture.time,'time',return_value=wall):
                    self.assertEqual(capture.read_control() is not None, active)

    def test_missing_control_never_accesses_cuda_or_tensors(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(capture, 'CONTROL', Path(directory)/'absent'), \
                patch.object(capture, '_checked', 0):
            # The only provided tensor attribute is shape. Any data/CUDA access
            # would fail here; disabled capture must return cleanly.
            capture.capture(None, SimpleNamespace(shape=(3072,6144)), None,None,None)
            self.assertIsNone(capture.read_control())

    def test_actual_layer_name_pattern(self):
        for name in ('model.layers.3.mlp.experts','model.layers.40.mlp.experts'):
            self.assertIsNotNone(capture.PATTERN.search(name))
        self.assertIsNone(capture.PATTERN.search('model.layers.3.mlp.shared_experts'))


if __name__ == '__main__': unittest.main()
