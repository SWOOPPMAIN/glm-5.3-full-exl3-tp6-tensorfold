"""CPU checks for required controls and per-prefill-chunk policy latching."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import amos_e3_policy as policy


class Policy(unittest.TestCase):
    def test_reject_unsupported_or_implicit_choices(self):
        for value in ({}, {'rows':32}, {'revision':'e33-a','rows':16},
                      {'revision':'e33-a','rows':32.0}, {'revision':'../a','rows':32},
                      {'revision':'e33-a','rows':32,'fallback':64}):
            with self.subTest(value=value),self.assertRaises(ValueError):policy.validate(value)

    def test_latch_at_first_moe_layer_and_missing_file_is_not_fallback(self):
        with tempfile.TemporaryDirectory() as d,patch.object(policy,'_current',None),patch.object(policy,'_digest',None):
            path=Path(d)/'rows.json'
            with self.assertRaises(FileNotFoundError):policy.latch('model.layers.3.mlp.experts',path)
            path.write_text(json.dumps(dict(revision='e33-control',rows=64)))
            self.assertEqual(policy.latch('model.layers.3.mlp.experts',path)['rows'],64)
            path.write_text(json.dumps(dict(revision='e33-candidate',rows=32)))
            self.assertEqual(policy.latch('model.layers.40.mlp.experts',path)['rows'],64)
            self.assertEqual(policy.latch('model.layers.3.mlp.experts',path)['rows'],32)
            path.unlink()
            with self.assertRaises(FileNotFoundError):policy.latch('model.layers.3.mlp.experts',path)


if __name__=='__main__':unittest.main()
