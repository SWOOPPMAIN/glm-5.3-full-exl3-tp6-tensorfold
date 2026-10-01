"""CPU checks for the full-model port's geometry and stored-byte reader."""
import copy
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import torch

from tensorfold.families.glm_moe_dsa.config import Config
from tensorfold.families.glm_moe_dsa.checkpoint import RankPieces, Fragment

FIXTURE = Path(__file__).parent / "fixtures/glm53_tp6"


class FullGLMGeometry(unittest.TestCase):
    def setUp(self):
        self.raw = json.loads((FIXTURE / "config.json").read_text())
        self.config = Config.from_dict(self.raw)

    def test_heads_are_complete_and_nonoverlapping(self):
        ranges = [self.config.head_range(r) for r in range(6)]
        self.assertEqual([h for lo, hi, _ in ranges for h in range(lo, hi)], list(range(64)))
        self.assertEqual([hi-lo for lo, hi, _ in ranges], [11, 11, 11, 11, 11, 9])
        self.assertEqual({width for _, _, width in ranges}, {11})

    def test_vocab_keeps_every_real_token_once(self):
        ranges = [self.config.vocab_range(r) for r in range(6)]
        self.assertEqual([t for lo, hi, _ in ranges for t in range(lo, hi)], list(range(154880)))
        self.assertTrue(all(width % 64 == 0 for _, _, width in ranges))
        self.assertEqual(sum(width-(hi-lo) for lo, hi, width in ranges), 256)

    def test_flash_and_nope_are_not_full_model(self):
        for key, value in (("model_type", "glm5_next"), ("qk_rope_head_dim", 0)):
            with self.subTest(key=key):
                raw = copy.deepcopy(self.raw); raw[key] = value
                with self.assertRaises(ValueError): Config.from_dict(raw)

    def test_missing_index_reuse_pattern_rejected(self):
        self.raw["indexer_types"] = []
        with self.assertRaises(ValueError): Config.from_dict(self.raw)

    def test_feed_forward_partition_geometry_drift_rejected(self):
        for key,value in (('intermediate_size',16384),('n_shared_experts',2)):
            with self.subTest(key=key):
                raw=copy.deepcopy(self.raw);raw[key]=value
                with self.assertRaises(ValueError):Config.from_dict(raw)

    def test_target_index_reuse_does_not_cross_into_mtp(self):
        sources = [self.config.indexer_source(i) for i in range(79)]
        self.assertEqual(sources[:11], [0, 1, 2, 2, 2, 2, 6, 6, 6, 6, 10])
        self.assertEqual(sources[74:], [74, 74, 74, 74, 78])
        self.raw['indexer_types'][0] = 'shared'
        with self.assertRaises(ValueError): Config.from_dict(self.raw)
        for value in (-1, 79, True):
            with self.assertRaises(ValueError): self.config.indexer_source(value)

    def test_wrong_rank_and_world_rejected(self):
        for rank, world in ((-1, 6), (6, 6), (0, 2)):
            with self.assertRaises(ValueError): self.config.head_range(rank, world)

    def test_absent_expert_uses_nonnegative_skip_sentinel(self):
        reader = RankPieces.__new__(RankPieces); reader.config = self.config
        parts = [Fragment(1, 2, 3, "x"), Fragment(17, 0, 4, "y")]
        mapping = reader.routing_map(parts)
        self.assertEqual((mapping[1], mapping[17], mapping[0], mapping[255]), (0, 1, 2, 2))
        with self.assertRaises(ValueError): reader.routing_map(parts + [parts[0]])


class StoredBytes(unittest.TestCase):
    def reader(self, root, header, payload):
        raw = json.dumps(header).encode()
        path = root / "one.safetensors"
        path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)
        reader = RankPieces.__new__(RankPieces)
        reader.root, reader._headers = root.resolve(), {}
        reader.files = {path.name: {"bytes": path.stat().st_size}}
        reader.index = {name: path.name for name in header}
        return reader

    def test_exact_bytes_without_numeric_conversion(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = bytes.fromhex("ed1faccb0000803f")
            reader = self.reader(Path(tmp), {
                "marker": {"dtype": "I32", "shape": [], "data_offsets": [0, 4]},
                "value": {"dtype": "F32", "shape": [1], "data_offsets": [4, 8]}}, data)
            self.assertEqual(reader.read_bytes("marker"), data[:4])
            self.assertEqual(reader.read_bytes("value"), data[4:])

    def test_overlapping_tensor_ranges_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            reader = self.reader(Path(tmp), {
                "a": {"dtype": "I32", "shape": [1], "data_offsets": [0, 4]},
                "b": {"dtype": "I32", "shape": [1], "data_offsets": [0, 4]}}, bytes(4))
            with self.assertRaises(ValueError): reader.read_bytes("a")

    def test_truncation_after_header_inspection_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            reader = self.reader(Path(tmp), {
                "a": {"dtype": "I32", "shape": [1], "data_offsets": [0, 4]}}, bytes(4))
            reader.tensor_meta("a")
            path = Path(tmp) / "one.safetensors"
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaises(ValueError): reader.read_bytes("a")

    def test_row_shards_keep_original_bits_and_do_not_read_entire_tensor(self):
        with tempfile.TemporaryDirectory() as tmp:
            values=torch.arange(105,dtype=torch.int16).reshape(7,3,5)
            payload=bytes(8)+values.numpy().tobytes()
            reader=self.reader(Path(tmp),{
                'prefix':{'dtype':'I32','shape':[2],'data_offsets':[0,8]},
                'a':{'dtype':'I16','shape':[7,3,5],
                'data_offsets':[8,len(payload)]}},payload)
            with patch.object(reader,'read_bytes',side_effect=AssertionError('whole read')):
                pieces=[reader.read_rows('a',lo,hi) for lo,hi in ((0,2),(2,6),(6,7))]
                self.assertTrue(torch.equal(torch.cat(pieces),values))
                self.assertEqual(reader.read_rows('a',7,7).shape,(0,3,5))
            pieces[0][0,0,0]=999
            self.assertEqual(reader.read_rows('a',0,1)[0,0,0],0)

    def test_bf16_shards_preserve_signed_zero_and_nan_payload_bits(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw=torch.tensor([0,-32768,0x3f80,0x7fc1,0x7f80,-128],dtype=torch.int16)
            payload=raw.numpy().tobytes()
            reader=self.reader(Path(tmp),{'a':{'dtype':'BF16','shape':[3,2],
                'data_offsets':[0,len(payload)]}},payload)
            self.assertTrue(torch.equal(reader.read_rows('a',1,3).view(torch.int16),raw[2:].reshape(2,2)))

    def test_invalid_row_ranges_and_changed_file_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            reader=self.reader(Path(tmp),{'a':{'dtype':'I32','shape':[2,2],
                'data_offsets':[0,16]}},bytes(16))
            for start,stop in ((-1,1),(1,0),(0,3),(True,1),(0,1.0)):
                with self.subTest(start=start,stop=stop),self.assertRaises(ValueError):
                    reader.read_rows('a',start,stop)
            reader.tensor_meta('a');p=Path(tmp)/'one.safetensors'
            p.write_bytes(p.read_bytes()[:-1])
            with self.assertRaises(ValueError):reader.read_rows('a',0,1)


if __name__ == "__main__":
    unittest.main()
