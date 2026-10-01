"""Correctness checks for the redistribution, not serving-performance claims."""

import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from amos_exl3_tp6 import normalize, owner, pieces, remap_routes
from shard_checkpoint import header, shard_file


class PlacementTests(unittest.TestCase):
    def test_every_piece_exactly_once_and_at_most_one_per_expert_per_rank(self):
        found = [p for rank in range(6) for p in pieces(rank)]
        self.assertEqual(sorted(found), [(e, r) for e in range(256) for r in range(4)])
        for rank in range(6):
            self.assertEqual(len({e for e, _ in pieces(rank)}), len(pieces(rank)))
        self.assertLessEqual(max(map(lambda r: len(pieces(r)), range(6))) -
                             min(map(lambda r: len(pieces(r)), range(6))), 1)

    def test_nonlinear_moe_sum_with_original_router_probabilities(self):
        # Dense reference computes whole expert matrices. Redistributed
        # execution applies SiLU independently to intact intermediate slices.
        rng = np.random.default_rng(451)
        x = rng.normal(size=(7, 12))
        gate = rng.normal(size=(256, 16, 12)) * .1
        up = rng.normal(size=gate.shape) * .1
        down = rng.normal(size=(256, 12, 16)) * .1
        ids = np.array([rng.choice(256, 8, replace=False) for _ in x])
        probs = rng.uniform(size=ids.shape)
        probs /= probs.sum(axis=1, keepdims=True)
        reference = np.zeros_like(x)
        distributed = np.zeros((6, *x.shape))
        for token, routes in enumerate(ids):
            for j, expert in enumerate(routes):
                g = gate[expert] @ x[token]
                u = up[expert] @ x[token]
                act = g / (1 + np.exp(-g)) * u
                reference[token] += probs[token, j] * (down[expert] @ act)
                for source in range(4):
                    span = slice(4 * source, 4 * (source + 1))
                    distributed[owner(int(expert), source), token] += (
                        probs[token, j] * (down[expert, :, span] @ act[span]))
        np.testing.assert_allclose(distributed.sum(axis=0), reference, atol=1e-14, rtol=1e-14)

    def test_global_routes_and_names(self):
        for rank in range(6):
            permutation = list(reversed(range(len(pieces(rank)))))
            mapping = remap_routes(permutation, rank)
            self.assertEqual(mapping.count(-1), 256 - len(pieces(rank)))
            for local, (expert, source) in enumerate(pieces(rank)):
                self.assertEqual(mapping[expert], permutation[local])
                name = f"model.layers.78.mlp.experts.{expert}.up_proj.rank{source}.suh"
                self.assertEqual(normalize(name, rank), f"model.layers.78.mlp.experts.{local}.up_proj.suh")
                self.assertIsNone(normalize(name, (rank + 1) % 6))

    def test_streamed_shards_preserve_every_tensor_byte(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            values = {"model.layers.3.self_attn.weight": b"dense123"}
            for expert in range(256):
                for rank in range(4):
                    name = f"model.layers.3.mlp.experts.{expert}.gate_proj.rank{rank}.trellis"
                    values[name] = hashlib.sha256(name.encode()).digest()
            hdr, payload = {}, b""
            for name, data in values.items():
                hdr[name] = {"dtype": "U8", "shape": [len(data)],
                             "data_offsets": [len(payload), len(payload) + len(data)]}
                payload += data
            encoded = json.dumps(hdr).encode()
            source = path / "model-layer-003.safetensors"
            source.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
            roots = [path / f"rank{r}" for r in range(6)]
            for root in roots:
                root.mkdir()
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            shard_file(source, roots, digest)
            observed = {}
            for rank, root in enumerate(roots):
                file = root / source.name
                entries, raw = header(file)
                data = file.read_bytes()[len(raw):]
                for name, value in entries.items():
                    if name == "__metadata__":
                        continue
                    start, end = value["data_offsets"]
                    self.assertEqual(data[start:end], values[name])
                    observed.setdefault(name, []).append(rank)
            self.assertEqual(set(observed), set(values))
            for name, ranks in observed.items():
                self.assertEqual(len(ranks), 1 if ".experts." in name else 6)
            shard_file(source, roots, digest)  # Verified restart.
            file = roots[2] / source.name
            with file.open("r+b") as stream:
                stream.seek(-1, 2)
                stream.write(b"X")
            with self.assertRaisesRegex(ValueError, "verification failed"):
                shard_file(source, roots, digest)


if __name__ == "__main__":
    unittest.main()
