"""Read existing TP6 fragments directly for TensorFold, without requantization.

The config's modelopt label is a vLLM dispatch shim. Only explicit TP6 placement,
the TR3 metadata, tensor shapes and codebook bytes determine this reader's format.
Header inspection and byte reads need no torch, CUDA, process group or network.
"""
import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import struct

from .config import Config

SOURCE_REVISION = "6d6bd738c0c1635513e0bd0fdf0302049bd820a9"
OWNER = "(4 * global_expert + original_tp_rank) % 6"
BYTES = {"BF16": 2, "F16": 2, "F32": 4, "I16": 2, "I32": 4, "I64": 8,
         "U8": 1, "I8": 1, "F64": 8, "U32": 4, "U16": 2}


@dataclass(frozen=True)
class Fragment:
    expert: int
    original_rank: int
    bits: int
    prefix: str


class RankPieces:
    def __init__(self, model_dir, rank):
        self.root = Path(model_dir).resolve()
        self.rank = rank
        if type(rank) is not int or not 0 <= rank < 6:
            raise ValueError("Rank must be in 0..5")
        self.raw_config = json.loads((self.root / "config.json").read_text())
        self.config = Config.from_dict(self.raw_config)
        placement = json.loads((self.root / "TP6_PLACEMENT.json").read_text())
        if (placement.get("schema") != "amos-tp4-pieces-on-tp6-v1"
                or placement.get("rank") != rank or placement.get("owner") != OWNER
                or placement.get("source_revision") != SOURCE_REVISION):
            raise ValueError("TP6 placement does not match this reader")
        self.verified = json.loads((self.root / "TP6_VERIFIED.json").read_text())
        if self.verified.get("rank") != rank or self.verified.get("source_revision") != SOURCE_REVISION:
            raise ValueError("TP6 verification receipt is for another rank or checkpoint")
        tail = self.raw_config.get("hybrid_tr3_tail", {})
        if (tail.get("format") != "exl3-trellis" or tail.get("codebook") != "mcg"
                or tail.get("mcg_multiplier") != 0xCBAC1FED or tail.get("tp") != 4
                or tail.get("bits_avg") != 3.25 or tail.get("k_values") != [3, 4]):
            raise ValueError("Expected original mixed K3/K4 MCG fragments")
        self.tiers = json.loads((self.root / "tier_bitmap.json").read_text())
        self.index = json.loads((self.root / "model.safetensors.index.json").read_text())["weight_map"]
        self.files = {f["path"]: f for f in self.verified["files"]}
        self._headers = {}

    def _file(self, name):
        if name not in self.files or Path(name).name != name:
            raise ValueError("Tensor file is outside the verified rank manifest")
        p = self.root / name
        if p.resolve().parent != self.root or p.stat().st_size != self.files[name]["bytes"]:
            raise ValueError("Rank file path or byte count changed")
        return p

    def _header(self, name):
        if name not in self._headers:
            path = self._file(name)
            with path.open("rb") as f:
                lead = f.read(8)
                if len(lead) != 8:
                    raise ValueError("Truncated safetensors header length")
                length = struct.unpack("<Q", lead)[0]
                if not 2 <= length <= min(64 * 1024**2, path.stat().st_size - 8):
                    raise ValueError("Invalid safetensors header extent")
                raw = f.read(length)
            header = json.loads(raw)
            extents = []
            for key, meta in header.items():
                if key == "__metadata__":
                    continue
                shape, dtype = meta["shape"], meta["dtype"]
                start, end = meta["data_offsets"]
                if (dtype not in BYTES or any(type(n) is not int or n < 0 for n in shape)
                        or type(start) is not int or type(end) is not int
                        or not 0 <= start <= end <= path.stat().st_size - length - 8
                        or end - start != math.prod(shape) * BYTES[dtype]):
                    raise ValueError(f"Invalid tensor extent: {key}")
                extents.append((start, end))
            ordered = sorted(extents)
            if any(left[1] > right[0] for left, right in zip(ordered, ordered[1:])):
                raise ValueError("Overlapping safetensors extents")
            self._headers[name] = (header, length + 8, hashlib.sha256(raw).hexdigest())
        return self._headers[name]

    def tensor_meta(self, name):
        filename = self.index[name]
        header, _, _ = self._header(filename)
        return header[name]

    def read_bytes(self, name):
        filename = self.index[name]
        header, base, _ = self._header(filename)
        start, end = header[name]["data_offsets"]
        with self._file(filename).open("rb") as f:
            f.seek(base + start)
            raw = f.read(end - start)
        if len(raw) != end - start:
            raise ValueError(f"Truncated tensor: {name}")
        return raw

    def read_tensor(self, name, device="cpu"):
        """Load exactly one stored tensor; GPU upload is explicit and opt-in."""
        import torch
        dtypes = {"I16": torch.int16, "I32": torch.int32, "I64": torch.int64,
                  "F16": torch.float16, "BF16": torch.bfloat16, "F32": torch.float32}
        meta = self.tensor_meta(name)
        value = torch.frombuffer(bytearray(self.read_bytes(name)), dtype=dtypes[meta["dtype"]])
        return value.reshape(meta["shape"]).to(device=device)

    def read_rows(self, name, start, stop, device="cpu"):
        """Read a contiguous first-axis shard without materializing the tensor.

        A vocabulary shard is hundreds of MiB; copying the full original table
        through bytes and bytearray would transiently occupy several GiB. One
        owned bytearray receives exactly the requested rows with readinto.
        Header/extent/manifest checks are the same as for an entire tensor.
        """
        import torch
        dtypes={"I16":torch.int16,"I32":torch.int32,"I64":torch.int64,
                "F16":torch.float16,"BF16":torch.bfloat16,"F32":torch.float32}
        filename=self.index[name];header,base,_=self._header(filename);meta=header[name]
        shape=meta['shape'];dtype=meta['dtype']
        if (not shape or type(start) is not int or type(stop) is not int
                or not 0<=start<=stop<=shape[0] or dtype not in dtypes):
            raise ValueError('Invalid original tensor row shard')
        row_bytes=math.prod(shape[1:])*BYTES[dtype];size=(stop-start)*row_bytes
        path=self._file(filename)
        if not size:return torch.empty((stop-start,*shape[1:]),dtype=dtypes[dtype],device=device)
        data=bytearray(size)
        with path.open('rb',buffering=0) as handle:
            handle.seek(base+meta['data_offsets'][0]+start*row_bytes)
            view=memoryview(data);done=0
            while done<size:
                count=handle.readinto(view[done:])
                if not count:raise ValueError('Truncated original tensor row shard: '+name)
                done+=count
        return torch.frombuffer(data,dtype=dtypes[dtype]).reshape(stop-start,*shape[1:]).to(device)

    def fragments(self, layer):
        if not self.config.dense_layers <= layer <= self.config.layers:
            raise ValueError("Expected target MoE layer 3..77 or draft layer 78")
        bits = self.tiers[str(layer)]["k"]
        if len(bits) != 256 or bits.count(3) != 192 or bits.count(4) != 64:
            raise ValueError("Original layer quantization changed")
        parts = [Fragment(e, p, bits[e], f"model.layers.{layer}.mlp.experts.{e}")
                 for e in range(256) for p in range(4) if (4 * e + p) % 6 == self.rank]
        expected_names = set()
        for part in parts:
            for projection in ("gate_proj", "up_proj", "down_proj"):
                k, n = (512, 6144) if projection == "down_proj" else (6144, 512)
                prefix = f"{part.prefix}.{projection}.rank{part.original_rank}"
                wanted = {"trellis": ("I16", [k // 16, n // 16, 16 * part.bits]),
                          "suh": ("F16", [k]), "svh": ("F16", [n]), "mcg": ("I32", [])}
                for field, (dtype, shape) in wanted.items():
                    name = prefix + "." + field
                    expected_names.add(name)
                    meta = self.tensor_meta(name)
                    if meta["dtype"] != dtype or meta["shape"] != shape:
                        raise ValueError(f"Wrong stored EXL3 shape or dtype: {name}")
                if struct.unpack("<I", self.read_bytes(prefix + ".mcg"))[0] != 0xCBAC1FED:
                    raise ValueError("Unexpected EXL3 codebook marker")
        prefix = f"model.layers.{layer}.mlp.experts."
        actual_names = {name for name in self.index if name.startswith(prefix)}
        if actual_names != expected_names:
            raise ValueError("Expert ownership differs from the TP6 placement")
        return parts

    def routing_map(self, parts):
        """Global expert ID -> local slot; absent experts use TensorFold's E sentinel."""
        result = [len(parts)] * self.config.experts
        for slot, part in enumerate(parts):
            if result[part.expert] != len(parts):
                raise ValueError("Two fragments of an expert unexpectedly share a rank")
            result[part.expert] = slot
        return result

    def prepare_experts(self, layer, device="cuda"):
        """Adapt original tensors to TensorFold's general mixed-width EXL3 kernel.

        Requires the caller's bounded GPU qualification environment. This is a
        loader, not proof of forward-pass fidelity or whole-model readiness.
        """
        from tensorfold.cuda.exl3.experts import prepare
        parts = self.fragments(layer)
        projections = []
        for projection in ("gate_proj", "up_proj", "down_proj"):
            triples = []
            for part in parts:
                prefix = f"{part.prefix}.{projection}.rank{part.original_rank}"
                triples.append(tuple(self.read_tensor(prefix + "." + field, device)
                                     for field in ("trellis", "suh", "svh")))
            projections.append(triples)
        return prepare(*projections, codebook="mcg", device=device), self.routing_map(parts)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--layers", type=int, nargs="+", default=[3, 40, 77, 78])
    a = p.parse_args()
    reader = RankPieces(a.model_dir, a.rank)
    layers = []
    for layer in a.layers:
        parts = reader.fragments(layer)
        layers.append(dict(layer=layer, fragments=len(parts),
                           k3=sum(p.bits == 3 for p in parts), k4=sum(p.bits == 4 for p in parts),
                           routing_map=reader.routing_map(parts)))
    print(json.dumps(dict(rank=a.rank, source_revision=SOURCE_REVISION, layers=layers,
        head_range=reader.config.head_range(a.rank), vocab_range=reader.config.vocab_range(a.rank),
        headers={k:v[2] for k,v in reader._headers.items()},
        validation="manifest byte counts, header extents, exact ownership, bit widths and codebook markers",
        full_file_hashes_recomputed=False, gpu_initialized=False, serving_ready=False)))


if __name__ == "__main__":
    main()
