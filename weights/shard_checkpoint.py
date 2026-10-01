#!/usr/bin/env python3
"""Stream a verified TP4 safetensors checkpoint into six lossless local copies.

Expert tensors are assigned, never decoded. Dense tensors remain replicated on
disk and are sliced by vLLM's normal TP loaders. Memory use is bounded by one
8 MiB transfer block plus safetensors headers. Output receipts record full hashes.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import time

from amos_exl3_tp6 import EXPERT_WEIGHT, TARGET_TP, owner, validate

BLOCK = 8 * 1024 * 1024


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(BLOCK), b""):
            h.update(chunk)
    return h.hexdigest()


def header(path):
    with path.open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError(f"truncated safetensors header: {path}")
        length = struct.unpack("<Q", raw)[0]
        if length > 64 * 1024 * 1024:
            raise ValueError(f"oversize safetensors header: {path}")
        raw += stream.read(length)
    tensors = json.loads(raw[8:])
    offset = 0
    for name, value in sorted(
        ((k, v) for k, v in tensors.items() if k != "__metadata__"),
        key=lambda kv: kv[1]["data_offsets"][0],
    ):
        start, end = value["data_offsets"]
        if start != offset or end < start:
            raise ValueError(f"invalid safetensors offsets: {path}/{name}")
        offset = end
    if len(raw) + offset != path.stat().st_size:
        raise ValueError(f"safetensors size mismatch: {path}")
    return tensors, raw


def destinations(name):
    match = EXPERT_WEIGHT.fullmatch(name)
    if match:
        return (owner(int(match["expert"]), int(match["rank"])),)
    if ".experts." in name:
        raise ValueError(f"unrecognized expert tensor: {name}")
    return range(TARGET_TP)


def shard_file(source, roots, expected_sha256):
    tensors, raw_header = header(source)
    records = sorted(((k, v) for k, v in tensors.items() if k != "__metadata__"),
                     key=lambda kv: kv[1]["data_offsets"][0])
    output_headers = [{} for _ in roots]
    sizes = [0] * len(roots)
    for name, value in records:
        start, end = value["data_offsets"]
        for rank in destinations(name):
            output_headers[rank][name] = {
                **value, "data_offsets": [sizes[rank], sizes[rank] + end - start]
            }
            sizes[rank] += end - start
    receipts = [r / (source.name + ".receipt.json") for r in roots]
    if all(p.exists() for p in receipts):
        for root, receipt in zip(roots, receipts):
            saved = json.loads(receipt.read_text())
            if (saved["source_sha256"] != expected_sha256
                    or sha256(root / source.name) != saved["sha256"]):
                raise ValueError(f"previous shard verification failed: {receipt}")
        return output_headers
    if any((r / source.name).exists() for r in roots):
        raise ValueError(f"incomplete prior commit; inspect shards of {source.name}")
    handles, digests = [], []
    source_digest = hashlib.sha256(raw_header)
    try:
        for root, values in zip(roots, output_headers):
            values["__metadata__"] = {"format": "pt", "placement": "amos-tp4-pieces-on-tp6-v1"}
            encoded = json.dumps(values, separators=(",", ":")).encode()
            encoded += b" " * (-len(encoded) % 8)
            prefix = struct.pack("<Q", len(encoded)) + encoded
            stream = (root / (source.name + ".partial")).open("wb")
            handles.append(stream)
            digests.append(hashlib.sha256(prefix))
            stream.write(prefix)
        with source.open("rb") as stream:
            stream.seek(len(raw_header))
            for name, value in records:
                remaining = value["data_offsets"][1] - value["data_offsets"][0]
                ranks = destinations(name)
                while remaining:
                    data = stream.read(min(BLOCK, remaining))
                    if not data:
                        raise ValueError(f"short input: {source}")
                    source_digest.update(data)
                    for rank in ranks:
                        handles[rank].write(data)
                        digests[rank].update(data)
                    remaining -= len(data)
        if source_digest.hexdigest() != expected_sha256:
            raise ValueError(f"source hash mismatch: {source}")
        for stream in handles:
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        for stream in handles:
            stream.close()
    for rank, root in enumerate(roots):
        partial = root / (source.name + ".partial")
        partial.rename(root / source.name)
        receipt = {"source": source.name, "source_sha256": expected_sha256,
                   "rank": rank, "sha256": digests[rank].hexdigest(),
                   "bytes": (root / source.name).stat().st_size}
        receipts[rank].write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"file": source.name, "status": "sharded-verified",
                      "rank_bytes": sizes}), flush=True)
    return output_headers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--wait", action="store_true", help="wait for atomic downloader commits")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    config = json.loads((args.source / "config.json").read_text())
    validate(config.get("hybrid_tr3_tail"), 6)
    roots = [args.destination / f"rank{rank}" for rank in range(TARGET_TP)]
    for root in roots:
        root.mkdir(parents=True, exist_ok=True)
    weights = [f for f in manifest["files"] if f["path"].endswith(".safetensors")]
    indices = [{} for _ in roots]
    for item in weights:
        if Path(item["path"]).name != item["path"]:
            raise ValueError("manifest paths must be flat")
        source = args.source / item["path"]
        while args.wait and not source.exists():
            print(json.dumps({"waiting": source.name}), flush=True)
            time.sleep(30)
        values = shard_file(source, roots, item["sha256"])
        for rank, entries in enumerate(values):
            indices[rank].update({name: source.name for name in entries if name != "__metadata__"})
    for item in manifest["files"]:
        name = item["path"]
        if name.endswith(".safetensors") or name in {"model.safetensors.index.json", "MANIFEST.sha256"}:
            continue
        if Path(name).name != name:
            raise ValueError("manifest paths must be flat")
        while args.wait and not (args.source / name).exists():
            time.sleep(30)
        for root in roots:
            shutil.copy2(args.source / name, root / name)
    for rank, root in enumerate(roots):
        (root / "model.safetensors.index.json").write_text(json.dumps({
            "metadata": {"total_size": sum((root / f["path"]).stat().st_size for f in weights)},
            "weight_map": indices[rank]}, indent=2) + "\n")
        (root / "TP6_PLACEMENT.json").write_text(json.dumps({
            "schema": "amos-tp4-pieces-on-tp6-v1", "rank": rank,
            "source_repository": manifest["repository"],
            "source_revision": manifest["revision"],
            "owner": "(4 * global_expert + original_tp_rank) % 6",
            "requires": "AMOS_EXL3_TP6_PIECES=1",
        }, indent=2) + "\n")
    print(json.dumps({"status": "complete", "ranks": list(map(str, roots))}), flush=True)


if __name__ == "__main__":
    main()
