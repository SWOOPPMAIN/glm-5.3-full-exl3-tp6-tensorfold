#!/usr/bin/env python3
"""Verify a transferred TP6 shard against pinned source hashes and ownership."""

import argparse
import json
from pathlib import Path

from amos_exl3_tp6 import EXPERT_WEIGHT, owner, pieces, validate
from shard_checkpoint import header, sha256


def verify(root, manifest_path, rank):
    manifest = json.loads(manifest_path.read_text())
    placement = json.loads((root / "TP6_PLACEMENT.json").read_text())
    expected = {
        "schema": "amos-tp4-pieces-on-tp6-v1", "rank": rank,
        "source_repository": manifest["repository"],
        "source_revision": manifest["revision"],
        "owner": "(4 * global_expert + original_tp_rank) % 6",
        "requires": "AMOS_EXL3_TP6_PIECES=1",
    }
    if placement != expected:
        raise ValueError("shard placement does not match the requested rank/source")
    config = json.loads((root / "config.json").read_text())
    validate(config.get("hybrid_tr3_tail"), 6)
    declared = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    actual, receipts, seen = {}, [], set()
    for item in manifest["files"]:
        name = item["path"]
        if not name.endswith(".safetensors"):
            continue
        if Path(name).name != name:
            raise ValueError("checkpoint paths must be flat")
        path = root / name
        receipt = json.loads((root / (name + ".receipt.json")).read_text())
        if (receipt["source_sha256"] != item["sha256"] or receipt["rank"] != rank
                or receipt["source"] != name or path.is_symlink()
                or path.stat().st_size != receipt["bytes"]
                or sha256(path) != receipt["sha256"]):
            raise ValueError(f"transferred shard verification failed: {name}")
        values, _ = header(path)
        for tensor_name in values:
            if tensor_name == "__metadata__":
                continue
            if tensor_name in actual:
                raise ValueError(f"duplicate tensor: {tensor_name}")
            actual[tensor_name] = name
            match = EXPERT_WEIGHT.fullmatch(tensor_name)
            if match:
                expert, source_rank = int(match["expert"]), int(match["rank"])
                if owner(expert, source_rank) != rank:
                    raise ValueError(f"foreign expert piece: {tensor_name}")
                seen.add(tensor_name)
            elif ".experts." in tensor_name:
                raise ValueError(f"unsupported expert tensor: {tensor_name}")
        receipts.append({"path": name, "sha256": receipt["sha256"], "bytes": receipt["bytes"]})
    if actual != declared:
        raise ValueError("weight index differs from verified tensor contents")
    expected_experts = {
        f"model.layers.{layer}.mlp.experts.{expert}.{projection}.rank{source}.{field}"
        for layer in range(3, 79)
        for expert, source in pieces(rank)
        for projection in ("gate_proj", "up_proj", "down_proj")
        for field in ("trellis", "suh", "svh", "mcg")
    }
    if seen != expected_experts:
        raise ValueError(f"expert payload set mismatch: missing={len(expected_experts-seen)}, "
                         f"extra={len(seen-expected_experts)}")
    # Tokenizer/config provenance matters as much as the large tensors. The
    # index is regenerated and the original manifest describes unsharded files.
    from download_checkpoint import check_hash, hashes
    for item in manifest["files"]:
        name = item["path"]
        if name.endswith(".safetensors") or name in {"model.safetensors.index.json", "MANIFEST.sha256"}:
            continue
        path = root / name
        if path.is_symlink() or path.stat().st_size != item["size"] or not check_hash(item, *hashes(path, item["size"])):
            raise ValueError(f"metadata verification failed: {name}")
    result = {"rank": rank, "source_revision": manifest["revision"],
              "verification": "full-file SHA256 and exact expert ownership",
              "files": receipts, "tensor_count": len(actual)}
    (root / "TP6_VERIFIED.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"rank": rank, "verified_files": len(receipts),
                      "tensors": len(actual), "status": "verified"}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--rank", type=int, required=True, choices=range(6))
    args = parser.parse_args()
    verify(args.root, args.manifest, args.rank)


if __name__ == "__main__":
    main()
