#!/usr/bin/env python3
"""Stage a pinned public HF checkpoint with bounded memory and hash verification."""

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import time
import urllib.request

BLOCK = 1024 * 1024


def hashes(path, size):
    sha256 = hashlib.sha256()
    git_sha1 = hashlib.sha1(f"blob {size}\0".encode())
    if path.exists():
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(BLOCK), b""):
                sha256.update(chunk)
                git_sha1.update(chunk)
    return sha256, git_sha1


def check_hash(item, sha256, git_sha1):
    return (sha256.hexdigest() == item["sha256"] if item.get("sha256")
            else git_sha1.hexdigest() == item["git_blob_sha1"])


def download(item, root, repo, revision):
    name = item["path"]
    rel = PurePosixPath(name)
    if rel.is_absolute() or ".." in rel.parts or len(rel.parts) != 1:
        raise ValueError(f"unexpected checkpoint path: {name}")
    dest = root / name
    partial = root / (name + ".partial")
    size = item["size"]
    if dest.exists():
        if dest.stat().st_size == size and check_hash(item, *hashes(dest, size)):
            print(json.dumps({"file": name, "status": "verified-existing"}), flush=True)
            return
        raise RuntimeError(f"existing file fails verification: {dest}")
    for attempt in range(6):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > size:
            raise RuntimeError(f"oversized partial file: {partial}")
        sha256, git_sha1 = hashes(partial, size)
        try:
            if offset < size:
                # A fresh resolve request avoids retaining expired signed CDN URLs.
                url = f"https://huggingface.co/{repo}/resolve/{revision}/{name}?download=true"
                headers = {"User-Agent": "amos-glm53-tp6-stager/1"}
                if offset:
                    headers["Range"] = f"bytes={offset}-"
                request = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(request, timeout=90) as response:
                    if offset and (response.status != 206 or not response.headers.get(
                            "Content-Range", "").startswith(f"bytes {offset}-")):
                        raise RuntimeError(f"server did not honor resume for {name}")
                    with partial.open("ab" if offset else "wb") as stream:
                        last_report = time.monotonic()
                        while chunk := response.read(BLOCK):
                            if offset + len(chunk) > size:
                                raise RuntimeError(f"response exceeds pinned size: {name}")
                            stream.write(chunk)
                            sha256.update(chunk)
                            git_sha1.update(chunk)
                            offset += len(chunk)
                            if time.monotonic() - last_report > 60:
                                print(json.dumps({"file": name, "bytes": offset,
                                                  "total": size}), flush=True)
                                last_report = time.monotonic()
                        stream.flush()
                        os.fsync(stream.fileno())
            if offset != size:
                raise RuntimeError(f"short response: {name}, {offset}/{size}")
            if not check_hash(item, sha256, git_sha1):
                raise RuntimeError(f"hash mismatch: {name}; retaining partial for inspection")
            partial.rename(dest)
            print(json.dumps({"file": name, "status": "verified", "bytes": size}), flush=True)
            return
        except Exception as error:
            print(json.dumps({"file": name, "attempt": attempt + 1,
                              "error": type(error).__name__ + ": " + str(error)}), flush=True)
            if attempt == 5 or "hash mismatch" in str(error):
                raise
            time.sleep(min(2 ** attempt, 20))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--reserve-gib", type=int, default=256)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    revision = manifest["revision"]
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        parser.error("revision must be an immutable Git commit")
    if not 1 <= args.workers <= 16:
        parser.error("workers must be between 1 and 16")
    args.destination.mkdir(parents=True, exist_ok=True)
    required = sum(max(0, f["size"] - sum(p.stat().st_size for p in (
        args.destination / f["path"], args.destination / (f["path"] + ".partial"))
        if p.exists())) for f in manifest["files"])
    free = shutil.disk_usage(args.destination).free
    if free < required + args.reserve_gib * 1024**3:
        parser.error(f"insufficient free space: {free} bytes, need {required} plus reserve")
    print(json.dumps({"repository": manifest["repository"], "revision": revision,
                      "required_bytes": required, "free_bytes": free}), flush=True)
    # Small metadata first, so the resharing tool can inspect its input contract.
    files = sorted(manifest["files"], key=lambda x: (x["size"] > 32 * BLOCK, x["path"]))
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(download, f, args.destination,
                                   manifest["repository"], revision) for f in files]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    (args.destination / "DOWNLOAD_VERIFIED.json").write_text(json.dumps({
        "repository": manifest["repository"], "revision": revision,
        "files": len(files), "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "verification": "Every byte checked against HF LFS SHA256 or Git blob SHA1",
    }, indent=2))


if __name__ == "__main__":
    main()
