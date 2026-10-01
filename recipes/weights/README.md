# Prepare the original 3.25 bpw weights

Run from the repository root. This recipe uses
[davidsyoung/GLM-5.3-EXL3-TR3-3.25bpw](https://huggingface.co/davidsyoung/GLM-5.3-EXL3-TR3-3.25bpw)
at revision `6d6bd738c0c1635513e0bd0fdf0302049bd820a9`.
Read its model terms before downloading. The checkpoint is a mixed-K TP4
export; a stock uniform-K EXL3 loader is incompatible.

## Download and reshard

Choose storage with room for the source, all six output directories, and a
working reserve. The downloader checks required source space and leaves a
256 GiB reserve by default. Resharding needs additional space: replicated
dense tensors mean six output directories are larger in total than the source.

```bash
export SOURCE_DIR=/mnt/models/glm53-exl3-3.25
export TP6_DIR=/mnt/models/glm53-exl3-tp6
python3 weights/download_checkpoint.py \
  weights/checkpoint-manifest.json "$SOURCE_DIR" --workers 2
python3 weights/shard_checkpoint.py \
  weights/checkpoint-manifest.json "$SOURCE_DIR" "$TP6_DIR"
for rank in 0 1 2 3 4 5; do
  python3 weights/verify_shard.py "$TP6_DIR/rank$rank" \
    weights/checkpoint-manifest.json --rank "$rank"
done
```

The resharing tool copies original compressed fragments; it never decodes or
requantizes them. Ownership is `(4 * global_expert + original_tp_rank) % 6`.
This exact convention must agree with the runtime; other TP6 recipes can use
a different, layer-dependent ownership rule.

## Transfer

Copy each `rankN/` to its assigned Spark, including metadata, per-file receipts,
and `TP6_PLACEMENT.json`. A NAS can hold the source and output shards. Its fast
network connection is useful for this transfer; serving uses local storage.

Run `verify_shard.py` again **on the receiving node** before mounting its
directory read-only at `/model`. Verification checks complete file hashes,
expert ownership, tensor index, and tokenizer/config metadata. Do not reuse
another rank's `TP6_VERIFIED.json`.

CPU geometry checks:

```bash
python3 -m pip install -r requirements-check.txt
python3 -m unittest discover -s weights -p 'test_placement.py'
```
