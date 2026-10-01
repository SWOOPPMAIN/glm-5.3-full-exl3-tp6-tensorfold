# Source provenance

`imports.json` records each exported file's origin and SHA256. It describes
the snapshot as imported; update a file's record when intentionally changing it.

| Component | Pin |
| --- | --- |
| Swoopp serving source | `amos-recipes` commit `173f4c5449a38d05057e1c342d1b44655b024ab8` |
| Swoopp TensorFold GPU-qualified source | `f06e85fe20ef8b626f5c1054b24d4b84410b9567` |
| TensorFold base | `9cd52ab4daba68ddd09be89be8f23ad43175e821` (v0.5.0) |
| Imported Mia recipe | `ed026ef92d1650120dada1294a112acb6c8f2f48` (52 patches) |
| Kindling E3 import | `0ecf21ebcccd924f7f7c47eda0333282d8e177f5` |
| Checkpoint | `6d6bd738c0c1635513e0bd0fdf0302049bd820a9` |

The source revision is distinct from a metadata-only ledger commit. Each
experiment identifies its executed source revision in the results; the
current snapshot matches TFP17.

TensorFold files under `src/` are byte-identical to the qualified revision.
Earlier GPU probe scripts replace a private fabric address with `MASTER_ADDR`;
TFP16 and TFP17 already used that environment variable in their executed source.
The benchmark client's default endpoint is loopback. No GPU or serving
benchmark was rerun as part of this export.

The retained [Mia patch manifest](../experimental/tensorfold/mia-recipe-provenance.json),
[RoCEnante source lock](../runtime/vllm/rocenante/source-lock.json), and
[checkpoint manifest](../weights/checkpoint-manifest.json) preserve finer-grained pins.
`base-image-lineage.json` records build inputs inherited from the earlier
full-model TP4 image. Its old model/drafter selections are intentionally
omitted: the current model pin is the checkpoint manifest above.

Raw fleet receipts, private deployment manifests, credentials, and application
responses remain outside this repository. Results are selected measurement
fields with original receipt hashes, not a wholesale copy of operational logs.
