# Source provenance

`imports.json` records each exported file's origin and SHA256. It describes
the snapshot as imported; update a file's record when intentionally changing it.

| Component | Pin |
| --- | --- |
| Historical serving source export | `amos-recipes` commit `173f4c5449a38d05057e1c342d1b44655b024ab8` |
| Bundled historical TensorFold source | `c79902ec67c9e516e67b9aca82a851c9b7a49c91` |
| TensorFold base | `9cd52ab4daba68ddd09be89be8f23ad43175e821` (v0.5.0) |
| Imported Mia recipe | `ed026ef92d1650120dada1294a112acb6c8f2f48` (52 patches) |
| Kindling E3 import | `0ecf21ebcccd924f7f7c47eda0333282d8e177f5` |
| Checkpoint | `6d6bd738c0c1635513e0bd0fdf0302049bd820a9` |

The source revision is distinct from a metadata-only ledger commit. Each
experiment identifies its executed source revision in the results; the
bundled executable TensorFold snapshot matches TFP21. Later local results through
TFP57 are published separately; their source revisions are recorded in each summary.

TensorFold files under `src/` are byte-identical to the qualified revision.
Earlier GPU probe scripts replace a private fabric address with `MASTER_ADDR`;
TFP16 through TFP21 already used that environment variable in their executed source.
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

## October 3 deployment closeout export

Current local deployment closeout: `bb49dce1a426a80e7c8c638a5b247b1e865dfb69`.
Selected public results retain SHA256 hashes of their original receipts. The P27
tuning file is imported exactly. Existing direct vLLM integration modules were
checked against the current workspace and were unchanged; this is not a complete
P27 image export. No later TensorFold source refresh is implied by the result reports.
Private node addresses, container IDs, credentials and application records remain local.

## E3 row32 serving qualification

Source/results revision: `1ec92c9d2a3ce7c556884fb662f1ea6f1da1027c`.
The policy wrapper and image builder are byte-identical exports. The row32 patch
retains the original Mia/ExLlamaV3 notices and native b12x epilogues. Published
serving results contain all timing samples and receipt hashes; the standalone
sample analyzer reproduces the local comparison. Native/router/Code/Chat acceptance
passed on the selected row32 image. The later copy-drafting comparison is now completed and exported separately; see
[its quality results, all samples and compatibility repairs](../recipes/vllm-tp6/COPY_DRAFTING.md).
Original adaptive MTP remains selected on the forward image.
