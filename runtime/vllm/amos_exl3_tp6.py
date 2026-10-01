"""Lossless placement of a TP4 EXL3 checkpoint's expert pieces on TP6.

Router IDs stay global (256). Each rank stores at most one of the four
512-wide pieces of an expert. The existing TP all-reduce adds those pieces.
No router probabilities, rotations, codebooks, or quantized codes change.
"""

from functools import lru_cache
import os
import re

SOURCE_TP = 4
TARGET_TP = 6
EXPERTS = 256
PIECE_WIDTH = 512
EXPERT_WEIGHT = re.compile(
    r"^(?P<prefix>.+\.experts)\.(?P<expert>\d+)\."
    r"(?P<projection>gate_proj|up_proj|down_proj)\.rank(?P<rank>\d+)\."
    r"(?P<field>trellis|suh|svh|mcg|mul1)$"
)


def enabled():
    return os.environ.get("AMOS_EXL3_TP6_PIECES") == "1"


def owner(expert, source_rank):
    if not 0 <= expert < EXPERTS or not 0 <= source_rank < SOURCE_TP:
        raise ValueError(f"invalid source expert piece: {expert}/{source_rank}")
    return (SOURCE_TP * expert + source_rank) % TARGET_TP


@lru_cache(maxsize=TARGET_TP)
def pieces(rank):
    if not 0 <= rank < TARGET_TP:
        raise ValueError(f"invalid TP6 rank: {rank}")
    return tuple((expert, source_rank)
                 for expert in range(EXPERTS)
                 for source_rank in range(SOURCE_TP)
                 if owner(expert, source_rank) == rank)


@lru_cache(maxsize=TARGET_TP)
def local_ids(rank):
    return {pair: i for i, pair in enumerate(pieces(rank))}


def normalize(name, rank):
    match = EXPERT_WEIGHT.fullmatch(name)
    if match is None:
        if ".experts." in name and ".rank" in name:
            raise ValueError(f"unsupported rank-sliced tensor: {name}")
        return name
    pair = int(match["expert"]), int(match["rank"])
    if owner(*pair) != rank:
        return None
    return (f"{match['prefix']}.{local_ids(rank)[pair]}."
            f"{match['projection']}.{match['field']}")


def validate(metadata, runtime_tp, intermediate_size=None):
    if runtime_tp != TARGET_TP:
        raise ValueError("AMOS_EXL3_TP6_PIECES requires exactly TP6")
    if not isinstance(metadata, dict) or (
        metadata.get("tp") != SOURCE_TP
        or metadata.get("experts_per_layer") != EXPERTS
        or metadata.get("codebook") != "mcg"
        or metadata.get("bits") != "mixed"
        or metadata.get("rotation_layout", "per_expert_v1") != "per_expert_v1"
    ):
        raise ValueError("TP6 piece placement requires mixed-K TP4 per-expert MCG weights")
    if intermediate_size is not None and intermediate_size != PIECE_WIDTH:
        raise ValueError(f"TP6 piece width must be {PIECE_WIDTH}, got {intermediate_size}")


def remap_routes(compact_to_tier, rank):
    """Return a global-router-to-physical-tier map; absent experts map to -1."""
    if len(compact_to_tier) != len(pieces(rank)):
        raise ValueError("compact route map does not match the local piece count")
    result = [-1] * EXPERTS
    for local, (expert, _) in enumerate(pieces(rank)):
        result[expert] = int(compact_to_tier[local])
    return result
