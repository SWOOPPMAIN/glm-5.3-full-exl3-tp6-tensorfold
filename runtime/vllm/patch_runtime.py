#!/usr/bin/env python3
"""Install the narrow TP6 piece-placement overlay on a pinned vLLM source tree."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"expected one patch anchor, got {text.count(old)}: {old[:100]}")
    return text.replace(old, new, 1)


def patch_exl3(text):
    text = replace_once(text, '''        if not self.exl3_preallocate:
            if borrowed and self.exl3_tp_slice is None:''', '''        if not self.exl3_preallocate:
            # Mixed K3/K4 pieces have different shapes, so this path cannot
            # use the uniform slab preallocator. On GB10, retaining every CPU
            # piece until postprocessing duplicates ~50GiB of weights in the
            # same physical RAM as CUDA. Stream each piece to its final device.
            if tp6_pieces_enabled() and loaded_weight.device != self.device:
                loaded_weight = loaded_weight.to(device=self.device, non_blocking=True)
                borrowed = False
            if borrowed and self.exl3_tp_slice is None:''')
    text = replace_once(text, "class Exl3MoEMethod(FusedMoEMethodBase):", "from vllm.amos_exl3_tp6 import (\n    enabled as tp6_pieces_enabled, pieces as tp6_pieces,\n    normalize as tp6_normalize, validate as tp6_validate,\n)\n\n\nclass Exl3MoEMethod(FusedMoEMethodBase):")
    text = replace_once(text, '        # The checkpoint stores one hidden-side rotation per layer with no TP', '''        if tp6_pieces_enabled():
            from vllm.distributed import get_tensor_model_parallel_world_size
            tp6_validate(self.rank_sliced_metadata, get_tensor_model_parallel_world_size())
            return tp6_normalize(name, get_tensor_model_parallel_rank())
        # The checkpoint stores one hidden-side rotation per layer with no TP''')
    text = replace_once(text, '            if checkpoint_tp != layer.exl3_tp_size:', '''            if tp6_pieces_enabled():
                tp6_validate(rank_sliced_metadata, layer.exl3_tp_size,
                             intermediate_size_per_partition)
                layer.exl3_tp6_pieces = tp6_pieces(layer.exl3_tp_rank)
            if checkpoint_tp != layer.exl3_tp_size and not tp6_pieces_enabled():''')
    text = replace_once(text, '            vllm_config = get_current_vllm_config_or_none()\n            scheduler_config = (', '''            if tp6_pieces_enabled():
                # Keep the router and MoE parallel config in the 256-expert
                # namespace. Only this quantizer's physical storage is compact.
                num_experts = len(layer.exl3_tp6_pieces)
                layer.local_num_experts = num_experts
            vllm_config = get_current_vllm_config_or_none()
            scheduler_config = (''')
    text = replace_once(text, '            layer.exl3_mixed_bitrate = len(set(layer.exl3_layer_bitrates)) > 1', '''            if tp6_pieces_enabled():
                layer.exl3_layer_bitrates = tuple(
                    layer.exl3_layer_bitrates[expert]
                    for expert, _source_rank in layer.exl3_tp6_pieces
                )
                if len(set(layer.exl3_layer_bitrates)) != 2:
                    raise ValueError("TP6 placement requires both bitrate tiers on every rank")
            layer.exl3_mixed_bitrate = len(set(layer.exl3_layer_bitrates)) > 1''')
    text = replace_once(text, '''        global_to_combined, descriptor_map = mixed_api.build_tiered_maps(
            tier_ids[0], tier_ids[1], device=device
        )''', '''        global_to_combined, descriptor_map = mixed_api.build_tiered_maps(
            tier_ids[0], tier_ids[1], device=device
        )
        if hasattr(layer, "exl3_tp6_pieces"):
            # Existing B12X route packing and output summation both mask -1.
            # Never renormalize the remaining router weights on this rank.
            global_map = torch.full((256,), -1, dtype=torch.int32, device=device)
            expert_ids = torch.tensor(
                [e for e, _r in layer.exl3_tp6_pieces], dtype=torch.long, device=device
            )
            global_map.index_copy_(0, expert_ids, global_to_combined)
            global_to_combined = global_map''')
    text = replace_once(text, '        route_num_experts = int(layer.local_num_experts)', '        route_num_experts = int(mixed["global_to_combined"].numel())')
    text = replace_once(text, '''        logger.info(
            "EXL3 mixed Trellis %s: tiers=%s",''', '''        # GB10 shares GPU and host RAM. Return released checkpoint slabs
        # between layers instead of retaining them until the profile RPC.
        # Live prepared weights and rotations are unaffected.
        if tp6_pieces_enabled():
            torch.cuda.empty_cache()
        logger.info(
            "EXL3 mixed Trellis %s: tiers=%s",''')
    return text


def patch_virtual_tp(text):
    return replace_once(text, '''def _get_moe_intermediate_local_alignment(model_config: ModelConfig) -> int:
''', '''def _get_moe_intermediate_local_alignment(model_config: ModelConfig) -> int:
    from vllm.amos_exl3_tp6 import enabled, validate
    if enabled():
        config = model_config.hf_text_config
        validate(getattr(config, "hybrid_tr3_tail", None), 6)
        if int(config.moe_intermediate_size) != 2048:
            raise ValueError("TP6 pieces require original MoE width 2048")
        # Routed pieces remain 512 wide. Shared dense experts use the usual
        # zero-padding loaders for 2048 -> 3072, including empty tail ranks.
        return 512
''')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("vllm_root", type=Path)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    lock = json.loads((here / "runtime-source-lock.json").read_text())
    transformed = {}
    for relative, transform in {
        "model_executor/layers/quantization/exl3.py": patch_exl3,
        "config/virtual_tp.py": patch_virtual_tp,
    }.items():
        path = args.vllm_root / relative
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != lock["input_sha256"][relative]:
            raise ValueError(f"source differs from pinned runtime: {path}")
        result = transform(data.decode())
        compile(result, str(path), "exec")
        transformed[path] = result
    for path, result in transformed.items():
        path.write_text(result)
    shutil.copy2(here / "amos_exl3_tp6.py", args.vllm_root / "amos_exl3_tp6.py")
    print(json.dumps({"patched": [str(p) for p in transformed]}))


if __name__ == "__main__":
    main()
