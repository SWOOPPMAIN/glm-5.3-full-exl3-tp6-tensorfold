"""Full GLM metadata and padding, independent of Flash's NoPE/KDA backbone."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    hidden: int
    layers: int
    heads: int
    q_lora: int
    kv_lora: int
    qk_nope: int
    qk_rope: int
    value: int
    vocab: int
    experts: int
    topk: int
    expert_width: int
    dense_layers: int
    indexer_types: tuple[str, ...]
    rope_theta: float
    norm_eps: float
    routed_scale: float
    eos: tuple[int, ...]

    @classmethod
    def from_dict(cls, raw):
        expected = dict(model_type="glm_moe_dsa", hidden_size=6144,
            num_hidden_layers=78, num_attention_heads=64, q_lora_rank=2048,
            kv_lora_rank=512, qk_nope_head_dim=192, qk_rope_head_dim=64,
            v_head_dim=256, vocab_size=154880, n_routed_experts=256,
            num_experts_per_tok=8, moe_intermediate_size=2048,
            intermediate_size=12288, n_shared_experts=1,
            first_k_dense_replace=3, num_nextn_predict_layers=1,
            n_group=1, topk_group=1, scoring_func="sigmoid",
            topk_method="noaux_tc", norm_topk_prob=True, hidden_act="silu",
            index_n_heads=32, index_head_dim=128, index_topk=2048,
            rope_interleave=True, indexer_rope_interleave=True)
        drift = {k: (raw.get(k), v) for k, v in expected.items() if raw.get(k) != v}
        if drift:
            raise ValueError(f"Unsupported full GLM geometry: {drift}")
        pattern = tuple(raw.get("indexer_types", ()))
        if len(pattern) != 78 or pattern[0] != 'full' or any(k not in ("full", "shared") for k in pattern):
            raise ValueError("Expected explicit 78-layer sparse-index reuse pattern")
        rope = raw.get("rope_parameters", {})
        if rope != {"rope_theta": 8000000, "rope_type": "default"}:
            raise ValueError("Unqualified RoPE parameters")
        if raw.get("routed_scaling_factor") != 2.5 or raw.get("rms_norm_eps") != 1e-5:
            raise ValueError("Unqualified normalization or routed scaling")
        return cls(6144, 78, 64, 2048, 512, 192, 64, 256, 154880,
                   256, 8, 2048, 3, pattern, 8000000.0, 1e-5, 2.5,
                   tuple(raw["eos_token_id"]))

    def head_range(self, rank, world=6):
        """Real heads assigned to a rank; every rank allocates 11 head slots."""
        if world != 6 or not 0 <= rank < world:
            raise ValueError("This port uses six ranks")
        width = (self.heads + world - 1) // world
        start = rank * width
        return start, min(start + width, self.heads), width

    def indexer_source(self, layer):
        """Layer supplying this target layer's selected tokens; draft owns its indexer.

        A shared target layer reuses the current forward's token list from the
        previous full indexer. It must not reuse that layer's attention cache.
        MTP iteration reuse is a separate request-scoped cache commit policy.
        """
        if type(layer) is not int or not 0 <= layer <= self.layers:
            raise ValueError('Expected target layer0..77 or draft layer78')
        if layer == self.layers:
            return layer
        while self.indexer_types[layer] == 'shared':
            layer -= 1
        return layer

    def vocab_range(self, rank, world=6, alignment=64):
        """Aligned local vocabulary, with padded token IDs excluded from sampling."""
        if world != 6 or not 0 <= rank < world or alignment <= 0:
            raise ValueError("Invalid vocabulary partition")
        width = ((self.vocab + world * alignment - 1) // (world * alignment)) * alignment
        start = rank * width
        return start, min(start + width, self.vocab), width
