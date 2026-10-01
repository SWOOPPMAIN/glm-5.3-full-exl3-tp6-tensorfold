"""End a full GLM chat response before it generates another conversation turn.

The checkpoint already stops at user/observation/end-of-text. Its generated
assistant header can otherwise start another answer inside the same response.
Only chat sampling defaults change. Explicit ignore_eos benchmarking and raw
completion requests keep their original behavior.
"""

ROLE_STOPS = {'<|system|>': 154826, '<|assistant|>': 154828}


def defaults(original, request, model_config, tokenizer):
    config = model_config.hf_text_config
    if (config.model_type != 'glm_moe_dsa' or request.ignore_eos
            or config.hidden_size != 6144 or config.num_hidden_layers != 78):
        return original
    for token, expected in ROLE_STOPS.items():
        if tokenizer.convert_tokens_to_ids(token) != expected:
            raise ValueError('GLM chat role tokens differ from the pinned tokenizer')
    return {**original, 'stop_token_ids': list(dict.fromkeys(
        [*(original.get('stop_token_ids') or []), *ROLE_STOPS.values()]))}
