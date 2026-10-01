#!/usr/bin/env python3
"""Pin attention-normalization reductions on the exact P23r2 model source."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

RELATIVE = 'model_executor/models/deepseek_v2.py'
EXPECTED = '00944713dae8f9d306e0c2bc0b9041a43eef22077cba160ea074b3cac48d1473'


def transform(raw):
    if hashlib.sha256(raw).hexdigest() != EXPECTED:
        raise ValueError('Stable attention norms require the pinned P23r2 model source')
    source = raw.decode()
    changes = [
        ('from vllm.model_executor.layers.layernorm import LayerNorm, RMSNorm\n',
         'from vllm.model_executor.layers.layernorm import LayerNorm, RMSNorm\n'
         'from vllm.amos_stable_attention_norm import AttentionRMSNorm, IndexerLayerNorm\n', 1),
        ('self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)',
         'self.q_a_layernorm = AttentionRMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)', 2),
        ('self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)',
         'self.kv_a_layernorm = AttentionRMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)', 2),
        ('self.k_norm = LayerNorm(self.head_dim, eps=1e-6)',
         'self.k_norm = IndexerLayerNorm(self.head_dim, eps=1e-6)', 1),
    ]
    for before, after, count in changes:
        if source.count(before) != count:
            raise ValueError('Pinned normalization constructor changed')
        source = source.replace(before, after)
    compile(source, RELATIVE, 'exec')
    return source


def main():
    roots = [Path('/usr/local/lib/python3.12/dist-packages/vllm'),
             Path('/opt/glm53-full/vllm/vllm')]
    prepared = [(root, transform((root/RELATIVE).read_bytes())) for root in roots]
    helper = Path(__file__).with_name('amos_stable_attention_norm.py')
    compile(helper.read_bytes(), str(helper), 'exec')
    for root, source in prepared:
        (root/RELATIVE).write_text(source)
        shutil.copy2(helper, root/helper.name)
    print(json.dumps({'before': EXPECTED, 'after': hashlib.sha256(prepared[0][1].encode()).hexdigest(),
                      'helper': hashlib.sha256(helper.read_bytes()).hexdigest(),
                      'source_weights_changed': False, 'all_request_sizes': True}))


if __name__ == '__main__':
    main()
