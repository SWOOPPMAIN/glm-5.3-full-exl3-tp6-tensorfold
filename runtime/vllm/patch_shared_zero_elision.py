#!/usr/bin/env python3
"""Add canonical shared partials and proven-empty work elision to P18R3."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

HASHES = {
    'model_executor/models/deepseek_v2.py': '156264a8139288482b3de1e336b94b8454d654ade12c6d4bfca379fb8c64f1e9',
    'model_executor/kernels/linear/mxfp8/b12x.py': '4488c81aefd59a44ecac5867d33a1a5ea1dbeb82e5009b93cf686940375575a4',
    'amos_shared_expert_padding.py': '04979570134066bfb86cf0e4f6ad92e0f74acb282e4e2715dca5954c14cebe3f',
}


def patch(root):
    prepared = {}
    for name, digest in HASHES.items():
        path = root/name
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError('Source differs from pinned P18R3: '+name)
        source = raw.decode()
        if name == 'model_executor/models/deepseek_v2.py':
            anchor = '        self.act_fn = SiluAndMul()\n\n    def forward(self, x):\n        gate_up, _ = self.gate_up_proj(x)\n'
            replacement = ('        self.act_fn = SiluAndMul()\n'
                '        from vllm.amos_shared_zero_elision import configure\n'
                '        self.amos_shared_zero = configure(prefix, intermediate_size, hidden_size)\n\n'
                '    def forward(self, x):\n'
                '        if self.amos_shared_zero:\n'
                '            if not (getattr(self.gate_up_proj, "amos_shared_zero_verified", False)\n'
                '                    and getattr(self.down_proj, "amos_shared_zero_verified", False)):\n'
                '                raise RuntimeError("Shared empty shard was not verified after loading")\n'
                '            return torch.zeros_like(x)\n'
                '        gate_up, _ = self.gate_up_proj(x)\n')
            if source.count(anchor) != 1:
                raise ValueError('Shared MLP forward anchor changed')
            source = source.replace(anchor,replacement)
        elif name == 'amos_shared_expert_padding.py':
            changes = {
                "AXIS = {'original_size':2048, 'padded_size':2304, 'tp_size':6, 'local_size':384}\n":
                    "AXIS = {'original_size':2048, 'padded_size':2304, 'tp_size':6, 'local_size':384}\n\n"
                    "def active_axis():\n    from vllm.amos_shared_zero_elision import axis\n    return axis(AXIS)\n",
                'return dict(AXIS)': 'return active_axis()',
                "plan.get('shared_expert_intermediate_size')!=AXIS": "plan.get('shared_expert_intermediate_size')!=active_axis()",
                "return AXIS['padded_size']": "return active_axis()['padded_size']",
            }
            for before,after in changes.items():
                if source.count(before) != 1:
                    raise ValueError('Shared axis anchor changed')
                source = source.replace(before,after)
        else:
            anchor = '        _register_b12x_mxfp8_linear_layer(layer)\n'
            if source.count(anchor) != 1:
                raise ValueError('Shared loaded-weight verification anchor changed')
            source = source.replace(anchor,
                '        from vllm.amos_shared_zero_elision import verify\n'
                '        verify(layer, weight, weight_scale)\n'+anchor)
        compile(source,str(path),'exec'); prepared[path] = source
    for path,source in prepared.items():
        path.write_text(source)
    shutil.copy2(Path(__file__).with_name('amos_shared_zero_elision.py'),root/'amos_shared_zero_elision.py')
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in prepared}


if __name__ == '__main__':
    print(json.dumps(patch(Path(sys.argv[1]))))
