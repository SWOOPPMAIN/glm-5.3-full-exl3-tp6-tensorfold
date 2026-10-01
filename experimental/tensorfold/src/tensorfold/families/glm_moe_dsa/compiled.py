"""Load the qualified SM121 expert extension; never compile during inference."""
import hashlib
import importlib.util
import json
from pathlib import Path

_loaded=None
PACKAGED_BINARY=Path('/opt/amos-tensorfold/compiled/tensorfold_exl3_experts_v1.so')


def load_experts(binary=None):
    global _loaded
    from tensorfold.cuda.exl3 import experts
    binary=Path(binary if binary is not None else PACKAGED_BINARY).resolve()
    manifest=json.loads(Path(__file__).with_name('experts-sm121.json').read_text())
    if hashlib.sha256(binary.read_bytes()).hexdigest()!=manifest['binary_sha256']:
        raise ValueError('Expert binary does not match the qualified build')
    source=Path(experts.__file__).parent
    for name,expected in manifest['sources'].items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Expert source differs from the compiled build: '+name)
    if _loaded is None:
        spec=importlib.util.spec_from_file_location('tensorfold_exl3_experts_v1',binary)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        for name in ('group','rot_in','grouped','gateup_epilogue','down_epilogue','down_combine'):
            if not callable(getattr(module,name,None)):
                raise ValueError('Missing compiled expert entry point: '+name)
        _loaded=module
        experts._ext=lambda:_loaded
    return _loaded


def require_experts():
    if _loaded is None:
        raise RuntimeError('Load the qualified SM121 expert binary before creating MoE layers')
