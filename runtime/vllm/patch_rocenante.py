#!/usr/bin/env python3
"""Port pinned upstream RoCEnante onto the retained GLM runtime's older API."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).parent / 'rocenante'
LOCK = json.loads((HERE / 'source-lock.json').read_text())


def replace(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f'patch anchor expected once: {old[:100]!r}')
    return text.replace(old, new)


def patch(vllm, b12x, patch_b12x=True):
    inputs = {}
    for name, digest in LOCK['base_files'].items():
        library, _, relative = name.split('/', 2)
        if library == 'b12x' and not patch_b12x:
            continue
        path = (vllm if library == 'vllm' else b12x) / relative
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise RuntimeError(f'unexpected source hash: {path}')
        inputs[path] = data.decode()
    for name, digest in LOCK['vendor_files'].items():
        if hashlib.sha256((HERE / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'vendor source changed: {name}')

    path = vllm / 'envs.py'
    text = inputs[path]
    text = replace(text, '    VLLM_ENABLE_PCIE_ALLREDUCE: bool = False',
        '    VLLM_ENABLE_ROCE_ALLREDUCE: bool = False\n'
        '    VLLM_ROCE_ALLREDUCE_MAX_SIZE: str = "2MB"\n'
        '    VLLM_ROCE_ALLGATHER_MAX_SIZE: str = "16MB"\n'
        '    VLLM_ENABLE_PCIE_ALLREDUCE: bool = False')
    anchor = '    "VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE": lambda: os.getenv('
    text = replace(text, anchor,
        '    "VLLM_ENABLE_ROCE_ALLREDUCE": lambda: bool(int(os.getenv("VLLM_ENABLE_ROCE_ALLREDUCE", "0"))),\n'
        '    "VLLM_ROCE_ALLREDUCE_MAX_SIZE": lambda: os.getenv("VLLM_ROCE_ALLREDUCE_MAX_SIZE", "2MB"),\n'
        '    "VLLM_ROCE_ALLGATHER_MAX_SIZE": lambda: os.getenv("VLLM_ROCE_ALLGATHER_MAX_SIZE", "16MB"),\n' + anchor)
    inputs[path] = text

    path = vllm / 'distributed/device_communicators/cuda_communicator.py'
    text = inputs[path]
    anchor = '        self.ca_comm: CustomAllreduce | None = None'
    text = replace(text, anchor,
        '        from .b12x_roce_all_reduce import B12xRoceAllReduce\n'
        '        self.b12x_ar_comm: B12xRoceAllReduce | None = None\n' + anchor)
    anchor = '        if use_custom_allreduce and self.aiter_ar_comm is None and self.world_size > 1:'
    text = replace(text, anchor,
        '        if use_custom_allreduce and envs.VLLM_ENABLE_ROCE_ALLREDUCE and self.world_size > 1:\n'
        '            self.b12x_ar_comm = B12xRoceAllReduce(\n'
        '                group=self.cpu_group, device_group=self.device_group, device=self.device\n'
        '            )\n\n' + anchor)
    # GB10 has no MNNVL multicast. Its speculative rendezvous can strand one
    # rank while peers advance to the next CPU collective. RoCEnante already
    # owns the fast path and PyNccl handles ineligible inputs.
    text = replace(text, anchor,
        '        if (use_custom_allreduce and self.aiter_ar_comm is None and self.world_size > 1\n'
        '                and (self.b12x_ar_comm is None or self.b12x_ar_comm.disabled)):')
    text = replace(text, '        all_potential_ar_backends = [',
        '        all_potential_ar_backends = [\n            "B12X_ROCENANTE",')
    text = replace(text, '        enabled_ar_backends: list[str] = []',
        '        enabled_ar_backends: list[str] = []\n'
        '        if self.b12x_ar_comm is not None and not self.b12x_ar_comm.disabled:\n'
        '            enabled_ar_backends.append(self.b12x_ar_comm.backend_name)')
    anchor = '    def all_reduce(self, input_):'
    text = replace(text, anchor, anchor + '\n'
        '        roce = self.b12x_ar_comm\n'
        '        if roce is not None and roce.should_custom_ar(input_):\n'
        '            return roce.custom_all_reduce(input_)')
    anchor = '    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:'
    text = replace(text, anchor, anchor + '\n'
        '        roce = self.b12x_ar_comm\n'
        '        if roce is not None and roce.should_all_gather(input_, dim):\n'
        '            return roce.all_gather(input_, dim)')
    anchor = '    def destroy(self):'
    text = replace(text, anchor, anchor + '\n'
        '        if self.b12x_ar_comm is not None:\n'
        '            self.b12x_ar_comm.close()\n'
        '            self.b12x_ar_comm = None')
    inputs[path] = text

    path = vllm / 'distributed/parallel_state.py'
    text = inputs[path]
    text = replace(text, '        maybe_ca_context = nullcontext()',
        '        maybe_roce_context = nullcontext()\n        maybe_ca_context = nullcontext()')
    anchor = '            ca_comm = self.device_communicator.ca_comm'
    text = replace(text, anchor,
        '            roce = getattr(self.device_communicator, "b12x_ar_comm", None)\n'
        '            if roce is not None:\n'
        '                maybe_roce_context = roce.capture(stream=stream)\n' + anchor)
    text = replace(text, '        with torch.cuda.stream(stream), maybe_ca_context, maybe_aiter_context:',
        '        with torch.cuda.stream(stream), maybe_roce_context, maybe_ca_context, maybe_aiter_context:')
    inputs[path] = text

    path = vllm / 'v1/worker/gpu_worker.py'
    text = inputs[path]
    text = replace(text,
        '            self.model_runner.load_model(load_dummy_weights=load_dummy_weights)\n',
        '            self.model_runner.load_model(load_dummy_weights=load_dummy_weights)\n'
        '\n        # Release temporary loading slabs before waiting for slower peers.\n'
        '        self._release_unoccupied_accelerator_memory()\n'
        '        logger.info("Post-load CUDA allocated=%.2f GiB reserved=%.2f GiB",\n'
        '                    torch.cuda.memory_allocated() / 2**30,\n'
        '                    torch.cuda.memory_reserved() / 2**30)\n')
    text = replace(text, 'class Worker(WorkerBase):',
        (HERE / 'worker-health-class.txt').read_text() + '\n\nclass Worker(WorkerBase):')
    text = replace(text, '        return self.model_runner.sample_tokens(grammar_output)',
        '        return self._b12x_roce_guarded(self.model_runner.sample_tokens(grammar_output))\n\n' +
        (HERE / 'worker-health-methods.txt').read_text())
    text = replace(text,
        '                output, ModelRunnerOutput | AsyncModelRunnerOutput | NoneType\n            ):\n                return output',
        '                output, ModelRunnerOutput | AsyncModelRunnerOutput | NoneType\n            ):\n                return self._b12x_roce_guarded(output)')
    inputs[path] = text

    if patch_b12x:
        path = b12x / 'comm/__init__.py'
        inputs[path] = replace(inputs[path], '_OP_MODULES = ("pcie",)', '_OP_MODULES = ("pcie", "roce")')
    # Validate every changed Python file before committing the overlay.
    for path, text in inputs.items():
        compile(text, str(path), 'exec')
    for path, text in inputs.items():
        path.write_text(text)
    if patch_b12x:
        shutil.copytree(HERE / 'roce', b12x / 'comm/roce', dirs_exist_ok=True)
    adapter = (HERE / 'b12x_roce_all_reduce.py').read_text()
    adapter = replace(adapter, 'device_communicators.b12x_pcie_all_reduce import',
        'device_communicators.custom_all_reduce import')
    (vllm / 'distributed/device_communicators/b12x_roce_all_reduce.py').write_text(adapter)
    print(f'Applied pinned RoCEnante port: {vllm}, {b12x}')


if __name__ == '__main__':
    patch(Path(sys.argv[1]), Path(sys.argv[2]), '--vllm-only' not in sys.argv)
