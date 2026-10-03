"""CPU checks for dispatch parity, decode isolation and chunk-latched controls."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock,patch

from build_e3_crossover import patch_dispatch


class Boundary(unittest.TestCase):
    def setUp(self):
        self.dir=tempfile.TemporaryDirectory();self.addCleanup(self.dir.cleanup)
        self.path=Path(self.dir.name)/'boundary.json'
        self.policy=types.ModuleType('vllm.amos_e3.policy')
        self.policy.latch=Mock(return_value={'rows':32});self.policy.apply=Mock(return_value='e3-output')
        package=types.ModuleType('vllm.amos_e3');package.__path__=[];package.policy=self.policy
        torch=types.ModuleType('torch');torch.cuda=types.SimpleNamespace(is_current_stream_capturing=Mock(return_value=False))
        dist=types.ModuleType('vllm.distributed');dist.get_tensor_model_parallel_rank=lambda:2
        modules={'vllm':types.ModuleType('vllm'),'vllm.amos_e3':package,'vllm.amos_e3.policy':self.policy,'torch':torch,'vllm.distributed':dist}
        scope=patch.dict(sys.modules,modules);scope.start();self.addCleanup(scope.stop)
        spec=importlib.util.spec_from_file_location('vllm.amos_e3.boundary',Path(__file__).with_name('amos_e3_boundary.py'))
        self.module=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.module)
        package.boundary=self.module;sys.modules['vllm.amos_e3.boundary']=self.module
        self.module.CONTROL=self.path
        latch=self.module.latch
        self.module.latch=lambda name,path=None:latch(name,path or self.path)
        env=patch.dict('os.environ',AMOS_TP6_E3_PREFILL='1',AMOS_EXL3_TP6_PIECES='1');env.start();self.addCleanup(env.stop)
        old=Path(__file__).with_name('amos_grouped_prefill.py').read_bytes().replace(b'from vllm.amos_e3 import runtime',b'from vllm.amos_e3 import policy as runtime')
        dispatch={};exec(compile(patch_dispatch(old),'candidate_dispatch','exec'),dispatch)
        self.dispatch=dispatch['apply_if_prefill']
        self.layer=types.SimpleNamespace(layer_name='model.layers.3.mlp.experts',exl3_max_num_batched_tokens=3072,
            exl3_hidden_size=6144,exl3_intermediate_size_per_partition=512)

    def write(self,boundary,revision='e35-test'):
        self.path.write_text(json.dumps(dict(revision=revision,native_max_rows=boundary)))

    def run_rows(self,rows):
        return self.dispatch(self.layer,types.SimpleNamespace(shape=(rows,6144)),None,None,max_decode_m=32)

    def test_native_decode_does_not_require_or_read_boundary(self):
        for rows in (1,4,20,32):self.assertIsNone(self.run_rows(rows))
        self.policy.latch.assert_not_called();self.policy.apply.assert_not_called()
        with self.assertRaises(FileNotFoundError):self.run_rows(33)

    def test_exact_same_dispatch_for_scoring_and_generation(self):
        for boundary in (512,32):
            self.write(boundary,'e35-boundary'+str(boundary))
            for rows in (33,65,128,256,384,512,513,3072):
                with self.subTest(boundary=boundary,rows=rows):
                    self.assertEqual(self.run_rows(rows),'e3-output' if rows>boundary else None)
            with self.assertRaises(ValueError):self.run_rows(3073)

    def test_chunk_latch_and_next_chunk_refresh(self):
        self.write(512);self.assertIsNone(self.run_rows(128))
        self.write(32,'e35-next');self.layer.layer_name='model.layers.40.mlp.experts'
        self.assertIsNone(self.run_rows(128))
        self.layer.layer_name='model.layers.3.mlp.experts';self.assertEqual(self.run_rows(128),'e3-output')

    def test_invalid_control_or_wrong_kernel_never_falls_back(self):
        for value in ({},{'revision':'e35-a','native_max_rows':64},{'revision':'e35-a','native_max_rows':32.0},
                      {'revision':'../x','native_max_rows':32},{'revision':'e35-a','native_max_rows':32,'fallback':512}):
            self.path.write_text(json.dumps(value))
            with self.assertRaises(ValueError):self.run_rows(256)
        self.write(32);self.policy.latch.return_value={'rows':64}
        with self.assertRaises(ValueError):self.run_rows(256)

    def test_graph_capture_defers_acknowledgment(self):
        self.write(32);torch=sys.modules['torch'];torch.cuda.is_current_stream_capturing.return_value=True
        self.assertEqual(self.run_rows(256),'e3-output');ack=self.path.with_suffix('.applied.json');self.assertFalse(ack.exists())
        torch.cuda.is_current_stream_capturing.return_value=False;self.assertEqual(self.run_rows(256),'e3-output')
        record=json.loads(ack.read_text());self.assertEqual(record['native_max_rows'],32)
        self.assertEqual(record['rank'],2);self.assertEqual(record['route'],'e3')


if __name__=='__main__':unittest.main()
