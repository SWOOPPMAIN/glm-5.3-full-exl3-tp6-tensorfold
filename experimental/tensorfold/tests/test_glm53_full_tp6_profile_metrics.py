import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('profile_metrics',Path(__file__).resolve().parents[1]/'tools/glm53_tp6_profile_metrics.py')
metrics=importlib.util.module_from_spec(spec);spec.loader.exec_module(metrics)

class ProfileMetricsTests(unittest.TestCase):
    def test_overlapping_streams_are_counted_once(self):
        self.assertEqual(metrics.union_us([(10,20),(0,30),(35,40),(40,45),(7,8)]),40)
    def test_disjoint_devices_are_not_merged(self):
        events=[dict(ph='X',cat='kernel',name='ncclKernel_AllGather',ts=0,dur=10,pid=0),
                dict(ph='X',cat='kernel',name='expert',ts=5,dur=10,pid=0),
                dict(ph='X',cat='kernel',name='expert',ts=5,dur=10,pid=1)]
        r=metrics.summarize({'traceEvents':events})
        self.assertEqual(r['kernel_sum_us'],30)
        self.assertEqual(r['devices']['0']['kernel_union_us'],15)
        self.assertEqual(r['devices']['1']['kernel_union_us'],10)
        self.assertEqual(r['nccl_kernel_sum_us'],10)
    def test_cpu_ranges_are_separate_from_gpu_activity(self):
        events=[dict(ph='X',cat='kernel',name='expert',ts=5,dur=10,pid=0),
                dict(ph='X',cat='user_annotation',name='tfp19.target.r5',ts=0,dur=30,pid=4),
                dict(ph='X',cat='cpu_op',name='ignored',ts=0,dur=300,pid=4)]
        r=metrics.summarize({'traceEvents':events})
        self.assertEqual(r['kernel_sum_us'],10)
        self.assertEqual(r['annotations']['tfp19.target.r5']['host_total_us'],30)
    def test_invalid_and_missing_gpu_data_are_rejected(self):
        for spans in [[(2,1)],[(float('nan'),1)],[(0,float('inf'))]]:
            with self.assertRaises(ValueError):metrics.union_us(spans)
        with self.assertRaises(ValueError):metrics.summarize({'traceEvents':[]})

if __name__=='__main__':unittest.main()
