"""Independent BF16 oracle regressions, including FP32 double-rounding traps."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import torch

spec=importlib.util.spec_from_file_location('glm53_reference_under_test',
    Path(__file__).resolve().parents[1]/'tools/glm53_tp6_reference.py')
reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)


class NearestBF16(unittest.TestCase):
    def test_midpoints_from_both_sides_and_even_ties(self):
        values=[1+2**-8+2**-28,1+2**-8-2**-28,-1-2**-8-2**-28,
                -1-2**-8+2**-28,1+2**-8,1+3*2**-8]
        expected=[1+2**-7,1.,-1-2**-7,-1.,1.,1+2**-6]
        result=reference.bf16_nearest(torch.tensor(values,dtype=torch.float64))
        self.assertEqual(result.float().tolist(),expected)

    def test_every_finite_bf16_value_roundtrips(self):
        codes=np.arange(65536,dtype=np.uint32)
        values=(codes<<16).view(np.float32);values=values[np.isfinite(values)]
        original=torch.from_numpy(values.copy()).double()
        self.assertTrue(torch.equal(reference.bf16_nearest(original).double(),original))

    def test_nonfinite_and_out_of_range_inputs_rejected(self):
        for value in (float('nan'),float('inf'),float('-inf'),1e100):
            with self.subTest(value=value),self.assertRaises(ValueError):
                reference.bf16_nearest(torch.tensor([value],dtype=torch.float64))


if __name__=='__main__':unittest.main()
