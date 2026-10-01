"""CPU ownership/allocation checks for shared model scratch; GPU parity is separate."""
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch
from tensorfold.families.glm_moe_dsa.decoder import DecoderWorkspace
from tensorfold.families.glm_moe_dsa.experts import RoutedLayer


def weights(layer,rank=0):
    norm=torch.ones(6144,dtype=torch.bfloat16)
    if layer<3:ffn=NS(width=2048)
    else:
        count=sum((4*e+p)%6==rank for e in range(256) for p in range(4))
        routed=RoutedLayer(NS(dims=6144,width=512,count=count),torch.empty(256,dtype=torch.int32))
        ffn=NS(gate=torch.empty((256,6144),dtype=torch.bfloat16),
               shared=NS(weights=NS(width=512 if rank<4 else 0)),routed=routed)
    return NS(layer=layer,rank=rank,input_norm=norm,post_norm=norm,
              attention=NS(weights=NS(real_heads=9 if rank==5 else 11)),ffn=NS(weights=ffn))


class SharedWorkspace(unittest.TestCase):
    def test_all_layers_bind_without_allocation_and_retain_addresses(self):
        dense,moe=weights(0),weights(3)
        arena=DecoderWorkspace(dense,moe,3,4096)
        addresses=[t.data_ptr() for t in (arena.normalized,arena.reduced,arena.attention.output,
                                         arena.moe.routed.kernel.y,arena.dense.output)]
        before=arena.nbytes()
        others=[weights(layer) for layer in (1,2,40,77,78,3)]
        with patch.object(torch,'empty',side_effect=AssertionError('bind allocated')), \
             patch.object(torch,'empty_like',side_effect=AssertionError('bind allocated')):
            for w in others:
                self.assertIs(arena.bind(w),arena)
                self.assertIs(arena.weights,w)
                if w.layer>=3:
                    self.assertIs(arena.ffn.weights,w.ffn.weights)
                    self.assertIs(arena.ffn.routed.layer,w.ffn.weights.routed)
                self.assertEqual(arena.nbytes(),before)
        self.assertEqual(addresses,[t.data_ptr() for t in (arena.normalized,arena.reduced,
            arena.attention.output,arena.moe.routed.kernel.y,arena.dense.output)])

    def test_rank_fragment_and_shared_geometry_are_enforced(self):
        arena=DecoderWorkspace(weights(0),weights(3),1,32)
        wrong=weights(78);wrong.ffn.weights.routed.weights.count-=1
        for w in (weights(0,1),weights(78,5),wrong):
            with self.assertRaises(ValueError):arena.bind(w)
        wrong=weights(78);wrong.ffn.weights.shared.weights.width=0
        with self.assertRaises(ValueError):arena.bind(wrong)
        # Last rank really has two padded attention heads and no shared shard.
        last=DecoderWorkspace(weights(0,5),weights(3,5),1,32)
        last.bind(weights(78,5))
        self.assertEqual(last.attention.real_heads,9)
        self.assertEqual(last.moe.shared.width,0)

    def test_independent_workspaces_do_not_alias(self):
        dense,moe=weights(0),weights(3)
        left,right=[DecoderWorkspace(dense,moe,1,32) for _ in range(2)]
        self.assertNotEqual(left.reduced.data_ptr(),right.reduced.data_ptr())
        self.assertNotEqual(left.moe.routed.kernel.y.data_ptr(),right.moe.routed.kernel.y.data_ptr())
        self.assertEqual(left.nbytes(),right.nbytes())


if __name__=='__main__':unittest.main()
