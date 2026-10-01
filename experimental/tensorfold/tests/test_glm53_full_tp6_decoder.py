"""CPU protocol tests; these do not qualify six-GPU execution."""
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
import torch
from tensorfold.families.glm_moe_dsa import decoder as module


def cpu_norm(x,w,out,skip,*,residual=None):
    z=x if residual is None else (x.float()+residual.float()).bfloat16()
    skip.copy_(z)
    v=z.double()
    out.copy_((v*torch.rsqrt(v.square().mean(-1,keepdim=True)+1e-5)*w.double()).bfloat16())
    return out,skip


class DecoderContract(unittest.TestCase):
    def test_group_must_be_explicit_six_rank_nccl(self):
        with patch.object(module.dist,'is_initialized',return_value=True), \
             patch.object(module.dist,'get_world_size',return_value=6) as world, \
             patch.object(module.dist,'get_rank',return_value=2) as rank, \
             patch.object(module.dist,'get_backend',return_value='nccl') as backend:
            group=object();owner=module.TP6Reduction(group,2)
            self.assertIs(owner.group,group)
            for invalid in (None,):
                with self.assertRaises(ValueError):module.TP6Reduction(invalid,2)
            world.return_value=5
            with self.assertRaises(ValueError):module.TP6Reduction(group,2)
            world.return_value=6;rank.return_value=1
            with self.assertRaises(ValueError):module.TP6Reduction(group,2)
            rank.return_value=2;backend.return_value='gloo'
            with self.assertRaises(ValueError):module.TP6Reduction(group,2)

    def test_both_reductions_precede_next_nonlinearity(self):
        torch.manual_seed(607);x=torch.randn(3,6144).bfloat16()
        initial_skip=torch.randn_like(x)
        weights=[torch.randn(6144).bfloat16() for _ in range(2)]
        # Independent collective inputs are computed from the full mathematical
        # layer, including each rank's BF16 rounding BEFORE its sum.
        for residual in (None,initial_skip):
            z=x if residual is None else (x.float()+residual.float()).bfloat16()
            h=(z.double()*torch.rsqrt(z.double().square().mean(-1,keepdim=True)+1e-5)*weights[0].double()).bfloat16()
            att_parts=[(h.float()*(r+1)/8).bfloat16() for r in range(6)]
            att=torch.stack(att_parts).float().sum(0).bfloat16()
            post_skip=(att.float()+z.float()).bfloat16()
            post=(post_skip.double()*torch.rsqrt(post_skip.double().square().mean(-1,keepdim=True)+1e-5)*weights[1].double()).bfloat16()
            mlp_parts=[(post.float().square()*(r+2)/16).bfloat16() for r in range(6)]
            expected=torch.stack(mlp_parts).float().sum(0).bfloat16()
            for rank in range(6):
                test=self
                class CheckedReduction(module.TP6Reduction):
                    def __init__(self):self.rank=rank;self.calls=0
                    def sum_into(self,local,out):
                        test.assertLess(self.calls,2)
                        wanted=(att_parts,mlp_parts)[self.calls][rank]
                        test.assertTrue(torch.equal(local,wanted),'Wrong input at collective boundary')
                        out.copy_((att,expected)[self.calls]);self.calls+=1;return out
                owner=CheckedReduction()
                attention=NS(forward=lambda hidden,*a,**kw:(hidden.float()*(rank+1)/8).bfloat16())
                ffn=NS(forward=lambda hidden,s:(hidden.float().square()*(rank+2)/16).bfloat16())
                w=NS(rank=rank,input_norm=weights[0],post_norm=weights[1],attention=attention,ffn=ffn)
                s=NS(weights=w,rows=3,attention=NS(output=torch.empty_like(x)),ffn=None,ffn_rows=2,
                     **{k:torch.empty_like(x) for k in
                     ('normalized','input_residual','post_residual','reduced')})
                before=x.clone();previous=None if residual is None else residual.clone()
                with patch.object(module,'hidden_rms',cpu_norm):
                    y,skip=module.Decoder(w,owner).forward(x,residual,None,None,None,None,None,s,None,scope=object())
                self.assertEqual(owner.calls,2)
                self.assertTrue(torch.equal(y,expected))
                self.assertTrue(torch.equal(skip,post_skip))
                self.assertTrue(torch.equal(x,before))
                if residual is not None:self.assertTrue(torch.equal(residual,previous))
                # Skip must not be included in the MLP sum: it is added exactly
                # once by the next layer (or final normalization).
                self.assertFalse(torch.equal(y,(expected.float()+post_skip.float()).bfloat16()))

    def test_decoder_rejects_identity_or_wrong_rank_reduction(self):
        with self.assertRaises(ValueError):module.Decoder(NS(rank=0),NS(rank=0,world_size=6))
        owner=module.TP6Reduction.__new__(module.TP6Reduction);owner.rank=5
        with self.assertRaises(ValueError):module.Decoder(NS(rank=0),owner)


if __name__=='__main__':unittest.main()
