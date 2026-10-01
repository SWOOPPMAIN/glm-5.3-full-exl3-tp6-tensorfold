"""CPU model assembly contracts; original-weight GPU evidence is separate."""
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock,patch

import torch
from tensorfold.families.glm_moe_dsa.config import Config
from tensorfold.families.glm_moe_dsa.decoder import TP6Reduction
from tensorfold.families.glm_moe_dsa.model import MTPWeights,MTPScratch,TargetForward,FullModel
from tensorfold.families.glm_moe_dsa.vocab import VocabWeights,VocabScratch


def owner(rank=0):
    result=TP6Reduction.__new__(TP6Reduction);result.rank=rank;result.group=object();result.gather=None
    return result


def reader(rank):
    cfg=Config.from_dict(json.loads((Path(__file__).parent/'fixtures/glm53_tp6/config.json').read_text()))
    def meta(name):
        shape=([6144,12288] if name.endswith('eh_proj.weight') else
               [154880,6144] if name in ('model.embed_tokens.weight','lm_head.weight') else [6144])
        return dict(dtype='BF16',shape=shape)
    result=NS(rank=rank,config=cfg,tensor_meta=Mock(side_effect=meta))
    result.read_rows=Mock(side_effect=lambda name,start,stop,device:torch.empty(
        (stop-start,*meta(name)['shape'][1:]),dtype=torch.bfloat16,device=device))
    result.read_tensor=Mock(side_effect=lambda name,device:torch.empty(meta(name)['shape'],dtype=torch.bfloat16,device=device))
    return result


class ModelContracts(unittest.TestCase):
    def test_vocabulary_only_reads_owned_rows_and_norm(self):
        for rank in range(6):
            r=reader(rank);w=VocabWeights(r,'meta')
            calls=[tuple(c.args[:3]) for c in r.read_rows.call_args_list]
            self.assertEqual(calls,[(name,rank*25856,min((rank+1)*25856,154880))
                for name in ('model.embed_tokens.weight','lm_head.weight')])
            r.read_tensor.assert_called_once_with('model.norm.weight','meta')
            self.assertEqual(w.embedding.shape,(25856,6144));self.assertEqual(w.head.shape,w.embedding.shape)

    def test_original_glue_geometry_and_dtype_enforced(self):
        for cls in (VocabWeights,MTPWeights):
            r=reader(0);r.tensor_meta=Mock(return_value=dict(dtype='F16',shape=[6144]))
            with self.assertRaises(ValueError):cls(r,'meta')
            r.read_rows.assert_not_called();r.read_tensor.assert_not_called()

    def test_eh_output_shards_cover_every_row_once(self):
        rows=[]
        for rank in range(6):
            r=reader(rank);w=MTPWeights(r,'meta')
            r.read_rows.assert_called_once_with('model.layers.78.eh_proj.weight',rank*1024,(rank+1)*1024,'meta')
            self.assertEqual(w.eh.shape,(1024,12288));rows.extend(range(rank*1024,(rank+1)*1024))
        self.assertEqual(rows,list(range(6144)))

    def test_head_workspace_does_not_scale_with_prefill_rows(self):
        small,large=[VocabScratch(n,'meta',logit_rows=17) for n in (1,3072)]
        self.assertEqual(small.logits.shape,large.logits.shape)
        self.assertEqual(large.embedding.shape,(3072,6144))
        self.assertEqual(MTPScratch(3072,'meta').concat.shape,(128,12288))
        for n in (0,129,True):
            with self.assertRaises(ValueError):VocabScratch(3072,'meta',logit_rows=n)

    def test_target_rejects_missing_reordered_or_foreign_layers(self):
        layers=[NS(layer=i,rank=0) for i in range(78)]
        wrong=list(layers);wrong[40]=NS(layer=40,rank=1)
        for value in (layers[:-1],list(reversed(layers)),wrong):
            with self.assertRaises(ValueError):TargetForward(value,None,owner())

    def test_target_runs_all_layers_and_adds_final_skip_once(self):
        layers=[NS(layer=i,rank=0) for i in range(78)];seen=[];scope=object();selection=object()
        caches=[NS(layer=i) for i in range(78)]
        def forward(decoder,hidden,residual,positions,bases,slots,cache,table,workspace,selected,*,scope,visible_tokens=None):
            self.assertIs(workspace.weights,decoder.weights)
            self.assertIs(cache,caches[decoder.weights.layer]);self.assertIs(selected,selection)
            seen.append((decoder.weights.layer,visible_tokens))
            # Simulated branches deliberately have different values so losing
            # either the previous residual or final branch changes the result.
            return hidden+1,torch.zeros_like(hidden)+2 if residual is None else residual+2
        workspace=NS(bind=lambda w:setattr(workspace,'weights',w))
        x=torch.zeros((1,6144),dtype=torch.bfloat16);out=torch.empty_like(x);res=torch.empty_like(x)
        def norm(hidden,weight,dst,skip,*,residual):
            skip.copy_(hidden+residual);dst.copy_(skip/2);return dst,skip
        with patch('tensorfold.families.glm_moe_dsa.decoder.Decoder.forward',forward), \
             patch('tensorfold.families.glm_moe_dsa.model.hidden_rms',side_effect=norm) as final_norm:
            model=TargetForward(layers,object(),owner())
            model.forward(x,None,None,None,caches,None,workspace,selection,out,res,scope=scope,visible_tokens=17)
            self.assertEqual(seen,[(i,17) for i in range(78)]);final_norm.assert_called_once()
            self.assertTrue(torch.equal(out,torch.full_like(x,117)))
            with self.assertRaises(ValueError):model.forward(x,None,None,None,caches[:-1],None,workspace,selection,out,res,scope=scope)

    def test_mtp_returns_once_normalized_recycle_and_uses_own_selection(self):
        model=FullModel.__new__(FullModel)
        model.weights=NS(mtp=NS(norm=object()))
        model.vocab=NS(embed=Mock(return_value='embedding'))
        model.mix=NS(forward=Mock(return_value='mixed'))
        hidden=torch.ones((1,6144),dtype=torch.bfloat16);skip=hidden*2
        model.draft=NS(weights=object(),forward=Mock(return_value=(hidden,skip)))
        workspace=NS(weights=model.weights,rows=1,vocab=object(),mtp=object(),
                     decoder=NS(bind=Mock()),mtp_selection=object(),target_selection=object(),
                     hidden=torch.empty_like(hidden),residual=torch.empty_like(hidden))
        ids=torch.tensor([7]);previous=torch.zeros_like(hidden);scope=object()
        def norm(x,w,out,residual_out,*,residual):
            residual_out.copy_(x+residual);out.copy_(residual_out/2);return out,residual_out
        with patch('tensorfold.families.glm_moe_dsa.model.hidden_rms',side_effect=norm) as final_norm:
            result=model.mtp_forward(ids,previous,None,None,None,object(),None,workspace,scope=scope,visible_tokens=17)
        final_norm.assert_called_once();self.assertTrue(torch.equal(result,hidden*1.5))
        self.assertIs(model.mix.forward.call_args.args[1],previous)
        self.assertIs(model.draft.forward.call_args.args[8],workspace.mtp_selection)
        self.assertEqual(model.draft.forward.call_args.kwargs,{'scope':scope,'visible_tokens':17})


if __name__=='__main__':unittest.main()
