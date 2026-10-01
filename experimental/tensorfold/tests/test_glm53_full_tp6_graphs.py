"""Host graph planning and actual backend dispatch with a CPU capture recorder.

The fake runtime substitutes CUDA capture/replay only. It runs a simple model
with borrowed output arenas so stale inputs, offsets and hidden aliases remain
observable. Six-node NCCL/full-weight graph qualification is a separate gate.
"""
from collections import OrderedDict
from dataclasses import replace
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch

from tensorfold.families.glm_moe_dsa.graph_plan import graph_key,graph_reserve
from tensorfold.families.glm_moe_dsa.projection_plan import REFERENCE_PLAN
from tensorfold.families.glm_moe_dsa.packed_backend import batch_counters
from tensorfold.families.glm_moe_dsa.graphs import GraphBackend
from tensorfold.families.glm_moe_dsa.request import Extent


class FakeModel:
    def __init__(self):
        self.hidden = torch.empty((17,6144),dtype=torch.bfloat16)
        self.head = torch.empty((17,31))
        self.calls = []

    def target_forward(self,ids,pos,bases,slots,caches,table,workspace,*,scope,visible_tokens):
        self.calls.append(('target',ids.tolist(),pos.tolist(),bases.tolist(),slots.tolist(),visible_tokens))
        self.hidden[:len(ids)] = (ids[:,None]+2*pos[:,None]+bases[:,None]%17+slots[:,None]%19
                                 +torch.arange(6144)[None,:]%13).to(torch.bfloat16)
        return self.hidden[:len(ids)]

    def mtp_forward(self,ids,hidden,pos,bases,slots,cache,table,workspace,*,scope,visible_tokens):
        self.calls.append(('mtp',ids.tolist(),pos.tolist(),bases.tolist(),slots.tolist(),visible_tokens))
        self.hidden[:len(ids)] = (hidden.float()+3*ids[:,None]+5*pos[:,None]
                                  +bases[:,None]%17+slots[:,None]%19).to(torch.bfloat16)
        return self.hidden[:len(ids)]

    def logits(self,hidden,workspace):
        self.calls.append(('head',len(hidden)))
        self.head[:len(hidden)].copy_(hidden[:,:31])
        return self.head[:len(hidden)]


class FakeGraph:
    def __init__(self,run): self.run,self.replays,self.resets=run,0,0
    def replay(self): self.replays+=1;self.run()
    def reset(self): self.resets+=1


class FakeRuntime:
    def __init__(self):
        self.alloc=0;self.growth=128;self.retired=[];self.graphs=[];self.allowed=[];self.reject=False
    def reserved(self): return self.alloc
    def capture(self,run):
        for _ in range(3):run()
        output=run();graph=FakeGraph(run);self.graphs.append(graph);self.alloc+=self.growth
        return graph,output
    def agree(self,okay): self.allowed.append(okay);return okay and not self.reject
    def retire(self,entry):
        self.retired.append(entry.key)
        if entry.graph is not None:entry.graph.reset()
        entry.graph=entry.output=entry.inputs=entry.hidden=None


def backend(max_graphs=16,capture_bytes=512*2**20):
    b=GraphBackend.__new__(GraphBackend)
    b.device=torch.device('cpu');b.capacity,b.rows,b.logit_rows,b.vocab=16384,17,9,31
    b.model,b.caches,b.table,b.workspace=FakeModel(),tuple(range(79)),object(),object()
    b.sampler=lambda logits,positions,sampling: logits.argmax(-1).tolist()
    b.projection_plan=REFERENCE_PLAN
    b.batch_counts=batch_counters()
    b.plan=graph_reserve(max_graphs,5,capture_bytes)
    b.entries=OrderedDict();b.thread=threading.get_ident();b.closed=False;b.broken=None
    b.captures=b.replays=b.evictions=b.eager=b.retained_growth=b.peak_retained_growth=0
    b.runtime=FakeRuntime();b.stream=Mock()
    def owner():
        if threading.get_ident()!=b.thread or b.closed or b.broken is not None:
            raise RuntimeError('unavailable')
    b._stream=owner  # substitute only CUDA stream ownership, preserving thread/failure ownership
    return b


class GraphTests(unittest.TestCase):
    def test_keys_follow_logical_width_and_boundaries(self):
        for visible,bound in [(1,2048),(2048,2048),(2049,4096),(4096,4096),(4097,8192),
                              (359999,524288),(360000,524288),(804000,804000)]:
            self.assertEqual(graph_key('target',5,804000,visible=visible).visible,bound)
        self.assertEqual(graph_key('target',1,512,visible=2).visible,512)
        self.assertEqual(graph_key('head',5,804000).visible,0)
        self.assertIsNone(graph_key('target',6,804000,visible=20))
        self.assertIsNone(graph_key('mtp',2,804000,visible=20))
        for kwargs in [dict(operation='bad',rows=1,capacity=804000,visible=10),
                       dict(operation='target',rows=True,capacity=804000,visible=10),
                       dict(operation='target',rows=1,capacity=804000),
                       dict(operation='target',rows=1,capacity=804000,visible=804001),
                       dict(operation='head',rows=1,capacity=804000,visible=1)]:
            with self.assertRaises(ValueError):graph_key(**kwargs)

    def test_changed_tokens_positions_and_physical_base_reuse_static_addresses(self):
        b=backend();first=Extent(0,6000,1);second=Extent(7000,6000,2)
        kept=b.target([2,3,5],13,first).clone()
        entry=next(iter(b.entries.values()));address=entry.inputs.data_ptr()
        got=b.target([7,11,13],29,second).clone()
        expected=torch.stack([(t+2*p+7000%17+(7000+p)%19+torch.arange(6144)%13).to(torch.bfloat16)
                              for t,p in zip([7,11,13],[29,30,31])])
        self.assertTrue(torch.equal(got,expected))
        self.assertFalse(torch.equal(kept,got))
        self.assertEqual(b.captures,1)
        self.assertEqual(b.replays,2)
        self.assertEqual(entry.inputs.data_ptr(),address)
        self.assertEqual(b.model.calls[-1],('target',[7,11,13],[29,30,31],[7000]*3,[7029,7030,7031],2048))

    def test_context_bucket_transition_captures_new_graph(self):
        b=backend();e=Extent(7000,6000,1)
        b.target([2],2047,e)
        b.target([3],2048,e)
        self.assertEqual([k.visible for k in b.entries],[2048,4096])
        b.target([4],4094,e)
        self.assertEqual(b.captures,2)
        b.target([5],4096,e)
        self.assertEqual(b.captures,3)
        self.assertEqual(b.model.calls[-1][-1],8192)

    def test_target_draft_head_interleaving_preserves_owned_hidden(self):
        b=backend();e=Extent(10,100,1)
        target=b.target([3],0,e).clone()
        draft=b.mtp([7],target,0,e).clone()
        expected=(target.float()+21+10%17+10%19).to(torch.bfloat16)
        self.assertTrue(torch.equal(draft,expected))
        b.target([8],3,e)
        got=b.mtp([2],draft,1,e).clone()
        expected=(draft.float()+6+5+10%17+11%19).to(torch.bfloat16)
        self.assertTrue(torch.equal(got,expected))
        chosen=[]
        b.sampler=lambda logits,positions,sampling: chosen.append(logits.clone()) or [0]*len(positions)
        b.sample(got,[2],None)
        changed=got.clone();changed[:,4]=99
        b.sample(changed,[7],None)
        self.assertTrue(torch.equal(chosen[0],got[:,:31].float()))
        self.assertTrue(torch.equal(chosen[1],changed[:,:31].float()))
        self.assertEqual(b.captures,3)

    def test_lru_is_bounded_and_retirement_precedes_new_capture(self):
        b=backend(max_graphs=2);e=Extent(0,100,1)
        b.target([2],0,e);b.target([2,3],0,e);b.target([3],1,e)
        first,second=b.runtime.graphs
        b.sample(torch.zeros(1,6144,dtype=torch.bfloat16),[1],None)
        self.assertEqual(len(b.entries),2)
        self.assertEqual(first.resets,0)
        self.assertEqual(second.resets,1)
        self.assertEqual(b.evictions,1)
        self.assertEqual([k.rows for k in b.runtime.retired],[2])
        b.target([2,3],1,e)
        self.assertEqual(b.captures,4)
        self.assertEqual(len(b.entries),2)

    def test_bad_metadata_never_changes_graph_cache_or_runs_model(self):
        b=backend();e=Extent(0,100,1)
        for tokens,start,extent in [([True],0,e),([31],0,e),([2],100,e),([2],0,Extent(16384,1,1))]:
            with self.assertRaises(ValueError):b.target(tokens,start,extent)
        self.assertFalse(b.model.calls)
        self.assertFalse(b.entries)
        self.assertIsNone(b.broken)
        b.target([2],0,e)
        key=next(iter(b.entries));entry=b.entries[key]
        with self.assertRaises(ValueError):b._fill(entry,[2],[2048],e,None)

    def test_collective_budget_rejection_latches_without_replay_or_eager_retry(self):
        for remote_rejection in (False,True):
            b=backend(capture_bytes=1);b.runtime.growth=b.plan['total']+1
            if remote_rejection:b.runtime.growth=0;b.runtime.reject=True
            with self.assertRaises(MemoryError):b.target([2],0,Extent(0,100,1))
            self.assertIsNotNone(b.failed_entry)
            self.assertEqual(b.replays,0)
            self.assertEqual(b.eager,0)
            with self.assertRaises(RuntimeError):b.target([2],0,Extent(0,100,1))
            self.assertEqual(len(b.runtime.graphs),1)

    def test_close_releases_graphs_once_and_rejects_reuse(self):
        b=backend();b.target([2],0,Extent(0,100,1));b.target([2,3],0,Extent(0,100,1))
        with patch('tensorfold.families.glm_moe_dsa.graphs.torch.cuda.current_stream',return_value=b.stream):
            b.close_graphs();b.close_graphs()
        self.assertTrue(b.closed)
        self.assertFalse(b.entries)
        self.assertEqual([g.resets for g in b.runtime.graphs],[1,1])
        with self.assertRaises(RuntimeError):b.target([2],0,Extent(0,100,1))

    def test_memory_reserve_bounds_inputs_and_configuration(self):
        p=graph_reserve()
        self.assertLess(p['total'],.502*2**30)
        self.assertGreater(p['inputs'],16*5*6144*2)
        for args in [(0,5,1),(65,5,1),(16,33,1),(16,5,2**30+1)]:
            with self.assertRaises(ValueError):graph_reserve(*args)


if __name__=='__main__':unittest.main()
