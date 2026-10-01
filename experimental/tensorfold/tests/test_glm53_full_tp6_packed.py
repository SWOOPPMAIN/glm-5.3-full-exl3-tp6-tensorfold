"""Packed state/control and actual row-metadata backend tests; GPU gate separate."""
from types import MethodType
import unittest
import torch

from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.glm_moe_dsa.request import RequestEngine,Extent
from tensorfold.families.glm_moe_dsa.request_ops import Call
from tensorfold.families.glm_moe_dsa.control import Replica,start_command,TransportBroken
from tensorfold.families.glm_moe_dsa.graph_plan import graph_reserve,graph_key
from tensorfold.families.glm_moe_dsa.packed_backend import packed_reserve
from test_glm53_full_tp6_request import CausalBackend,serial
from test_glm53_full_tp6_control import Fleet
from test_glm53_full_tp6_graphs import backend
import test_glm53_full_tp6_scheduler as scheduler_tests


def enable_batch(b):
    b.groups=[]
    def batch(self,calls):
        assert 1<=len(calls)<=4 and len({c.operation for c in calls})==1
        limit=self.logit_rows if calls[0].operation=='sample' else self.rows
        assert sum(c.rows for c in calls)<=limit
        self.groups.append((calls[0].operation,tuple(c.rows for c in calls)))
        result=[]
        for call in calls:
            value=getattr(self,call.operation)(*call.args)
            result.append(value.clone() if isinstance(value,torch.Tensor) else value)
        return result
    b.batch=MethodType(batch,b)
    return b


class PackedCore(unittest.TestCase):
    def test_mixed_phases_sampling_depths_and_rejections_match_independent_serial(self):
        b=enable_batch(CausalBackend(mismatch=range(0,300,3)));e=RequestEngine(b,context_limit=128)
        settings=[None,Sampling(13,.7,7,.92),Sampling(91,.9,0,.9,.02),None]
        prompts=[[2,3],[4,8,1]*9,[7]*6,[5,9,2]*4]
        requests=[e.start('first',prompts[0],max_tokens=23,draft_tokens=0)]
        e.step_many(requests,[False])
        for i in range(1,4):requests.append(e.start(str(i),prompts[i],max_tokens=23-i,sampling=settings[i],draft_tokens=[0,1,4,8][i]))
        for _ in range(120):
            active=[r for r in requests if r.status in ('prefill','decode')]
            if not active:break
            # Changing client order and membership cannot change seeded output.
            active.reverse();e.step_many(active,[False]*len(active))
        self.assertTrue(all(r.status=='finished' for r in requests))
        for i,r in enumerate(requests):self.assertEqual(r.output,serial(prompts[i],23-i,settings[i]))
        self.assertTrue(any(len(shape)>1 for _,shape in b.groups))
        self.assertTrue(all(sum(shape)<=(b.logit_rows if op=='sample' else b.rows) for op,shape in b.groups))
        self.assertTrue(any(r.accepted<r.drafted for r in requests))

    def test_cancellation_retained_resume_and_physical_reuse(self):
        b=enable_batch(CausalBackend());e=RequestEngine(b,context_limit=128)
        a=e.start('a',[2,3,5],max_tokens=10);other=e.start('other',[1]*20,max_tokens=20)
        e.step_many([a,other],[False,True]);self.assertEqual(other.status,'cancelled');self.assertFalse(other.output)
        e.drop(other)
        while a.status!='finished':e.step_many([a],[False])
        prompt=[*a.tokens,7,3];kept=len(a.tokens);old=a.extent
        a=e.start('continued',prompt,max_tokens=8,resume=a)
        c=e.start('replacement',[9,3],max_tokens=9)
        self.assertEqual(a.reused_tokens,kept);self.assertEqual(a.extent.base,old.base)
        while True:
            active=[r for r in (a,c) if r.status in ('prefill','decode')]
            if not active:break
            e.step_many(active,[False]*len(active))
        self.assertEqual(a.output,serial(prompt,8,None));self.assertEqual(c.output,serial([9,3],9,None))
        e.drop(a);e.drop(c);self.assertEqual(e.pool.free,[(0,b.capacity)])

    def test_eos_and_zero_draft_members_leave_group_independently(self):
        b=enable_batch(CausalBackend());b.eos=(serial([2,5],1,None)[0],)
        e=RequestEngine(b,context_limit=128)
        a=e.start('eos',[2,5],max_tokens=10);c=e.start('long',[2,5],max_tokens=7,draft_tokens=0,ignore_eos=True)
        e.step_many([a,c],[False,False]);self.assertEqual(a.status,'finished');self.assertEqual(len(a.output),1)
        while c.status!='finished':e.step_many([c],[False])
        self.assertEqual(c.output,serial([2,5],7,None))

    def test_bad_group_is_rejected_before_any_work_and_failure_retains_leases(self):
        b=enable_batch(CausalBackend());e=RequestEngine(b,context_limit=128)
        a=e.start('a',[1,2],max_tokens=8);c=e.start('c',[2,3],max_tokens=8)
        for rs,flags in [([a,a],[False,False]),([a,c],[False]),([a],[1]),([] ,[])]:
            with self.assertRaises(ValueError):e.step_many(rs,flags)
        self.assertFalse(b.groups);self.assertFalse(a.tokens)
        b.fault=True
        with self.assertRaises(RuntimeError):e.step_many([a,c],[False,False])
        self.assertEqual([a.status,c.status],['failed','failed']);self.assertEqual(len(e.pool.leases),2)
        with self.assertRaises(ValueError):e.step_many([a,c],[False,False])
        e.drop(a);e.drop(c);self.assertFalse(e.pool.leases)


class PackedBackendTests(unittest.TestCase):
    def test_changed_batch_membership_replays_with_per_row_cache_bases(self):
        b=backend();b.plan=graph_reserve(64,32)
        a,c=Extent(0,100,1),Extent(300,100,2)
        calls=[Call('target',([2,3],0,a)),Call('target',([7,9,11],17,c))]
        got=b.batch(calls);saved=[v.clone() for v in got]
        self.assertEqual(b.captures,1);self.assertEqual(b.model.calls[-1][3],[0,0,300,300,300])
        changed=[Call('target',([5],3,c)),Call('target',([1,4,6,8],9,a))]
        out=b.batch(changed);self.assertEqual(b.captures,1)
        self.assertEqual(b.model.calls[-1][3],[300,0,0,0,0])
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(got,saved)))
        expected=backend()
        for value,call in zip(out,changed):self.assertTrue(torch.equal(value,expected.target(*call.args)))
        self.assertEqual(b.batch_counts['target']['multi_request_groups'],2)

    def test_packed_mtp_and_head_preserve_hidden_and_sampling_policy(self):
        b=backend();b.plan=graph_reserve(64,32)
        a,c=Extent(0,100,1),Extent(300,100,2)
        hidden=[torch.full((2,6144),3,dtype=torch.bfloat16),torch.full((3,6144),5,dtype=torch.bfloat16)]
        calls=[Call('mtp',([2,3],hidden[0],1,a)),Call('mtp',([7,8,9],hidden[1],8,c))]
        got=b.batch(calls)
        self.assertEqual(next(iter(b.entries)).operation,'mtp_batch')
        expected=backend()
        for value,call in zip(got,calls):self.assertTrue(torch.equal(value,expected.mtp(*call.args)))
        seen=[];policies=[None,Sampling(9,.8,7,.95)]
        b.sampler=lambda logits,pos,s: seen.append((logits.clone(),pos,s)) or [3]*len(pos)
        output=b.batch([Call('sample',(got[0],[2,3],policies[0])),Call('sample',(got[1],[9,10,11],policies[1]))])
        self.assertEqual(output,[[3,3],[3,3,3]])
        self.assertEqual([x[1] for x in seen],[[2,3],[9,10,11]])
        self.assertEqual([x[2] for x in seen],policies)
        self.assertTrue(torch.equal(seen[0][0],got[0][:,:31].float()))
        self.assertEqual(b.batch_counts['sample']['groups'],1)
        self.assertEqual(graph_key('mtp',2,804000,visible=10,max_rows=32),None)
        self.assertEqual(graph_key('mtp_batch',20,804000,visible=10,max_rows=32).rows,20)

    def test_overlap_and_invalid_geometry_are_rejected_before_model_execution(self):
        b=backend();a=Extent(0,100,1);c=Extent(90,100,2)
        for calls in [[],[Call('target',([2],0,a)),Call('target',([3],0,c))],
                      [Call('target',([2],0,a)),Call('sample',(torch.zeros(1,6144,dtype=torch.bfloat16),[1],None))],
                      [Call('target',([2]*18,0,a))]]:
            with self.assertRaises(ValueError):b.batch(calls)
        self.assertFalse(b.entries);self.assertFalse(b.model.calls)
        self.assertGreater(packed_reserve(3072),200*2**20)


class PackedControl(unittest.TestCase):
    def test_real_six_store_commands_and_preparation_rejection(self):
        def configure(rank,core):enable_batch(core.backend)
        fleet=Fleet(configure=configure);self.addCleanup(fleet.close)
        for i in range(4):fleet.dispatch(start_command(str(i),[i+1,3,8]*4,max_tokens=12,draft_tokens=i))
        before=fleet.head.completed
        with self.assertRaises(ValueError):fleet.dispatch(dict(op='step_many',args=dict(keys=['0','0'],cancelled=[False,False])))
        self.assertEqual(fleet.head.completed,before)
        while True:
            keys=[k for k,r in fleet.head.replica.core.requests.items() if r.status in ('prefill','decode')]
            if not keys:break
            fleet.dispatch(dict(op='step_many',args=dict(keys=keys,cancelled=[False]*len(keys))))
        for rank,c in fleet.controllers.items():
            for i in range(4):self.assertEqual(c.replica.core.requests[str(i)].output,serial([i+1,3,8]*4,12,None))
        fleet.close();self.assertFalse(fleet.errors);self.assertEqual(fleet.master.num_keys(),1)

    def test_partial_batch_failure_latches_all_ranks_without_reissuing(self):
        def configure(rank,core):enable_batch(core.backend);core.backend.fault=rank==3
        fleet=Fleet(configure=configure)
        for i in range(2):fleet.dispatch(start_command(str(i),[i+1,3],max_tokens=7))
        with self.assertRaises(TransportBroken):fleet.dispatch(dict(op='step_many',args=dict(keys=['0','1'],cancelled=[False,False])))
        with self.assertRaises(TransportBroken):fleet.dispatch(dict(op='step_many',args=dict(keys=['0','1'],cancelled=[False,False])))
        fleet.close();self.assertEqual(set(fleet.errors),{1,2,3,4,5})
        for c in fleet.controllers.values():self.assertEqual(len(c.replica.core.pool.leases),2)


class PackedSchedulerTests(scheduler_tests.SchedulerTests):
    def fleet(self,**kwargs):
        existing=kwargs.pop('configure',None)
        def configure(rank,b):
            enable_batch(b)
            if existing:existing(rank,b)
        return super().fleet(packed=True,configure=configure,**kwargs)

if __name__=='__main__':unittest.main()
