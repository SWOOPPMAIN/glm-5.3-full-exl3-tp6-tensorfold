"""Exact six-GPU communication tests, called only by the guarded fleet probe."""
import gc
import torch
import torch.distributed as dist

from tensorfold.families.glm_moe_dsa.decoder import TP6Reduction
from tensorfold.families.glm_moe_dsa.memory import tensor_storage_bytes
from tensorfold.families.glm_moe_dsa.reduction_plan import reduction_plan


def qualify(rank, report, check, save, timing):
    reference=TP6Reduction(dist.group.WORLD,rank).prepare(3072,'cuda')
    candidate=TP6Reduction(dist.group.WORLD,rank).prepare(3072,'cuda',bulk_min_rows=1)
    for owner,threshold in ((reference,None),(candidate,1)):
        assert tensor_storage_bytes(owner,device_type='cuda',skip=('group',))==reduction_plan(3072,bulk_min_rows=threshold)['total']
    def oracle(local):
        # Independent CPU arithmetic, in source-rank order. In particular this
        # does not use the production Triton sum or an unordered Torch reduce.
        expected=torch.empty_like(local)
        for start in range(0,len(local),128):
            block=local[start:start+128]
            gathered=torch.empty((6*len(block),6144),dtype=torch.bfloat16,device='cuda')
            dist.all_gather_into_tensor(gathered,block)
            parts=gathered.cpu().reshape(6,len(block),6144).float()
            value=parts[0].clone()
            for source in range(1,6):value=value+parts[source]
            expected[start:start+len(block)].copy_(value.bfloat16())
        return expected
    shapes=(1,5,6,7,17,127,128,129,255,256,257,1024,2053,3071,3072)
    report['collective_timings']={}
    for rows in shapes:
        torch.manual_seed(530614+rank+rows)
        local=torch.randn((rows,6144),dtype=torch.bfloat16,device='cuda')
        # Different columns move small terms around catastrophic cancellation;
        # wrong rank ordering or intermediate BF16 rounding changes the answer.
        patterns=((2**24,1,-2**24,1,0,0),(256,1,-256,1,0,0),
                  (1,2**24,-2**24,0,0,0),(0,0,2**24,1,-2**24,1))
        for col,values in enumerate(patterns):local[:,col]=values[rank]
        before=local.clone();expected=oracle(local)
        old=torch.empty_like(local);new=torch.empty_like(local)
        reference.sum_into(local,old)
        candidate.send.fill_(float('nan'));candidate.shard.fill_(float('nan'))
        candidate.gather.fill_(float('nan'));candidate.sum_into(local,new)
        check(old,expected,f'collective{rows}-reference-cpu')
        check(new,expected,f'collective{rows}-sharded-cpu')
        check(local,before,f'collective{rows}-input-preserved')
        padded=6*((rows+5)//6)
        if padded>rows:
            check(candidate.gather[rows:padded],torch.zeros_like(candidate.gather[rows:padded]),f'collective{rows}-zero-padding')
        # Exercise the intended automatic cutoff with a real communicator.
        candidate.bulk_min_rows=256;candidate.sum_into(local,new)
        check(new,expected,f'collective{rows}-threshold256')
        candidate.bulk_min_rows=1
        item={}
        for name,owner in (('gather',reference),('sharded',candidate)):
            fn=lambda:owner.sum_into(local,new)
            for _ in range(3):fn()
            item[name]=timing(fn,repeats=11)
        report['collective_timings'][str(rows)]=item
        if rows in (1,17,255,256,2053,3072):
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):candidate.sum_into(local,new)
            torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize();dist.barrier()
            graph=torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(graph,stream=stream):candidate.sum_into(local,new)
                for step in range(2):
                    local.copy_(torch.randn_like(local));expected=oracle(local)
                    candidate.send.fill_(float('nan'));graph.replay()
                    check(new,expected,f'collective{rows}-changed-graph{step}')
                item['sharded_graph']=timing(graph.replay,repeats=11)
            finally:graph.reset()
            del graph,stream
        save(f'collective{rows}-complete')
        del local,before,expected,old,new,fn,owner
    # Alias errors must be rejected before any collective is entered.
    x=torch.ones((17,6144),dtype=torch.bfloat16,device='cuda')
    for label,local,out in (
        ('in-place',x,x),('gather-alias',candidate.gather[:17],x),
        ('send-alias',x,candidate.send[:17]),('sum-alias',candidate.shard[:17],x)):
        try:candidate.sum_into(local,out)
        except ValueError:report['cases'].append(dict(case='reject-'+label,passed=True,exact=True))
        else:raise AssertionError('Accepted unsafe alias: '+label)
    del reference,candidate,x,local,out
    torch.cuda.synchronize();gc.collect();torch.cuda.empty_cache();dist.barrier()
    save('collective-qualification-complete')
