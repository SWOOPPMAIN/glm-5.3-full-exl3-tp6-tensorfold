"""Real GPU tests for empty-tile skipping and larger attention row batches."""
import gc,math
import torch
from tensorfold.families.glm_moe_dsa.attention import AttentionScratch,attend
from tensorfold.families.glm_moe_dsa.cache import LayerCache,write_mla


def qualify(rank,report,check,save,timing):
    torch.manual_seed(530615+rank)
    capacity=364128
    latent=torch.randn((capacity,512),dtype=torch.bfloat16,device='cuda')*.2
    rope=torch.randn((capacity,64),dtype=torch.bfloat16,device='cuda')*.2
    cache=LayerCache(0,capacity,'cuda',indexer=False)
    slots=torch.arange(capacity,device='cuda');write_mla(latent,rope,slots,cache)
    counts0=torch.tensor([0,1,17,31,32,33,511,512,513,1023,1024,1025,1535,1536,1537,2047,2048,
                          17,33,512,2048,-3,3000],dtype=torch.int32,device='cuda')
    count_list=counts0.cpu().tolist();n=len(count_list)
    pos0=torch.tensor([max(c,0) for c in count_list],dtype=torch.int64,device='cuda')
    pos0[16]=359999;pos0[17]=4095;pos0[18]=32;pos0[19]=511;pos0[20]=2047;pos0[22]=2047
    base0=torch.zeros(n,dtype=torch.int64,device='cuda');base0[17]=360032;base0[18]=capacity-10;base0[20]=-20
    tok0=torch.full((n,2048),-1,dtype=torch.int32,device='cuda')
    for row,count in enumerate(count_list):
        width=max(0,min(2048,count))
        if width:tok0[row,:width]=torch.linspace(0,int(pos0[row]),width,device='cuda').round().int()
    tok0[19].fill_(-1)  # A nonzero count with no readable keys.
    tok0[18,0]=-1;tok0[18,32]=pos0[18]+1
    qa0=torch.randn((n,11,512),dtype=torch.bfloat16,device='cuda')*.2
    qr0=torch.randn((n,11,64),dtype=torch.bfloat16,device='cuda')*.2
    def repeated(x,rows):return x.repeat(((rows+n-1)//n,)+(1,)*(x.ndim-1))[:rows].contiguous()
    def oracle(qa,qr,lc,kc,scales,real):
        expected=torch.zeros_like(qa,dtype=torch.float64)
        for row,count in enumerate(count_list):
            ids=tok0[row,:max(0,min(2048,count))].long()
            valid=(ids>=0)&(ids<=pos0[row])&(ids+base0[row]>=0)&(ids+base0[row]<capacity)
            addresses=ids[valid]+base0[row]
            if not len(addresses):continue
            values=lc[addresses] if scales is None else lc.view(torch.uint8)[addresses].view(lc.dtype)
            if scales is not None:
                values=(values.float().reshape(-1,4,128)*scales[addresses,:,None]).reshape(-1,512).bfloat16()
            scores=(qa[row,:real].double()@values.double().T+qr[row,:real].double()@kc[addresses].double().T)/16
            expected[row,:real]=scores.softmax(-1)@values.double()
        return expected
    def close(got,expected,label):
        x,y=got.double(),expected.double();denom=float(y.square().mean())
        same=torch.equal(x,y);relative=float((x-y).square().mean().sqrt())/math.sqrt(denom) if denom else (0. if same else float('inf'))
        cosine=float(torch.nn.functional.cosine_similarity(x.flatten(),y.flatten(),dim=0)) if denom else float(same)
        ok=bool(torch.isfinite(x).all()) and relative<.006 and cosine>.99995
        report['cases'].append(dict(case=label,passed=ok,exact=False,relative_rms=relative,cosine=cosine,tolerance=.006))
        save(label);assert ok,(label,relative,cosine)
    report['attention_component_timings']={}
    for compact in (False,True):
        lc,kc,scales=(cache.latent,cache.rope,cache.scales) if compact else (latent,rope,None)
        for real in (9,11):
            prefix=f'compact{compact}-heads{real}'
            qa=qa0.clone();qr=qr0.clone()
            if real==9:qa[:,9:]=float('nan');qr[:,9:]=float('nan')
            ref_small=torch.empty_like(qa);scratch=AttentionScratch(n,'cuda',real)
            attend(qa,qr,lc,kc,tok0,counts0,pos0,base0,ref_small,scratch,latent_scales=scales)
            expected=oracle(qa,qr,lc,kc,scales,real)
            for row in range(n):close(ref_small[row:row+1],expected[row:row+1],prefix+f'-fp64-row{row}')
            del expected
            # 259 creates tails for128/256.3072 checks every row of a full batch.
            for rows in (259,3072):
                inputs=[repeated(x,rows) for x in (qa,qr,tok0,counts0,pos0,base0)]
                reference=repeated(ref_small,rows);out=torch.empty_like(reference)
                qa_b,qr_b,tokens,counts,positions,bases=inputs
                fixed=AttentionScratch(rows,'cuda',real)
                def run(arena,destination=out):
                    return attend(qa_b,qr_b,lc,kc,tokens,counts,positions,bases,destination,arena,latent_scales=scales)
                run(fixed);check(out,reference,prefix+f'-fixed-rows{rows}')
                if rows==3072:
                    report['attention_component_timings'][prefix]={'fixed128':timing(lambda:run(fixed),repeats=3)}
                for part in (128,256,512,1024):
                    candidate=AttentionScratch(rows,'cuda',real,part_rows=part,skip_empty=True)
                    candidate.po.fill_(float('nan'));candidate.pm.fill_(float('nan'));candidate.pl.fill_(float('nan'))
                    out.fill_(float('nan'));run(candidate)
                    check(out,reference,prefix+f'-skip-part{part}-rows{rows}')
                    if rows==3072:
                        report['attention_component_timings'][prefix]['skip'+str(part)]=timing(lambda:run(candidate),repeats=3)
                    if rows==259 and part in (128,1024):
                        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(stream):run(candidate)
                        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
                        graph=torch.cuda.CUDAGraph()
                        saved=[v.clone() for v in inputs]
                        try:
                            with torch.cuda.graph(graph,stream=stream):run(candidate)
                            for step in (1,3):
                                for target,value in zip(inputs,saved):target.copy_(value.roll(step,0))
                                expected_graph=torch.empty_like(out);run(fixed,expected_graph)
                                graph.replay();check(out,expected_graph,prefix+f'-graph-part{part}-change{step}')
                            for target,value in zip(inputs,saved):target.copy_(value)
                        finally:graph.reset()
                        del graph,stream,saved,expected_graph
                    del candidate
                del fixed,run,inputs,reference,out,qa_b,qr_b,tokens,counts,positions,bases
            del qa,qr,ref_small,scratch
    # Clear all temporary device storage before the full-model load.
    del latent,rope,cache,slots,counts0,pos0,base0,tok0,qa0,qr0,lc,kc,scales,oracle,repeated
    torch.cuda.synchronize();gc.collect();torch.cuda.empty_cache();save('attention-component-complete')
