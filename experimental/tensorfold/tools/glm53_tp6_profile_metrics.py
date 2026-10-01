"""Summarize GPU kernel intervals without double-counting concurrent streams."""
import math
from collections import defaultdict


def union_us(intervals):
    spans = sorted(intervals)
    end = None
    total = 0.
    for start, stop in spans:
        if not all(isinstance(x,(int,float)) and math.isfinite(x) for x in (start,stop)) or stop<start:
            raise ValueError('Invalid trace interval')
        if end is None or start>end:
            total += stop-start
        elif stop>end:
            total += stop-end
        end = stop if end is None else max(end,stop)
    return total


def summarize(trace):
    devices = defaultdict(list)
    names = defaultdict(lambda: dict(count=0,total_us=0.))
    annotations = defaultdict(list)
    for event in trace['traceEvents']:
        if event.get('ph')!='X':continue
        category=event.get('cat','')
        if category!='kernel' and category!='user_annotation':continue
        start=event['ts'];stop=start+event['dur']
        union_us([(start,stop)])
        name=event['name']
        if category=='kernel':
            devices[str(event['pid'])].append((start,stop))
            names[name]['count']+=1
            names[name]['total_us']+=event['dur']
        elif name.startswith('tfp19.'):
            annotations[name].append((start,stop))
    if not devices:raise ValueError('Profiler returned no CUDA kernel events')
    gpu={device:dict(kernel_union_us=union_us(spans),span_us=max(b for _,b in spans)-min(a for a,_ in spans),kernel_count=len(spans)) for device,spans in devices.items()}
    ranked=sorted((dict(name=name,**value) for name,value in names.items()),key=lambda r:r['total_us'],reverse=True)
    return dict(devices=gpu,kernel_sum_us=sum(r['total_us'] for r in ranked),
                nccl_kernel_sum_us=sum(r['total_us'] for r in ranked if 'nccl' in r['name'].lower()),
                top_kernels=ranked[:25],annotations={name:dict(count=len(spans),host_total_us=sum(b-a for a,b in spans)) for name,spans in annotations.items()},
                caveat='Kernel sums may overlap across streams. Union is per-device active kernel time between first and last kernel, not GPU utilization outside this short profiled window. Profiling perturbs timings.')
