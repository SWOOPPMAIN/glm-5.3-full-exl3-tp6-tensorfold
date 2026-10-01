"""Bounded target/MTP/head CUDA graphs over the admitted full-model workspace.

Each entry owns its capture inputs and an independent allocator pool. Target,
MTP and projection may replay in any serial order; model outputs remain borrowed
workspace views and must be retained by the request core before its next pass.
No graph failure retries eagerly or launches a replacement worker.

See PyTorch2.10 CUDA semantics, Graph memory management, for address lifetime
and independent/shared graph-pool requirements. Distributed GPU qualification
is mandatory before serving this adapter.
"""
from collections import OrderedDict
from dataclasses import asdict, dataclass
import threading

import torch
import torch.distributed as dist

from .graph_plan import graph_key, graph_reserve
from .request_backend import FullModelBackend


@dataclass
class Entry:
    key: object
    inputs: object = None
    hidden: object = None
    graph: object = None
    output: object = None
    growth: int = 0


class CudaGraphRuntime:
    """CUDA operations isolated from the host cache/ownership protocol."""
    def __init__(self, backend, group):
        self.backend, self.group = backend, group
        self.capture_stream = torch.cuda.Stream(device=backend.device)
        self.admission = torch.empty(1, dtype=torch.int32, device=backend.device)

    def reserved(self):
        return torch.cuda.memory_reserved(self.backend.device)

    def capture(self, run):
        stream = self.capture_stream
        stream.wait_stream(self.backend.stream)
        with torch.cuda.stream(stream):
            for _ in range(3):
                run()
        self.backend.stream.wait_stream(stream)
        self.backend.stream.synchronize()
        dist.barrier(group=self.group)
        graph = torch.cuda.CUDAGraph()
        try:
            # No pool hint: independent pools avoid cross-entry allocation
            # lifetime assumptions when requests interleave in varying order.
            with torch.cuda.graph(graph, stream=stream):
                output = run()
            return graph, output
        except Exception:
            self.backend.stream.synchronize()
            graph.reset()
            raise

    def agree(self, okay):
        self.admission.fill_(int(okay))
        dist.all_reduce(self.admission, op=dist.ReduceOp.MIN, group=self.group)
        return bool(self.admission.item())

    def retire(self, entry):
        self.backend.stream.synchronize()
        if entry.graph is not None:
            entry.graph.reset()
        entry.graph = entry.output = entry.inputs = entry.hidden = None


class GraphBackend(FullModelBackend):
    def __init__(self, model, caches, table, workspace, *, group, sampler=None,
                 max_graphs=16, max_rows=5, capture_bytes=512*2**20):
        super().__init__(model,caches,table,workspace,sampler=sampler)
        if (not dist.is_initialized() or dist.get_world_size(group)!=6
                or dist.get_backend(group)!='nccl' or dist.get_rank(group)!=model.weights.rank
                or model.vocab.reduction.group is not group or max_rows>self.rows):
            raise ValueError('Graph backend requires the admitted TP6 group and row workspace')
        self.plan = graph_reserve(max_graphs,max_rows,capture_bytes)
        self.entries = OrderedDict()
        self.thread = threading.get_ident()
        self.closed = False
        self.broken = None
        self.captures = self.replays = self.evictions = self.eager = 0
        self.retained_growth = self.peak_retained_growth = 0
        self.runtime = CudaGraphRuntime(self,group)

    def _stream(self):
        super()._stream()
        if hasattr(self,'thread') and (threading.get_ident()!=self.thread or self.closed or self.broken is not None):
            raise RuntimeError('Request graphs require their healthy owning worker') from self.broken

    def control_state(self):
        """Only deterministic host state: included in six-rank command agreement."""
        return dict(kind='full_tp6_graphs_v1',plan=self.plan,projections=asdict(self.projection_plan),
                    packed=self.batch_counts,entries=[asdict(k) for k in self.entries],captures=self.captures,
                    replays=self.replays,evictions=self.evictions,eager=self.eager,closed=self.closed)

    def _fill(self, entry, tokens, positions, extent, hidden, *, bases=None):
        if entry.inputs is not None:
            if (len(tokens)!=entry.key.rows or len(positions)!=entry.key.rows
                    or max(positions)+1>entry.key.visible or entry.key.visible>self.capacity):
                raise ValueError('Graph replay exceeds its captured logical context bound')
            bases = [extent.base]*len(tokens) if bases is None else bases
            if len(bases) != len(tokens):
                raise ValueError('Packed cache bases do not match row count')
            metadata = [list(tokens),positions,bases,[b+p for b,p in zip(bases,positions)]]
            # Blocking host copy keeps the temporary CPU packet alive until its
            # bytes are consumed, including consecutive target/MTP submissions.
            entry.inputs.copy_(torch.tensor(metadata,dtype=torch.int64),non_blocking=False)
        if entry.hidden is not None:
            entry.hidden.copy_(hidden)

    def _run(self, entry):
        key = entry.key
        if key.operation == 'head':
            return self.model.logits(entry.hidden,self.workspace)
        ids,pos,bases,slots = entry.inputs.unbind(0)
        if key.operation == 'target':
            return self.model.target_forward(ids,pos,bases,slots,self.caches[:78],self.table,self.workspace,
                                             scope=object(),visible_tokens=key.visible)
        return self.model.mtp_forward(ids,entry.hidden,pos,bases,slots,self.caches[78],self.table,self.workspace,
                                      scope=object(),visible_tokens=key.visible)

    def _retire(self, entry):
        self.runtime.retire(entry)
        self.retained_growth -= entry.growth
        self.evictions += 1

    def _execute(self, key, *, tokens=None, positions=None, extent=None, hidden=None, bases=None):
        self._stream()
        entry = None
        try:
            if key in self.entries:
                entry = self.entries.pop(key)
                self.entries[key] = entry
                self._fill(entry,tokens,positions,extent,hidden,bases=bases)
            else:
                if len(self.entries) >= self.plan['max_graphs']:
                    _,old = self.entries.popitem(last=False)
                    self._retire(old)
                before = self.runtime.reserved()
                entry = Entry(key)
                if key.operation != 'head':
                    entry.inputs = torch.empty((4,key.rows),dtype=torch.int64,device=self.device)
                if key.operation != 'target':
                    entry.hidden = torch.empty((key.rows,6144),dtype=torch.bfloat16,device=self.device)
                self._fill(entry,tokens,positions,extent,hidden,bases=bases)
                entry.graph,entry.output = self.runtime.capture(lambda: self._run(entry))
                entry.growth = max(0,self.runtime.reserved()-before)
                allowed = self.retained_growth+entry.growth <= self.plan['total']
                # All ranks reject an over-budget capture before anyone replays
                # its NCCL graph. Actual growth may differ with the uneven shards.
                if not self.runtime.agree(allowed):
                    raise MemoryError('A TP6 rank exceeded the admitted graph growth allowance')
                self.entries[key] = entry
                self.retained_growth += entry.growth
                self.peak_retained_growth = max(self.peak_retained_growth,self.retained_growth)
                self.captures += 1
            entry.graph.replay()
            self.replays += 1
            return entry.output
        except Exception as exc:
            self.broken = exc
            # Keep a failed capture's exact buffers/graph alive for owner-driven
            # synchronized cleanup; do not retry a possibly partial collective.
            self.failed_entry = entry
            raise

    def _key(self, operation, tokens, start, extent):
        positions,visible = self._host_positions(tokens,start,extent)
        return graph_key(operation,len(tokens),self.capacity,visible=visible,
                         max_rows=self.plan['max_rows']),positions

    @torch.inference_mode()
    def target(self,tokens,start,extent):
        key,positions = self._key('target',tokens,start,extent)
        if key is None:
            self.eager += 1
            return super().target(tokens,start,extent)
        return self._execute(key,tokens=tokens,positions=positions,extent=extent)

    @torch.inference_mode()
    def mtp(self,tokens,hidden,start,extent):
        key,positions = self._key('mtp',tokens,start,extent)
        if key is None:
            self.eager += 1
            return super().mtp(tokens,hidden,start,extent)
        self._hidden(hidden,len(tokens))
        return self._execute(key,tokens=tokens,positions=positions,extent=extent,hidden=hidden)

    def _hidden(self,hidden,rows):
        if (hidden.shape!=(rows,6144) or hidden.dtype!=torch.bfloat16 or hidden.device!=self.device):
            raise ValueError('Graph input must be the owning normalized hidden rows')

    @torch.inference_mode()
    def sample(self,hidden,positions,sampling):
        self._stream()
        if not 1 <= len(positions) <= self.logit_rows:
            raise ValueError('Sample rows exceed the admitted vocabulary workspace')
        self._hidden(hidden,len(positions))
        key = graph_key('head',len(positions),self.capacity,max_rows=self.plan['max_rows'])
        if key is None:
            self.eager += 1
            return super().sample(hidden,positions,sampling)
        logits = self._execute(key,hidden=hidden)
        return self.sampler(logits,positions,sampling)

    def _batch_forward(self, operation, metadata, hidden, visible):
        tokens,positions,bases,_ = metadata
        key = graph_key('target' if operation == 'target' else 'mtp_batch',len(tokens),
                        self.capacity,visible=visible,max_rows=self.plan['max_rows'])
        if key is None:
            self.eager += 1
            return super()._batch_forward(operation,metadata,hidden,visible)
        return self._execute(key,tokens=tokens,positions=positions,bases=bases,hidden=hidden)

    def _batch_logits(self, hidden):
        key = graph_key('head',len(hidden),self.capacity,max_rows=self.plan['max_rows'])
        if key is None:
            self.eager += 1
            return super()._batch_logits(hidden)
        return self._execute(key,hidden=hidden)

    def close_graphs(self):
        """Call after all ranks stop requests, before destroying the communicator.

        This also permits explicit cleanup of a failed backend after its owner
        establishes that the same CUDA work has finished. No automatic retry.
        """
        if threading.get_ident()!=self.thread or torch.cuda.current_stream(self.device)!=self.stream:
            raise RuntimeError('Graph cleanup requires its owning worker/stream')
        if self.closed:
            return
        self.stream.synchronize()
        seen = set()
        for entry in [*self.entries.values(),getattr(self,'failed_entry',None)]:
            if entry is not None and id(entry) not in seen:
                self.runtime.retire(entry)
                seen.add(id(entry))
        self.entries.clear()
        self.failed_entry = None
        self.retained_growth = 0
        self.closed = True
