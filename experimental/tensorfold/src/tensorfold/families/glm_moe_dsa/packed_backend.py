"""Bounded multi-request operations over one existing model/cache/workspace.

Each forward row carries its own logical position and physical cache base.
Requests retain disjoint leases; outputs are copied before arena reuse. Sampling
keeps each request's own seed, position and policy, even when its head is packed.
"""
import torch
from .request_ops import Call


class PackedBackend:
    @torch.inference_mode()
    def batch(self, calls):
        self._stream()
        if (not isinstance(calls, list) or not 1 <= len(calls) <= 4
                or any(type(c) is not Call for c in calls)
                or len({c.operation for c in calls}) != 1):
            raise ValueError('Packed work requires one to four matching request operations')
        operation = calls[0].operation
        total = sum(c.rows for c in calls)
        limit = self.logit_rows if operation == 'sample' else self.rows
        if any(c.rows < 1 for c in calls) or total > limit:
            raise ValueError('Packed operation exceeds its admitted row capacity')
        if operation == 'sample':
            for call in calls:
                hidden, positions, sampling = call.args
                self._hidden(hidden, len(positions))
                if any(type(p) is not int or p < 0 for p in positions):
                    raise ValueError('Invalid packed sampling position')
            hidden = torch.cat([c.args[0] for c in calls], dim=0)
            logits = self._batch_logits(hidden)
            start = 0; result = []
            for call in calls:
                _, positions, sampling = call.args
                result.append(self.sampler(logits[start:start+call.rows], positions, sampling))
                start += call.rows
        else:
            metadata = [[], [], [], []]
            visible = 0; extents = []; hidden_parts = []
            for call in calls:
                if operation == 'target': tokens, start, extent = call.args
                else:
                    tokens, hidden, start, extent = call.args
                    self._hidden(hidden, len(tokens)); hidden_parts.append(hidden)
                positions, bound = self._host_positions(tokens, start, extent)
                if (any(type(v) is not int for v in (extent.base, extent.size, extent.generation))
                        or extent.generation < 1):
                    raise ValueError('Packed calls require valid physical owners')
                extents.append((extent.base, extent.base+extent.size))
                visible = max(visible, bound)
                for dest, values in zip(metadata, (tokens, positions, [extent.base]*len(tokens),
                                                  [extent.base+p for p in positions])):
                    dest.extend(values)
            spans = sorted(extents)
            if any(first[1] > second[0] for first, second in zip(spans, spans[1:])):
                raise ValueError('Packed requests must own disjoint cache extents')
            hidden = torch.cat(hidden_parts, dim=0) if hidden_parts else None
            borrowed = self._batch_forward(operation, metadata, hidden, visible)
            result = []; start = 0
            for call in calls:
                result.append(borrowed[start:start+call.rows].clone())
                start += call.rows
        counts = self.batch_counts[operation]
        counts['groups'] += 1; counts['requests'] += len(calls); counts['rows'] += total
        counts['multi_request_groups'] += int(len(calls) > 1)
        return result

    def _batch_forward(self, operation, metadata, hidden, visible):
        ids, positions, bases, slots = torch.tensor(metadata, dtype=torch.int64, device=self.device).unbind(0)
        if operation == 'target':
            return self.model.target_forward(ids,positions,bases,slots,self.caches[:78],self.table,
                                             self.workspace,scope=object(),visible_tokens=visible)
        return self.model.mtp_forward(ids,hidden,positions,bases,slots,self.caches[78],self.table,
                                      self.workspace,scope=object(),visible_tokens=visible)

    def _batch_logits(self, hidden):
        return self.model.logits(hidden, self.workspace)


def batch_counters():
    return {name:dict(groups=0,requests=0,rows=0,multi_request_groups=0)
            for name in ('target','mtp','sample')}


def packed_reserve(rows, max_requests=4):
    """Additional temporary allowance for suspended rounds and concatenation.

    Request-owned retained tensors remain in request_plan. This conservative
    extra bound accounts for suspended borrowed-result copies plus one packed
    input and output, without another model workspace or weight copy.
    """
    if (type(rows) is not int or not 1 <= rows <= 3072
            or type(max_requests) is not int or not 1 <= max_requests <= 4):
        raise ValueError('Invalid packed temporary geometry')
    return (max_requests+2)*rows*6144*2
