"""Single-worker full-GLM request state machine over one admitted TP6 backend.

The worker must issue the same commands on all six ranks. This module does not
start a communicator, allocate model weights, or register a serving family.
Target cache positions name input tokens; MTP position p combines target hidden
p with token p+1. Rejected writes stay outside the committed prefix and are
rewritten before a future query can see them. No cache tensors are copied to
keep a conversation: its extent stays leased until explicitly dropped.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import threading
from typing import Callable

import torch

from tensorfold.engine.exact_sampling import Sampling


@dataclass(frozen=True)
class Extent:
    base: int
    size: int
    generation: int


class CachePool:
    """Deterministic contiguous leases; generation IDs reject stale owners."""

    def __init__(self, capacity: int):
        if type(capacity) is not int or not 1 <= capacity <= 1048576:
            raise ValueError('Invalid full-GLM cache pool capacity')
        self.capacity = capacity
        self.free = [(0, capacity)]
        self.leases: dict[int, Extent] = {}
        self.generation = 0

    def allocate(self, size: int) -> Extent:
        if type(size) is not int or not 1 <= size <= self.capacity:
            raise ValueError('Invalid request reservation')
        for i, (base, count) in enumerate(self.free):
            if count >= size:
                self.generation += 1
                extent = Extent(base, size, self.generation)
                self.leases[extent.generation] = extent
                self.free[i:i+1] = [(base+size, count-size)] if count > size else []
                return extent
        raise MemoryError('No contiguous cache extent fits this request')

    def require(self, extent: Extent):
        if self.leases.get(extent.generation) is not extent:
            raise ValueError('Cache extent is no longer owned by this request')

    def grow(self, extent: Extent, size: int) -> Extent:
        self.require(extent)
        if type(size) is not int or not 1 <= size <= self.capacity:
            raise ValueError('Invalid request reservation')
        if size <= extent.size:
            return extent
        extra = size-extent.size
        for i, (base, count) in enumerate(self.free):
            if base == extent.base+extent.size and count >= extra:
                grown = Extent(extent.base, size, extent.generation)
                self.leases[extent.generation] = grown
                self.free[i:i+1] = [(base+extra, count-extra)] if count > extra else []
                return grown
        raise MemoryError('Retained prefix cannot grow in place; keep its lease until explicitly dropped')

    def release(self, extent: Extent):
        self.require(extent)
        del self.leases[extent.generation]
        spans = sorted([*self.free, (extent.base, extent.size)])
        merged = []
        for base, size in spans:
            if merged and merged[-1][0]+merged[-1][1] == base:
                old, count = merged.pop()
                merged.append((old, count+size))
            else:
                merged.append((base, size))
        self.free = merged


@dataclass(eq=False)
class Request:
    key: str
    extent: Extent
    prompt: tuple[int, ...]
    max_tokens: int
    sampling: Sampling | None
    draft_tokens: int
    eos: frozenset[int]
    tokens: list[int] = field(default_factory=list)  # target inputs actually committed
    output: list[int] = field(default_factory=list)
    mtp_end: int = 0
    pending_hidden: torch.Tensor | None = None     # target [mtp_end:len(tokens)]
    last_hidden: torch.Tensor | None = None
    pending: int | None = None                     # emitted, not yet target-processed
    status: str = 'prefill'
    finish_reason: str | None = None
    rounds: int = 0
    drafted: int = 0
    accepted: int = 0
    reused_tokens: int = 0


@dataclass(frozen=True)
class Step:
    tokens: tuple[int, ...] = ()
    finished: bool = False
    reason: str | None = None
    drafted: int = 0
    accepted: int = 0


def validate_sampling(sampling):
    if sampling is None:
        return
    if (not isinstance(sampling, Sampling) or type(sampling.seed) is not int
            or not all(math.isfinite(x) for x in
                       (sampling.temperature, sampling.top_p, sampling.min_p))
            or sampling.temperature < 0 or not 0 < sampling.top_p <= 1
            or not 0 <= sampling.min_p <= 1):
        raise ValueError('Invalid sampling parameters')


class RequestEngine:
    """Chunked prefill and recursive MTP with serial keyed target verification.

    ``backend`` supplies target/mtp/sample/synchronize and rows/logit_rows,
    capacity/vocab/eos. Returned hidden rows are borrowed until its next forward.
    One worker serializes steps for up to four requests over the same workspace.
    Kept extents count against pool capacity; no second full-model arena is made.
    This is the execution core, not the HTTP scheduler or a six-rank command bus.
    """

    def __init__(self, backend, *, context_limit=360000, max_requests=4):
        if (type(context_limit) is not int or not 1 <= context_limit <= backend.capacity
                or type(max_requests) is not int or not 1 <= max_requests <= 4
                or not 1 <= backend.logit_rows <= backend.rows <= 3072):
            raise ValueError('Invalid admitted request engine geometry')
        self.backend = backend
        self.pool = CachePool(backend.capacity)
        self.context_limit, self.max_requests = context_limit, max_requests
        self.requests: dict[str, Request] = {}
        self.thread = threading.get_ident()
        self.busy = False

    def _worker(self):
        if threading.get_ident() != self.thread or self.busy:
            raise RuntimeError('Full TP6 execution requires one non-reentrant worker')

    def _owned(self, request):
        if self.requests.get(request.key) is not request:
            raise ValueError('Request does not belong to this engine')
        self.pool.require(request.extent)

    def _tokens(self, tokens):
        result = tuple(tokens)
        if not result or any(type(t) is not int or not 0 <= t < self.backend.vocab for t in result):
            raise ValueError('Expected nonempty in-vocabulary token IDs')
        return result

    def start(self, key, prompt, *, max_tokens, sampling=None, draft_tokens=4,
              ignore_eos=False, resume: Request | None = None):
        self._worker()
        prompt = self._tokens(prompt)
        validate_sampling(sampling)
        if (not isinstance(key, str) or not key or type(max_tokens) is not int or max_tokens < 1
                or len(prompt)+max_tokens > self.context_limit
                or type(draft_tokens) is not int or not 0 <= draft_tokens < self.backend.logit_rows
                or type(ignore_eos) is not bool):
            raise ValueError('Invalid request or draft/context limits')
        if key in self.requests and self.requests[key] is not resume:
            raise ValueError('Request key is already owned')
        # Finished, cancelled and failed requests still own hidden buffers and
        # cache extents. Count retained conversations too, or repeated failures
        # could exceed the four-request memory reserve without active work.
        if len(self.requests)-(resume is not None) >= self.max_requests:
            raise MemoryError('All admitted request slots are retained; drop one or resume its prefix')
        needed = len(prompt)+max_tokens
        if resume is None:
            r = Request(key, self.pool.allocate(needed), prompt, max_tokens, sampling,
                        draft_tokens, frozenset() if ignore_eos else frozenset(self.backend.eos))
        else:
            self._owned(resume)
            if (resume.status != 'finished' or not resume.tokens or resume.last_hidden is None
                    or tuple(resume.tokens) != prompt[:len(resume.tokens)]):
                raise ValueError('Resume requires the entire retained target prefix of a finished request')
            extent = self.pool.grow(resume.extent, needed)
            r = Request(key, extent, prompt, max_tokens, sampling, draft_tokens,
                        frozenset() if ignore_eos else frozenset(self.backend.eos),
                        tokens=list(resume.tokens), mtp_end=resume.mtp_end,
                        pending_hidden=resume.pending_hidden, last_hidden=resume.last_hidden,
                        reused_tokens=len(resume.tokens))
            # Ownership moves atomically; stale Request objects cannot write/release it.
            del self.requests[resume.key]
            resume.status = 'transferred'
            resume.pending_hidden = resume.last_hidden = None
        self.requests[key] = r
        return r

    def drop(self, request):
        self._worker()
        self._owned(request)
        # Even a cancelled/failed forward must finish writes before reuse.
        self.backend.synchronize()
        self.pool.release(request.extent)
        del self.requests[request.key]
        request.pending_hidden = request.last_hidden = None
        request.status = 'released'

    def _append_hidden(self, request, hidden):
        owned = hidden.clone()
        if request.pending_hidden is not None:
            owned = torch.cat((request.pending_hidden, owned), dim=0)
        request.pending_hidden = owned
        request.last_hidden = owned[-1:].clone()

    def _absorb(self, request, next_tokens):
        n = len(next_tokens)
        if n == 0:
            return None
        hidden = request.pending_hidden
        if hidden is None or n > len(hidden) or request.mtp_end+n > len(request.tokens):
            raise RuntimeError('MTP absorption does not match the committed target prefix')
        last = None
        for start in range(0, n, self.backend.rows):
            stop = min(n, start+self.backend.rows)
            last = self.backend.mtp(next_tokens[start:stop], hidden[start:stop],
                                    request.mtp_end+start, request.extent).clone()
        request.mtp_end += n
        request.pending_hidden = hidden[n:].clone() if n < len(hidden) else None
        return last[-1:]

    def _sample(self, hidden, positions, request):
        result = list(self.backend.sample(hidden, positions, request.sampling))
        if len(result) != len(positions) or any(type(t) is not int or not 0 <= t < self.backend.vocab for t in result):
            raise RuntimeError('Sampler returned invalid token IDs')
        return result

    def _finish(self, request):
        if request.output and request.output[-1] in request.eos:
            request.status, request.finish_reason = 'finished', 'stop'
        elif len(request.output) == request.max_tokens:
            request.status, request.finish_reason = 'finished', 'length'

    def step(self, request, *, cancelled: Callable[[], bool] = lambda: False):
        """One prefill chunk or verified decode round; cancellation is rank-coordinated.

        The controller must distribute one cancellation decision to all ranks;
        independent HTTP callbacks on each worker would desynchronize collectives.
        """
        self._worker()
        self._owned(request)
        if request.status not in ('prefill', 'decode'):
            raise ValueError('Request is not active')
        self.busy = True
        try:
            def stopped():
                if not cancelled():
                    return False
                self.backend.synchronize()
                request.status, request.finish_reason = 'cancelled', 'cancelled'
                return True
            if stopped():
                return Step(finished=True, reason='cancelled')
            if request.status == 'prefill':
                start = len(request.tokens)
                if start < len(request.prompt):
                    part = request.prompt[start:start+self.backend.rows]
                    hidden = self.backend.target(part, start, request.extent)
                    if stopped():
                        return Step(finished=True, reason='cancelled')
                    self._append_hidden(request, hidden)
                    request.tokens.extend(part)
                # One known token beyond this target chunk is valid MTP input.
                nxt = request.prompt[request.mtp_end+1:len(request.tokens)+1]
                self._absorb(request, nxt)
                if stopped():
                    return Step(finished=True, reason='cancelled')
                if len(request.tokens) < len(request.prompt):
                    return Step()
                token = self._sample(request.last_hidden, [len(request.tokens)], request)[0]
                if stopped():
                    return Step(finished=True, reason='cancelled')
                request.pending = token
                request.output.append(token)
                request.status = 'decode'
                self._finish(request)
                return Step((token,), request.status == 'finished', request.finish_reason)

            start = len(request.tokens)
            # Every unabsorbed target row has its next token, ending at pending.
            next_tokens = [*request.tokens[request.mtp_end+1:], request.pending]
            draft_hidden = self._absorb(request, next_tokens)
            if request.mtp_end != start or request.pending_hidden is not None:
                raise RuntimeError('Canonical MTP prefix did not reach the target boundary')
            room = request.max_tokens-len(request.output)
            depth = min(request.draft_tokens, room-1)
            drafts = []
            for j in range(depth):
                if stopped():
                    return Step(finished=True, reason='cancelled')
                token = self._sample(draft_hidden, [start+j+1], request)[0]
                drafts.append(token)
                if token in request.eos:
                    break
                if j+1 < depth:
                    draft_hidden = self.backend.mtp([token], draft_hidden, start+j,
                                                    request.extent).clone()
                # Recursive writes do NOT advance request.mtp_end.
            if stopped():
                return Step(finished=True, reason='cancelled')
            inputs = [request.pending, *drafts]
            hidden = self.backend.target(inputs, start, request.extent)
            if stopped():
                return Step(finished=True, reason='cancelled')
            hidden = hidden.clone()  # survives head projection and the next MTP pass
            sampled = self._sample(hidden, list(range(start+1, start+1+len(inputs))), request)
            if stopped():
                return Step(finished=True, reason='cancelled')
            keep = 1
            for i, proposal in enumerate(drafts):
                if sampled[i] != proposal or sampled[i] in request.eos:
                    break
                keep += 1
            emitted = sampled[:keep]
            request.tokens.extend(inputs[:keep])
            self._append_hidden(request, hidden[:keep])
            request.pending = emitted[-1]
            request.output.extend(emitted)
            request.rounds += 1
            request.drafted += len(drafts)
            request.accepted += keep-1
            self._finish(request)
            return Step(tuple(emitted), request.status == 'finished', request.finish_reason,
                        len(drafts), keep-1)
        except Exception:
            request.status, request.finish_reason = 'failed', 'error'
            raise
        finally:
            self.busy = False
