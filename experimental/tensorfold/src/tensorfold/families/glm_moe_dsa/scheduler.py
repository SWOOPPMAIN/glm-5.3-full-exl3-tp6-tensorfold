"""One TP6 model worker with bounded client queues and retained-prefix reuse.

The factory runs in the owning worker and must construct rank zero's backend,
RequestEngine and RequestController there. Followers run controller.follow().
Callbacks run on client threads; only the worker emits collective commands.
Packed mode combines compatible operations across active requests; scalar mode remains available for matched measurements.
"""
from dataclasses import dataclass, field
import heapq
import queue
import threading
import time

from .control import Replica, start_command


class SlowConsumer(RuntimeError):
    pass


@dataclass(eq=False)
class Ticket:
    key: str
    command: dict
    background: bool
    box: queue.Queue
    cancel: threading.Event = field(default_factory=threading.Event)
    finished: threading.Event = field(default_factory=threading.Event)
    error: Exception | None = None
    stats: dict = field(default_factory=dict)
    pulse_pending: bool = False
    admitted_at: float = 0
    first_at: float = 0
    prefill_s: float = 0
    min_rows: int = 0

    def emit(self, tokens):
        if self.cancel.is_set():
            return
        if not tokens and self.pulse_pending:
            return
        try:
            if not tokens:
                self.pulse_pending = True
            self.box.put_nowait(list(tokens))
        except queue.Full:
            if not tokens:
                self.pulse_pending = False
            self.error = SlowConsumer('Client output queue filled; this request was cancelled')
            self.cancel.set()


class RequestScheduler:
    def __init__(self, factory, *, draft_tokens=4, output_chunks=32, waiting_limit=32, packed=False):
        if (type(packed) is not bool or type(draft_tokens) is not int or draft_tokens < 0
                or type(output_chunks) is not int or output_chunks < 1
                or type(waiting_limit) is not int or waiting_limit < 1):
            raise ValueError('Invalid scheduler bounds')
        self.factory, self.draft_tokens = factory, draft_tokens
        self.packed = packed
        self.output_chunks, self.waiting_limit = output_chunks, waiting_limit
        self.condition = threading.Condition()
        self.waiting, self.active = [], []
        self.queued = set()
        self.sequence = 0
        self.stopping = False
        self.error = None
        self.ready = threading.Event()
        self.thread = threading.Thread(target=self._loop, name='glm53-tp6-requests', daemon=True)
        self.thread.start()
        # Loading the model happens before creating the scheduler; only worker
        # ownership and existing buffers are initialized by this factory.
        if not self.ready.wait(30):
            with self.condition:
                self.stopping = True
                self.condition.notify_all()
            raise TimeoutError('Same TP6 scheduler worker is still initializing; no replacement was launched')
        if self.error is not None:
            raise RuntimeError('TP6 scheduler initialization failed') from self.error

    def submit(self, prompt, max_tokens, sampling, on_tokens=None, *, draft=True, stop_eos=True, background=False):
        if any(type(flag) is not bool for flag in (draft, stop_eos, background)):
            raise ValueError('Request flags must be booleans')
        with self.condition:
            if self.error is not None or self.stopping:
                raise RuntimeError('TP6 scheduler is unavailable') from self.error
            if len(self.queued) >= self.waiting_limit:
                raise RuntimeError('TP6 waiting queue is full')
            self.sequence += 1
            key = f'request-{self.sequence}'
            command = start_command(key, prompt, max_tokens=max_tokens, sampling=sampling,
                draft_tokens=self.draft_tokens if draft else 0, ignore_eos=not stop_eos)
            ticket = Ticket(key, command, background, queue.Queue(self.output_chunks))
            heapq.heappush(self.waiting, (int(background), self.sequence, ticket))
            self.queued.add(ticket)
            self.condition.notify()
        callback_error = None
        try:
            while not ticket.finished.is_set() or not ticket.box.empty():
                try:
                    tokens = ticket.box.get(timeout=.05)
                except queue.Empty:
                    if ticket.finished.is_set():
                        continue
                    tokens = []  # client disconnects remain observable while queued/in a long forward
                if not tokens:
                    ticket.pulse_pending = False
                if ticket.cancel.is_set() or callback_error is not None:
                    continue
                if on_tokens is not None:
                    try:
                        if on_tokens(tokens):
                            ticket.cancel.set()
                    except Exception as exc:
                        callback_error = exc
                        ticket.cancel.set()
            if callback_error is not None:
                raise callback_error
            if ticket.error is not None:
                raise ticket.error
            return ticket.stats
        finally:
            # Also stop abandoned generators/caller interrupts at a command boundary.
            ticket.cancel.set()

    def _admit(self, ticket):
        """Plan against a host-only copy; never evict state for invalid input."""
        core = self.controller.replica.core
        args = ticket.command['args']
        # The qualified start code validates everything before pool allocation.
        # A fresh host pool distinguishes bad input from temporarily busy slots.
        from copy import copy
        from .request import CachePool
        trial = copy(core)
        trial.pool, trial.requests = CachePool(core.pool.capacity), {}
        Replica(trial).prepare(ticket.command)
        prompt = tuple(args['prompt'])
        finished = [r for r in core.requests.values() if r.status == 'finished']
        candidate = max((r for r in finished if r.tokens and tuple(r.tokens) == prompt[:len(r.tokens)]),
                        key=lambda r: len(r.tokens), default=None)
        # At most four retained leases, deterministic oldest-first eviction.
        victims = [r for r in core.requests.values()
                   if r.status not in ('prefill', 'decode') and r is not candidate]
        command = dict(op='start', args={**args, 'resume': candidate.key if candidate else None})
        while True:
            try:
                self.controller.replica.prepare(command)
                break
            except MemoryError:
                if victims:
                    victim = victims.pop(0)
                elif candidate is not None:
                    victim, candidate = candidate, None
                    command['args']['resume'] = None
                else:
                    return False  # active requests own capacity; continue their work
                self.controller.dispatch(dict(op='drop', args=dict(key=victim.key)))
        result = self.controller.dispatch(command)
        ticket.admitted_at = time.perf_counter()
        ticket.stats['cached'] = result['reused_tokens']
        self.active.append(ticket)
        return True

    def _dequeued(self, ticket):
        with self.condition:
            self.queued.discard(ticket)

    def _finish(self, ticket, result=None):
        now = time.perf_counter()
        result = result or {}
        ticket.stats.update(prefill_s=round(ticket.prefill_s, 4),
            decode_s=round(max(0, now-ticket.first_at), 4) if ticket.first_at else 0.,
            rounds=result.get('rounds', 0), drafted=result.get('total_drafted', 0),
            accepted=result.get('total_accepted', 0), min_rows=ticket.min_rows,
            drafts=bool(ticket.command['args']['draft_tokens']),
            reason=result.get('reason', 'cancelled'), cached=ticket.stats.get('cached', 0))
        ticket.finished.set()

    def _loop(self):
        current = None
        try:
            self.controller = self.factory()
            core = self.controller.replica.core
            if self.controller.rank != 0 or self.draft_tokens >= core.backend.logit_rows:
                raise ValueError('Scheduler needs rank zero and admitted draft geometry')
            if self.packed and not callable(getattr(core.backend, 'batch', None)):
                raise ValueError('Packed scheduler requires an admitted batch backend')
            self.limit, self.eos = core.context_limit, core.backend.eos
            self.ready.set()
            while True:
                with self.condition:
                    while not self.waiting and not self.active and not self.stopping:
                        self.condition.wait()
                    stopping = self.stopping
                    pending = [heapq.heappop(self.waiting) for _ in range(len(self.waiting))]
                if stopping:
                    for _, _, ticket in pending:
                        self._dequeued(ticket)
                        self._finish(ticket)
                    for ticket in self.active:
                        ticket.cancel.set()
                else:
                    blocked = []
                    for priority, seq, current in pending:
                        if current.cancel.is_set():
                            self._dequeued(current)
                            self._finish(current)
                            continue
                        try:
                            admitted = self._admit(current)
                        except (ValueError, TypeError, MemoryError) as exc:
                            current.error = exc
                            self._dequeued(current)
                            self._finish(current)
                            continue
                        if not admitted:
                            blocked.append((priority, seq, current))
                        else:
                            self._dequeued(current)
                    with self.condition:
                        for entry in blocked:
                            heapq.heappush(self.waiting, entry)
                    current = None
                batch = list(self.active)
                group_start = time.perf_counter()
                grouped = self.controller.dispatch(dict(op='step_many', args=dict(
                    keys=[t.key for t in batch], cancelled=[t.cancel.is_set() for t in batch]))) if self.packed and batch else None
                for current in batch:
                    start = group_start if grouped is not None else time.perf_counter()
                    result = grouped[current.key] if grouped is not None else self.controller.dispatch(dict(op='step', args=dict(
                        key=current.key, cancelled=current.cancel.is_set())))
                    if result['phase'] == 'prefill':
                        current.prefill_s += time.perf_counter()-start
                    if result['tokens'] and not current.first_at:
                        current.first_at = time.perf_counter()
                    if result['phase'] == 'decode' and result['reason'] != 'cancelled':
                        rows = result['drafted']+1
                        current.min_rows = min(current.min_rows or rows, rows)
                    current.emit(result['tokens'])  # empty pulses detect prefill disconnects
                    if result['finished']:
                        if result['reason'] == 'cancelled' or current.cancel.is_set():
                            self.controller.dispatch(dict(op='drop', args=dict(key=current.key)))
                        self.active.remove(current)
                        self._finish(current, result)
                    current = None
                if stopping and not self.active:
                    self.controller.dispatch(dict(op='close', args={}))
                    return
        except Exception as exc:
            with self.condition:
                self.error, self.stopping = exc, True
                pending_tickets = [entry[2] for entry in self.waiting]
                self.waiting.clear()
                # Include drained admissions too: a failed exchange can happen
                # before they are added to active or returned to the heap.
                drained = [entry[2] for entry in locals().get('pending', [])]
                tickets = {*self.active, *pending_tickets, *drained, *self.queued}
                self.queued.clear()
                if current is not None:
                    tickets.add(current)
                for ticket in tickets:
                    if not ticket.finished.is_set():
                        ticket.error = exc
                        self._finish(ticket)
            # Failed transport keeps exact leases/exchange for process-owner
            # inspection. Never issue cleanup collectives on a broken group.
        finally:
            self.ready.set()

    def close(self, timeout=30):
        if threading.current_thread() is self.thread:
            raise RuntimeError('Scheduler cannot join itself')
        with self.condition:
            self.stopping = True
            self.condition.notify_all()
        self.thread.join(timeout)
        if self.thread.is_alive():
            raise TimeoutError('Same TP6 worker is still closing; no replacement was launched')
        if self.error is not None:
            raise RuntimeError('TP6 worker failed; inspect its original transport') from self.error


class ServingEngine:
    """Interface consumed by tensorfold.cuda.server.App; family remains opt-in."""
    tp, concurrent = 6, True

    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.limit, self.eos = scheduler.limit, scheduler.eos

    def generate(self, prompt, max_tokens, sampling=None, on_tokens=None, *, draft=True,
                 stop_eos=True, background=False):
        return self.scheduler.submit(prompt, max_tokens, sampling, on_tokens,
                                     draft=draft, stop_eos=stop_eos, background=background)

    def close(self):
        self.scheduler.close()
