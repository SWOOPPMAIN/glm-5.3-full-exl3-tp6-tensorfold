"""Ordered TP6 request commands over the existing CPU rendezvous store.

Rank zero owns external decisions. A command has a prepare barrier before model
execution and a result/state agreement barrier before any output reaches a
client. Idle followers block on their own CPU doorbell keys; no idle NCCL call
runs. Errors latch the controller, retain the failed exchange for inspection,
and never retry a partially executed command or launch replacement workers.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import timedelta
import hashlib
import json

from tensorfold.engine.exact_sampling import Sampling
from .doorbell import RequestDoorbell

VERSION = 1
MAX_PACKET_BYTES = 8*1024**2


class TransportBroken(RuntimeError):
    pass


def encode(value):
    data = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    if len(data) > MAX_PACKET_BYTES:
        raise ValueError('TP6 command exceeds the bounded host packet size')
    return data


def decode(data):
    if not isinstance(data, bytes) or not 0 < len(data) <= MAX_PACKET_BYTES:
        raise ValueError('Invalid TP6 packet size/type')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate TP6 packet field')
            result[key] = value
        return result
    def nonfinite(value):
        raise ValueError('Non-finite TP6 packet value')
    return json.loads(data, object_pairs_hook=unique, parse_constant=nonfinite)


def digest(value):
    return hashlib.sha256(encode(value)).hexdigest()


def fields(value, expected):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError('Unexpected TP6 command fields')


def start_command(key, prompt, *, max_tokens, sampling=None, draft_tokens=4,
                  ignore_eos=False, resume=None):
    return dict(op='start', args=dict(key=key, prompt=list(prompt), max_tokens=max_tokens,
        sampling=None if sampling is None else asdict(sampling), draft_tokens=draft_tokens,
        ignore_eos=ignore_eos, resume=resume))


class Replica:
    """One rank's request core; all state changes pass through prepared commands."""

    def __init__(self, core):
        self.core, self.closed = core, False
        self.hashes = {}

    def fingerprint(self):
        # Extend hashes only with new tokens; a 360K prefix is not rehashed on
        # every decode step. Object identity is local bookkeeping, never sent.
        state = []
        for key, request in sorted(self.core.requests.items()):
            entry = self.hashes.get(key)
            if entry is None or entry['request'] is not request:
                entry = dict(request=request, prompt=digest(request.prompt), n=0, out_n=0,
                             tokens=hashlib.sha256(), output=hashlib.sha256())
                self.hashes[key] = entry
            for attr, count, target in (('tokens', 'n', request.tokens), ('output', 'out_n', request.output)):
                if len(target) < entry[count]:
                    raise RuntimeError('A committed request prefix moved backwards')
                for token in target[entry[count]:]:
                    entry[attr].update(token.to_bytes(4, 'little'))
                entry[count] = len(target)
            state.append(dict(key=key, prompt=entry['prompt'], tokens=entry['tokens'].hexdigest(),
                output=entry['output'].hexdigest(), target_end=len(request.tokens), mtp_end=request.mtp_end,
                pending=request.pending, hidden_rows=0 if request.pending_hidden is None else len(request.pending_hidden),
                extent=asdict(request.extent), status=request.status, reason=request.finish_reason,
                max_tokens=request.max_tokens, draft_tokens=request.draft_tokens,
                sampling=None if request.sampling is None else asdict(request.sampling), eos=sorted(request.eos),
                rounds=request.rounds, drafted=request.drafted, accepted=request.accepted,
                reused=request.reused_tokens))
        self.hashes = {k:v for k,v in self.hashes.items() if k in self.core.requests}
        return digest(dict(closed=self.closed, requests=state, free=self.core.pool.free,
                           generation=self.core.pool.generation, context=self.core.context_limit,
                           capacity=self.core.pool.capacity, max_requests=self.core.max_requests,
                           backend=self.core.backend.control_state() if hasattr(self.core.backend,'control_state') else None))

    def prepare(self, command):
        self.core._worker()
        if self.closed:
            raise RuntimeError('TP6 request replica is closed')
        fields(command, ('op', 'args'))
        op, args = command['op'], command['args']
        if op == 'start':
            fields(args, ('key', 'prompt', 'max_tokens', 'sampling', 'draft_tokens', 'ignore_eos', 'resume'))
            if not isinstance(args['prompt'], list):
                raise ValueError('Prompt must be a token list')
            sampling = args['sampling']
            if sampling is not None:
                fields(sampling, ('seed', 'temperature', 'top_k', 'top_p', 'min_p'))
                if type(sampling['top_k']) is not int or sampling['top_k'] < 0:
                    raise ValueError('Invalid top-k in command')
                sampling = Sampling(**sampling)
            resume = args['resume']
            if resume is not None:
                if not isinstance(resume, str) or resume not in self.core.requests:
                    raise ValueError('Unknown retained prefix')
                resume = self.core.requests[resume]
            kwargs = {**args, 'sampling': sampling, 'resume': resume}
            self.core.preview_start(**kwargs)
            return op, kwargs
        if op == 'step':
            fields(args, ('key', 'cancelled'))
            if type(args['cancelled']) is not bool:
                raise ValueError('Cancellation must be one leader-owned boolean')
            self._request(args['key'], active=True)
        elif op == 'step_many':
            fields(args, ('keys', 'cancelled'))
            if not isinstance(args['keys'], list) or not isinstance(args['cancelled'], list):
                raise ValueError('Packed keys and cancellation flags must be lists')
            requests = [self._request(key, active=True) for key in args['keys']]
            self.core.preview_many(requests, args['cancelled'])
        elif op == 'drop':
            fields(args, ('key',))
            self._request(args['key'])
        elif op == 'close':
            fields(args, ())
        else:
            raise ValueError('Unknown TP6 request operation')
        return op, args

    def _request(self, key, *, active=False):
        if not isinstance(key, str) or key not in self.core.requests:
            raise ValueError('Unknown request key')
        r = self.core.requests[key]
        self.core._owned(r)
        if active and r.status not in ('prefill', 'decode'):
            raise ValueError('Request is not active')
        return r

    def apply(self, prepared):
        op, args = prepared
        if op == 'start':
            r = self.core.start(**args)
            return dict(key=r.key, extent=asdict(r.extent), reused_tokens=r.reused_tokens)
        if op == 'step':
            r = self._request(args['key'], active=True)
            phase = r.status
            result = self.core.step(r, cancelled=lambda: args['cancelled'])
            return dict(phase=phase, **asdict(result), rounds=r.rounds,
                        total_drafted=r.drafted, total_accepted=r.accepted,
                        target_end=len(r.tokens), mtp_end=r.mtp_end)
        if op == 'step_many':
            requests = [self._request(key, active=True) for key in args['keys']]
            phases = [r.status for r in requests]
            steps = self.core.step_many(requests, args['cancelled'])
            return {r.key:dict(phase=phase, **asdict(step), rounds=r.rounds,
                        total_drafted=r.drafted, total_accepted=r.accepted,
                        target_end=len(r.tokens), mtp_end=r.mtp_end)
                    for r,phase,step in zip(requests,phases,steps)}
        if op == 'drop':
            self.core.drop(self._request(args['key']))
            return dict(dropped=args['key'])
        if op == 'close':
            for r in list(self.core.requests.values()):
                self.core.drop(r)
            if hasattr(self.core.backend,'close_graphs'):
                self.core.backend.close_graphs()
            self.closed = True
            return dict(closed=True)
        raise RuntimeError('Prepared operation was lost')


class RequestController:
    """One controller per rank, sharing a unique generation and TCPStore.

    Rank zero calls dispatch; other ranks call follow. An observation timeout
    preserves pending sequence/keys and latches failure; it is never a retry.
    The process owner can inspect the same live workers and exchange afterward.
    """

    def __init__(self, replica, store, rank, *, generation, timeout=timedelta(seconds=300),
                 idle_timeout=timedelta(hours=24)):
        if timeout.total_seconds() <= 0 or idle_timeout.total_seconds() <= 0:
            raise ValueError('Positive explicit control timeouts required')
        self.replica, self.store, self.rank = replica, store, rank
        self.bell = RequestDoorbell(store, rank, generation=generation)
        self.timeout, self.idle_timeout = timeout, idle_timeout
        self.broken = None
        self.pending = None
        self.completed = 0

    def _key(self, seq, suffix):
        return f'tf_glm53_tp6/{self.bell.generation}/command/{seq}/{suffix}'

    def _set(self, seq, suffix, value):
        self.store.set(self._key(seq, suffix), encode(value))

    def _get(self, seq, suffix):
        key = self._key(seq, suffix)
        self.store.wait([key], self.timeout)
        return decode(self.store.get(key))

    def _healthy(self):
        if self.broken is not None:
            raise TransportBroken('TP6 control is latched after an incomplete or failed exchange') from self.broken
        if self.replica.closed or self.pending is not None:
            raise TransportBroken('TP6 controller is closed or already has a pending command')

    def _result(self, prepared, proceed):
        try:
            if not proceed:
                raise TransportBroken('A rank rejected command preparation')
            result = self.replica.apply(prepared)
            return result, dict(ok=True, result=digest(result), state=self.replica.fingerprint())
        except Exception as exc:
            return None, dict(ok=False, error=type(exc).__name__)

    def dispatch(self, command):
        if self.rank != 0:
            raise ValueError('Only rank zero dispatches request commands')
        self._healthy()
        # Client validation fails locally before publishing any wakeup.
        prepared = self.replica.prepare(command)
        before = self.replica.fingerprint()
        seq = self.bell.sequence+1
        packet = dict(version=VERSION, generation=self.bell.generation, sequence=seq,
                      before=before, command=command)
        data = encode(packet)
        packet_hash = hashlib.sha256(data).hexdigest()
        try:
            self.pending = seq
            self.store.set(self._key(seq, 'packet'), data)
            assert self.bell.publish() == seq
            ready = dict(ok=True, packet=packet_hash, state=before)
            self._set(seq, 'ready/0', ready)
            readiness = [self._get(seq, f'ready/{rank}') for rank in range(6)]
            proceed = all(r == ready for r in readiness)
            self._set(seq, 'go', proceed)
            result, local = self._result(prepared, proceed)
            self._set(seq, 'result/0', local)
            outcomes = [self._get(seq, f'result/{rank}') for rank in range(6)]
            success = proceed and local.get('ok') and all(r == local for r in outcomes)
            self._set(seq, 'finish', bool(success))
            consumed = [self._get(seq, f'consumed/{rank}') for rank in range(1, 6)]
            if not success or not all(c is True for c in consumed):
                raise TransportBroken('Six-rank prepare/result agreement failed')
            # All followers acknowledged completion; no rank still reads these.
            suffixes = ['packet', 'go', 'finish']
            suffixes += [f'{kind}/{rank}' for kind in ('ready', 'result') for rank in range(6)]
            suffixes += [f'consumed/{rank}' for rank in range(1, 6)]
            for suffix in suffixes:
                self.store.delete_key(self._key(seq, suffix))
            self.pending = None
            self.completed += 1
            return result
        except Exception as exc:
            self.broken = exc
            raise TransportBroken(f'TP6 command {seq} failed; inspect this exchange before further work') from exc

    def follow_once(self):
        if self.rank == 0:
            raise ValueError('Rank zero cannot follow request commands')
        self._healthy()
        try:
            seq = self.bell.wait(self.idle_timeout)
            self.pending = seq
            prepared = None
            try:
                data = self.store.get(self._key(seq, 'packet'))
                packet = decode(data)
                fields(packet, ('version', 'generation', 'sequence', 'before', 'command'))
                if (type(packet['version']) is not int or packet['version'] != VERSION
                        or type(packet['sequence']) is not int or packet['sequence'] != seq
                        or packet['generation'] != self.bell.generation
                        or packet['before'] != self.replica.fingerprint()):
                    raise ValueError('TP6 command generation, sequence or prior state differs')
                prepared = self.replica.prepare(packet['command'])
                ready = dict(ok=True, packet=hashlib.sha256(data).hexdigest(), state=packet['before'])
            except Exception as exc:
                ready = dict(ok=False, error=type(exc).__name__)
            self._set(seq, f'ready/{self.rank}', ready)
            proceed = self._get(seq, 'go')
            if type(proceed) is not bool or (proceed and prepared is None):
                raise TransportBroken('Invalid execute decision from leader')
            _, outcome = self._result(prepared, proceed)
            self._set(seq, f'result/{self.rank}', outcome)
            success = self._get(seq, 'finish')
            self._set(seq, f'consumed/{self.rank}', True)
            if success is not True or not outcome.get('ok'):
                raise TransportBroken('Leader rejected six-rank result agreement')
            self.pending = None
            self.completed += 1
            return not self.replica.closed
        except Exception as exc:
            self.broken = exc
            raise TransportBroken('TP6 follower exchange failed; no command was retried') from exc

    def follow(self):
        while self.follow_once():
            pass
