"""Six independent TCPStore connections and real CPU request cores.

The forwards use the causal serial oracle; these are transport/lifetime tests,
not numerical GPU or serving throughput qualification.
"""
from copy import deepcopy
from datetime import timedelta
import threading
import time
import unittest

import torch.distributed as dist

from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.glm_moe_dsa.control import (
    Replica, RequestController, TransportBroken, decode, encode, start_command)
from tensorfold.families.glm_moe_dsa.request import RequestEngine
from test_glm53_full_tp6_request import CausalBackend, finish, serial

GENERATION = 'control_test_generation_530607'


class StoreView:
    def __init__(self, store, rank, mutate=None):
        self.store, self.rank, self.mutate = store, rank, mutate

    def __getattr__(self, name):
        return getattr(self.store, name)

    def get(self, key):
        value = self.store.get(key)
        return self.mutate(self.rank, key, value) if self.mutate else value


class Fleet:
    def __init__(self, *, mutate=None, configure=None, delay_rank=None):
        self.master = dist.TCPStore('127.0.0.1', 0, 6, True, timedelta(seconds=4), wait_for_workers=False)
        self.clients = [StoreView(self.master, 0, mutate)]+[
            StoreView(dist.TCPStore('127.0.0.1', self.master.port, 6, False, timedelta(seconds=4)), r, mutate)
            for r in range(1, 6)]
        self.errors, self.controllers, self.backends = {}, {}, {}
        self.ready = threading.Barrier(6, timeout=4)
        def create(rank):
            backend = CausalBackend()
            core = RequestEngine(backend, context_limit=128)
            if configure:
                configure(rank, core)
            controller = RequestController(Replica(core), self.clients[rank], rank,
                generation=GENERATION, timeout=timedelta(seconds=3), idle_timeout=timedelta(seconds=4))
            self.controllers[rank], self.backends[rank] = controller, backend
            return controller
        def worker(rank):
            try:
                controller = create(rank)
                self.ready.wait()
                if rank == delay_rank:
                    time.sleep(.06)
                controller.follow()
            except Exception as exc:
                self.errors[rank] = exc
        self.threads = [threading.Thread(target=worker, args=(r,), daemon=True) for r in range(1, 6)]
        for t in self.threads:
            t.start()
        self.head = create(0)
        self.ready.wait()

    def dispatch(self, command):
        return self.head.dispatch(command)

    def join(self):
        for thread in self.threads:
            thread.join(5)
        assert all(not t.is_alive() for t in self.threads), 'same followers still live after bounded test'

    def close(self):
        if self.head.broken is None and not self.head.replica.closed:
            self.dispatch(dict(op='close', args={}))
        self.join()


def step(key, cancelled=False):
    return dict(op='step', args=dict(key=key, cancelled=cancelled))


class ControlTests(unittest.TestCase):
    def fleet(self, **kwargs):
        fleet = Fleet(**kwargs)
        self.addCleanup(fleet.close)
        return fleet

    def test_six_rank_interleaving_cancellation_retained_resume_and_idle_wake(self):
        fleet = self.fleet(delay_rank=5)
        smp = Sampling(41, .8, 5, .92)
        prompts = {'a': [2, 5, 3]*5, 'b': [1, 9, 7]*6, 'c': [3, 7, 2]*4, 'd': [5, 2, 8]*7}
        for key, prompt in prompts.items():
            fleet.dispatch(start_command(key, prompt, max_tokens=15, sampling=smp))
        active = set(prompts)
        outputs = {k: [] for k in active}
        while active:
            for key in sorted(active):
                result = fleet.dispatch(step(key, cancelled=key == 'c'))
                outputs[key].extend(result['tokens'])
                if result['finished']:
                    active.remove(key)
        for key, prompt in prompts.items():
            self.assertEqual(outputs[key], [] if key == 'c' else serial(prompt, 15, smp))
        held = fleet.head.replica.core.requests['a']
        prompt = list(held.tokens)+[2, 3]
        reused = len(held.tokens)
        fleet.dispatch(dict(op='drop', args=dict(key='b')))
        admitted = fleet.dispatch(start_command('again', prompt, max_tokens=8, sampling=smp, resume='a'))
        self.assertEqual(admitted['reused_tokens'], reused)
        got = []
        while True:
            result = fleet.dispatch(step('again'))
            got.extend(result['tokens'])
            if result['finished']:
                break
        self.assertEqual(got, serial(prompt, 8, smp))
        for backend in fleet.backends.values():
            # Cancelled c never executed any model pass.
            self.assertFalse(any(item[0] == 3 for item in backend.targets))
        fleet.close()
        self.assertFalse(fleet.errors)
        self.assertEqual(len({c.completed for c in fleet.controllers.values()}), 1)
        self.assertEqual(fleet.master.num_keys(), 1)
        for c in fleet.controllers.values():
            self.assertEqual(c.replica.core.pool.free, [(0, 512)])

    def test_bad_client_admission_does_not_wake_or_break_workers(self):
        fleet = self.fleet()
        commands = [start_command('', [1], max_tokens=2), start_command('bad', [True], max_tokens=2),
                    start_command('bad', [1], max_tokens=129), step('unknown'),
                    dict(op='unknown', args={})]
        malformed = start_command('bad', [1], max_tokens=2, sampling=Sampling(0))
        malformed['args']['sampling']['top_k'] = True
        commands.append(malformed)
        for command in commands:
            with self.assertRaises(ValueError):
                fleet.dispatch(command)
        self.assertEqual(fleet.head.bell.sequence, 0)
        self.assertIsNone(fleet.head.broken)
        fleet.dispatch(start_command('ok', [1, 2], max_tokens=2))
        fleet.close()
        self.assertFalse(fleet.errors)

    def test_preflight_divergence_latches_before_any_forward(self):
        def configure(rank, core):
            if rank == 5:
                core.context_limit = 127
        fleet = self.fleet(configure=configure)
        with self.assertRaises(TransportBroken):
            fleet.dispatch(start_command('a', [1, 2], max_tokens=3))
        fleet.join()
        self.assertEqual(set(fleet.errors), set(range(1, 6)))
        for c in fleet.controllers.values():
            self.assertIsNotNone(c.broken)
            self.assertEqual(c.pending, 1)
            self.assertFalse(c.replica.core.requests)
        self.assertGreater(fleet.master.num_keys(), 1)
        with self.assertRaises(TransportBroken):
            fleet.dispatch(start_command('a', [1, 2], max_tokens=3))
        self.assertEqual(fleet.head.bell.sequence, 1)

    def test_malformed_generation_sequence_and_schema_are_rejected(self):
        for field, value in [('generation', 'stale_generation_530607'), ('sequence', 0),
                             ('version', True), ('surprise', 1)]:
            def mutate(rank, key, data):
                if rank == 4 and key.endswith('/packet'):
                    packet = decode(data)
                    packet[field] = value
                    return encode(packet)
                return data
            fleet = Fleet(mutate=mutate)
            try:
                with self.assertRaises(TransportBroken):
                    fleet.dispatch(start_command('a', [1], max_tokens=2))
                fleet.join()
                self.assertTrue(all(not b.targets for b in fleet.backends.values()))
                self.assertTrue(all(not c.replica.core.requests for c in fleet.controllers.values()))
            finally:
                fleet.close()

    def test_forward_failure_retains_lease_and_does_not_retry(self):
        fleet = self.fleet()
        fleet.dispatch(start_command('a', [1, 2], max_tokens=3))
        fleet.backends[3].fault = True
        with self.assertRaises(TransportBroken):
            fleet.dispatch(step('a'))
        fleet.join()
        self.assertEqual(len(fleet.backends[3].targets), 1)
        for c in fleet.controllers.values():
            self.assertIsNotNone(c.broken)
            self.assertIn('a', c.replica.core.requests)
            self.assertEqual(c.pending, 2)

    def test_result_disagreement_suppresses_output(self):
        def mutate(rank, key, data):
            if rank == 0 and key.endswith('/result/5'):
                value = decode(data)
                value['state'] = 'different'
                return encode(value)
            return data
        fleet = self.fleet(mutate=mutate)
        with self.assertRaises(TransportBroken):
            fleet.dispatch(start_command('a', [1], max_tokens=2))
        fleet.join()
        self.assertTrue(all(c.broken is not None for c in fleet.controllers.values()))

    def test_strict_json(self):
        for data in (b'{"a":1,"a":2}', b'{"n":NaN}', b'', 'not bytes'):
            with self.assertRaises(ValueError):
                decode(data)
        with self.assertRaises(ValueError):
            encode({'value': float('inf')})

    def test_preview_preserves_retained_tensor_and_host_ownership(self):
        core = RequestEngine(CausalBackend(), context_limit=128)
        r = core.start('a', [2, 3, 5], max_tokens=6)
        finish(core, r)
        before = Replica(core).fingerprint()
        hidden, last, extent = r.pending_hidden, r.last_hidden, r.extent
        preview = core.preview_start('b', [*r.tokens, 7, 8], max_tokens=8, resume=r)
        self.assertGreater(preview.size, extent.size)
        self.assertEqual(before, Replica(core).fingerprint())
        self.assertIs(r.pending_hidden, hidden)
        self.assertIs(r.last_hidden, last)
        self.assertIs(r.extent, extent)
        self.assertEqual(r.status, 'finished')
        with self.assertRaises(ValueError):
            core.preview_start('b', [9], max_tokens=8, resume=r)
        self.assertEqual(before, Replica(core).fingerprint())
        admitted = core.start('b', [*r.tokens, 7, 8], max_tokens=8, resume=r)
        self.assertEqual(admitted.extent, preview)


if __name__ == '__main__':
    unittest.main()
