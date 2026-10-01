"""Concurrent client threads drive six real TCPStore request replicas."""
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
import threading
import time
import unittest
import torch.distributed as dist

from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.glm_moe_dsa.control import Replica, RequestController, TransportBroken
from tensorfold.families.glm_moe_dsa.request import RequestEngine
from tensorfold.families.glm_moe_dsa.scheduler import RequestScheduler, ServingEngine, SlowConsumer
from test_glm53_full_tp6_request import CausalBackend, serial


class ScheduledFleet:
    def __init__(self, *, configure=None, **options):
        self.store = dist.TCPStore('127.0.0.1', 0, 6, True, timedelta(seconds=4), wait_for_workers=False)
        clients = [self.store]+[dist.TCPStore('127.0.0.1', self.store.port, 6, False, timedelta(seconds=4))
                               for _ in range(5)]
        self.controllers, self.backends, self.errors = {}, {}, {}
        self.barrier = threading.Barrier(6, timeout=4)
        def create(rank):
            b = CausalBackend()
            if configure: configure(rank, b)
            core = RequestEngine(b, context_limit=128)
            c = RequestController(Replica(core), clients[rank], rank, generation='scheduler_test_530617',
                                  timeout=timedelta(seconds=4), idle_timeout=timedelta(seconds=5))
            self.controllers[rank], self.backends[rank] = c, b
            self.barrier.wait()
            return c
        def follow(rank):
            try: create(rank).follow()
            except Exception as exc: self.errors[rank] = exc
        self.threads = [threading.Thread(target=follow, args=(r,), daemon=True) for r in range(1, 6)]
        for t in self.threads: t.start()
        self.scheduler = RequestScheduler(lambda: create(0), **options)
        self.engine = ServingEngine(self.scheduler)

    def close(self):
        try:
            self.scheduler.close(5)
        finally:
            for t in self.threads: t.join(5)
            assert all(not t.is_alive() for t in self.threads)


class SchedulerTests(unittest.TestCase):
    def fleet(self, **kwargs):
        f = ScheduledFleet(**kwargs)
        self.addCleanup(f.close)
        return f

    def test_concurrent_clients_serial_parity_and_worker_callback_ownership(self):
        f = self.fleet()
        start = threading.Barrier(8)
        def client(i):
            prompt = [1+i, 2, 5]*6
            sampling = Sampling(10+i, .8, 7, .9)
            tokens, callback_threads = [], set()
            start.wait(4)
            def emit(new):
                callback_threads.add(threading.get_ident())
                tokens.extend(new)
            stats = f.engine.generate(prompt, 20, sampling, emit)
            self.assertEqual(tokens, serial(prompt, 20, sampling))
            self.assertEqual(callback_threads, {threading.get_ident()})
            self.assertNotIn(f.scheduler.thread.ident, callback_threads)
            self.assertGreater(stats['rounds'], 0)
            self.assertEqual(stats['reason'], 'length')
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(client, range(8)))
        f.close()
        self.assertFalse(f.errors)
        self.assertEqual(f.store.num_keys(), 1)
        for b in f.backends.values():
            self.assertEqual(len({row[0] for row in b.targets}), 8)

    def test_kept_prefix_resumes_and_invalid_client_preserves_it(self):
        f = self.fleet()
        prompt, out = [2, 5, 3]*4, []
        f.engine.generate(prompt, 12, None, lambda t: out.extend(t))
        retained = list(f.controllers[0].replica.core.requests.values())[0]
        first_owner, first_hidden = retained.extent, retained.last_hidden
        with self.assertRaises(ValueError):
            f.engine.generate([1], 999)
        self.assertIs(retained.extent, first_owner)
        self.assertIs(retained.last_hidden, first_hidden)
        follow = prompt+out+[7, 3]
        got = []
        stats = f.engine.generate(follow, 9, None, lambda t: got.extend(t))
        self.assertEqual(got, serial(follow, 9, None))
        self.assertEqual(stats['cached'], len(prompt)+len(out)-1)
        self.assertEqual(retained.status, 'transferred')

    def test_cancellation_during_prefill_and_callback_exception_are_local(self):
        def configure(rank, b):
            b.rows, b.logit_rows = 5, 5
        f = self.fleet(configure=configure)
        callbacks = []
        def cancel(tokens):
            callbacks.append(tokens)
            return True
        stats = f.engine.generate([2, 5, 8]*30, 10, None, cancel)
        self.assertEqual(callbacks, [[]])
        self.assertEqual(stats['reason'], 'cancelled')
        self.assertTrue(all(not c.replica.core.requests for c in f.controllers.values()))
        def broken_callback(tokens):
            raise LookupError('caller failed')
        with self.assertRaisesRegex(LookupError, 'caller failed'):
            f.engine.generate([1, 2, 3], 20, None, broken_callback)
        got = []
        f.engine.generate([3, 5, 1], 7, None, lambda t: got.extend(t))
        self.assertEqual(got, serial([3, 5, 1], 7, None))

    def test_slow_consumer_does_not_block_other_clients(self):
        f = self.fleet(output_chunks=2)
        entered, release = threading.Event(), threading.Event()
        def slow(tokens):
            if tokens:
                entered.set()
                self.assertTrue(release.wait(4))
        with ThreadPoolExecutor(max_workers=2) as pool:
            blocked = pool.submit(f.engine.generate, [2, 3], 90, None, slow)
            self.assertTrue(entered.wait(4))
            got = []
            healthy = pool.submit(f.engine.generate, [1, 2], 7, None, lambda t: got.extend(t))
            healthy.result(4)
            self.assertEqual(got, serial([1, 2], 7, None))
            release.set()
            with self.assertRaises(SlowConsumer):
                blocked.result(4)
        self.assertIsNone(f.scheduler.error)

    def test_temporary_cache_exhaustion_queues_without_failing(self):
        def configure(rank, b): b.capacity = 150
        f = self.fleet(configure=configure)
        start = threading.Barrier(4)
        def client(i):
            prompt = [i+1, 5, 7]*25
            start.wait(4)
            got = []
            f.engine.generate(prompt, 20, None, lambda t: got.extend(t))
            self.assertEqual(got, serial(prompt, 20, None))
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(client, range(4)))
        self.assertIsNone(f.scheduler.error)

    def test_failure_reaches_all_queued_clients_and_latches_worker(self):
        def configure(rank, b): b.fault = rank == 3
        f = ScheduledFleet(configure=configure)
        try:
            start = threading.Barrier(7)
            def client(i):
                start.wait(4)
                with self.assertRaises((TransportBroken, RuntimeError)):
                    f.engine.generate([2, 3], 20)
            with ThreadPoolExecutor(max_workers=7) as pool:
                list(pool.map(client, range(7)))
            with self.assertRaises(RuntimeError): f.engine.generate([2, 3], 5)
            self.assertIsNotNone(f.scheduler.error)
        finally:
            with self.assertRaises(RuntimeError): f.close()
        self.assertEqual(set(f.errors), set(range(1, 6)))

    def test_idle_workers_remain_on_cpu_and_close_reclaims_all(self):
        f = self.fleet()
        self.assertTrue(all(not b.targets for b in f.backends.values()))
        f.engine.generate([2, 4], 3)
        n = f.controllers[0].completed
        time.sleep(.03)
        self.assertEqual(n, f.controllers[0].completed)
        f.close()
        for c in f.controllers.values():
            self.assertTrue(c.replica.closed)
            self.assertEqual(c.replica.core.pool.free, [(0, 512)])


if __name__ == '__main__': unittest.main()
