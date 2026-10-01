"""Independent causal oracle for the actual request engine, without GPUs.

The fake forward has a reused output buffer and separately owned physical target
and MTP caches. It checks target-hidden versus recursive-MTP-hidden alignment;
serial generation never calls RequestEngine. Real-weight GPU parity is separate.
"""
import unittest
from unittest.mock import patch

import numpy as np
import torch

from tensorfold.engine.exact_sampling import Sampling, choose, choose_rows
from tensorfold.families.glm_moe_dsa.request import CachePool, RequestEngine

VOCAB = 31


def advance(h, token, position):
    return (h*17+token*13+position+7) % 100003


def history(tokens):
    h = 0
    for p, t in enumerate(tokens):
        h = advance(h, t, p)
    return h


def scores(h):
    return np.array([((h*(t+3)+19*t) % 97)/13-3.0 for t in range(VOCAB)], dtype=np.float64)


def serial(prompt, count, sampling, eos=()):
    tokens = list(prompt)
    out = []
    for _ in range(count):
        logits = scores(history(tokens))
        token = int(np.argmax(logits)) if sampling is None or sampling.temperature == 0 else choose(
            logits, np.arange(VOCAB), len(tokens), sampling)
        out.append(token)
        tokens.append(token)
        if token in eos:
            break
    return out


class CausalBackend:
    rows, logit_rows, capacity, vocab, eos = 11, 9, 512, VOCAB, ()

    def __init__(self, mismatch=()):
        self.target_cache, self.mtp_cache = {}, {}
        self.arena = torch.empty((self.rows, 4), dtype=torch.float64)
        self.targets, self.mtps, self.samples = [], [], []
        self.mismatch = set(mismatch)
        self.syncs = 0
        self.fault = False

    def _positions(self, tokens, start, extent):
        assert 0 <= start < start+len(tokens) <= extent.size
        assert extent.base+extent.size <= self.capacity
        return list(range(start, start+len(tokens)))

    def target(self, tokens, start, extent):
        positions = self._positions(tokens, start, extent)
        self.targets.append((extent.generation, start, tuple(tokens)))
        if self.fault:
            raise RuntimeError('injected target failure')
        for p, token in zip(positions, tokens):
            self.target_cache[extent.base+p] = (extent.generation, token)
        for i, p in enumerate(positions):
            cache = [self.target_cache[extent.base+j] for j in range(p+1)]
            assert all(owner == extent.generation for owner, _ in cache), 'foreign or uninitialized prefix'
            prefix = [token for _, token in cache]
            self.arena[i] = torch.tensor([history(prefix), p, prefix[-1], 0])
        return self.arena[:len(tokens)]

    def mtp(self, tokens, hidden, start, extent):
        positions = self._positions(tokens, start, extent)
        inputs = hidden.clone()  # real MTP consumes these before overwriting its output
        self.mtps.append((extent.generation, start, tuple(tokens), inputs.tolist()))
        for i, (p, token) in enumerate(zip(positions, tokens)):
            h, prev_pos, _, kind = (int(x) for x in inputs[i])
            if kind == 0:  # canonical target hidden p, with token p+1
                cache = [self.target_cache[extent.base+j] for j in range(p+1)]
                assert all(owner == extent.generation for owner, _ in cache)
                assert prev_pos == p and h == history([t for _, t in cache]), 'wrong target/MTP shift'
            else:  # previous MTP normalized hidden at p-1, not a target row
                owner, prior = self.mtp_cache[extent.base+p-1]
                assert owner == extent.generation and prev_pos == p-1
                assert tuple(inputs[i].tolist()) == prior, 'lost recursive MTP hidden'
            # All causal MTP positions must have been written by this owner.
            assert all(self.mtp_cache[extent.base+j][0] == extent.generation for j in range(p))
            row = (advance(h, token, p+1), p, token, 1)
            self.mtp_cache[extent.base+p] = (extent.generation, row)
            self.arena[i] = torch.tensor(row)
        return self.arena[:len(tokens)]

    def sample(self, hidden, positions, sampling):
        assert len(hidden) == len(positions) <= self.logit_rows
        logits = []
        for row, position in zip(hidden, positions):
            h, p, _, kind = (int(x) for x in row)
            assert position == p+1+kind, 'sample uses the wrong absolute token position'
            values = scores(h)
            if kind and position in self.mismatch:
                values = np.roll(values, 1)
            logits.append(values)
        values = np.stack(logits)
        self.samples.append((tuple(positions), tuple(int(r[3]) for r in hidden)))
        if sampling is None or sampling.temperature == 0:
            return [int(i) for i in values.argmax(axis=1)]
        ids = np.broadcast_to(np.arange(VOCAB), values.shape)
        return choose_rows(values, ids, positions, sampling)

    def synchronize(self):
        self.syncs += 1


def finish(engine, request):
    seen = []
    for _ in range(200):
        step = engine.step(request)
        seen.extend(step.tokens)
        if step.finished:
            return seen
    raise AssertionError('request failed to finish')


class RequestTests(unittest.TestCase):
    def engine(self, *, mismatch=(), rows=11, eos=()):
        backend = CausalBackend(mismatch)
        backend.rows, backend.logit_rows, backend.eos = rows, min(rows, 9), tuple(eos)
        return RequestEngine(backend, context_limit=128), backend

    def test_serial_parity_for_greedy_keyed_sampling_and_rejections(self):
        samplings = [None, Sampling(0, .8, 5, .9), Sampling(1234, 1.1, 0, .8, .1)]
        for sampling in samplings:
            for depth in (0, 1, 4, 8):
                for bad in ((), range(0, 100), range(5, 100, 3)):
                    with self.subTest(sampling=sampling, depth=depth, bad=tuple(bad)[:3]):
                        engine, backend = self.engine(mismatch=bad)
                        prompt = [2, 4, 8, 1, 3, 7, 9, 6, 8, 2, 5, 3, 8, 7, 2]
                        request = engine.start('a', prompt, max_tokens=25, sampling=sampling, draft_tokens=depth)
                        self.assertEqual(finish(engine, request), serial(prompt, 25, sampling))
                        self.assertEqual(request.tokens, prompt+request.output[:-1])
                        self.assertEqual(len(request.pending_hidden), len(request.tokens)-request.mtp_end)
                        self.assertEqual(request.finish_reason, 'length')
                        self.assertLessEqual(len(request.output), request.max_tokens)
                        self.assertLessEqual(request.accepted, request.drafted)

    def test_each_rejection_depth_commits_only_the_accepted_inputs(self):
        prompt = [2, 5, 1]
        for reject in range(4):
            engine, backend = self.engine(mismatch=[len(prompt)+1+reject])
            request = engine.start('a', prompt, max_tokens=12, draft_tokens=4)
            first = engine.step(request)
            step = engine.step(request)
            self.assertEqual(step.accepted, reject)
            self.assertEqual(len(request.tokens), len(prompt)+1+reject)
            self.assertEqual(request.mtp_end, len(prompt))
            self.assertEqual(len(request.pending_hidden), reject+1)
            rest = finish(engine, request)
            self.assertEqual(list(first.tokens+step.tokens)+rest, serial(prompt, 12, None))

    def test_four_requests_interleave_without_cache_or_workspace_aliases(self):
        engine, backend = self.engine(mismatch=range(8, 100, 4))
        requests = [engine.start(str(i), [i+1]*(i+4), max_tokens=19, sampling=Sampling(i), draft_tokens=4)
                    for i in range(4)]
        got = {r.key: [] for r in requests}
        for _ in range(100):
            for r in reversed(requests):
                if r.status in ('prefill', 'decode'):
                    got[r.key].extend(engine.step(r).tokens)
            if all(r.status == 'finished' for r in requests):
                break
        for r in requests:
            self.assertEqual(got[r.key], serial(r.prompt, r.max_tokens, r.sampling))
        with self.assertRaises(ValueError):
            engine.start('0', [3], max_tokens=1)
        for r in requests:
            engine.drop(r)
        self.assertEqual(engine.pool.free, [(0, backend.capacity)])

    def test_slot_cap_and_failed_admission_preserve_live_requests(self):
        engine, backend = self.engine()
        requests = [engine.start(str(i), [1], max_tokens=3) for i in range(4)]
        before = dict(engine.pool.leases)
        with self.assertRaises(MemoryError):
            engine.start('fifth', [2], max_tokens=3)
        self.assertEqual(engine.pool.leases, before)
        for r in requests:
            self.assertEqual(finish(engine, r), serial(r.prompt, 3, None))
        with self.assertRaises(MemoryError):
            engine.start('fifth', [2], max_tokens=3)
        engine.drop(requests[0])
        fifth = engine.start('fifth', [2], max_tokens=3)
        self.assertEqual(finish(engine, fifth), serial([2], 3, None))

    def test_exact_prompt_replay_keeps_its_head_and_uses_new_sampling(self):
        engine, backend = self.engine()
        prompt = [3, 9, 7, 1, 2, 5]
        old = engine.start('old', prompt, max_tokens=1)
        finish(engine, old)
        calls = len(backend.targets)
        new = engine.start('new', prompt, max_tokens=1, sampling=Sampling(987), resume=old)
        self.assertEqual(finish(engine, new), serial(prompt, 1, new.sampling))
        self.assertEqual(len(backend.targets), calls)
        self.assertEqual(new.reused_tokens, len(prompt))
        self.assertEqual(old.status, 'transferred')
        with self.assertRaises(ValueError):
            engine.drop(old)

    def test_followup_absorbs_all_kept_target_rows_with_new_next_tokens(self):
        engine, backend = self.engine(mismatch=range(8, 100, 4))
        old = engine.start('conversation', [2, 4, 1, 8], max_tokens=13, draft_tokens=4)
        finish(engine, old)
        prefix = list(old.tokens)
        prompt = [*prefix, old.pending, 3, 5, 7, 9]
        new = engine.start('conversation', prompt, max_tokens=17, sampling=Sampling(345), resume=old)
        calls = len(backend.targets)
        self.assertEqual(finish(engine, new), serial(prompt, 17, new.sampling))
        self.assertEqual(backend.targets[calls][1], len(prefix))
        self.assertEqual(new.reused_tokens, len(prefix))
        self.assertEqual(new.tokens, prompt+new.output[:-1])

    def test_nonmatching_prefix_and_failed_growth_do_not_destroy_kept_state(self):
        engine, backend = self.engine()
        old = engine.start('old', [2, 3], max_tokens=1)
        finish(engine, old)
        neighbour = engine.start('next', [4], max_tokens=4)
        for prompt in ([7, 8], [2, 3, 4, 5]):
            with self.assertRaises((ValueError, MemoryError)):
                engine.start('new', prompt, max_tokens=5, resume=old)
            self.assertIs(engine.requests['old'], old)
            self.assertEqual(old.status, 'finished')
        self.assertEqual(finish(engine, neighbour), serial([4], 4, None))

    def test_eos_at_first_token_and_within_verified_window_never_overemits(self):
        prompt = [3, 5, 6, 1]
        expected = serial(prompt, 20, None)
        for eos in (expected[0], expected[2], expected[5]):
            engine, backend = self.engine(eos=[eos])
            r = engine.start('eos', prompt, max_tokens=20)
            self.assertEqual(finish(engine, r), serial(prompt, 20, None, [eos]))
            self.assertEqual(r.finish_reason, 'stop')
            self.assertEqual(r.tokens, prompt+r.output[:-1])
        engine, backend = self.engine(eos=expected)
        r = engine.start('ignore', prompt, max_tokens=20, ignore_eos=True)
        self.assertEqual(finish(engine, r), expected)
        self.assertEqual(r.finish_reason, 'length')

    def test_cancel_after_target_keeps_uncommitted_rows_invisible_on_reuse(self):
        engine, backend = self.engine()
        old = engine.start('cancel', [1, 7, 2, 9], max_tokens=10)
        step = engine.step(old, cancelled=lambda: bool(backend.targets))
        self.assertEqual(step.tokens, ())
        self.assertEqual(old.tokens, [])
        self.assertEqual(old.status, 'cancelled')
        base = old.extent.base
        engine.drop(old)
        new = engine.start('reuse', [3, 2], max_tokens=16)
        self.assertEqual(new.extent.base, base)
        self.assertNotEqual(new.extent.generation, old.extent.generation)
        self.assertEqual(finish(engine, new), serial([3, 2], 16, None))
        self.assertGreaterEqual(backend.syncs, 2)

    def test_cancel_mid_draft_emits_nothing_and_does_not_advance_target(self):
        engine, backend = self.engine()
        r = engine.start('cancel', [1, 2, 3], max_tokens=20)
        engine.step(r)
        before = list(r.tokens)
        calls = len(backend.mtps)
        step = engine.step(r, cancelled=lambda: len(backend.mtps) >= calls+2)
        self.assertEqual(step.tokens, ())
        self.assertEqual(r.tokens, before)
        self.assertEqual(len(r.output), 1)
        self.assertEqual(r.mtp_end, len(before))
        self.assertEqual(r.status, 'cancelled')

    def test_cancel_during_sampling_discards_the_unpublished_result(self):
        for during_prefill in (True, False):
            engine, backend = self.engine()
            r = engine.start('cancel', [1, 2, 3], max_tokens=10, draft_tokens=0)
            if not during_prefill:
                engine.step(r)
            previous = list(r.output)
            samples = len(backend.samples)
            event = engine.step(r, cancelled=lambda: len(backend.samples) > samples)
            self.assertFalse(event.tokens)
            self.assertEqual(r.output, previous)
            self.assertEqual(r.status, 'cancelled')

    def test_failure_keeps_extent_owned_until_writes_are_synchronized(self):
        engine, backend = self.engine()
        r = engine.start('bad', [2, 3], max_tokens=6)
        backend.fault = True
        with self.assertRaisesRegex(RuntimeError, 'injected'):
            engine.step(r)
        self.assertEqual(r.status, 'failed')
        engine.pool.require(r.extent)
        with patch.object(backend, 'synchronize', side_effect=RuntimeError('unfinished writes')):
            with self.assertRaises(RuntimeError):
                engine.drop(r)
        engine.pool.require(r.extent)
        backend.fault = False
        engine.drop(r)
        self.assertFalse(engine.pool.leases)

    def test_input_limits_reject_before_cache_allocation_or_model_calls(self):
        engine, backend = self.engine()
        cases = [dict(prompt=[]), dict(prompt=[-1]), dict(prompt=[VOCAB]), dict(prompt=[True]),
                 dict(max_tokens=0), dict(max_tokens=True), dict(max_tokens=128),
                 dict(draft_tokens=9), dict(draft_tokens=True),
                 dict(sampling=Sampling(1, float('nan'))), dict(sampling=Sampling(1, 1, 3, .8, 2))]
        for changes in cases:
            args = dict(prompt=[1], max_tokens=5)
            args.update(changes)
            with self.assertRaises(ValueError):
                engine.start('bad', **args)
            self.assertFalse(engine.pool.leases)
            self.assertFalse(backend.targets)
        with patch('threading.get_ident', return_value=-1):
            with self.assertRaises(RuntimeError):
                engine.start('wrong-worker', [1], max_tokens=2)

    def test_tiny_chunks_and_output_budgets(self):
        for rows in (1, 2, 5, 11):
            for budget in range(1, 10):
                engine, backend = self.engine(rows=rows)
                r = engine.start('small', [4, 3, 2, 1, 6, 9], max_tokens=budget,
                                 draft_tokens=min(4, rows-1))
                self.assertEqual(finish(engine, r), serial(r.prompt, budget, None))
                self.assertEqual(len(r.output), budget)
                self.assertTrue(all(len(tokens) <= rows for _, _, tokens in backend.targets))


class PoolTests(unittest.TestCase):
    def test_fragmentation_coalescing_and_stale_generation(self):
        pool = CachePool(100)
        a, b, c = (pool.allocate(n) for n in (20, 30, 40))
        pool.release(b)
        with self.assertRaises(MemoryError):
            pool.allocate(35)
        grown = pool.grow(a, 45)
        with self.assertRaises(ValueError):
            pool.require(a)
        self.assertEqual(grown.base, 0)
        pool.release(c)
        pool.release(grown)
        self.assertEqual(pool.free, [(0, 100)])
        replacement = pool.allocate(100)
        with self.assertRaises(ValueError):
            pool.release(grown)
        self.assertEqual(replacement.base, 0)


if __name__ == '__main__':
    unittest.main()
