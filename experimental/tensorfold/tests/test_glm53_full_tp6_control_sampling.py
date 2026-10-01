"""CPU decision wire tests; NCCL/device execution is a separate GPU gate."""
import threading
import unittest
import torch
from tensorfold.families.glm_moe_dsa.control_sampling import broadcast_decision


class SamplingDecisionTests(unittest.TestCase):
    def run_six(self, choose, *, follower_rows=3):
        packets = [torch.empty(11, dtype=torch.int64) for _ in range(6)]
        ready, consumed = threading.Barrier(6), threading.Barrier(6)
        results, errors, calls = {}, {}, []
        def run(rank):
            def sample(logits, positions, sampling):
                calls.append(rank)
                return choose(logits, positions, sampling)
            def broadcast(packet):
                ready.wait(3)
                if rank:
                    packet.copy_(packets[0])
                consumed.wait(3)
            n = 3 if rank == 0 else follower_rows
            try:
                results[rank] = broadcast_decision(torch.zeros(n, 31), list(range(10, 10+n)), None,
                    rank=rank, packet=packets[rank], broadcast=broadcast, choose=sample)
            except Exception as exc:
                errors[rank] = exc
        threads = [threading.Thread(target=run, args=(r,)) for r in range(6)]
        for t in threads: t.start()
        for t in threads: t.join(4)
        self.assertFalse(any(t.is_alive() for t in threads))
        self.assertEqual(calls, [0])
        return results, errors

    def test_only_leader_samples_and_every_rank_receives_same_ids(self):
        results, errors = self.run_six(lambda *_: [3, 7, 12])
        self.assertFalse(errors)
        self.assertEqual(results, {r: [3, 7, 12] for r in range(6)})

    def test_leader_failure_and_bad_ids_reach_every_rank(self):
        def fail(*_): raise ValueError('sampler failed')
        for choose in (fail, lambda *_: [31, 1, 2], lambda *_: [2], lambda *_: [True, 2, 3]):
            results, errors = self.run_six(choose)
            self.assertFalse(results)
            self.assertEqual(set(errors), set(range(6)))
            self.assertTrue(all(isinstance(e, RuntimeError) for e in errors.values()))

    def test_fixed_size_packet_exposes_follower_geometry_mismatch(self):
        results, errors = self.run_six(lambda *_: [3, 7, 12], follower_rows=2)
        self.assertEqual(results, {0: [3, 7, 12]})
        self.assertEqual(set(errors), set(range(1, 6)))


if __name__ == '__main__': unittest.main()
