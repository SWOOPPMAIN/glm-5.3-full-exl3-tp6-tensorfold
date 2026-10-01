"""Use a real TCPStore to exercise delayed followers and timeout observations."""
from datetime import timedelta
import unittest
import torch.distributed as dist
from tensorfold.families.glm_moe_dsa.doorbell import RequestDoorbell


class DoorbellTest(unittest.TestCase):
    def setUp(self):
        self.store=dist.TCPStore('127.0.0.1',0,1,True,timedelta(seconds=2))
        self.head=RequestDoorbell(self.store,0,generation='test_generation_530607')
        self.workers=[RequestDoorbell(self.store,r,generation='test_generation_530607') for r in range(1,6)]

    def test_fast_follower_cannot_consume_delayed_followers_wakeup(self):
        for sequence in (1,2,3):
            self.assertEqual(self.head.publish(),sequence)
            # The first worker has deleted its key before any other worker even
            # begins waiting. All remaining workers must still observe the bell.
            for worker in self.workers:
                self.assertEqual(worker.wait(timedelta(seconds=1)),sequence)
            self.assertEqual(self.store.num_keys(),1)  # TCPStore's own init key

    def test_previous_generation_does_not_satisfy_new_request(self):
        self.head.publish()
        old=self.workers[0]._key(1,1)
        fresh=RequestDoorbell(self.store,1,generation='different_generation_530607')
        self.assertTrue(self.store.check([old]))
        self.assertFalse(self.store.check([fresh._key(1,1)]))
        self.assertEqual(fresh.sequence,0)

    def test_failed_observation_keeps_pending_sequence(self):
        # Real TCPStore throws on timeout. Avoid its noisy socket timeout log by
        # substituting only the observation failure; key delivery stays real.
        class FailOnce:
            def __init__(self,store):self.store=store;self.first=True
            def set(self,*a):return self.store.set(*a)
            def delete_key(self,*a):return self.store.delete_key(*a)
            def wait(self,*a):
                if self.first:self.first=False;raise TimeoutError('observation only')
                return self.store.wait(*a)
        worker=RequestDoorbell(FailOnce(self.store),3,generation='test_generation_530607')
        with self.assertRaises(TimeoutError):worker.wait(timedelta(seconds=1))
        self.assertEqual(worker.sequence,0)
        self.head.publish()
        self.assertEqual(worker.wait(timedelta(seconds=1)),1)

    def test_rank_roles(self):
        with self.assertRaises(ValueError):self.head.wait(timedelta(seconds=1))
        with self.assertRaises(ValueError):self.workers[0].publish()
        with self.assertRaises(ValueError):RequestDoorbell(self.store,6,generation='test_generation_530607')


if __name__=='__main__':unittest.main()
