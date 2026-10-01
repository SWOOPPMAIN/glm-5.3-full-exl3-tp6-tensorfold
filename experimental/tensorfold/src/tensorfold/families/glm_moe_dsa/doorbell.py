"""Wake five TP6 workers without leaving a GPU collective spinning when idle.

Adapts the idle-store idea from MiaAI-Lab/TensorFold (commit eec0205d657c).
Each follower owns a separate key: the two-rank recipe's shared-key deletion
would let an early follower consume a wakeup before another follower sees it.
The engine supplies a unique generation and the same sequence on all ranks.
"""
from datetime import timedelta
import re


class RequestDoorbell:
    def __init__(self,store,rank,*,generation):
        if (type(rank) is not int or not 0<=rank<6 or not isinstance(generation,str)
                or not re.fullmatch(r'[a-zA-Z0-9_-]{16,128}',generation)
                or not all(callable(getattr(store,name,None)) for name in ('set','wait','delete_key'))):
            raise ValueError('Expected a store, TP6 rank and unique generation identifier')
        self.store,self.rank,self.generation=store,rank,generation
        self.sequence=0

    def _key(self,sequence,rank):
        return f'tf_glm53_tp6/{self.generation}/{sequence}/{rank}'

    def publish(self):
        """Rank0 signals one request to all five followers before its header.

        A store error is a transport failure for the owning engine. It must not
        start another request or silently skip a partially published sequence.
        """
        if self.rank!=0:
            raise ValueError('Only rank0 publishes request wakeups')
        self.sequence+=1
        for rank in range(1,6):self.store.set(self._key(self.sequence,rank),b'1')
        return self.sequence

    def wait(self,timeout):
        """A follower waits on the host, then removes only its own key.

        On a timeout the sequence stays unchanged, so the owner can observe the
        same pending request again. This class never retries or restarts ranks.
        """
        if self.rank==0 or not isinstance(timeout,timedelta) or timeout.total_seconds()<=0:
            raise ValueError('Follower waits require a positive explicit timeout')
        next_sequence=self.sequence+1;key=self._key(next_sequence,self.rank)
        self.store.wait([key],timeout)
        self.store.delete_key(key)
        self.sequence=next_sequence
        return self.sequence
