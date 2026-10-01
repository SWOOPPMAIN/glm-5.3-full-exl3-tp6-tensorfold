"""Immutable BF16 projection geometry, scoped to one full-model call.

Only output tiles and pipeline stages vary. K tiles stay64, four warps and
FP32 accumulation stay fixed. Numerical qualification is still required for
any candidate because compiler layout changes may alter hardware arithmetic.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True)
class LinearTile:
    m: int = 16
    n: int = 64
    stages: int = 2

    def __post_init__(self):
        if (type(self.m) is not int or self.m not in (16,32,64)
                or type(self.n) is not int or self.n not in (16,32,64,128)
                or type(self.stages) is not int or self.stages not in (2,3)):
            raise ValueError('Unadmitted BF16 output tile or pipeline depth')


@dataclass(frozen=True)
class ProjectionPlan:
    decode: LinearTile = LinearTile()
    prefill: LinearTile = LinearTile()
    bulk_min_rows: int = 256

    def __post_init__(self):
        if (type(self.decode) is not LinearTile or self.decode.m!=16
                or type(self.prefill) is not LinearTile
                or type(self.bulk_min_rows) is not int or self.bulk_min_rows not in (128,256)):
            raise ValueError('Unadmitted projection plan')

    def tile(self,rows):
        if type(rows) is not int or not 1<=rows<=3072:
            raise ValueError('Projection row count exceeds admitted model workspace')
        return self.prefill if rows>=self.bulk_min_rows else self.decode


REFERENCE_TILE=LinearTile()
REFERENCE_PLAN=ProjectionPlan()
_current=ContextVar('glm53_tp6_projection_plan',default=REFERENCE_PLAN)


def current_projection_plan():
    return _current.get()


@contextmanager
def projection_scope(plan):
    if type(plan) is not ProjectionPlan:
        raise ValueError('Projection scope requires an immutable admitted plan')
    token=_current.set(plan)
    try:yield
    finally:_current.reset(token)
