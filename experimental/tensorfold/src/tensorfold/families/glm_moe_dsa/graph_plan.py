"""Deterministic decode graph geometry; physical cache offsets are not keys."""
from dataclasses import dataclass
from .indexer_plan import score_width


@dataclass(frozen=True)
class GraphKey:
    operation: str
    rows: int
    visible: int


def graph_key(operation, rows, capacity, *, visible=None, max_rows=5):
    if (operation not in ('target', 'mtp', 'head') or type(capacity) is not int
            or not 1 <= capacity <= 1048576 or type(rows) is not int
            or not 1 <= rows <= 3072 or type(max_rows) is not int or not 1 <= max_rows <= 17):
        raise ValueError('Invalid decode graph operation/row geometry')
    if operation == 'head':
        if visible is not None:
            raise ValueError('Vocabulary projection has no context bound')
        return GraphKey(operation, rows, 0) if rows <= max_rows else None
    # Even eager-size inputs must have a valid caller-established logical bound.
    bound = min(capacity, score_width(capacity, visible))
    if visible is None:
        raise ValueError('Request graphs require an explicit logical visible bound')
    limit = 1 if operation == 'mtp' else max_rows
    return GraphKey(operation, rows, bound) if rows <= limit else None


def graph_reserve(max_graphs=16, max_rows=5, capture_bytes=512*2**20):
    """Additional allowance; allocator growth is also checked on every capture.

    CUDA/NCCL driver allocations remain in the existing runtime reserve and
    host guard. This is an admission allowance, not a prediction of graph size.
    """
    if (type(max_graphs) is not int or not 1 <= max_graphs <= 32
            or type(max_rows) is not int or not 1 <= max_rows <= 17
            or type(capture_bytes) is not int or not 1 <= capture_bytes <= 2**30):
        raise ValueError('Invalid bounded graph reserve')
    inputs = max_graphs*max_rows*(4*8+6144*2)
    return dict(max_graphs=max_graphs,max_rows=max_rows,capture_bytes=capture_bytes,
                inputs=inputs,total=inputs+capture_bytes+4)
