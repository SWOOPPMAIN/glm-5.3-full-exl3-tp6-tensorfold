"""Host-side visible-token bounds; no device synchronization during a forward."""


def visible_token_bound(positions, capacity):
    """Compute the bound from the same validated CPU positions sent to the GPU.

    Graph reuse requires every replay's positions to fit its captured bound.
    Cache bases are physical offsets and must not be added to this logical bound.
    """
    if (type(capacity) is not int or not 1 <= capacity <= 1048576
            or not isinstance(positions, (list, tuple)) or not positions
            or any(type(p) is not int or not 0 <= p < capacity for p in positions)):
        raise ValueError('Expected nonempty valid host query positions and cache capacity')
    return max(positions)+1


def score_width(capacity, visible_tokens=None):
    """Bucket a caller-verified visible bound without changing scratch addresses.

    Omitted bounds retain the full-capacity path. The planner must establish
    max(query positions)+1 <= visible_tokens on every eager call/graph replay.
    A smaller bound is never inferred from GPU data or a previous request.
    """
    if type(capacity) is not int or not 1 <= capacity <= 1048576:
        raise ValueError('Invalid indexer capacity')
    if visible_tokens is None:
        return max(2048, capacity)
    if type(visible_tokens) is not int or not 1 <= visible_tokens <= capacity:
        raise ValueError('Visible-token bound must fit the indexer capacity')
    return max(2048, min(capacity, 1 << (visible_tokens-1).bit_length()))
