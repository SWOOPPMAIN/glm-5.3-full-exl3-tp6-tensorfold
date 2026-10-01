"""CPU-only shape/pointer contract for original 512-wide expert fragments."""


def packed_projection_bytes(bits, hidden, intermediate):
    if bits not in (3, 4) or hidden % 128 or intermediate % 128:
        raise ValueError('Requires K3/K4 and full 128-element rotation blocks')
    return hidden * intermediate * bits // 8


def projection_offsets(bits, experts, hidden, intermediate):
    stride = packed_projection_bytes(bits, hidden, intermediate)
    return dict(gate=[expert * stride for expert in range(experts)],
                up=[(experts + expert) * stride for expert in range(experts)],
                down=[expert * stride for expert in range(experts)])


def scratch_bytes(tokens, topk, hidden, intermediate):
    rows = tokens * topk
    return rows * (2 * hidden + 2 * intermediate) * 2 + tokens * hidden * 4


def reference_routes(ids, weights, mapping, experts, tile_rows=64):
    """CPU oracle for sentinel omission, duplicate routes, and segment bounds."""
    records = []
    for token, (token_ids, token_weights) in enumerate(zip(ids, weights, strict=True)):
        for logical, weight in zip(token_ids, token_weights, strict=True):
            local = mapping[logical]
            if local >= 0:
                if local >= experts:
                    raise ValueError('Local route outside expert table')
                records.append((local, token, weight))
    records.sort(key=lambda row: row[0])
    segments = []
    offset = 0
    for expert in range(experts):
        count = sum(row[0] == expert for row in records)
        for row in range(0, count, tile_rows):
            segments.append((expert, offset + row, min(tile_rows, count - row)))
        offset += count
    return records, segments
