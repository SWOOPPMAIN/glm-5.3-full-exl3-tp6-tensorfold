"""Host-only geometry for a fixed-order six-rank hidden-state reduction."""


def reduction_plan(rows, *, bulk_min_rows=None):
    if type(rows) is not int or not 1 <= rows <= 3072:
        raise ValueError('TP6 reduction requires 1..3072 rows')
    if bulk_min_rows is not None and (type(bulk_min_rows) is not int or not 1 <= bulk_min_rows <= 3072):
        raise ValueError('Bulk reduction threshold must be None or 1..3072')
    enabled = bulk_min_rows is not None and rows >= bulk_min_rows
    shard_rows = (rows + 5) // 6 if enabled else 0
    padded_rows = 6 * shard_rows
    gather_rows = 6 * rows
    return dict(rows=rows, bulk_min_rows=bulk_min_rows, gather_rows=gather_rows,
                padded_rows=padded_rows, shard_rows=shard_rows,
                total=(gather_rows + padded_rows + shard_rows) * 6144 * 2)
