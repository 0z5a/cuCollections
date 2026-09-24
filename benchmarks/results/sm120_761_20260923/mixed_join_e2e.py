# SPDX-FileCopyrightText: Copyright (c) 2026 0z5a
# SPDX-License-Identifier: Apache-2.0

"""Exercise libcudf's mixed join from the pylibcudf consumer API."""

import json
import statistics
import time

import cupy as cp
import numpy as np
import pyarrow as pa
import pylibcudf as plc


LEFT = plc.expressions.TableReference.LEFT
RIGHT = plc.expressions.TableReference.RIGHT
PREDICATE = plc.expressions.Operation(
    plc.expressions.ASTOperator.LESS,
    plc.expressions.ColumnReference(0, LEFT),
    plc.expressions.ColumnReference(0, RIGHT),
)
NULLS = plc.types.NullEquality.EQUAL


def tables(keys, values):
    return (
        plc.Table.from_arrow(pa.table({"key": keys})),
        plc.Table.from_arrow(pa.table({"value": values})),
    )


def join(algorithm, left, right):
    return algorithm(left[0], right[0], left[1], right[1], PREDICATE, NULLS)


left = tables([1, 1, 2, 3, 4, 4], [1, 4, 7, 1, 2, 8])
right = tables([1, 1, 2, 4, 5], [3, 5, 9, 3, 10])
inner = join(plc.join.mixed_inner_join, left, right)
pairs = sorted(zip(inner[0].to_arrow().to_pylist(), inner[1].to_arrow().to_pylist()))
assert pairs == [(0, 0), (0, 1), (1, 1), (2, 2), (4, 3)], pairs
semi = join(plc.join.mixed_left_semi_join, left, right)
rows = sorted(semi.to_arrow().to_pylist())
assert rows == [0, 1, 2, 4], rows
print(json.dumps({"check": "mixed_join_correctness", "pairs": len(pairs), "semi_rows": rows}), flush=True)

rng = np.random.default_rng(761)
for size in (1 << 20, 1 << 22):
    keys = np.arange(size, dtype=np.int32)
    left = tables(keys, rng.integers(0, 1 << 20, size=size, dtype=np.int32))
    right = tables(rng.permutation(keys), rng.integers(0, 1 << 20, size=size, dtype=np.int32))
    for name, algorithm in (("mixed_inner_join", plc.join.mixed_inner_join), ("mixed_left_semi_join", plc.join.mixed_left_semi_join)):
        for _ in range(2):
            result = join(algorithm, left, right)
            cp.cuda.runtime.deviceSynchronize()
        samples = []
        for _ in range(10):
            start = time.perf_counter()
            result = join(algorithm, left, right)
            cp.cuda.runtime.deviceSynchronize()
            samples.append((time.perf_counter() - start) * 1000)
        print(json.dumps({"join": name, "rows_per_side": size, "median_ms": statistics.median(samples), "min_ms": min(samples), "max_ms": max(samples), "repetitions": len(samples)}), flush=True)
