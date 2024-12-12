#!/usr/bin/env python3

import numpy as np
from scipy.signal import correlate

from aoc_util import np_raw_table, run_aoc

XMAS = np.frombuffer(b"XMAS", dtype="uint8")
XMAS_BINARY = np.equal.outer([XMAS], XMAS)
XMAS_DIAG_BINARY = XMAS_BINARY * np.identity(4, dtype=bool)[:, :, np.newaxis]
X_MAS_BINARY = np.equal.outer(
    np.frombuffer(b"M.S.A.M.S", dtype="uint8").reshape((3, 3)), XMAS
)


def aoc04(table):
    def count_pattern(pattern):
        threshold = np.sum(pattern)
        rotated = (np.rot90(pattern, r) for r in range(4))
        return sum(
            int(np.sum(correlate(data, p, mode="valid") == threshold)) for p in rotated
        )

    data = np.equal.outer(table, XMAS).astype("uint8")
    yield count_pattern(XMAS_BINARY) + count_pattern(XMAS_DIAG_BINARY)
    yield count_pattern(X_MAS_BINARY)


if __name__ == "__main__":
    run_aoc(aoc04, transform=np_raw_table)
