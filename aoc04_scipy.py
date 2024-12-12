#!/usr/bin/env python3

import numpy as np
from scipy.signal import correlate

from aoc_util import np_raw_table, run_aoc

XMAS = np.frombuffer(b"XMAS", dtype="uint8")
XMAS_LR = np.equal.outer(XMAS, [XMAS])
XMAS_DIAG = XMAS_LR * np.identity(4, dtype=bool)
X_MAS = np.equal.outer(
    XMAS,
    np.frombuffer(b"M.S.A.M.S", dtype="uint8").reshape((3, 3)),
)


def aoc04(table):
    def count_pattern(pattern):
        threshold = np.sum(pattern)
        rotated = (np.rot90(pattern, r, axes=(1, 2)) for r in range(4))
        return sum(
            int(np.sum(correlate(data, p, mode="valid") == threshold)) for p in rotated
        )

    data = np.equal.outer(XMAS, table).astype("uint8")
    yield count_pattern(XMAS_LR) + count_pattern(XMAS_DIAG)
    yield count_pattern(X_MAS)


if __name__ == "__main__":
    run_aoc(aoc04, transform=np_raw_table)
