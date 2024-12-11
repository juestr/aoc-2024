#!/usr/bin/env python3

import numpy as np

from aoc_util import np_raw_table, run_aoc

SHIFT_NESW = (
    (slice(1, None), slice(None, -1)),
    ((slice(None), slice(1, None)), (slice(None), slice(None, -1))),
    (slice(None, -1), slice(1, None)),
    ((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
)


def aoc10(topo: np.ndarray):
    def init_reach9s():
        NINES = np.sum(topo == 9)
        NINES_BYTES = (NINES + 7) // 8
        cycling_bits = 1 << (np.arange(NINES, dtype="u1") % 8)
        byte_offs = np.identity(NINES_BYTES, dtype="u1")[np.arange(NINES) // 8]
        reach9s = np.zeros((*topo.shape, NINES_BYTES), dtype="u1")
        reach9s[topo == 9] = cycling_bits[:, np.newaxis] * byte_offs
        return reach9s

    # We'll track reachability of every 9 in topo with its own bit in
    # a multi-byte packed bitmask stored in the 3rd dimension of reach9s.
    # First each 9 can reach itself.
    reach9s = init_reach9s()

    # work our way backwards
    for stage in (8, 7, 6, 5, 4, 3, 2, 1, 0):
        new_reach9s = np.zeros_like(reach9s)
        for src, dst in SHIFT_NESW:
            new_reach9s[dst] |= reach9s[src]
        new_reach9s *= (topo == stage)[..., np.newaxis]
        reach9s = new_reach9s
    yield np.sum(np.bitwise_count(reach9s))

    # This is actually easier than #1, since we can use a 2d int array
    # to track only the number of 9s reachable.
    scores = np.zeros(topo.shape, dtype=int)
    scores[topo == 9] = 1
    for stage in (8, 7, 6, 5, 4, 3, 2, 1, 0):
        new_scores = np.zeros_like(scores)
        for src, dst in SHIFT_NESW:
            new_scores[dst] += scores[src]
        new_scores *= topo == stage
        scores = new_scores
    yield np.sum(scores)


if __name__ == "__main__":
    run_aoc(aoc10, transform=(np_raw_table, dict(offs=ord("0"))))
