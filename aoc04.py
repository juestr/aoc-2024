#!/usr/bin/env python3

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from aoc_util import np_raw_table, run_aoc

XMAS_LR = np.frombuffer(b"XMAS", dtype="uint8")[[(0, 1, 2, 3), (3, 2, 1, 0)]]
XMAS_DIAG_MASK = np.identity(4, dtype=bool)
XMAS_DIAG = XMAS_LR[:, np.newaxis, :] * XMAS_DIAG_MASK[np.newaxis, :, :]
XMAS_DIAG2 = np.fliplr(XMAS_DIAG)
XMAS_DIAG_MASK2 = np.fliplr(XMAS_DIAG_MASK)
X_MAS = np.array(
    [
        t := np.frombuffer(b"M.S.A.M.S", dtype="uint8").reshape((3, 3)),
        np.rot90(t, 1),
        np.rot90(t, 2),
        np.rot90(t, 3),
    ]
)
X_MAS_MASK = X_MAS[0] != ord(".")


def count_patterns(xs, patterns, mask=None, axis=None):
    """In xs, count any occurance of patterns[0], patterns[1], ..."""

    shape = patterns.shape[1:]
    mask = np.ones(shape, dtype=bool) if mask is None else mask
    windows = sliding_window_view(xs, shape, axis=axis).reshape((-1, *shape))
    return int(
        np.sum(
            np.logical_and.reduce(
                # - outer join on first axis, apply == on common pattern axes
                (windows[np.newaxis, :] == patterns[:, np.newaxis])[..., mask],
                axis=-1,
            )
        )
    )


def aoc04(data):
    yield sum(
        (
            count_patterns(data, XMAS_LR, axis=0),
            count_patterns(data, XMAS_LR, axis=1),
            count_patterns(data, XMAS_DIAG, mask=XMAS_DIAG_MASK),
            count_patterns(data, XMAS_DIAG2, mask=XMAS_DIAG_MASK2),
        )
    )
    yield count_patterns(data, X_MAS, mask=X_MAS_MASK)


if __name__ == "__main__":
    run_aoc(aoc04, transform=np_raw_table)
