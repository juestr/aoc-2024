#!/usr/bin/env python3

from collections import Counter

import numpy as np
from scipy.signal import correlate

from aoc_util import run_aoc


def aoc22(data):
    def derive_next(secret):
        out = secret.copy()
        out ^= out << 6
        out &= 0xFFFFFF
        out ^= out >> 5
        out ^= out << 11
        out &= 0xFFFFFF
        return out

    price = np.pad(data[:, None], ((0, 0), (0, 2000)))
    for i in range(2000):
        price[:, i + 1] = derive_next(price[:, i])
    yield np.sum(price[:, 2000])

    price %= 10
    delta4sig = correlate(
        price[:, 1:] - price[:, :-1] + 10,
        [100 ** np.arange(4, dtype="int32")],
        mode="valid",
    )
    ctr = Counter()
    for d4s_row, price_row in zip(delta4sig, price[:, 4:]):
        uniques, uidx = np.unique(d4s_row, return_index=True)
        for d4s, p in zip(uniques, price_row[uidx]):
            ctr[d4s] += p
    [(_, bananas)] = ctr.most_common(1)
    yield bananas


if __name__ == "__main__":
    run_aoc(aoc22, read=(np.loadtxt, dict(dtype="int32")))
