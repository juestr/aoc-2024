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
    delta = price[:, 1:] - price[:, :-1]
    delta4sig = correlate(delta + 10, [[1_000_000, 10_000, 100, 1]], mode="valid")
    unique_delta4sig = np.zeros_like(delta4sig)
    for i in range(delta4sig.shape[0]):
        _, uidx = np.unique(delta4sig[i], return_index=True)
        unique_delta4sig[i, uidx] = 1
    delta4sig *= unique_delta4sig

    ctr = Counter(delta4sig.flat)
    del ctr[0]
    high = 0
    for d4s, n in ctr.most_common():
        if n * 9 < high:
            break
        high = max(high, np.sum(price[:, 4:] * (delta4sig == d4s)))
    yield high


if __name__ == "__main__":
    run_aoc(
        aoc22,
        read=(np.loadtxt, dict(dtype="int32")),
        np_printoptions=dict(linewidth=160, threshold=3000, edgeitems=8),
    )
