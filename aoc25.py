#!/usr/bin/env python3

import numpy as np
from funcy import select

from aoc_util import run_aoc


def aoc25(data):
    raw = bytes(select(set(".#"), data), "ASCII")
    schematics = (np.frombuffer(raw, dtype="uint8") == ord("#")).reshape(-1, 7 * 5)
    locks = schematics[schematics[:, 0] == 1]
    keys = schematics[schematics[:, 0] == 0]
    overlap = np.sum(locks[:, None, :] * keys[None, :, :], axis=-1)
    yield np.sum(overlap == 0)


if __name__ == "__main__":
    run_aoc(aoc25)
