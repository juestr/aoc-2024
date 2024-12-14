#!/usr/bin/env python3

import re

import numpy as np
from funcy import lmap, re_iter
from scipy import ndimage

from aoc_util import dbg, np_condense, run_aoc

EXAMPLE_ROWS = 12


def setup(data):
    pattern = r"^p=(\d+),(\d+) v=(-?\d+),(-?\d+)"
    robots = np.array([lmap(int, i) for i in re_iter(pattern, data, re.M)])
    size = np.array([101, 103] if robots.shape[0] != EXAMPLE_ROWS else [11, 7])
    return (robots, size)


def aoc14(robots, size):
    def simulate(time):
        return (robots[:, 0:2] + time * robots[:, 2:4]) % size

    def qcount(a, b):
        return int(np.sum(a & b))

    def image(xy):
        img = np.zeros(size[::-1], dtype=int)
        np.add.at(img, tuple(xy[:, ::-1].T), 1)
        return img

    area = simulate(100)
    dbg(np_condense(image(area)), t="at time 100")
    size_h = size // 2
    qW = area[:, 0] < size_h[0]
    qE = area[:, 0] > size_h[0]
    qN = area[:, 1] < size_h[1]
    qS = area[:, 1] > size_h[1]
    yield qcount(qN, qW) * qcount(qN, qE) * qcount(qS, qW) * qcount(qS, qE)

    if robots.shape[0] != EXAMPLE_ROWS:
        parts_threshold = robots.shape[0] // 3
        for time in range(np.prod(size)):
            area = simulate(time)
            img = image(area)
            _, parts = ndimage.label(img)
            if parts <= parts_threshold:
                dbg(np_condense(img), t=f"{time=} {parts=}")
                yield time
                return
        assert False, "time exceeded"


if __name__ == "__main__":
    run_aoc(
        aoc14,
        transform=setup,
        np_printoptions=dict(linewidth=300, threshold=500000, edgeitems=10),
    )
