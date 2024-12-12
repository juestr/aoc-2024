#!/usr/bin/env python3

import numpy as np
from scipy import ndimage, signal

from aoc_util import np_raw_table, run_aoc

EDGE = np.array([[1, -1]], dtype="int8")
EDGES = [np.rot90(EDGE, r) for r in range(4)]
STRAIGHT_EDGE = np.array([[-1, -1], [1, 1]], dtype="int8")
STRAIGHT_EDGES = [np.rot90(STRAIGHT_EDGE, r) for r in range(4)]


def count_patterns(xs, patterns, threshold):
    return sum(
        np.sum(signal.correlate(xs, pattern, mode="full") >= threshold)
        for pattern in patterns
    )


def aoc12(garden):
    def plant_costs(plant):
        regions, n = ndimage.label(garden == plant)
        return sum(region_costs(regions == i) for i in range(1, n + 1))

    def region_costs(region):
        area = np.sum(region)
        fence = count_patterns(region, EDGES, 1)
        straight_edges = count_patterns(region, STRAIGHT_EDGES, 2)
        return np.array([area * fence, area * (fence - straight_edges)])

    yield from sum(map(plant_costs, np.unique(garden)))


if __name__ == "__main__":
    run_aoc(aoc12, transform=np_raw_table)
