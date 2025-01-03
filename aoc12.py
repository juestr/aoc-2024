#!/usr/bin/env python3

import numpy as np
from scipy import ndimage, signal

from aoc_util import np_raw_table, run_aoc

EDGE = np.array([[1, -1]], dtype="int8")
STRAIGHT_EDGE = np.array([[-1, -1], [1, 1]], dtype="int8")


def aoc12(garden):
    def count_pattern(xs, pattern, threshold):
        return sum(
            np.sum(np.abs(signal.correlate(xs, p, mode="full")) >= threshold)
            for p in (pattern, pattern.T)
        )

    def plant_costs(plant):
        # return array of shape (2,) holding results for both parts
        regions, n = ndimage.label(garden == plant)
        return sum(region_costs(regions == i) for i in range(1, n + 1))

    def region_costs(region):
        area = np.sum(region)
        fence = count_pattern(region, EDGE, 1)
        straight_edges = count_pattern(region, STRAIGHT_EDGE, 2)
        return np.array([area * fence, area * (fence - straight_edges)])

    yield from sum(map(plant_costs, np.unique(garden)))


if __name__ == "__main__":
    run_aoc(aoc12, transform=np_raw_table)
