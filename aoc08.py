#!/usr/bin/env python3

import numpy as np
from funcy import count, lmap, lmapcat

from aoc_util import dbg, np_raw_table, run_aoc


def aoc08(data):
    def get_group(name):
        mask = ANTENNA_NAMES == name
        return ANTENNA_Y[mask], ANTENNA_X[mask]

    def get_valid_indices(y, x):
        not_diag_mask = np.logical_not(np.identity(y.shape[0], dtype=bool))
        mask = np.logical_and.reduce(
            (not_diag_mask, 0 <= y, y < ROWS, 0 <= x, x < COLS)
        )
        return np.column_stack((y[mask], x[mask]))

    def find_antinodes(antennas):
        y, x = antennas
        dy = np.subtract.outer(y, y)
        dx = np.subtract.outer(x, x)
        return get_valid_indices(
            y[:, np.newaxis] + dy,
            x[:, np.newaxis] + dx,
        )

    ROWS, COLS = data.shape
    ANTENNA_Y, ANTENNA_X = (data != ord(".")).nonzero()
    ANTENNA_NAMES = data[(ANTENNA_Y, ANTENNA_X)]
    ANTENNA_GROUPS = lmap(get_group, np.unique(ANTENNA_NAMES))

    antinode_groups = lmap(find_antinodes, ANTENNA_GROUPS)
    antinodes = np.unique(np.concatenate(antinode_groups), axis=0)
    dbg(antinodes, t="antinodes")
    yield antinodes.shape[0]

    def find_antinodes2(antennas):
        y, x = antennas
        dy = np.subtract.outer(y, y)
        dx = np.subtract.outer(x, x)
        for i in count():
            antinodes = get_valid_indices(
                y[:, np.newaxis] + dy * i,
                x[:, np.newaxis] + dx * i,
            )
            if not antinodes.size:
                return
            yield antinodes

    antinode_groups2 = lmapcat(find_antinodes2, ANTENNA_GROUPS)
    antinodes2 = np.unique(np.concatenate(antinode_groups2), axis=0)
    dbg(antinodes2, t="antinodes2")
    yield antinodes2.shape[0]


if __name__ == "__main__":
    run_aoc(aoc08, transform=np_raw_table, np_printoptions=dict(threshold=5000))
