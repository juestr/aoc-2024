#!/usr/bin/env python3

from itertools import count, pairwise

import numpy as np

from aoc_util import np_raw_table, run_aoc

NEVER = 999999999
MOVES = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=int)


def aoc06(data):
    def walk(heading, pos, start=0, extra_obstacle=None):
        visited[visited > start] = NEVER
        for step in count(start):
            visited[heading, *pos] = step
            nextpos = pos + MOVES[heading]
            if np.any(nextpos < 0) or np.any(nextpos >= data.shape):
                return "exit"
            elif visited[heading, *nextpos] < step:
                return "loop"
            elif obstacles[*nextpos] or np.all(nextpos == extra_obstacle):
                heading = (heading + 1) % 4
            else:
                pos = nextpos

    start_pos = np.transpose(np.nonzero(data == ord("^")))[0]
    obstacles = data == ord("#")
    visited = np.full((4, *data.shape), NEVER, dtype=int)
    result = walk(0, start_pos)
    assert result == "exit"
    yield np.sum(np.logical_or.reduce(visited < NEVER, axis=0))

    first_visit_steps = set(np.min(visited, axis=0).flat) - {NEVER}
    path_unsorted = (visited < NEVER).nonzero()
    sorter = np.argsort(visited[path_unsorted])
    path = np.array([idx[sorter] for idx in path_unsorted]).T
    path_pairs_backwards = list(enumerate(pairwise(path)))[::-1]
    loops = sum(
        walk(heading, [y, x], step, extra_obstacle=[y2, x2]) == "loop"
        for step, ((heading, y, x), (_, y2, x2)) in path_pairs_backwards
        if step + 1 in first_visit_steps
    )
    yield loops


if __name__ == "__main__":
    run_aoc(aoc06, transform=np_raw_table)
