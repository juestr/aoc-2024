#!/usr/bin/env python3

import numpy as np
from scipy.sparse import diags_array
from scipy.sparse.csgraph import dijkstra

from aoc_util import dbg, run_aoc


def aoc18(data):
    def create_full_grid_graph():
        diags = np.ones((4, SIZE - 1), dtype="float32")
        diags[0:2, COLS - 1 :: COLS] = 0  # no wrapping around at end of rows
        return diags_array(
            diags, offsets=(-1, 1, -COLS, COLS), shape=(SIZE, SIZE), format="dok"
        )

    def add_walls(graph, sel):
        # clearing whole rows and cols in dok array is effective but very slow
        graph[WALLS[sel], :] = 0
        graph[:, WALLS[sel]] = 0
        return graph

    def shortest_path(graph):
        distances = dijkstra(graph, indices=[START], min_only=True, directed=False)
        return distances[END]

    ROWS, COLS, BYTES1 = (71, 71, 1024) if len(data) > 1000 else (7, 7, 12)
    SIZE = ROWS * COLS
    START = 0
    END = ROWS * COLS - 1
    WALLS = data[:, 0] * COLS + data[:, 1]

    graph = add_walls(create_full_grid_graph(), np.s_[:BYTES1])
    yield int(shortest_path(graph))

    low = BYTES1
    low_graph = graph
    high = len(data)
    while high - low > 1:
        mid = (low + high) // 2
        dbg(f"trying {mid} bytes ({low=} {high=})")
        graph = add_walls(low_graph.copy(), np.s_[low:mid])
        if shortest_path(graph) == np.inf:
            high = mid
        else:
            low = mid
            low_graph = graph
    yield ",".join(map(str, data[low]))


if __name__ == "__main__":
    run_aoc(aoc18, read=(np.loadtxt, dict(delimiter=",", dtype=int)))
