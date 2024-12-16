#!/usr/bin/env python3

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import sparse

from aoc_util import np_raw_table, run_aoc


def aoc16(maze):
    def create_graph(cost_move=1, cost_turn=1000):
        def node(heading, row=0, col=0):
            return heading * TURN + row * COLS + col

        def get_connections(walls, axis):
            return np.logical_and.reduce(
                sliding_window_view(walls, 2, axis=axis).reshape(-1, 2), axis=1
            ).nonzero()[0]

        ROWS, COLS = np.array(maze.shape) - 2  # skip outer walls
        TURN = ROWS * COLS
        walls = maze != ord("#")
        horiz = get_connections(walls[1:-1, 1:], axis=1)
        vert = get_connections(walls[1:-1, 1:-1], axis=0)
        graph = sparse.dok_array((TURN * 4, TURN * 4), dtype=int)
        graph[node(1, 0, horiz), node(1, 0, horiz + 1)] = cost_move  # move E
        graph[node(3, 0, horiz + 1), node(3, 0, horiz)] = cost_move  # move W
        graph[node(0, 1, vert), node(0, 0, vert)] = cost_move  # move N
        graph[node(2, 0, vert), node(2, 1, vert)] = cost_move  # move S
        (space,) = walls[1:-1, 1:-1].ravel().nonzero()
        for h in range(4):
            space_a = node(h, 0, space)
            space_b = node((h + 1) % 4, 0, space)
            graph[space_a, space_b] = cost_turn  # turn right
            graph[space_b, space_a] = cost_turn  # turn left

        start = node(1, 0, (maze[1:-1, 1:-1] == ord("S")).ravel().nonzero()[0][0])
        endN = (maze[1:-1, 1:-1] == ord("E")).ravel().nonzero()[0][0]
        return graph, start, [node(h, 0, endN) for h in range(4)]

    # Create a directed graph with 4 direction nodes per tile excl. outer walls
    graph, start, ends = create_graph()

    # First we search from the 4 possible end states backwards to the start
    distances = sparse.csgraph.dijkstra(graph.T, indices=ends, min_only=True)
    distance = int(distances[start])
    yield distance

    # Now we do a search forward from the start
    distances2 = sparse.csgraph.dijkstra(graph, indices=[start], min_only=True)
    # Nodes on the shortest paths must have the minimum distance to both ends in sum
    path_nodes = (distances + distances2) == distance
    tiles = np.sum(np.logical_or.reduce(path_nodes.reshape(4, -1), axis=0))
    yield tiles


if __name__ == "__main__":
    run_aoc(
        aoc16,
        transform=np_raw_table,
        np_printoptions=dict(linewidth=300, threshold=500000, edgeitems=10),
    )
