#!/usr/bin/env python3

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import correlate
from scipy.sparse import dok_array
from scipy.sparse.csgraph import dijkstra

from aoc_util import np_raw_table, run_aoc

CHEAT1 = np.array([[1, -1, 1]])

C2 = 20
C2x2p1 = C2 * 2 + 1
_C2_tri = np.tri(C2 + 1)
_C2_tri_nz = _C2_tri.nonzero()
_C2_tri[:] = np.inf
_C2_tri[_C2_tri_nz] = C2 - abs(_C2_tri_nz[0] - _C2_tri_nz[1])
_C2_upper = np.hstack((_C2_tri[:, -1:0:-1], _C2_tri))
# cost of a max 20 move from center, 0,1,2,... around it, np.inf out of range
C2_COST = np.vstack((_C2_upper, _C2_upper[-2::-1, :]))


def aoc20(track):
    def create_graph():
        def get_connections(path, axis):
            return np.logical_and.reduce(
                sliding_window_view(path, 2, axis=axis).reshape(-1, 2), axis=1
            ).nonzero()[0]

        horiz = get_connections(PATH[1:-1, 1:], axis=1)
        vert = get_connections(PATH[1:-1, 1:-1], axis=0)
        graph = dok_array((ROWS * COLS, ROWS * COLS), dtype=int)
        graph[horiz, horiz + 1] = 1  # move E
        graph[horiz + 1, horiz] = 1  # move W
        graph[vert + COLS, vert] = 1  # move N
        graph[vert, vert + COLS] = 1  # move S
        start = (track[1:-1, 1:-1] == ord("S")).ravel().nonzero()[0][0]
        end = (track[1:-1, 1:-1] == ord("E")).ravel().nonzero()[0][0]
        return graph, start, end

    ROWS, COLS = np.array(track.shape) - 2  # skip outer PATH
    PATH = track != ord("#")
    graph, start, end = create_graph()
    distances = dijkstra(graph, indices=end, min_only=True)

    dist1 = distances.reshape(ROWS, COLS)
    y, x = (correlate(PATH[1:-1, 1:-1], CHEAT1, mode="valid") == 2).nonzero()
    cheats1_h = np.abs(dist1[y, x] - dist1[y, x + 2]) - 2
    y, x = (correlate(PATH[1:-1, 1:-1], np.rot90(CHEAT1), mode="valid") == 2).nonzero()
    cheats1_v = np.abs(dist1[y, x] - dist1[y + 2, x]) - 2
    yield np.sum(cheats1_h >= 100) + np.sum(cheats1_v >= 100)

    dist2 = np.pad(distances.reshape((ROWS, COLS)), C2, constant_values=np.nan)
    dist2[dist2 == np.inf] = np.nan
    cheats2 = (
        sliding_window_view(dist2, (C2x2p1, C2x2p1)).reshape(-1, C2x2p1, C2x2p1).copy()
    )
    cheats2 -= cheats2[:, C2x2p1 // 2, C2x2p1 // 2][:, None, None]
    cheats2 -= C2_COST[None, :]
    yield np.sum(cheats2 >= 100)


if __name__ == "__main__":
    run_aoc(
        aoc20,
        transform=np_raw_table,
        np_printoptions=dict(linewidth=160, threshold=500, edgeitems=25),
    )
