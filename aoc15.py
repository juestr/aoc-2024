#!/usr/bin/env python3
# ruff: noqa: E701

from enum import IntEnum

import numpy as np
from funcy import select

from aoc_util import np_raw_table, run_aoc

Cell = IntEnum("Cell", "FREE BOX WALL BOX_L BOX_R".split(), start=0)


def setup(data):
    part1, part2 = data.split("\n\n")
    (wh,) = np_raw_table(part1 + "\n")
    [y], [x] = (wh == ord("@")).nonzero()
    for c, t in [(".", Cell.FREE), ("@", Cell.FREE), ("O", Cell.BOX), ("#", Cell.WALL)]:
        wh[wh == ord(c)] = t
    moves = select(set("<>v^"), part2)
    return wh, moves, int(y), int(x)


def aoc15(warehouse, moves, start_y, start_x):
    def get_move_view(wh, y, x, mv, doubled=False):
        # The first return value is a *writeable* view into wh,
        # anchored at (y, x) and facing into the move direction.
        # The view is a simple line by default, but a wider field when
        # moving doubled boxes vertically.
        # fmt: off
        match mv:
            case "^": return wh[y-1::-1, slice(None) if doubled else x], y-1, x
            case "v": return wh[y+1:, slice(None) if doubled else x], y+1, x
            case ">": return wh[y, x+1:], y, x+1
            case "<": return wh[y, x-1::-1], y, x-1
        # fmt: on

    def move(wh, y, x, mv):
        line, new_y, new_x = get_move_view(wh, y, x, mv)
        first_wall = (line == Cell.WALL).nonzero()[0][0]
        match tuple((line == Cell.FREE).nonzero()[0]):
            case (i, *_) if i < first_wall:
                boxes = line[0:i].copy()
                line[0] = Cell.FREE
                line[1 : i + 1] = boxes
                return new_y, new_x
            case _:
                return y, x

    def move2(wh, y, x, mv):
        if mv in "<>":
            return move(wh, y, x, mv)
        else:
            field, new_y, new_x = get_move_view(wh, y, x, mv, doubled=True)
            move_indices = ([], [])
            depth, pressure = 0, [x]
            while pressure:
                next_pressure = []
                for pos, cell in zip(pressure, field[depth, pressure]):
                    match cell:
                        case Cell.BOX_L:
                            next_pressure.extend((pos, pos + 1))
                        case Cell.BOX_R:
                            next_pressure.extend((pos - 1, pos))
                        case Cell.WALL:
                            return y, x  # abort, move is impossible
                move_indices[0].extend([depth] * len(next_pressure))
                move_indices[1].extend(next_pressure)
                pressure = next_pressure
                depth += 1
            boxes = field[move_indices]
            field[move_indices] = Cell.FREE
            field[1:][move_indices] = boxes
            return new_y, new_x

    def double(warehouse):
        boxes = warehouse == Cell.BOX
        wh = np.empty_like(warehouse, shape=np.array(warehouse.shape) * [1, 2])
        wh[:, 0::2] = warehouse
        wh[:, 1::2] = warehouse
        wh[:, 0::2][boxes] = Cell.BOX_L
        wh[:, 1::2][boxes] = Cell.BOX_R
        return wh

    def sum_coordinates(xs):
        return np.sum([[100, 1]] @ np.array(xs.nonzero()))

    # part 1
    wh, y, x = warehouse.copy(), start_y, start_x
    for mv in moves:
        y, x = move(wh, y, x, mv)
    yield sum_coordinates(wh == Cell.BOX)

    # part 2
    wh2, y, x = double(warehouse), start_y, start_x * 2
    for mv in moves:
        y, x = move2(wh2, y, x, mv)
    yield sum_coordinates(wh2 == Cell.BOX_L)


if __name__ == "__main__":
    run_aoc(aoc15, transform=setup)
