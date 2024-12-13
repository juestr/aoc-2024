#!/usr/bin/env python3

import re

from funcy import lmap, map, mapcat, re_iter

from aoc_util import run_aoc


def setup(data):
    pattern = r"""^Button A: X\+(\d+), Y\+(\d+)
Button B: X\+(\d+), Y\+(\d+)
Prize: X=(\d+), Y=(\d+)"""
    return ([lmap(int, m) for m in re_iter(pattern, data, re.M)],)


def aoc13(machines):
    def solve(machine):
        ax, ay, bx, by, px, py = machine
        a, a_not_int = divmod((-bx * py + by * px), (ax * by - ay * bx))
        b, b_not_int = divmod((ax * py - ay * px), (ax * by - ay * bx))
        if not (a_not_int or b_not_int):
            yield a * 3 + b

    def add_1e13(machine):
        *coeff, px, py = machine
        return [*coeff, px + int(1e13), py + int(1e13)]

    yield sum(mapcat(solve, machines))
    yield sum(mapcat(solve, map(add_1e13, machines)))


if __name__ == "__main__":
    run_aoc(aoc13, transform=setup)
