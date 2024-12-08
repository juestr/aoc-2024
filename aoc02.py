#!/usr/bin/env python3

from itertools import combinations, pairwise

from aoc_util import run_aoc


def aoc02(reports):
    increasing = {1, 2, 3}
    decreasing = {-1, -2, -3}

    def safe(report):
        steps = set(a - b for a, b in pairwise(report))
        return steps <= increasing or steps <= decreasing

    yield sum(map(safe, reports))

    def safe2(report):
        return safe(report) or any(
            safe(dampened) for dampened in combinations(report, len(report) - 1)
        )

    yield sum(map(safe2, reports))


if __name__ == "__main__":
    run_aoc(aoc02, split="lines_fields", apply=int)
