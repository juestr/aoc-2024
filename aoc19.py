#!/usr/bin/env python3

from functools import cache, partial
from aoc_util import run_aoc


def setup(data):
    lines = data.splitlines()
    return (lines[0].split(", "), lines[2:])


def aoc19(patterns, designs):
    @cache
    def check(agg, design):
        return not design or agg(
            check(agg, design[len(p) :]) for p in patterns if design.startswith(p)
        )

    yield sum(map(partial(check, any), designs))
    yield sum(map(partial(check, sum), designs))


if __name__ == "__main__":
    run_aoc(aoc19, transform=setup)
