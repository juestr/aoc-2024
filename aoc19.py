#!/usr/bin/env python3

from functools import cache

from aoc_util import run_aoc


def setup(data):
    lines = data.splitlines()
    return (lines[0].split(", "), lines[2:])


def aoc19(patterns, designs):
    @cache
    def validate(design):
        if not design:
            return True
        else:
            return any(
                validate(design[len(p) :]) for p in patterns if design.startswith(p)
            )

    yield sum(map(validate, designs))

    @cache
    def count(design):
        if not design:
            return 1
        else:
            return sum(
                count(design[len(p) :]) for p in patterns if design.startswith(p)
            )

    yield sum(map(count, designs))


if __name__ == "__main__":
    run_aoc(aoc19, transform=setup)
