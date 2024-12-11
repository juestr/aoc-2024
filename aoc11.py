#!/usr/bin/env python3

from collections import Counter

from funcy import iterate, last, take

from aoc_util import run_aoc


def blink(stones):
    def replace(stone):
        if stone == 0:
            yield 1
        else:
            xstr = str(stone)
            xmid, xodd = divmod(len(xstr), 2)
            if xodd:
                yield stone * 2024
            else:
                yield int(xstr[:xmid])
                yield int(xstr[xmid:])

    newstones = Counter()
    for stone, n in stones.items():
        for newstone in replace(stone):
            newstones[newstone] += n
    return newstones


def aoc11(numbers):
    stones = Counter(numbers)
    stones_seq = iterate(blink, stones)  # includes initial
    yield last(take(26, stones_seq)).total()
    yield last(take(50, stones_seq)).total()


if __name__ == "__main__":
    run_aoc(aoc11, split="fields", apply=int)
