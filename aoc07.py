#!/usr/bin/env python3

from aoc_util import run_aoc


def parse(line):
    head, tail = line.split(": ")
    return int(head), tuple(map(int, tail.split()))


def aoc07(equations):
    def test(result, xs, check):
        def inner(result, pos):
            last = xs[pos]
            if pos == 0:
                return last == result
            else:
                return check(result, last, lambda r: inner(r, pos - 1))

        return inner(result, len(xs) - 1)

    def check_add_mul(result, last, cont):
        return (last <= result and cont(result - last)) or (
            result % last == 0 and cont(result // last)
        )

    def check_add_mul_cat(result, last, cont):
        return check_add_mul(result, last, cont) or (
            (r := str(result)).endswith((s := str(last)))
            and len(r) > len(s)
            and cont(int(r[: -len(s)]))
        )

    yield sum(r for r, xs in equations if test(r, xs, check_add_mul))
    yield sum(r for r, xs in equations if test(r, xs, check_add_mul_cat))


if __name__ == "__main__":
    run_aoc(aoc07, split="lines", apply=parse)
