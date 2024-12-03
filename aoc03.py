#!/usr/bin/env python3

import re

from aoc_util import run_aoc

def aoc03(input):

    def muls(m):
        return int(m[1]) * int(m[2])

    mulsre = re.compile(r"mul\((\d{1,3}),(\d{1,3})\)")
    yield sum(map(muls, mulsre.finditer(input)))

    def muls2(m):
        nonlocal flag
        match m[1] or m[2]:
            case "do()":       flag = 1; return 0
            case "don't()":    flag = 0; return 0
            case _:            return flag * int(m[3]) * int(m[4])

    mulsre2 = re.compile(r"(do\(\))|(don\'t\(\))|mul\((\d{1,3}),(\d{1,3})\)")
    flag = 1
    yield sum(map(muls2, mulsre2.finditer(input)))


if __name__ == '__main__':
    run_aoc(aoc03)
