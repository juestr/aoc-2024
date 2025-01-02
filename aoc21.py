#!/usr/bin/env python3

from functools import cache
from itertools import starmap
from operator import itemgetter

from funcy import autocurry, map, pairwise, some

from aoc_util import run_aoc


def aoc21(codes):
    @cache
    def best_moves(keypad_ref, a, b):
        # only consider moves with at most 1 direction change
        p1, p2, avoid = itemgetter(a, b, "⛔")(KEYPADS[keypad_ref])
        d = p2 - p1
        mv_v = "^v"[d.real > 0] * int(abs(d.real))
        mv_h = "<>"[d.imag > 0] * int(abs(d.imag))
        if p1.imag == avoid.imag and p2.real == avoid.real:
            return (mv_h + mv_v + "A",)
        elif p2.imag == avoid.imag and p1.real == avoid.real:
            return (mv_v + mv_h + "A",)
        elif not mv_v or not mv_h:
            return (mv_v + mv_h + "A",)
        else:
            return (mv_v + mv_h + "A", mv_h + mv_v + "A")

    @autocurry
    @cache
    def button_presses(keypad_ref, level, current, button):
        # For robot at #level (0 is last),
        # his arm over button "current" on keypad,
        # having to press "button" on keypad,
        # return number of button presses required at the last robot's keypad.
        moves = best_moves(keypad_ref, current, button)
        if level == 0:
            return len(some(moves))
        else:
            return min(map(button_presses_seq(ROBOT_KEYPAD, level - 1, "A"), moves))

    @autocurry
    def button_presses_seq(keypad_ref, level, current, buttons):
        return sum(
            starmap(button_presses(keypad_ref, level), pairwise(current + buttons))
        )

    @autocurry
    def complexity(robots, code):
        return int(code[:-1]) * button_presses_seq(
            NUMERIC_KEYPAD, robots - 1, "A", code
        )

    KEYPADS = tuple(
        {b: i // 3 + i % 3 * 1j for i, b in enumerate(buttons)}
        for buttons in ("789456123⛔0A", "⛔^A<v>")
    )
    NUMERIC_KEYPAD, ROBOT_KEYPAD = 0, 1

    yield sum(map(complexity(3), codes))
    yield sum(map(complexity(26), codes))


if __name__ == "__main__":
    run_aoc(aoc21, split="lines")
