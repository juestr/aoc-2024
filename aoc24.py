#!/usr/bin/env python3

import re
from functools import cache
from itertools import product
from operator import and_, or_, xor
from random import randint

from funcy import re_iter

from aoc_util import info, run_aoc


def setup(data):
    GATE_OPS = {"AND": and_, "OR": or_, "XOR": xor}
    w_re = re.compile(r"^(.{3}): (\d+)", re.M)
    g_re = re.compile(r"^(.{3}) (\D{2,3}) (.{3}) -> (.{3})", re.M)
    wires = {name: int(value) for name, value in re_iter(w_re, data)}
    gates = {c: (GATE_OPS[g], a, b) for a, g, b, c in re_iter(g_re, data)}
    return wires, gates


def aoc24(wires, gates):
    def gates_evaluator(wires=wires, gates=gates):
        @cache
        def eval_wire(name):
            if name in wires:
                return wires[name]
            else:
                op, a, b = gates[name]
                return op(eval_wire(a), eval_wire(b))

        return eval_wire

    zwires = sorted((w for w in gates if w.startswith("z")), reverse=True)
    zwire_vals = map(gates_evaluator(), zwires)
    yield int("".join(map(str, zwire_vals)), base=2)

    if len(wires) < 50:
        info("skipping part 2 for example data")
        return

    def z(n):
        return f"z{n:02}"

    def swap_gates(gates, g1, g2):
        new_gates = gates.copy()
        new_gates[g1] = gates[g2]
        new_gates[g2] = gates[g1]
        return new_gates

    def eval_as_int(gates, x, y):
        xbits = f"{x:045b}"[::-1]
        ybits = f"{y:045b}"[::-1]
        wires = {
            f"{xy}{n:02}": int(v)
            for xy, bits in [("x", xbits), ("y", ybits)]
            for n, v in enumerate(bits)
        }
        eval_wire = gates_evaluator(wires, gates)
        zbits = [eval_wire(f"z{n:02}") for n in range(46)]
        return int("".join(map(str, zbits[::-1])), 2)

    def find_lowest_faulty_bit(gates, stop_on=0, repeat=1000):
        @cache
        def gates_used(name):
            if name not in gates:
                return set()
            else:
                _, a, b = gates[name]
                return {name} | gates_used(a) | gates_used(b)

        min_faulty_bit = 100
        for _ in range(repeat):
            x = randint(0, 1 << 45 - 1)
            y = randint(0, 1 << 45 - 1)
            fault = abs(x + y - eval_as_int(gates, x, y))
            if fault:
                faulty_bit = fault.bit_length() - 1
                min_faulty_bit = min(min_faulty_bit, faulty_bit)
                if min_faulty_bit <= stop_on:
                    break
        if min_faulty_bit < 100:
            good_gates = (
                gates_used(z(min_faulty_bit - 1)) if min_faulty_bit > 0 else set()
            )
            good_gates |= {z(i) for i in range(min_faulty_bit - 1)}
            faulty_gates = gates_used(z(min_faulty_bit)) - good_gates
            swapable_gates = all_gates - good_gates
            return min_faulty_bit, faulty_gates, swapable_gates
        else:
            return None, set(), set()

    def search(gates, faulty_bit, faulty_gates, swapable_gates, swaps=(), level=0):
        for g1, g2 in product(faulty_gates, swapable_gates):
            if g1 != g2:
                swapped_gates = swap_gates(gates, g1, g2)
                try:
                    faulty_bit2, faulty_gates2, swapable_gates2 = (
                        find_lowest_faulty_bit(swapped_gates, stop_on=faulty_bit)
                    )
                except RecursionError:
                    # a very cheap trick to avoid checking for illegal cycles explicitly
                    pass
                else:
                    if faulty_bit2 is None:
                        return ",".join(sorted((*swaps, g1, g2)))
                    elif faulty_bit2 > faulty_bit and level < 4:
                        result = search(
                            swapped_gates,
                            faulty_bit2,
                            faulty_gates2,
                            swapable_gates2,
                            swaps=(*swaps, g1, g2),
                            level=level + 1,
                        )
                        if result:
                            return result
        return None

    all_gates = set(gates.keys())
    faulty_bit, faulty_gates, swapable_gates = find_lowest_faulty_bit(gates)
    yield search(gates, faulty_bit, faulty_gates, swapable_gates)


if __name__ == "__main__":
    run_aoc(aoc24, transform=setup)
