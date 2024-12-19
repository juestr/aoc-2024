#!/usr/bin/env python3

import logging
import re
from dataclasses import dataclass, field

from funcy import chunks, map, re_find

from aoc_util import dbg, info, run_aoc

ITABLE, INAME, ILITERALOP = [], [], []


def instruction(f):
    ITABLE.append(f)
    INAME.append(f.__name__)
    return f


def instruction_litop(f):
    ILITERALOP.append(len(ITABLE))
    return instruction(f)


@dataclass
class Device:
    a: int
    b: int
    c: int
    ip: int = field(default=0, kw_only=True)
    prog: list[int]
    o: list[int] = field(default_factory=list, init=False)
    ctr: int = field(default=0, init=False)

    def halted(self):
        return self.ip >= len(self.prog)

    def step(self):
        ip = self.ip
        opcode, operand = self.prog[ip : ip + 2]
        opv = operand if opcode in ILITERALOP else self.combo_op(operand)
        dbg(f"{self.ctr:5} STEP  {INAME[opcode]} {opv}  ({ip=} {opcode=} {operand=})")
        ITABLE[opcode](self, opv)
        self.ip += 2
        self.ctr += 1
        dbg("           ", self)

    def run(self):
        while not self.halted():
            self.step()

    def outs(self, f=str):
        return ",".join(map(f, self.o))

    def disasm(self, str=bin):
        def opsym(opc, x):
            if opc in ILITERALOP:
                return str(x)
            else:
                return str(x) if x < 4 else chr(ord("A") + x - 4)

        return [
            f"{ip:4}  {INAME[opcode]} {opsym(opcode, operand)}"
            for ip, (opcode, operand) in enumerate(chunks(2, self.prog))
        ]

    def combo_op(self, operand):
        match operand:
            case 4:
                return self.a
            case 5:
                return self.b
            case 6:
                return self.c
            case x if 0 <= x <= 3:
                return x
            case _:
                assert False, "illegal operand"

    @instruction
    def adv(self, operand):
        self.a >>= operand

    @instruction_litop
    def bxl(self, operand):
        self.b ^= operand

    @instruction
    def bst(self, operand):
        self.b = operand % 8

    @instruction_litop
    def jnz(self, operand):
        if self.a:
            self.ip = operand - 2

    @instruction_litop
    def bxc(self, operand):
        self.b ^= self.c

    @instruction
    def out(self, operand):
        x = operand % 8
        self.o.append(x)

    @instruction
    def bdv(self, operand):
        self.b = self.a >> operand

    @instruction
    def cdv(self, operand):
        self.c = self.a >> operand


def setup(data):
    pattern = r"""A: (\d+).*B: (\d+).*C: (\d+).*Program: (\d+(?:,\d+)*)"""
    a, b, c, p = re_find(pattern, data, re.M | re.DOTALL)
    cmds = tuple(map(int, p.split(",")))
    return (Device(int(a), int(b), int(c), cmds),)


def aoc17(device):
    dbg(device, t="initial")
    device.run()
    yield device.outs()

    # let's analyze the program
    dbg(*device.disasm(), "\n", t="disasm", s="\n", l=logging.INFO)

    if device.prog == (2, 4, 1, 2, 7, 5, 4, 7, 1, 3, 5, 5, 0, 3, 3, 0):
        # Optimized single loop iteration for the officially given program.
        def out(a):
            a3 = a & 0b111
            return a3 ^ ((a >> (a3 ^ 0b010)) & 0b111) ^ 0b001

        # Every loop iteration shifts A by 3 bits to the right.
        # We note that the last program value thus only depends on the first 3 bits of A,
        # the second last on the first 6 bits, and so on. We can search from the back
        # and build A in chunks of 3 bits.
        def search(expect, step, a=0):
            for i in range(step == len(expect), 0b1000):  # first 3 bits must not be 000
                new_a = a | i
                if out(new_a) == expect[step]:
                    dbg(f"step {step:2}:", i, bin(new_a))
                    if step == 0:
                        yield new_a
                    else:
                        yield from search(expect, step - 1, new_a << 3)

        yield next(search(device.prog, 15))

    else:
        info("skipping part 2 for non official mission")


if __name__ == "__main__":
    run_aoc(aoc17, transform=setup)
