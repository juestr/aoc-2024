#!/usr/bin/env python3

from math import sumprod

from funcy import accumulate, constantly, count, interleave, lmap, lmapcat, map, repeat

from aoc_util import dbg, run_aoc

SP = type("Space", (), {"__int__": constantly(0), "__repr__": constantly(".")})()


def aoc09(data):
    def checksum(disk):
        return sumprod(map(int, disk), range(len(disk)))

    def disk_str(disk):
        return " ".join(map(str, disk))

    spans = lmap(int, data.strip())
    disk = lmapcat(repeat, interleave(count(), repeat(SP)), spans)
    dbg(disk, apply=disk_str)

    start, end = 0, len(disk) - 1
    while ...:
        while disk[start] is not SP and start < end:
            start += 1
        while disk[end] is SP and start < end:
            end -= 1
        if start < end:
            disk[start], disk[end] = disk[end], SP
        else:
            break
    dbg(disk, apply=disk_str)
    yield checksum(disk)

    disk = lmapcat(repeat, interleave(count(), repeat(SP)), spans)
    disk_objs = lmap(list, zip(accumulate(spans, initial=0), spans))
    spaces = disk_objs[1::2]
    for f_start, f_len in reversed(disk_objs[0::2]):
        for i, (sp_start, sp_len) in enumerate(spaces):
            if sp_start >= f_start:
                break
            if sp_len >= f_len:
                disk[sp_start : sp_start + f_len] = disk[f_start : f_start + f_len]
                disk[f_start : f_start + f_len] = repeat(SP, f_len)
                if sp_len == f_len:
                    del spaces[i]
                else:
                    spaces[i][0] += f_len
                    spaces[i][1] -= f_len
                break
    dbg(disk, apply=disk_str)
    yield checksum(disk)


if __name__ == "__main__":
    run_aoc(aoc09)
