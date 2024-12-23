#!/usr/bin/env python3

from funcy import lsplit, map, zipdict
from graphlib import TopologicalSorter

from aoc_util import run_aoc


def setup(lines):
    to_ints = lambda line, sep: tuple(int(x) for x in line.split(sep))
    sepidx = lines.index("")
    rules = {to_ints(line, "|") for line in lines[:sepidx]}
    updates = [to_ints(line, ",") for line in lines[sepidx + 1 :]]
    return rules, updates


def aoc05(rules, updates):
    def filter_rules(update):
        return filter(set(update).issuperset, rules)

    def check(update):
        page_index = zipdict(update, range(len(update)))
        return all(page_index[p1] < page_index[p2] for p1, p2 in filter_rules(update))

    def tsort(update):
        sorter = TopologicalSorter()
        for p, x in filter_rules(update):
            sorter.add(x, p)
        return list(sorter.static_order())

    def middle_page(update):
        return update[len(update) // 2]

    valid, invalid = lsplit(check, updates)
    yield sum(map(middle_page, valid))
    yield sum(map(middle_page, map(tsort, invalid)))


if __name__ == "__main__":
    run_aoc(aoc05, split="lines", transform=setup)
