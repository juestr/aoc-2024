#!/usr/bin/env python3

import networkx as nx
from funcy import first, map

from aoc_util import run_aoc


def aoc23(lines):
    G = nx.from_edgelist(line.split("-") for line in lines)
    triples_with_t = [
        c for c in nx.enumerate_all_cliques(G) if len(c) == 3 and "t" in map(first, c)
    ]
    yield len(triples_with_t)

    party = max(nx.enumerate_all_cliques(G), key=len)
    yield ",".join(sorted(party))


if __name__ == "__main__":
    run_aoc(aoc23, split="lines")
