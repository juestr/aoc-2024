#!/usr/bin/env python3

import pandas as pd

from aoc_util import read_pd_table, run_aoc


def aoc01(df):
    xs1 = df[0].sort_values(ignore_index=1)
    xs2 = df[1].sort_values(ignore_index=1)
    yield (xs1 - xs2).abs().sum()

    c = xs2.value_counts()
    yield (xs1 * xs1.map(lambda x: c.get(x, default=0))).sum()


if __name__ == "__main__":
    run_aoc(aoc01, read=read_pd_table)
