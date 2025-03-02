#!/usr/bin/env python3
import sys

import pandas as pd

if len(sys.argv) < 2:
    print(f'Usage: {sys.argv[0]} FILE')
    exit(1)

filename = sys.argv[1]
cols = []

df = pd.read_csv(filename, header=None)
for i in range(3, df.columns.size, 2):
    prev = i - 2
    diff_sec = df[i] - df[prev]
    diff_nsec = df[i + 1] - df[prev + 1]
    diff = diff_sec * 1e6 + diff_nsec / 1e3
    cols.append(diff)

res = pd.concat(cols, axis=1)
res.to_csv(filename, header=False, index=False)