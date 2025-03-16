#!/usr/bin/env python3
import sys

import pandas as pd

#
#  calculate the end-to-end latency
#

if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} OUTPUT_DIR")
    exit(1)

output_dir = sys.argv[1]
names = ["1KB", "4KB", "16KB", "64KB", "256KB", "1MB", "4MB"]

cols = []
for name in names:
    df_pub = pd.read_csv(f"{output_dir}/pub-{name}.csv", header=None)
    df_sub = pd.read_csv(f"{output_dir}/sub-{name}.csv", header=None)
    diff = (df_sub - df_pub) * 1e6
    cols.append(diff)

res = pd.concat(cols, axis=1)
res.columns = names

output_file = f"{output_dir}/e2e-latency.csv"
res.to_csv(output_file, index=False)
print(f"{output_file} is generated.")