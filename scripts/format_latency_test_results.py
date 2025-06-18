#!/usr/bin/env python3
import sys

import pandas as pd

#
#  calculate the end-to-end latency and the latency of each part
#

if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} OUTPUT_DIR")
    exit(1)

output_dir = sys.argv[1]
names = ["4KB", "16KB", "64KB", "256KB", "1MB", "4MB"]

def get_e2e_latency(df_pub, df_sub):
    diff_sec = df_sub[5] - df_pub[1]
    diff_nsec = df_sub[6] - df_pub[2]
    diff = diff_sec * 1e6 + diff_nsec / 1e3
    return diff

def dump_latency_of_each_part(df, output_prefix):
    cols = []
    for i in range(2, df.columns.size, 2):
        prev = i - 2
        diff_sec = df.iloc[:, i] - df.iloc[:, prev]
        diff_nsec = df.iloc[:, i + 1] - df.iloc[:, prev + 1]
        diff = diff_sec * 1e6 + diff_nsec / 1e3
        cols.append(diff)
    res = pd.concat(cols, axis=1)
    output_name = f"{output_prefix}-breakdown.csv"
    res.to_csv(output_name, header=False)
    print(f"{output_name} is generated.")

cols = []
for name in names:
    pub_prefix = f"{output_dir}/pub-{name}"
    sub_prefix = f"{output_dir}/sub-{name}"

    df_pub = pd.read_csv(f"{pub_prefix}-1.csv", header=None, index_col=0)
    df_sub = pd.read_csv(f"{sub_prefix}-1.csv", header=None, index_col=0)

    # end-to-end latency
    cols.append(get_e2e_latency(df_pub, df_sub))

    # latency of each part
    dump_latency_of_each_part(df_pub.drop(columns=[1, 2]), pub_prefix)
    dump_latency_of_each_part(df_sub.drop(columns=[5, 6]), sub_prefix)

res = pd.concat(cols, axis=1)
res.columns = names

output_name = f"{output_dir}/e2e-latency.csv"
res.to_csv(output_name, index=False)
print(f"{output_name} is generated.")