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
output_dir_1p1s = f"{output_dir}/1p1s"
output_dir_mpns = f"{output_dir}/mpns"

names = ["1KB", "4KB", "16KB", "64KB", "256KB", "1MB", "4MB"]
table = {}

def get_e2e_latency(df_pub, df_sub):
    diff_sec = df_sub[5] - df_pub[1]
    diff_nsec = df_sub[6] - df_pub[2]
    diff = diff_sec * 1e6 + diff_nsec / 1e3
    return diff

def dump_latency_of_each_part(df, output_prefix):
    cols = []
    for i in range(3, df.columns.size, 2):
        prev = i - 2
        diff_sec = df[i] - df[prev]
        diff_nsec = df[i + 1] - df[prev + 1]
        diff = diff_sec * 1e6 + diff_nsec / 1e3
        cols.append(diff)
    res = pd.concat(cols, axis=1)
    output_name = f"{output_prefix}-breakdown.csv"
    res.to_csv(output_name, header=False)
    print(f"{output_name} is generated.")

# Calculate 1p1s latencies

cols_e2e = []
for name in names:
    pub_prefix = f"{output_dir_1p1s}/pub-{name}"
    sub_prefix = f"{output_dir_1p1s}/sub-{name}"

    df_pub = pd.read_csv(f"{pub_prefix}-1.csv", header=None, index_col=0)
    df_sub = pd.read_csv(f"{sub_prefix}-1.csv", header=None, index_col=0)

    # end-to-end latency
    cols_e2e.append(get_e2e_latency(df_pub, df_sub))

    # latency of each part
    dump_latency_of_each_part(df_pub, pub_prefix)
    dump_latency_of_each_part(df_sub, sub_prefix)

res = pd.concat(cols_e2e, axis=1)
res.columns = names

output_name = f"{output_dir_1p1s}/e2e-latency.csv"
res.to_csv(output_name, index=False)
print(f"{output_name} is generated.")

# Calculate mpns latencies

mpns_list = [(1, 2), (1, 4), (2, 1), (4, 1), (2, 2)]
table = {}

for mpns in mpns_list:
    name = f"{mpns[0]}p{mpns[1]}s"

    df_pub_list = []
    for i in range(mpns[0]):
        pub_prefix = f"{output_dir_mpns}/pub-{name}-{i + 1}"
        df_pub = pd.read_csv(f"{pub_prefix}.csv", header=None, index_col=0)

        # latency of each part
        dump_latency_of_each_part(df_pub, pub_prefix)

        df_pub_list.append(df_pub)
    df_pub = pd.concat(df_pub_list).sort_index()

    diff_list = []
    for i in range(mpns[1]):
        sub_prefix = f"{output_dir_mpns}/sub-{name}-{i + 1}"
        df_sub = pd.read_csv(f"{sub_prefix}.csv", header=None, index_col=0)

        # latency of each part
        dump_latency_of_each_part(df_sub, sub_prefix)

        # end-to-end latency
        df_sub = df_sub.sort_index()
        diff_list.append(get_e2e_latency(df_pub, df_sub))
    table[name] = pd.concat(diff_list)

output_name = f"{output_dir_mpns}/e2e-latency-mpns.csv"
pd.DataFrame.from_dict(table, orient="index").T.to_csv(output_name, index=False)
print(f"{output_name} is generated.")
