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
mpns_list = [(1, 2), (1 ,4), (1, 8), (2, 1), (4, 1), (8, 1)]
queue_size = 128

def get_e2e_latency(df_pub, df_sub):
    diff_sec = df_sub[5] - df_pub[1]
    diff_nsec = df_sub[6] - df_pub[2]
    diff = diff_sec * 1e6 + diff_nsec / 1e3
    return diff

for mpns in mpns_list:
    np = mpns[0]
    ns = mpns[1]

    name = f"{np}p{ns}s"
    pub_prefix = f"{output_dir}/pub-{name}"
    sub_prefix = f"{output_dir}/sub-{name}"

    diff_list = []
    for i in range(np):
        df_pub = pd.read_csv(f"{pub_prefix}-{i + 1}.csv", header=None, index_col=0)

        # drop wram-up rounds
        df_pub = df_pub.drop(df_pub.index[:3].to_list())

        for j in range(ns):
            df_sub = pd.read_csv(f"{sub_prefix}-{j + 1}.csv", header=None, index_col=0)

            # end-to-end latency
            diff_list.append(get_e2e_latency(df_pub, df_sub).dropna().reset_index(drop=True))

    res = pd.concat(diff_list, axis=1, ignore_index=True)
    output_name = f"{output_dir}/e2e-latency-{name}.csv"
    res.to_csv(output_name, header=False, index=False)
    print(f"{output_name} is generated.")