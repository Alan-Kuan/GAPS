#!/usr/bin/env python3
import pandas as pd

OUTPUT_DIR = "outputs/python"

names = ["1KB", "4KB", "16KB", "64KB", "256KB", "1MB", "4MB"]

table = {}

for name in names:
    with open(f"{OUTPUT_DIR}/pub-{name}") as f:
        pub_list = [float(line) * 1e6 for line in f]

    with open(f"{OUTPUT_DIR}/sub-{name}") as f:
        sub_list = [float(line) * 1e6 for line in f]

    diff_list = [s - p for s, p in zip(sub_list, pub_list)]
    table[name] = diff_list

output_file = "python_results.csv"
pd.DataFrame(table).to_csv(output_file, index=False)
print(f"{output_file} is generated")