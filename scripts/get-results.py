#!/usr/bin/env python3
from os import path

import pandas as pd

SCRIPT_DIR = path.dirname(path.realpath(__file__))
PROJECT_DIR = path.dirname(SCRIPT_DIR)
OUTPUT_1_DIR = path.join(PROJECT_DIR, 'outputs/1p1s')
# OUTPUT_2_DIR = path.join(PROJECT_DIR, 'outputs/mpns')

names = ['1KB', '4KB', '16KB', '64KB', '256KB', '1MB', '4MB', '16MB']
table = {}

def getDiff(pub_csv, sub_csv):
    diff_nsec = sub_csv[2] - pub_csv[2]

    ltz_idx = diff_nsec < 0
    diff_nsec[ltz_idx] += 10**9
    sub_csv.loc[ltz_idx, 1] -= 1
    diff_sec = sub_csv[1] - pub_csv[1]

    return diff_sec * 10**6 + diff_nsec / 10**3

# Calculate 1p1s latencies

for name in names:
    pub_csv = pd.read_csv(f'{OUTPUT_1_DIR}/pub-{name}-1.csv', header=None)
    sub_csv = pd.read_csv(f'{OUTPUT_1_DIR}/sub-{name}-1.csv', header=None)
    table[name] = getDiff(pub_csv, sub_csv)

pd.DataFrame(table).to_csv(f'{PROJECT_DIR}/results.csv', index=False)
print(f'results.csv is generated at {PROJECT_DIR}')

# Calculate mpns latencies

# mpns_list = [(2, 1), (4, 1), (8, 1), (1, 2), (1, 4), (1, 8), (4, 4)]
# table = {}
#
# for mpns in mpns_list:
#     name = f'{mpns[0]}p{mpns[1]}s'
#
#     pub_csv_list = []
#     for i in range(mpns[0]):
#         pub_csv_list.append(pd.read_csv(f'{OUTPUT_2_DIR}/pub-{name}-{i + 1}.csv', header=None))
#     pub_csv = pd.concat(pub_csv_list)
#
#     diff_list = []
#     for i in range(mpns[1]):
#         sub_csv = pd.read_csv(f'{OUTPUT_2_DIR}/sub-{name}-{i + 1}.csv', header=None)
#         sub_csv.sort_values(by=0)
#         diff_list.append(getDiff(pub_csv, sub_csv))
#     table[name] = pd.concat(diff_list)
#
# pd.DataFrame(table).to_csv(f'{PROJECT_DIR}/results-mpns.csv', index=False)
# print(f'results-mpns.csv is generated at {PROJECT_DIR}')
