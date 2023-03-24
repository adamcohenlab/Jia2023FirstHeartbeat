import os
import argparse
from parse import *

parser = argparse.ArgumentParser()
parser.add_argument("rootdir", type=str)
parser.add_argument("startidx", type=int)
parser.add_argument("endidx", type=int)

args = parser.parse_args()


found_indices = set([])

for f in os.listdir(args.rootdir):
    if ".tif" in f or ".csv" in f: 
        res = search("t{:d}_", f)
        if res is not None:
            found_indices.add(res[0])

missing_indices = []
for idx in range(args.startidx, args.endidx+1):
    if idx not in found_indices:
        missing_indices.append(idx)
print(missing_indices)
print(len(missing_indices))
