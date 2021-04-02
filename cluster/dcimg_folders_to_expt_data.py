import argparse
import os
import numpy as np
import pandas as pd
from parse import search

parser = argparse.ArgumentParser()
parser.add_argument("rootpath", type=str)
args = parser.parse_args()
rootpath = args.rootpath

entries = []

for folder in os.listdir(rootpath):
    if os.path.isdir(os.path.join(rootpath, folder)):
        print(folder)
        # res = parse("It's {}, I love it!", "It's spam, I love it!")
        res = search("{hh:2d}{mm:2d}{ss:2d}_p1", folder)
        # print(res)
        if res is not None:
            entries.append(("%02d:%02d:%02d" % (res['hh'], res['mm'], res['ss']), folder))

print(entries)
df = pd.DataFrame(entries, columns=["start_time", "file_name"])
df.to_csv(os.path.join(rootpath,"experiment_data.csv"),index=False)
