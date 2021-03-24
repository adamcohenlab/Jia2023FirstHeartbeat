import argparse
import os
import numpy as np
import pandas as pd
from parse import parse

parser = argparse.ArgumentParser()
parser.add_argument("rootpath", type=str)
args = parser.parse_args()
rootpath = args.rootpath

entries = []

for folder in os.listdir(rootpath):
    if os.path.isdir(os.path.join(rootpath, folder)):
        res = parse("{hh:2d}{mm:2d}{ss:2d}_p1_t1", folder)
        entries.append(("%2d:%2d:%2d" % (res['hh'], res['mm'], res['ss']), folder))

df = pd.DataFrame(entries, columns=["start_time", "file_name"])
df.to_csv(os.path.join(rootpath,"experiment_data.csv"))