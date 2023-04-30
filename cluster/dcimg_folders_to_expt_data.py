import argparse
import os
import numpy as np
import pandas as pd
from parse import search

parser = argparse.ArgumentParser()
parser.add_argument("rootpath", type=str)
parser.add_argument("--is_folder", default=1, type=int)
args = parser.parse_args()
rootpath = args.rootpath
is_folder = args.is_folder != 0

entries = []

os.makedirs(os.path.join(rootpath, "analysis"), exist_ok=True)

for folder in os.listdir(rootpath):
    if os.path.isdir(os.path.join(rootpath, folder)) == is_folder:
        print(folder)
        res = search("{hh:2d}{mm:2d}{ss:2d}", folder)
        if res is not None:
            if is_folder:
                if len(os.listdir(os.path.join(rootpath, folder))) > 0:
                    entries.append(("%02d:%02d:%02d" % (res['hh'], res['mm'], res['ss']), folder))
            else:
                if os.path.splitext(folder)[1] in [".tif", ".tiff", ".jpg", ".png", ".jpeg"]:
                    entries.append(("%02d:%02d:%02d" % (res['hh'], res['mm'], res['ss']), folder))

print(entries)
df = pd.DataFrame(entries, columns=["start_time", "file_name"])
df.to_csv(os.path.join(rootpath,"analysis","experiment_data.csv"),index=False)
