import argparse
import subprocess
import os
import shutil
import re
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("rootpath", type=str)
parser.add_argument("--output_dir", default=None)
parser.add_argument("--scale_factor", type=int, default=2)
parser.add_argument("--remove_from_start", type=int, default=0)
parser.add_argument("--remove_from_end", type=int, default=0)
parser.add_argument("--zsc_threshold", type=float, default=2)
parser.add_argument("--upper", type=int, default=0)
parser.add_argument("--fs", default=10.2, type=float)
parser.add_argument("--start_from_downsampled", default=0, type=int)
args = parser.parse_args()

rootpath = args.rootpath
output_dir = args.output_dir

if output_dir is None:
    output_dir = rootpath

expt_info = pd.read_csv(os.path.join(rootpath, "analysis/experiment_data.csv")).sort_values("start_time").reset_index()
start_times = [datetime.strptime(t,"%H:%M:%S") for t in list(expt_info["start_time"])]
offsets = [s - start_times[0] for s in start_times]
offsets = [o.seconds for o in offsets]
expt_info["offset"] = offsets

expt_tags = []
for fn in expt_info["file_name"]:
    if "stim" in fn:
        expt_tags.append("stim")
    elif "recovery" in fn:
        expt_tags.append("recovery")
    elif "pacing" in fn:
        expt_tags.append("prepost")
    else:
        expt_tags.append("NA")
expt_info["tag"] = expt_tags
expt_info = expt_info.set_index("tag")
del expt_info["index"]

for f in expt_info.loc["stim"]["file_name"]:
    sh_line = ["sbatch", "SpikeCounter/cluster/preprocess.sh", os.path.join(rootpath, f), output_dir, str(args.remove_from_start),\
              str(args.remove_from_end), str(args.scale_factor),\
              str(args.zsc_threshold), str(args.upper), str(args.fs),\
              str(args.start_from_downsampled)]
    print(sh_line)
    subprocess.run(sh_line)
