import argparse
import skimage.io as skio
import os
import pandas as pd
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument("rootdir", help="Root directory")
parser.add_argument("--subfolder", type=str, default="None")
parser.add_argument("--s", type=str, default = "0.05")
parser.add_argument("--n_knots", type=str, default="15")
parser.add_argument("--sta_before_s", type=str, default="1")
parser.add_argument("--sta_after_s", type=str, default="1")
parser.add_argument("--normalize_height", type=str, default="1")
parser.add_argument("--bootstrap_n", type=str, default="0")
parser.add_argument("--stim_channel", type=str, default="None")

args = parser.parse_args()
    
expt_info = pd.read_csv(os.path.join(args.rootdir,"analysis", args.subfolder, "experiment_data.csv")).sort_values("start_time")

for f in expt_info["file_name"]:
    sh_line = ["sbatch", "/n/holyscratch01/cohen_lab/bjia/SpikeCounter/cluster/sta_snapt.sh", args.rootdir, f, \
                args.subfolder, args.s, args.n_knots, \
               args.sta_before_s, args.sta_after_s, args.normalize_height, args.bootstrap_n, args.stim_channel]
    print(sh_line)
    subprocess.run(sh_line)
    time.sleep(0.5)