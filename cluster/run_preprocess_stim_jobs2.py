import argparse
import subprocess
import os
import shutil
import re
import numpy as np
import pandas as pd
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("rootpath", type=str)
parser.add_argument("crosstalk_channel", type=str)
parser.add_argument("--output_dir", default=None)
parser.add_argument("--scale_factor", type=int, default=2)
parser.add_argument("--remove_from_start", type=int, default=0)
parser.add_argument("--remove_from_end", type=int, default=0)
parser.add_argument("--start_from_downsampled", default=0, type=int)
parser.add_argument("--n_pcs", default=20, type=int)
parser.add_argument("--skewness_threshold", default=2, type=float)
parser.add_argument("--left_shoulder_freq", default=16, type=float)
parser.add_argument("--right_shoulder_freq", default=19, type=float)
args = parser.parse_args()

rootpath = args.rootpath
output_dir = args.output_dir

if output_dir is None:
    output_dir = rootpath
    
expt_info = pd.read_csv(os.path.join(rootpath,"analysis","experiment_data.csv")).sort_values("start_time").reset_index()

    
for f in expt_info["file_name"]:
    sh_line = ["sbatch", "SpikeCounter/cluster/preprocess_stim2.sh", rootpath, f, args.crosstalk_channel, output_dir, str(args.remove_from_start),\
              str(args.remove_from_end), str(args.scale_factor),\
              str(args.start_from_downsampled), str(args.n_pcs), str(args.skewness_threshold), str(args.left_shoulder_freq), str(args.right_shoulder_freq)]
    print(sh_line)
    subprocess.run(sh_line)