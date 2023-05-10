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
parser.add_argument("expected_stims", type=int)
parser.add_argument("--expt_info", type=str, default="analysis/experiment_data.csv")
parser.add_argument("--output_dir", default=None)
parser.add_argument("--scale_factor", type=int, default=2)
parser.add_argument("--remove_from_start", type=int, default=0)
parser.add_argument("--remove_from_end", type=int, default=0)
parser.add_argument("--zsc_threshold", type=float, default=2)
parser.add_argument("--upper", type=int, default=0)
parser.add_argument("--fs", default=10.2, type=float)
parser.add_argument("--start_from_downsampled", default=0, type=int)
parser.add_argument("--expected_stim_width", default=3, type=int)
parser.add_argument("--fallback_mask_path", default="None")
parser.add_argument("--skewness_threshold", default=0)
parser.add_argument("--n_pcs", default=50)
parser.add_argument("--crosstalk_mask", default="None", type=str)
args = parser.parse_args()

rootpath = args.rootpath
output_dir = args.output_dir

if output_dir is None:
    output_dir = rootpath

if args.crosstalk_mask == "None":
    crosstalk_mask = "None"
else:
    crosstalk_mask = os.path.join(rootpath, args.crosstalk_mask)

    
expt_info = pd.read_csv(os.path.join(rootpath,args.expt_info), dtype=str).sort_values("start_time")
print(expt_info)

for s in ["downsampled", "stim_frames_removed", "corrected", "denoised"]:
    os.makedirs(os.path.join(output_dir, s), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "analysis", s), exist_ok=True)
    expt_info.to_csv(os.path.join(output_dir, "analysis", s, "experiment_data.csv"), index=False)

for f in expt_info["file_name"]:
    sh_line = ["sbatch", "SpikeCounter/cluster/preprocess_widefield_stim.sh", rootpath, f, str(args.expected_stims), output_dir, str(args.remove_from_start),\
              str(args.remove_from_end), str(args.scale_factor),\
              str(args.zsc_threshold), str(args.upper), str(args.fs),\
              str(args.start_from_downsampled), str(args.expected_stim_width),\
               args.fallback_mask_path, str(args.n_pcs), str(args.skewness_threshold), crosstalk_mask]
    print(sh_line)
    subprocess.run(sh_line)
