import argparse
import subprocess
import os
import pandas as pd
import time

parser = argparse.ArgumentParser()
parser.add_argument("rootpath", type=str)
parser.add_argument("--expt_info", type=str, default="analysis/experiment_data.csv")
parser.add_argument("--um_per_px", default=0.265 * 4, type=float)
parser.add_argument("--hard_cutoff", default=0.005, type=float)
parser.add_argument("--downsample_factor", default=8, type=int)
parser.add_argument("--window_size_s", default=4, type=float)
parser.add_argument("--sta_before_s", default=2, type=float)
parser.add_argument("--sta_after_s", default=5, type=float)
parser.add_argument("--frame_start", default=0, type=int)
parser.add_argument("--frame_end", default=-0, type=int)


args = parser.parse_args()
rootpath = args.rootpath

expt_info = pd.read_csv(os.path.join(rootpath, args.expt_info), dtype=str).sort_values(
    "start_time"
)

for i in range(expt_info.shape[0]):
    f = expt_info.iloc[i]["file_name"]
    sh_line = [
        "sbatch",
        "/n/home11/bjia/SpikeCounter/cluster/simultaneous_voltage_calcium.sh",
        rootpath,
        f,
        str(args.um_per_px),
        str(args.hard_cutoff),
        str(args.downsample_factor),
        str(args.window_size_s),
        str(args.sta_before_s),
        str(args.sta_after_s),
        str(args.frame_start),
        str(args.frame_end),
    ]
    print(sh_line)
    subprocess.run(sh_line, check=True)
    time.sleep(0.5)