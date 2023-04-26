import argparse
import subprocess
import os
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("rootpath", type=str)
parser.add_argument("--expt_info", type=str, default="analysis/experiment_data.csv")
parser.add_argument("--um_per_px", default=0.265 * 4, type=float)
parser.add_argument("--hard_cutoff", default=0.005, type=float)
parser.add_argument("--downsample_factor", default=16, type=int)
parser.add_argument("--window_size", default=111, type=int)
parser.add_argument("--sta_before", default=40, type=int)
parser.add_argument("--sta_after", default=100, type=int)


args = parser.parse_args()

rootpath = args.rootpath
output_dir = args.output_dir
bg_const = args.bg_const

if output_dir is None:
    output_dir = rootpath

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
        str(args.window_size),
        str(args.sta_before),
        str(args.sta_after),
    ]
    print(sh_line)
    subprocess.run(sh_line, check=True)
