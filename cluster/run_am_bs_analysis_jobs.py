from pathlib import Path
import argparse
import os
import pandas as pd
import subprocess


SPIKECOUNTER_PATH = Path(os.getenv("SPIKECOUNTER_PATH"))
assert SPIKECOUNTER_PATH is not None

parser = argparse.ArgumentParser()
parser.add_argument("rootdir", help="Root directory")
parser.add_argument("--subfolder", type=str, default="None")

args = parser.parse_args()
    
expt_info = pd.read_csv(os.path.join(args.rootdir,"analysis", args.subfolder, "experiment_data.csv")).sort_values("start_time")

for f in expt_info["file_name"]:
    sh_line = ["sbatch", str(SPIKECOUNTER_PATH/"cluster/activation_map_bootstrap.sh"), args.rootdir, f, \
                args.subfolder]
    print(sh_line)
    subprocess.run(sh_line)