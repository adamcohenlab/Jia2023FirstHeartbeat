from pathlib import Path
import argparse
import subprocess
import os
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("rootpath", type=str)
parser.add_argument("--output_dir", default="analysis", type=str)
args = parser.parse_args()

SPIKECOUNTER_PATH = Path(os.getenv("SPIKECOUNTER_PATH"))
assert SPIKECOUNTER_PATH is not None

rootpath = args.rootpath
output_dir = args.output_dir
expt_info = pd.read_csv(Path(rootpath,"analysis","experiment_data.csv")).sort_values("start_time").reset_index()


for f in expt_info["file_name"]:
    if os.path.exists(Path(rootpath, "analysis/automasks/%s_mask.tif" % f)):
        sh_line = ["sbatch", str(SPIKECOUNTER_PATH/"cluster/clicky_calcium.sh"),
                   os.path.join(rootpath, "%s.tif" % f),
                   os.path.join(rootpath, "analysis/automasks/%s_mask.tif" % f),
                   os.path.join(rootpath, output_dir)]
        print(sh_line)
        subprocess.run(sh_line)
