import argparse
import subprocess
import os
import shutil
import re
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("rootpath", type=str)
parser.add_argument("roi_path", type=str)
parser.add_argument("--output_dir", type=str, default="None")
args = parser.parse_args()
rootpath = args.rootpath

output_dir = args.output_dir
if output_dir == "None":
    output_dir = os.path.join(rootpath, "analysis")

for f in os.listdir(rootpath):
    if ".tif" in f:
        sh_line = ["sbatch", "SpikeCounter/cluster/clicky_calcium.sh", os.path.join(rootpath, f), args.roi_path, output_dir]
        print(sh_line)
        subprocess.run(sh_line)
