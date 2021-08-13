import argparse
import subprocess
import os
import shutil
import re
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("rootpath", type=str)
args = parser.parse_args()
rootpath = args.rootpath

for f in os.listdir(rootpath):
    if ".tif" in f:
        sh_line = ["sbatch", "SpikeCounter/cluster/clicky_calcium.sh", os.path.join(rootpath, f), os.path.join(rootpath, "analysis/ROIs.tif"), os.path.join(rootpath, "analysis")]
        print(sh_line)
        subprocess.run(sh_line)
