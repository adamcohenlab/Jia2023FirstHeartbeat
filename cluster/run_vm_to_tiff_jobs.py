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

for folder in os.listdir(rootpath):
    if os.path.isdir(os.path.join(rootpath, folder)):
        subprocess.run(["sbatch", "SpikeCounter/cluster/vm_to_tiff.sh", os.path.join(rootpath, folder), rootpath])