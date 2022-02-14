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
    fullpath = os.path.join(rootpath, folder)
    if os.path.isdir(fullpath):
        if "frames.bin" in os.listdir(fullpath) or "Sq_camera.bin" in os.listdir(fullpath):
            sh_line = ["sbatch", "SpikeCounter/cluster/vm_to_tiff.sh", fullpath, rootpath]
            print(sh_line)
            subprocess.run(sh_line)