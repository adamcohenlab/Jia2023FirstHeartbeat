from pathlib import Path
import argparse
import subprocess
import os

parser = argparse.ArgumentParser()
parser.add_argument("rootpath", type=str)
args = parser.parse_args()

SPIKECOUNTER_PATH = Path(os.getenv("SPIKECOUNTER_PATH"))
assert SPIKECOUNTER_PATH is not None

rootpath = args.rootpath

for folder in os.listdir(rootpath):
    fullpath = os.path.join(rootpath, folder)
    if os.path.isdir(fullpath) and "output_data_py.mat" in os.listdir(fullpath):
        sh_line = ["sbatch", str(SPIKECOUNTER_PATH/"/cluster/register_dualcam.sh"), 
                    rootpath, folder, os.path.join(rootpath, "analysis", "tform_flipud.npz")]
        print(sh_line)
        subprocess.run(sh_line)