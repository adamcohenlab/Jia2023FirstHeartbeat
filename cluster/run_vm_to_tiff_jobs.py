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
    if os.path.isdir(fullpath):
        if "frames.bin" in os.listdir(fullpath) or "Sq_camera.bin" in os.listdir(fullpath):
            sh_line = ["sbatch", str(SPIKECOUNTER_PATH/"cluster/vm_to_tiff.sh"), fullpath, rootpath]
            print(sh_line)
            subprocess.run(sh_line)