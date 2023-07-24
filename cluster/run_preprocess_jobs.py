from pathlib import Path
import argparse
import subprocess
import os

parser = argparse.ArgumentParser()
parser.add_argument("rootpath", type=str)
parser.add_argument("--output_dir", default=None)
parser.add_argument("--scale_factor", type=int, default=2)
parser.add_argument("--remove_from_start", type=int, default=0)
parser.add_argument("--remove_from_end", type=int, default=0)
args = parser.parse_args()


SPIKECOUNTER_PATH = Path(os.getenv("SPIKECOUNTER_PATH"))
assert SPIKECOUNTER_PATH is not None

rootpath = args.rootpath
output_dir = args.output_dir

if output_dir is None:
    output_dir = rootpath

for f in os.listdir(rootpath):
    if ".tif" in f:
        sh_line = ["sbatch", str(SPIKECOUNTER_PATH/"cluster/preprocess.sh"),
                   os.path.join(rootpath, f), output_dir, str(args.remove_from_start),
                  str(args.remove_from_end), str(args.scale_factor)]
        print(sh_line)
        subprocess.run(sh_line)
