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
t_end = 50000

for f in os.listdir(rootpath):
    if ".mat" in f:
        cmd = ["sbatch", str(SPIKECOUNTER_PATH/"simulations/QIF/cluster/cluster_run_LIF_from_state.sh"),
               os.path.join(rootpath, f), str(t_end), str(30), rootpath]
        print(cmd)
        subprocess.run(cmd)
