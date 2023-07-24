from pathlib import Path
import os
import argparse
import subprocess
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("rootpath", type=str)
args = parser.parse_args()

SPIKECOUNTER_PATH = Path(os.getenv("SPIKECOUNTER_PATH"))
assert SPIKECOUNTER_PATH is not None

rootpath = args.rootpath

n_steps = 4
t_end = 50000
dt_set = 0.02

# rs = [0.6, 0.66666667, 0.73333333, 0.8, 0.86666667, 0.93333333]
sigmas = np.logspace(-2, 0, num=15)

rs = np.linspace(0, 2, num=26)
# sigmas = np.linspace(0.2, 0.8, num=15)

for sigma in sigmas:
    for i in range(len(rs)-1):
        r_min = rs[i]
        r_max = rs[i+1]
        cmd = ["sbatch", SPIKECOUNTER_PATH/"simulations/QIF/cluster/cluster_run_LIF_white.sh",
               str(sigma), str(r_min), str(r_max), str(n_steps), str(t_end),
               str(dt_set), str(1), rootpath]
        print(cmd)
        subprocess.run(cmd)
