from pathlib import Path
import argparse
import subprocess
import os
import numpy as np
import time
from parse import *

SPIKECOUNTER_PATH = Path(os.getenv("SPIKECOUNTER_PATH"))
assert SPIKECOUNTER_PATH is not None

parser = argparse.ArgumentParser()
parser.add_argument("rootpath", type=str)
parser.add_argument("bifurcation", type=str)
parser.add_argument("I_mean_min", type=float)
parser.add_argument("I_mean_max", type=float)
parser.add_argument("I_std_min", type=float)
parser.add_argument("I_std_max", type=float)
parser.add_argument("--n_repeats", type=int, default=1)
parser.add_argument("--sigma", type=float, default=1)
parser.add_argument("--D", type=float, default=0.5)
args = parser.parse_args()
bifurcation = args.bifurcation
rootpath = args.rootpath


I_means = np.linspace(args.I_mean_min, args.I_mean_max, num=15)
I_stds = np.logspace(np.log10(args.I_std_min), np.log10(args.I_std_max), num=31)
n_steps = 3
duration = 50000
for I_std in I_stds:
    for i in range(len(I_means)-1):
        I_mean_min = I_means[i]
        I_mean_max = I_means[i+1]
        cmd = ["sbatch", str(SPIKECOUNTER_PATH/"simulations/MorrisLecar/cluster/batch_sim_pde_I.sh"),
               rootpath, bifurcation, str(args.D),
               str(I_mean_min), str(I_mean_max), str(I_std), str(args.sigma),
               str(duration), str(n_steps)]
        print(cmd)
        subprocess.run(cmd)
        time.sleep(0.5)

