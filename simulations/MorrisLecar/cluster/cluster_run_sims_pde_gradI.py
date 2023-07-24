from pathlib import Path
import os
import argparse
import subprocess
import numpy as np
import time
from parse import *

SPIKECOUNTER_PATH = Path(os.getenv("SPIKECOUNTER_PATH"))
assert SPIKECOUNTER_PATH is not None

parser = argparse.ArgumentParser()
parser.add_argument("rootpath", type=str)
parser.add_argument("bifurcation", type=str)
parser.add_argument("I_mean", type=float)
parser.add_argument("I_std", type=float)
parser.add_argument("--n_repeats", type=int, default=1)
parser.add_argument("--min_sigma", type=float, default=-np.inf)
parser.add_argument("--max_sigma", type=float, default=np.inf)
parser.add_argument("--min_D", type=float, default=-np.inf)
parser.add_argument("--max_D", type=float, default=np.inf)
parser.add_argument("--min_height", type=float, default=-np.inf)
parser.add_argument("--max_height", type=float, default=np.inf)
args = parser.parse_args()
bifurcation = args.bifurcation
rootpath = args.rootpath

total_blocks = 45
n_jobs = 5


sigmas = np.logspace(np.log10(args.min_sigma), np.log10(args.max_sigma), num=n_jobs)
Ds = np.logspace(np.log10(args.min_D), np.log10(args.max_D), num=16)
# Ds = np.logspace(np.log10(args.min_D), np.log10(args.max_D), num=5)

n_steps = 3
duration = 10000
heights= np.array([0, 0.1, 0.2, 0.5, 1])/4.55 * args.I_mean
intercept = 12

for h in heights:
    for D in Ds:
        for i in list(range(len(sigmas)-1)):
            sigma_min = sigmas[i]
            sigma_max = sigmas[i+1]
            cmd = ["sbatch", str(SPIKECOUNTER_PATH/"simulations/MorrisLecar/cluster/batch_sim_pde_gradI.sh"),
                   rootpath, bifurcation, str(D),
                   str(args.I_mean), str(args.I_std*args.I_mean), str(h), str(intercept), str(sigma_min), str(sigma_max),
                   str(duration), str(int(total_blocks/n_jobs))]
            print(cmd)
            subprocess.run(cmd)
            time.sleep(0.5)

