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
parser.add_argument("I_std", type=float)
parser.add_argument("--n_repeats", type=int, default=1)
parser.add_argument("--min_sigma", type=float, default=-np.inf)
parser.add_argument("--max_sigma", type=float, default=np.inf)
args = parser.parse_args()
bifurcation = args.bifurcation
rootpath = args.rootpath

total_blocks = 21
n_jobs = 3

# sigma = 0.03
# sigma = 0.3
sigma = 3
D = 0.5
n_steps = 3
duration = 10000
heights= np.logspace(-3, 0, num=n_jobs)
intercept = 12
Is = np.linspace(args.I_mean_min, args.I_mean_max, num=36)

for I in Is:
    for i in list(range(len(heights)-1)):
            height_min = heights[i]
            height_max = heights[i+1]
            cmd = ["sbatch", str(SPIKECOUNTER_PATH/"simulations/MorrisLecar/batch_sim_pde_gradI_vary_height.sh"), rootpath, bifurcation, str(D),\
                   str(I), str(args.I_std), str(height_min), str(height_max), str(intercept), str(sigma), \
                   str(duration), str(int(total_blocks/n_jobs))]
            print(cmd)
            subprocess.run(cmd)
            time.sleep(0.5)