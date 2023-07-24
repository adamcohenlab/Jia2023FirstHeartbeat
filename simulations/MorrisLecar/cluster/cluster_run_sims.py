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
parser.add_argument("--from_state", type=bool, default=False)
parser.add_argument("--n_repeats", type=int, default=1)
parser.add_argument("--burn_in", type=float, default=1000)
parser.add_argument("--min_sigma", type=float, default=-np.inf)
parser.add_argument("--max_sigma", type=float, default=np.inf)
parser.add_argument("--min_I", type=float, default=-np.inf)
parser.add_argument("--max_I", type=float, default=np.inf)
args = parser.parse_args()
print(args.from_state)
bifurcation = args.bifurcation
rootpath = args.rootpath

os.makedirs(args.rootpath, exist_ok=True)

n_steps = 3
t_end = 50000
dt = 0.002

    
if args.from_state:
    for f in os.listdir(args.rootpath):
        params = parse("%s_sigma_{sigma:f}_I_{I:f}.npz" % bifurcation, f)
        if params is not None:
            if params["sigma"] >= args.min_sigma and params["sigma"] <= args.max_sigma \
            and params["I"] >= args.min_I and params["I"] <= args.max_I:
                state_path = os.path.join(args.rootpath, f)
                cmd = ["sbatch", str(SPIKECOUNTER_PATH/"simulations/MorrisLecar/cluster/sim_from_state.sh"), bifurcation,\
                      str(t_end), args.rootpath, str(args.n_repeats),\
                       state_path]
                print(cmd)
                subprocess.run(cmd)
                time.sleep(0.5)
else:
    total_blocks = 12
    n_jobs=3
    
    # total_blocks = 24
    # n_jobs=6
    

    sigmas = np.logspace(np.log10(args.min_sigma), np.log10(args.max_sigma), num=20)
    Is = np.linspace(args.min_I, args.max_I, num=n_jobs+1)
    for sigma in sigmas:
        for i in range(len(Is)-1):
            I_min = Is[i]
            I_max = Is[i+1]
            cmd = ["sbatch", str(SPIKECOUNTER_PATH/"simulations/MorrisLecar/cluster/batch_sim.sh"),
                    rootpath, bifurcation, str(I_min),
                   str(I_max), str(sigma), str(int(total_blocks/n_jobs)),
                   str(t_end), str(dt), str(args.n_repeats), "None", str(args.burn_in)]
            print(cmd)
            subprocess.run(cmd)
            time.sleep(0.5)
            
