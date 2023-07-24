from pathlib import Path
import argparse
import subprocess
import os
import numpy as np

SPIKECOUNTER_PATH = Path(os.getenv("SPIKECOUNTER_PATH"))
assert SPIKECOUNTER_PATH is not None

parser = argparse.ArgumentParser()
parser.add_argument("rootpath", type=str)
parser.add_argument("bifurcation", type=str)
parser.add_argument("D", type=float)
parser.add_argument("I_mean", type=float)
parser.add_argument("I_std", type=float)
parser.add_argument("I_height_min", type=float)
parser.add_argument("I_height_max", type=float)
parser.add_argument("I_intercept", type=int)
parser.add_argument("sigma", type=float)
parser.add_argument("duration", type=float)
parser.add_argument("n_steps", type=int)

args = parser.parse_args()

print(args.n_steps)
print(args)


I_heights = np.logspace(np.log10(args.I_height_min), np.log10(args.I_height_max), args.n_steps, endpoint=False)

for I_height in I_heights:
    cmd = ["python3", str(SPIKECOUNTER_PATH/"simulations/MorrisLecar/single_pde_sim_gradI.py"),
           args.rootpath, str(args.bifurcation),
           str(args.D), str(args.I_mean), str(args.I_std), str(I_height), str(args.I_intercept),
           str(args.sigma), str(args.duration)]
    subprocess.run(cmd)
    