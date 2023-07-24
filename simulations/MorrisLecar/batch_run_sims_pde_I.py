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
parser.add_argument("I_mean_min", type=float)
parser.add_argument("I_mean_max", type=float)
parser.add_argument("I_std", type=float)
parser.add_argument("sigma", type=float)
parser.add_argument("duration", type=float)
parser.add_argument("n_steps", type=int)

args = parser.parse_args()

Is = np.linspace(args.I_mean_min, args.I_mean_max, args.n_steps, endpoint=False)

for I_mean in Is:
    cmd = ["python3", str(SPIKECOUNTER_PATH/"simulations/MorrisLecar/single_pde_sim.py"),
           args.rootpath, str(args.bifurcation),
           str(args.D), str(I_mean), str(args.I_std), str(args.sigma), str(args.duration)]
    subprocess.run(cmd)
    