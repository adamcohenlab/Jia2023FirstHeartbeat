from pathlib import Path
import argparse
import subprocess
import os
import numpy as np

SPIKECOUNTER_PATH = Path(os.getenv("SPIKECOUNTER_PATH"))
assert SPIKECOUNTER_PATH is not None

parser = argparse.ArgumentParser()
parser.add_argument("rootpath", type=str)
parser.add_argument("bif", type=str)
parser.add_argument("I_min", type=float)
parser.add_argument("I_max", type=float)
parser.add_argument("sigma", type=float)
parser.add_argument("n_steps", type=int)
parser.add_argument("t_end", type=float)
parser.add_argument("dt", type=float)
parser.add_argument("--state_path", type=str, default="None")
parser.add_argument("--n_repeats", type=int, default=1)
parser.add_argument("--burn_in", type=float, default=1000)

args = parser.parse_args()

Is = np.linspace(args.I_min, args.I_max, args.n_steps, endpoint=False)

for I in Is:
    cmd = ["python3", str(SPIKECOUNTER_PATH/"simulations/MorrisLecar/single_sim.py"),
           args.bif, str(I), str(args.sigma), str(args.t_end), str(args.dt),
           args.rootpath, "--state_path", args.state_path, "--n_repeats",
           str(args.n_repeats), "--burn_in", str(args.burn_in)]
    subprocess.run(cmd)