from pathlib import Path
import argparse
import subprocess
import os
import numpy as np
import time

parser = argparse.ArgumentParser()
parser.add_argument("rootpath", type=str)
args = parser.parse_args()

SPIKECOUNTER_PATH = Path(os.getenv("SPIKECOUNTER_PATH"))
assert SPIKECOUNTER_PATH is not None

rootpath = args.rootpath
t_end = 50000
results = np.load(os.path.join(rootpath, "results.npz"), allow_pickle=True)

n_peaks = results["n_peaks"]
mat_files = results["mat_files"]
rs = results["rs"]
start_idx = np.argwhere(rs > 0.65).ravel()[0]

for i in range(n_peaks.shape[0]):
    for j in range(start_idx, n_peaks.shape[1]):
        if n_peaks[i,j] < 20000:
            f = mat_files[i,j]
            cmd = ["sbatch", str(SPIKECOUNTER_PATH/"simulations/QIF/cluster/cluster_run_LIF_from_state.sh"),
                   os.path.join(rootpath, f), str(t_end), str(75), rootpath]
            print(cmd)
            subprocess.run(cmd)
            time.sleep(0.5)