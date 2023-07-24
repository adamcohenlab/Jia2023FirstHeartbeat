from pathlib import Path
import argparse
import subprocess
import os
import numpy as np
import pandas as pd
import time

SPIKECOUNTER_PATH = Path(os.getenv("SPIKECOUNTER_PATH"))
assert SPIKECOUNTER_PATH is not None

parser = argparse.ArgumentParser()
parser.add_argument("rootpath", type=str)
args = parser.parse_args()
rootpath = args.rootpath

t_end = 50000

results = pd.read_csv(os.path.join(rootpath, "subcritical_hopf_stringent.csv"))
results = results[(results["n_peaks"] < 500)*np.isfinite(results["isi_cv"])]
# start_idx = np.argwhere(rs > 0.65).ravel()[0]

# for i in range(1):
for i in range(results.shape[0]):
    f = results.iloc[i]["file_name"]
    folder = results.iloc[i]["folder"]
    bif = results.iloc[i]["bifurcation"]
    
    cmd = ["sbatch", str(SPIKECOUNTER_PATH/"simulations/MorrisLecar/cluster/sim_from_state.sh"), bif,
           str(t_end), os.path.join(rootpath, folder), str(10),
           os.path.join(rootpath, folder, f)]
    print(cmd)
    subprocess.run(cmd)
    time.sleep(0.5)
