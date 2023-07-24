from pathlib import Path
import argparse
import subprocess
import os
import numpy as np
import pandas as pd
import time
from parse import *

parser = argparse.ArgumentParser()
parser.add_argument("rootpath", type=str)
args = parser.parse_args()

SPIKECOUNTER_PATH = Path(os.getenv("SPIKECOUNTER_PATH"))
assert SPIKECOUNTER_PATH is not None

rootpath = args.rootpath
df = pd.read_csv(os.path.join(rootpath, "all_simulation_stats.csv"))
I_mean = df["I_mean"].values[0]
I_std = df["I_std"].values[0]
I_intercept = df["I_intercept"].values[0]
# duration = df["duration"].values[0]
duration = 10000
bifurcation = df["bifurcation"].values[0]
df_by_slope = df.set_index("I_height")

for slope in df_by_slope.index.unique():
    slope_df = df_by_slope.loc[slope]
    pv = pd.pivot_table(slope_df, index=["D"], columns=["sigma"], values="phase_drift_rate")
    failed_jobs = [(pv[col][np.isnan(pv[col])].index[i], col) for col in pv.columns\
             for i in range(len(pv[col][np.isnan(pv[col])]))]
    for D, sigma in failed_jobs:
        cmd = ["sbatch", str(SPIKECOUNTER_PATH/"simulations/MorrisLecar/cluster/single_sim_pde_gradI.sh"),
               rootpath, bifurcation, str(D),
               str(I_mean), str(I_std), str(slope), str(int(I_intercept)), str(sigma), str(duration)]
        print(cmd)
        subprocess.run(cmd)
        time.sleep(0.5)
