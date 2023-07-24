from pathlib import Path
import argparse
import subprocess
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("rootpath", type=str)
parser.add_argument("--lamb", type=str, default="1")
args = parser.parse_args()


SPIKECOUNTER_PATH = Path(os.getenv("SPIKECOUNTER_PATH"))
assert SPIKECOUNTER_PATH is not None

rootpath = args.rootpath

n_steps = 4
t_end = 50000
dt_set = 0.02
# rs = [0.6, 0.66666667, 0.73333333, 0.8, 0.86666667, 0.93333333]

# rs = np.linspace(-0.05, 0.2, num=31)
# rs = np.linspace(1, 1.5, num=11)
# rs = np.linspace(-10, -5, num=31)
# rs = np.linspace(-2, -0.05, num=31)
# rs = np.linspace(-0.05, 4, num=31)
# rs = np.linspace(-5, -2, num=31)

# rs = np.linspace(-0.21, -0.05, num=31)
# rs = np.linspace(-0.415, -0.21, num=31)
# rs = np.linspace(-0.620, -0.415, num=31)
# rs = np.linspace(0.2, 0.405, num=31)
# rs = np.linspace(-1, -0.620, num=31)
# rs = np.linspace(0.405, 1, num=31)
# rs = np.linspace(1, 2, num=31)
# rs = np.linspace(-2, 1, num=31)
# rs = np.linspace(2, 4, num=31)


# sigmas = np.logspace(-2, 0, num=15)
# sigmas = np.logspace(0,1, num=8)
# rs = np.linspace(-0.1, 0.2, num=61)
# rs = np.linspace(-2, -0.1, num=61)
# rs = np.linspace(-5, -2, num=31)
# rs = np.linspace(0.2, 0.405, num=31)
# rs = np.linspace(0.405, 1, num=31)
# rs = np.linspace(1, 2, num=31)
rs = np.linspace(2, 4, num=31)


sigmas = np.array([1.00000000e-02, 1.33352143e-02, 1.77827941e-02, 2.37137371e-02,
       3.16227766e-02, 4.21696503e-02, 5.62341325e-02, 7.49894209e-02,
       1.00000000e-01, 1.33352143e-01, 1.77827941e-01, 2.37137371e-01,
       3.16227766e-01, 4.21696503e-01, 5.62341325e-01, 7.49894209e-01,
       1.00000000e+00, 1.33352143e+00, 1.44444444e+00, 1.77827941e+00,
       1.88888889e+00, 2.37137371e+00, 2.77777778e+00,
       3.16227766e+00, 3.22222222e+00, 3.66666667e+00, 4.11111111e+00,
       4.21696503e+00, 4.55555556e+00, 5.00000000e+00, 5.62341325e+00,
       6.66666667e+00, 7.49894209e+00, 8.33333333e+00, 1.00000000e+01,
       1.16666667e+01, 1.33333333e+01, 1.38949549e+01, 1.50000000e+01,
       1.66666667e+01, 1.83333333e+01, 1.93069773e+01, 2.00000000e+01,
       2.68269580e+01, 2.68269580e+01, 3.72759372e+01, 5.17947468e+01,
       7.19685673e+01, 1.00000000e+02])
# sigmas = np.logspace(1, 2, num=8)
# sigmas = np.linspace(5, 20, num=10)
# sigmas = np.linspace(1,5,num=10)
# q = np.load(os.path.join("/n/holyscratch01/cohen_lab/bjia/20220926_QIF/", "big_sigmas.npz"))
# sigmas = q["sigmas"]

rs = np.linspace(0, 2, num=26)
sigmas = np.linspace(0.2, 0.8, num=15)
os.makedirs(rootpath, exist_ok=True)

for sigma in sigmas:
    for i in range(len(rs)-1):
        r_min = rs[i]
        r_max = rs[i+1]
        cmd = ["sbatch", str(SPIKECOUNTER_PATH/"cluster/QIF/cluster/cluster_run_LIF.sh"),
               str(sigma), args.lamb, str(r_min), str(r_max), str(n_steps),
               str(t_end), str(dt_set), str(1), rootpath]
        print(cmd)
        subprocess.run(cmd)
