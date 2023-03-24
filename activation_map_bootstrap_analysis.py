import argparse
import skimage.io as skio
import os
from spikecounter.analysis import images, traces
from spikecounter.ui import visualize
from spikecounter import utils
import numpy as np
import itertools
import pickle
import mat73
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("rootdir", help="Root directory")
parser.add_argument("exptname", help="Experiment Name")
parser.add_argument("--subfolder", default="None", type=str)
parser.add_argument("--um_per_px", default=1.06, type=float)


args = parser.parse_args()

rootdir = args.rootdir
exptname = args.exptname
subfolder = args.subfolder
um_per_px = args.um_per_px

datadir = os.path.join(rootdir, "analysis", subfolder, exptname, "bootstrap")
os.makedirs(os.path.join(datadir, "QA_plots"), exist_ok=True)

with open(os.path.join(datadir, "all_combinations.pickle"), "rb") as f:
    all_combos = pickle.load(f)
    
matdata = mat73.loadmat(os.path.join(rootdir, exptname, "output_data_py.mat"))["dd_compat_py"]
ddict, t = utils.traces_to_dict(matdata)
dt = np.mean(np.diff(t))

erred_files = {}

data = []

for i, _ in enumerate(all_combos):
    filename = "%s_snapt_bs%d" % (exptname, i)
    try:
        beta = np.load(os.path.join(datadir, "%s.npz" % filename))["beta"]
        amplitude = (np.abs(beta[2] - 1)/beta[5])**2 #SNR
        db = 20*np.log10(amplitude)
        db[db < 0] = np.nan
        q = images.analyze_wave_dynamics(beta, dt, um_per_px, deltax=13)
        if q is None:
            continue
        else:
            rd, Tsmoothed, Tsmoothed_dv, divergence, v = q
        rowdata = (rootdir, exptname) + rd
        data.append(rowdata)
#             print(rowdata)
    except Exception as e:
        erred_files[filename] = e
        continue
            
    fig1, axes = visualize.plot_wave_analysis(db, rd, Tsmoothed, Tsmoothed_dv,\
                   divergence, v, filename)
    plt.savefig(os.path.join(datadir, "QA_plots", "waveplots_%s.tif" % filename))

with open(os.path.join(datadir, "QA_plots", "erred_files.pickle"), "wb") as f:
    pickle.dump(erred_files, f)

data = pd.DataFrame(data, columns = ['rootdir', 'file_name', 'mean_speed', 'median_speed', 'loi_x',\
                                                                  'loi_y', 'loi_x_dv', 'loi_y_dv'])
data.to_csv(os.path.join(datadir, "wavemap_data.csv"), index=False)