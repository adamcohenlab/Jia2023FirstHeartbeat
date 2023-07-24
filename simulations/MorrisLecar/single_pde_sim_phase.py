import sys
from pathlib import Path
import os
SPIKECOUNTER_PATH = os.getenv("SPIKECOUNTER_PATH")
sys.path.append(SPIKECOUNTER_PATH)
sys.path.append(SPIKECOUNTER_PATH/"simulations/MorrisLecar")

from spikecounter.analysis import traces
from pde import PDEBase, MemoryStorage, ScalarField, plot_kymograph, UnitGrid, FieldCollection, movie
from detools import detools
from sklearn.preprocessing import StandardScaler
from sklearn import neighbors, cluster, decomposition
import argparse
from skimage import exposure, transform, segmentation, feature
import numpy as np
import pickle
import uuid
from scipy import signal, optimize, interpolate, fft
from skimage import io as skio
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import colorcet as cc
from datetime import datetime


parser = argparse.ArgumentParser()
parser.add_argument("rootdir", type=str)
parser.add_argument("bif", type=str)
parser.add_argument("D", type=float)
parser.add_argument("I_mean", type=float)
parser.add_argument("I_std", type=float)
parser.add_argument("sigma", type=float)
parser.add_argument("duration", type=float)
parser.add_argument("--write_video", default=0, type=int)
parser.add_argument("--video_sampling_rate", default=1, type=float)
parser.add_argument("--plot_n_example_traces", default=0, type=int)
parser.add_argument("--plot_clusters", default=0, type=int)
args = parser.parse_args()

os.makedirs(args.rootdir, exist_ok=True)

def prep_visualize(array, stride=1, rescale=10, dtype=np.uint8):
    downsampled_t = array[::stride]
    if rescale > 1:
        downsampled_t = transform.rescale(downsampled_t, (1,\
            rescale, rescale), preserve_range=True, anti_aliasing=False,\
                order=0)
    return exposure.rescale_intensity(downsampled_t, out_range=(0,255)).astype(dtype)


def correct_phase(phase):
    all_peaks = []
    corrected_phase = np.copy(phase)
    for row in np.arange(phase.shape[1]):
        for col in np.arange(phase.shape[2]):
            pks, _ = signal.find_peaks(-np.diff(phase[:,row,col]), height=5)
            for idx, pk in enumerate(pks):
                corrected_phase[(pk+1):,row,col] += np.pi*2
            all_peaks.append(pks)
    return corrected_phase, all_peaks


params = {}
field_size = (10,10)
params["snic"] = {
    "E_rev": np.array([-80, 60, -90]),
    "n_h": -25,
    "g": np.array([8, 20, 10]),
    "m_h": -20,
    "k_m": 15,
    "k_n": 5,
    "tau": 1,
    "C": 1,
    "D":1,
    "I": None,
    "V_init": -61,
    "n_init": 0.01
}

params["saddle_node"] = {
    "E_rev": np.array([-80, 60, -90]),
    "n_h": -25,
    "g": np.array([8, 20, 10]),
    "m_h": -20,
    "k_m": 15,
    "k_n": 5,
    "tau": 0.14,
    "C": 1,
    "D":0.1,
    "I": None,
    "V_init": -62,
    "n_init": 0.01
}
params["supercritical_hopf"] = {
    "E_rev": np.array([-78, 60, -90]),
    "n_h": -45,
    "g": np.array([8, 20, 10]),
    "m_h": -20,
    "k_m": 15,
    "k_n": 5,
    "tau": 1,
    "C": 1,
    "D":1,
    "I": None,
    "V_init": -55,
    "n_init": 0.1
}
params["subcritical_hopf"] = {
    "E_rev": np.array([-78, 60, -90]),
    "n_h": -45,
    "g": np.array([1, 4, 4]),
    "m_h": -30,
    "k_m": 7,
    "k_n": 5,
    "tau": 1,
    "C": 1,
    "D":0.05,
    "I": None,
    "V_init": -49,
    "n_init": 0.3
}
params["subcritical_hopf2"] = {
    "E_rev": np.array([-78, 60, -90]),
    "n_h": -45,
    "g": np.array([1, 4, 4]),
    "m_h": -30,
    "k_m": 7,
    "k_n": 5,
    "tau": 1,
    "C": 1,
    "D":0.1,
    "I": None,
    "V_init": -49,
    "n_init": 0.3
}


min_bif_values = {"snic": 4.6, "saddle_node": 4.6, "subcritical_hopf": 50, "supercritical_hopf": 20}


block_size = 10000
n_blocks = int(args.duration//block_size + 1*(args.duration%block_size !=0))
print(n_blocks)
params = params[args.bif]
params["D"] = args.D
params["I"] = args.I_mean*np.ones(field_size)+ np.random.randn(*field_size)*args.I_std
g = UnitGrid(field_size)


V_init = ScalarField(g, data=np.ones(field_size)*params["V_init"] + np.random.randn(*field_size)*0)
n_init = ScalarField(g, data=np.ones(field_size)*params["n_init"] + np.random.randn(*field_size)*0.01)
del params["V_init"]
del params["n_init"]
pde = detools.INapIkPDE(params, sigma=args.sigma)

# Empirically determine the phase function of the system
params_ode = params.copy()
params_ode["I"] = np.max(params["I"])
theta, crit = detools.find_max_dVdt(params_ode, bif)

# Initialize stuff for saving the results
mean_isi = np.zeros(field_size[0]*field_size[1])
std_isi = np.zeros(field_size[0]*field_size[1])
first_spike_times = np.ones(field_size[0]*field_size[1])*np.nan
pk_times = [[] for i in range(field_size[0]*field_size[1])]

sampling_rate = 0.01

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
subfolder_name = "%s_%s_%s" % (args.bif, timestamp, str(uuid.uuid4())[:10])
subfolder_path = os.path.join(args.rootdir, subfolder_name)
os.mkdir(subfolder_path)
dt = 0.002


n_ticks_sta = int(10/sampling_rate)
length_scale = 1
time_scale = 1
fft_portraits = []
for j in range(n_blocks):
    start = time.perf_counter()
    fc = FieldCollection([V_init, n_init])
    storage = MemoryStorage()
    pde.solve(fc, t_range=min(args.duration, block_size),\
              dt=dt,tracker=storage.tracker(sampling_rate))

    data = np.array(storage.data)
    times = np.array(storage.times)
    n = data[:,1]
    V = data[:,0]
    
    rel_y = data - crit[None,:,None,None]
    rel_y[:,1] *= 100
    sos = signal.butter(7, 2, fs=1/dt, output="sos")
    mean_rel_y = np.mean(rel_y, axis=0)
    filtered_rel_y = signal.sosfiltfilt(sos, rel_y-mean_rel_y, axis=0) + mean_rel_y
    angles = np.arctan2(filtered_rel_y[:,1], filtered_rel_y[:,0])
    nr = n.reshape((n.shape[0], -1)).T
    tan = np.arctan2(n-crit[1], V-crit[0])
    tan[tan < 0] = np.pi*2 + tan[tan < 0]
    angles[angles < 0] = np.pi*2 + angles[angles<0]
    radii = np.sum((rel_y)**2, axis=1)**0.5
    locs = np.argwhere((np.diff(np.sign(angles-theta), prepend=0) > 0) & (radii > 0.1)).ravel()
    
    
    corrected_phase, pks = correct_phase(phase)
    fft_portraits.append(np.abs(fft.fftshift(fft.fft2(corrected_phase -\
                    np.median(corrected_phase, axis=(1,2))[:,None,None]), axes=(1,2))).mean(axis=0))

    for i in range(field_size[0]*field_size[1]):
        pk_times[i].extend(list(times[pks[i]]+j*block_size))
    
    if j == n_blocks-1:
        if args.plot_n_example_traces > 0:
            selected_samples = list(np.random.choice(len(pk_times), args.plot_n_example_traces, replace = False))
            fig1, ax1 = plt.subplots(figsize=(8, 0.5*len(selected_samples)))
            offset = np.max(np.percentile(nr, 90, axis=1) - np.percentile(nr, 10, axis=1))*1.5
            for idx, s in enumerate(selected_samples):
                ax1.plot(times, nr[s, :]+ idx*offset)
                ax1.plot(times[pks[s]], nr[s,pks[s]] + idx*offset, "rx")
            start_time = np.percentile(np.concatenate(pk_times), 50)
            ax1.set_xlim(start_time - 1, start_time + 20)
            plt.savefig(os.path.join(subfolder_path, "example_peak_detect.svg"))

    V_init = ScalarField(g, data=V[-1])
    n_init = ScalarField(g, data=n[-1])
    end = time.perf_counter()
    print("%.1f seconds elapsed for PDE sims" % (end-start))

for i in range(len(pk_times)):
    isi = np.diff(pk_times[i])
    mean_isi[i] = np.nanmean(isi)
    std_isi[i] = np.nanstd(isi)
    if len(pks[i]) > 0:
        first_spike_times[i] = pk_times[i][0]
    else:
        first_spike_times[i] = np.nan

fft_portrait = np.array(fft_portraits).mean(axis=0)
        
params["sigma"] = args.sigma
params["I_std"] = args.I_std
params["I_mean"] = args.I_mean
params["dt"] = dt

with open(os.path.join(subfolder_path, "params.pickle"), "wb") as pckl:
    pickle.dump({**vars(args), **params}, pckl)

np.savez(os.path.join(subfolder_path, "stats.npz"), mean_isi=mean_isi, std_isi=std_isi, first_spike_times=first_spike_times, fft_portrait=fft_portrait)

if bool(args.write_video):
    vid = prep_visualize(n, stride=int((1/args.video_sampling_rate)//sampling_rate))
    skio.imsave(os.path.join(subfolder_path, "%s_vid.tif" % args.bif), vid)
    
    phase_vid = prep_visualize(phase, stride=int((1/args.video_sampling_rate)//sampling_rate))
    skio.imsave(os.path.join(subfolder_path, "%s_phase_vid.tif" % args.bif), phase_vid)