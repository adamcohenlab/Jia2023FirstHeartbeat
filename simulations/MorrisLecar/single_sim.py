import sys
from pathlib import Path
import os
SPIKECOUNTER_PATH = os.getenv("SPIKECOUNTER_PATH")
sys.path.append(SPIKECOUNTER_PATH/"simulations/MorrisLecar")

import numpy as np
from scipy import signal
import argparse
from detools import detools
import time

parser = argparse.ArgumentParser()
parser.add_argument("bifurcation", type=str)
parser.add_argument("I", type=float)
parser.add_argument("sigma", type=float)
parser.add_argument("t_end", type=float)
parser.add_argument("dt", type=float)
parser.add_argument("output_path", type=str)
parser.add_argument("--state_path", type=str, default="None")
parser.add_argument("--n_repeats", type=int, default=1)
parser.add_argument("--burn_in", type=float, default=1000)
args = parser.parse_args()

os.makedirs(args.output_path, exist_ok=True)

bif = args.bifurcation

if bif not in ["snic", "saddle_node", "supercritical_hopf", \
                       "subcritical_hopf"]:
    raise Exception("Invalid bifurcation type")
params = {}
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
    "tau": 0.159,
    "C": 1,
    "D":0.1,
    "I": None,
    "V_init": -62,
    "n_init": 0.01
}
params["supercritical_hopf"] = {
    "E_rev": np.array([-78, 60, -90]),
    "n_h": -45,
    "g": np.array([8, 22, 10]),
    "m_h": -23.4,
    "k_m": 12.826,
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

def onecell_euler(de, ts, dt, y0):
    t = np.arange(ts[0], ts[1], step=dt)
    y = np.zeros((len(t), len(y0)))
    y[0,:] = y0
#     print(y.shape)
    for i in range(1,len(t)):
#         print(de(y[i-1,0], y[i-1,1], t[i]))
        y[i,:] = y[i-1,:] + np.array(de(y[i-1,0], y[i-1,1], t[i]))*dt
    return t, y

def correct_phase(phase):
    all_peaks = []
    corrected_phase = np.copy(phase).reshape(len(phase), -1)
    for j in range(corrected_phase.shape[1]):
        pks, _ = signal.find_peaks(-np.diff(corrected_phase[:,j]), height=5)
        for idx, pk in enumerate(pks):
            corrected_phase[(pk+1):,j] += np.pi*2
        all_peaks.append(pks)
    corrected_phase = corrected_phase.reshape(phase.shape)
    return corrected_phase, all_peaks

if args.state_path != "None":
    state_data = np.load(args.state_path)
    I = state_data["I"]
    sigma = state_data["sigma"]
    dt = state_data["dt"]
    y0 = state_data["y_end"]
    isi_mu = state_data["isi_mu"]
    isi_std = state_data["isi_std"]
    n_peaks = state_data["n_peaks"]
    last_peak = state_data["last_peak"]
    t_total = state_data["t_total"]
    amplitude_distribution = state_data["amplitude_distribution"]
else:
    I = args.I
    sigma = args.sigma
    dt = args.dt
    isi_mu = 0
    isi_std = 0
    t_total = 0
    n_peaks = 0
    last_peak = -1
    amplitude_distribution = np.zeros(100)
    y0 = np.array([params[bif]["V_init"], params[bif]["n_init"]]) + np.random.randn(2)*np.array([5,0.01])

params_ode = params[bif].copy()
params_ode["I"] = I
del params_ode["V_init"]
del params_ode["n_init"]
dE = detools.gen_ode(**params_ode, f = lambda x,t: np.random.randn()*sigma)

if args.burn_in > 0 and args.state_path == "None":
    t, y = onecell_euler(dE, (0, args.burn_in), dt, y0)
    y0 = y[-1,:]

theta, crit = detools.find_max_dVdt(params_ode, bif)

for i in range(args.n_repeats):
    t1 = time.perf_counter()
    t, y = onecell_euler(dE, (0, args.t_end), dt, y0)
    
    rel_y = y - crit
    rel_y[:,1] *= 100
    sos = signal.butter(7, 2, fs=1/dt, output="sos")
    mean_rel_y = np.mean(rel_y, axis=0)
    filtered_rel_y = signal.sosfiltfilt(sos, rel_y-mean_rel_y, axis=0) + mean_rel_y
    angles = np.arctan2(filtered_rel_y[:,1], filtered_rel_y[:,0])
    angles[angles < 0] = np.pi*2 + angles[angles<0]
    radii = np.sum((rel_y)**2, axis=1)**0.5
    sgns = np.sign(angles-theta)
    locs = np.argwhere((np.diff(sgns, prepend=sgns[0]) > 0) & (radii > 20)).ravel()
    # tan[tan < 0] = np.pi*2 + tan[tan < 0]
    # phase = phase_func(tan)[:, None]
    
#     _, locs = correct_phase(phase)

    all_peaks = t[locs] + i*args.t_end + t_total
    
    if last_peak == -1:
        isi = np.diff(all_peaks)
    else:
        isi = np.diff(np.insert(all_peaks, 0, last_peak))
    if len(isi) > 0:
        isi_mu = (isi_mu*max(n_peaks-1,0) + np.sum(isi))/(max(n_peaks-1, 0) + len(isi))
        isi_std = ((isi_std**2*max(n_peaks-1,0) + np.sum((isi-isi_mu)**2))/(max(n_peaks-1,0) + len(isi)))**0.5
    n_peaks = n_peaks + len(all_peaks)
    if len(all_peaks)>0:
        last_peak = all_peaks[-1]
    y0 = y[-1,:]
    print("%d seconds elapsed on repeat %d" % (time.perf_counter()-t1, i))

t_total = args.t_end*args.n_repeats + t_total
y_end = y[-1,:]
f = n_peaks/t_total
# print(f)
# print(isi_std)
# print(isi_mu)
np.savez(os.path.join(args.output_path, "%s_sigma_%f_I_%f.npz" % (args.bifurcation, sigma, I)), sigma=sigma, I=I, isi_mu=isi_mu, isi_std=isi_std,\
        f=f,y_end=y_end, dt=dt, n_peaks=n_peaks, last_peak=last_peak,t_total=t_total, amplitude_distribution=amplitude_distribution, all_isis=isi)