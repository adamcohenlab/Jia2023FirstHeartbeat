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
from scipy import signal, optimize, interpolate
from skimage import io as skio
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import colorcet as cc


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

def prep_visualize(array, stride=1, dtype=np.uint8):
    downsampled_t = array[::stride]
    return exposure.rescale_intensity(downsampled_t, out_range=(0,255)).astype(dtype)

def find_all_peaks(traces, peak_params={"prominence":(None, None)}):
    
    pks = [signal.find_peaks(traces[i], **peak_params)\
               for i in range(traces.shape[0])]
    pks,pk_data = zip(*pks)

    prominences = [pd["prominences"] for pd in pk_data]
    thresholds = [max(np.nanmean(prom)*0.3, 0.2) for prom in prominences]
    pks = [pk[prominences[i]>thresholds[i]] for i, pk in enumerate(pks)]
    prominences = [prom[prom>thresholds[i]] for i, prom in enumerate(prominences)]
    return pks


params = {}
field_size = (10,10)
dt = 0.002
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
params_ode = params[bif].copy()
params_ode["I"] = np.max(params[bif]["I"])
initial_guesses = {"snic": [-60, -29], "saddle_node": [-60, -29], "supercritical_hopf": [-40], "subcritical_hopf2": [-40]}
ode = detools.gen_ode(**params_ode)
nullV, nulln = detools.gen_nullclines(**params_ode)
critical_points= optimize.fsolve(lambda x: nullV(x) - nulln(x), initial_guesses[bif])
n_crit = nulln(critical_points)
if len(critical_points) >1:
    crit = (critical_points[-1], n_crit[-1])
else:
    crit = (critical_points[0], n_crit[0])

# Identify a start point for the phase plot simulation. Choose a point on the n-nullcline with maximum dV/dt - this is probably close to the upstroke of the action potential.
v_test = np.linspace(-70, - 10, 100)
n_test = nulln(v_test)
dVdt, _ = ode(v_test, n_test)
v_i = v_test[np.argmax(dVdt)]
n_i = nulln(v_i)

t_max = 50
dt = 0.001
n_steps = int(np.round(t_max/dt))
zs = np.zeros([n_steps,2])

zs[0,:] = np.array([v_i, n_i])
for i in range(1,n_steps):
    V_t, n_t = ode(zs[i-1,0], zs[i-1,1], i*dt)
    zs[i,:] = zs[i-1,:] + np.array([V_t, n_t])*dt
phase = np.arctan2((zs[:,1]-crit[1]), (zs[:,0]-crit[0]))
zero_crossings = np.argwhere((phase*np.roll(phase, 1) < 0) & \
                (np.diff(phase, prepend=[0]) > 0)).ravel()
if len(zero_crossings) < 2:
    phase_func = lambda x: None
else:
    single_rot_2pi = np.copy(test_phase[zero_crossings[-2]:zero_crossings[-1]])
    pi_crossing = np.argwhere(np.diff(single_rot_2pi) < 0).ravel()[0]
    single_rot_2pi[pi_crossing+1:] = 2*np.pi + single_rot_2pi[pi_crossing+1:]
    phase_func = interpolate.interp1d(single_rot_2pi, np.linspace(0, 1, len(single_rot_2pi), endpoint=False)*np.pi*2)

# Initialize stuff for saving the results
mean_isi = np.zeros(field_size[0]*field_size[1])
std_isi = np.zeros(field_size[0]*field_size[1])
first_spike_times = np.ones(field_size[0]*field_size[1])*np.nan
all_areas = []
pk_times = [[] for i in range(field_size[0]*field_size[1])]

sampling_rate = 0.01

subfolder_name = str(uuid.uuid4())
while os.path.exists(os.path.join(args.rootdir, subfolder_name)):
    subfolder_name = str(uuid.uuid4())
subfolder_path = os.path.join(args.rootdir, subfolder_name)
os.mkdir(subfolder_path)

n_ticks_sta = int(10/sampling_rate)
length_scale = 1
time_scale = 1
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
    nr = n.reshape((n.shape[0], -1)).T
    Vr = V.reshape((V.shape[0], -1)).T

    mean_trace = nr.mean(axis=0)
    # plt.plot(mean_trace)

    pks = find_all_peaks(nr)
    for i in range(field_size[0]*field_size[1]):
        pk_times[i].extend(list(times[pks[i]]+j*block_size))
    if j == n_blocks-1:
        sta_index = int(np.prod(field_size)//2)
        sta = traces.get_sta(nr[sta_index], pks[sta_index], 0, n_ticks_sta)
        local_minima, _ = signal.find_peaks(-sta)
        if len(local_minima)> 0:
            if args.plot_clusters == 1:
                fig1, ax1 = plt.subplots(figsize=(4,4))
                ax1.plot(sta)
                ax1.plot(local_minima, sta[local_minima], "rx")
            length_scale = 10*np.sqrt(times[local_minima[0]]*args.D*2)
            time_scale = times[local_minima[0]]
            plt.savefig(os.path.join(subfolder_path, "local_minima.svg"))
                             
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

X, Y = np.meshgrid(np.arange(field_size[0]), np.arange(field_size[1]))
coords = np.array([X,Y]).reshape((2,-1)).T
data_to_cluster = []
for i in range(len(pk_times)):
    loc_data = np.column_stack((coords[i,:]*np.ones((len(pk_times[i]),2)), pk_times[i]))
    isi = np.diff(pk_times[i])
    data_to_cluster.append(loc_data)
    mean_isi[i] = np.nanmean(isi)
    std_isi[i] = np.nanstd(isi)
    if len(pks[i]) > 0:
        first_spike_times[i] = pk_times[i][0]
    else:
        first_spike_times[i] = np.nan


data_to_cluster = np.concatenate(data_to_cluster, axis=0)
print(data_to_cluster.shape)
all_watersheds = []
if data_to_cluster.shape[0] > 0:
    distance_scaled = np.copy(data_to_cluster)
    print("length scale: %f" % length_scale)
    # time_scale = np.nanmean(mean_isi)
    print("time scale: %f" % time_scale)
    distance_scaled[:,:2] /= length_scale
    start = time.perf_counter()
    model = cluster.DBSCAN(eps=time_scale*0.5, min_samples=2)
    labels = model.fit_predict(distance_scaled)
    print("%d core samples" % len(model.core_sample_indices_))
    label_values = np.sort(np.unique(labels))
    end = time.perf_counter()
    print("%.1f seconds elapsed for clustering" % (end-start))
    print("%d merged spikes detected" % len(label_values))
    if args.plot_clusters == 1:
        pca = decomposition.PCA(n_components=2)
        dimred = pca.fit_transform(distance_scaled)
        cm = mpl.cm.get_cmap("cet_glasbey")
        n_points = 500
        indices = np.random.choice(distance_scaled.shape[0], \
                    size=min(n_points, distance_scaled.shape[0]), replace=False)
        colors = [cm(labels[i]) for i in indices]
        fig1, ax1 = plt.subplots(figsize=(6,6))
        ax1.scatter(dimred[indices,0],dimred[indices,1], color=colors, s=1.25)
        plt.savefig(os.path.join(subfolder_path, "cluster.svg"))
    start = time.perf_counter()
    for l in label_values:
        if l == -1:
            all_areas.append(np.ones(np.sum(labels==-1)))
        else:
            ss = StandardScaler()
            subclust_data = data_to_cluster[labels==l]
            wave_timings = np.ones(field_size)*np.inf
            for i in range(subclust_data.shape[0]):
                wave_timings[int(subclust_data[i,0]),int(subclust_data[i,1])] = subclust_data[i,2]
            cluster_membership = np.isfinite(wave_timings)

            region_labels = segmentation.watershed(wave_timings, mask=cluster_membership)
            all_watersheds.append(region_labels)
            region_lv = np.unique(region_labels)
            areas = np.zeros_like(region_lv)
            for idx, l in enumerate(region_lv):
                areas[idx] = np.count_nonzero(region_labels==l)
            all_areas.append(areas)
    end = time.perf_counter()
    print("%.1f seconds elapsed for watershedding" % (end-start))

if len(all_areas) > 0:
    all_areas = np.concatenate(all_areas)
else:
    all_areas = np.array([])

params["sigma"] = args.sigma
params["I_std"] = args.I_std
params["I_mean"] = args.I_mean

with open(os.path.join(subfolder_path, "params.pickle"), "wb") as pckl:
    pickle.dump(vars(args), pckl)

np.savez(os.path.join(subfolder_path, "stats.npz"), all_areas=all_areas, mean_isi=mean_isi, std_isi=std_isi, first_spike_times=first_spike_times)

if bool(args.write_video):
    vid = prep_visualize(n, stride=int((1/args.video_sampling_rate)//sampling_rate))
    skio.imsave(os.path.join(subfolder_path, "%s_vid.tif" % args.bif), vid)
    if len(all_watersheds) > 0:
        all_watersheds = np.array(all_watersheds)
        print(all_watersheds.shape)
        indices = np.sort(np.random.choice(all_watersheds.shape[0], size=min(500, all_watersheds.shape[0]), replace=False))
        skio.imsave(os.path.join(subfolder_path, "%s_watersheds.tif" % args.bif), all_watersheds[indices])