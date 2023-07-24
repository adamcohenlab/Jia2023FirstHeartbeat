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
import os
from scipy import signal, optimize, interpolate
from skimage import io as skio
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import colorcet as cc
from datetime import datetime
import collections


parser = argparse.ArgumentParser()
parser.add_argument("rootdir", type=str)
parser.add_argument("bif", type=str)
parser.add_argument("D", type=float)
parser.add_argument("I_mean", type=float)
parser.add_argument("I_std", type=float)
parser.add_argument("I_height", type=float)
parser.add_argument("I_intercept", type=int)
parser.add_argument("sigma", type=float)
parser.add_argument("duration", type=float)
parser.add_argument("--n_y", default=25, type=int)
parser.add_argument("--n_x", default=1, type=int)
parser.add_argument("--write_video", default=0, type=int)
parser.add_argument("--video_sampling_rate", default=1, type=float)
parser.add_argument("--plot_n_example_traces", default=0, type=int)
parser.add_argument("--plot_clusters", default=0, type=int)
args = parser.parse_args()


def prep_visualize(array, stride=1, rescale=10, dtype=np.uint8):
    downsampled_t = array[::stride]
    if rescale > 1:
        downsampled_t = transform.rescale(downsampled_t, (1,\
            rescale, rescale), preserve_range=True, anti_aliasing=False,\
                order=0)
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
field_size = (args.n_y,args.n_x)
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

block_size = 10000
n_blocks = int(args.duration//block_size + 1*(args.duration%block_size !=0))
print(n_blocks)

params = params[args.bif]
params["D"] = args.D
effective_slope = args.I_height/field_size[0]
params["I"] = args.I_mean*np.ones(field_size)+ np.random.randn(*field_size)*args.I_std + \
        (np.arange(field_size[0])[:,None]-args.I_intercept)*effective_slope

min_bif_values = {"snic": 4.6, "saddle_node": 4.6, "subcritical_hopf": 50, "supercritical_hopf": 20}
g = UnitGrid(field_size)
del params["V_init"]
del params["n_init"]

# Empirically determine the phase function of the system
params_ode = params.copy()
params_ode["I"] = np.max(params["I"])
theta, crit = detools.find_max_dVdt(params_ode, args.bif)
print(crit)
V_init = ScalarField(g, data=np.ones(field_size)*crit[0] + np.random.randn(*field_size)*1)
n_init = ScalarField(g, data=np.ones(field_size)*crit[1] + np.random.randn(*field_size)*0.01)



pde = detools.INapIkPDE(params, sigma=args.sigma)
mean_isi = np.zeros(field_size[0]*field_size[1])
std_isi = np.zeros(field_size[0]*field_size[1])
first_spike_times = np.ones(field_size[0]*field_size[1])*np.nan
all_areas = []
pk_times = [[] for i in range(field_size[0]*field_size[1])]

sampling_rate = 0.01

# Burn in
fc = FieldCollection([V_init, n_init])
storage = MemoryStorage()
pde.solve(fc, t_range=500,\
              dt=dt,tracker=storage.tracker(sampling_rate))
data = np.array(storage.data)
V_init = ScalarField(g, data=data[-1,0])
n_init = ScalarField(g, data=data[-1,1])
fc = FieldCollection([V_init, n_init])

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
    
    rel_y = data - crit[None,:,None,None]
    rel_y[:,1] *= 100
    sos = signal.butter(7, 7.5, fs=1/dt, output="sos")
    mean_rel_y = np.mean(rel_y, axis=0)
    filtered_rel_y = signal.sosfiltfilt(sos, rel_y-mean_rel_y, axis=0) + mean_rel_y
    angles = np.arctan2(filtered_rel_y[:,1], filtered_rel_y[:,0]).astype(np.float32)
    print(angles.shape)
    angles[angles < 0] = np.pi*2 + angles[angles<0]
    radii = np.sum((rel_y)**2, axis=1)**0.5
    sgns = np.sign(angles-theta)
    is_pk = ((np.diff(sgns, prepend=sgns[0][None,:], axis=0) > 0) & (radii > 20)).reshape(angles.shape[0],-1)
    print(is_pk.shape)
    pk_times = [times[is_pk[:,i]]+j*block_size for i in range(is_pk.shape[1])]
    print(np.sum(is_pk))
    
    corrected_phase, _ = correct_phase(angles)
    corrected_phase = corrected_phase[::10]
    phase_diff = corrected_phase - corrected_phase[:,-1,-1][:,None,None]

    V_init = ScalarField(g, data=data[-1,0])
    n_init = ScalarField(g, data=data[-1,1])
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
    if len(pk_times[i]) > 0:
        first_spike_times[i] = pk_times[i][0]
    else:
        first_spike_times[i] = np.nan

os.makedirs(args.rootdir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
subfolder_name = "%s_%s_%s" % (args.bif, timestamp, str(uuid.uuid4())[:10])
subfolder_path = os.path.join(args.rootdir, subfolder_name)
os.makedirs(subfolder_path, exist_ok=True)
print(subfolder_path)

data_to_cluster = np.concatenate(data_to_cluster, axis=0)
# print(data_to_cluster[:10,:])
all_watersheds = []
all_wave_origins = []
all_wave_timings = []
if data_to_cluster.shape[0] > 0:
    distance_scaled = np.copy(data_to_cluster)
    print("length scale: %f" % length_scale)
        # time_scale = np.nanmean(mean_isi)
    print("time scale: %f" % time_scale)
    start = time.perf_counter()
    if np.all(np.array(field_size) > 1):
        distance_scaled[:,:2] /= length_scale
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
                wave_origins = np.zeros_like(region_lv)
                wave_timings = np.zeros_like(region_lv, dtype=float)
                for idx, l in enumerate(region_lv):
                    areas[idx] = np.count_nonzero(region_labels==l)
                    wave_timings_region = np.copy(wave_timings)
                    wave_timings_region[region_labels != l] = np.inf
                    wave_origins[idx] = np.argmin(wave_timings_region)
                all_wave_timings.extend(list(wave_timings))
                all_areas.extend(list(areas))
                all_wave_origins.extend(list(wave_origins))
    else:
        # For 1D simulations
        curr_clusters = []
        keep_axis = np.argwhere(np.array(field_size) !=1).ravel()[0]
        n_units = field_size[keep_axis]
        data_to_cluster = data_to_cluster[:,[keep_axis,-1]]
        data_to_cluster = data_to_cluster[data_to_cluster[:,1].argsort()]
        print(data_to_cluster[:10,:])
        single_oscillator_period = np.median(np.diff(data_to_cluster[data_to_cluster[:,0]==n_units-1,1]))
        keep_indices = np.argwhere(np.abs(np.diff(data_to_cluster[:,0])) == 1)
        diffs = data_to_cluster[keep_indices+1,1] - data_to_cluster[keep_indices,1]
        max_diff = max(np.median(diffs)*4, single_oscillator_period/(n_units-3))
        print(np.max(diffs), max_diff)
        curr_clusters = []
        start_times = []
        start_locs = []
        cluster_size = []

        for i in range(data_to_cluster.shape[0]):
            prepend = False
            if len(curr_clusters) == 0:
                curr_clusters.append(collections.deque([data_to_cluster[i,:]]))
            else:
                k = len(curr_clusters)
                remove_clusters = 0
                for j in range(len(curr_clusters)):
                    if data_to_cluster[i,0] - curr_clusters[j][-1][0] == 1:
                        if data_to_cluster[i,1] - curr_clusters[j][-1][1] <= max_diff and data_to_cluster[i,1] - curr_clusters[j][-1][1] > 0:
                            k = j
                            break
                        elif (data_to_cluster[i,1] - curr_clusters[j][-1][1] > max_diff) & (curr_clusters[j][-1][1] > curr_clusters[j][0][1]):
                            remove_clusters = j+1
                    elif data_to_cluster[i,0] - curr_clusters[j][0][0] == -1:
                        if data_to_cluster[i,1] - curr_clusters[j][0][1] <= max_diff and data_to_cluster[i,1] - curr_clusters[j][0][1] > 0:
                            k = j
                            prepend=True
                            break
                        elif (data_to_cluster[i,1] - curr_clusters[j][0][1] > max_diff) & (curr_clusters[j][-1][1] < curr_clusters[j][0][1]):
                            remove_clusters = j+1
                if k == len(curr_clusters):
                    curr_clusters.append(collections.deque([data_to_cluster[i,:]]))
                else:
                    if prepend:
                        curr_clusters[k].appendleft(data_to_cluster[i])
                    else:
                        curr_clusters[k].append(data_to_cluster[i])

                clusters_to_remove = curr_clusters[:remove_clusters]
                curr_clusters = curr_clusters[remove_clusters:]
                curr_clusters = curr_clusters[:k-remove_clusters] + curr_clusters[k-remove_clusters+1:] + [curr_clusters[k-remove_clusters]]
                for cl in clusters_to_remove:
                    sorted_cl = sorted(cl, key=lambda x: x[1])
                    all_wave_timings.append(sorted_cl[0][1])
                    all_wave_origins.append(sorted_cl[0][0])
                    all_areas.append(len(sorted_cl))
        for cl in curr_clusters:
            sorted_cl = sorted(cl, key=lambda x: x[1])
            all_wave_timings.append(sorted_cl[0][1])
            all_wave_origins.append(sorted_cl[0][0])
            all_areas.append(len(sorted_cl))
    end = time.perf_counter()
    print("%.1f seconds elapsed for watershedding" % (end-start))


params["sigma"] = args.sigma
params["I_std"] = args.I_std
params["I_mean"] = args.I_mean
params["I_height"] = args.I_height
params["I_intercept"] = args.I_intercept

with open(os.path.join(subfolder_path, "params.pickle"), "wb") as pckl:
    pickle.dump({**vars(args), **params}, pckl)

np.savez(os.path.join(subfolder_path, "stats.npz"), all_areas=all_areas, mean_isi=mean_isi, std_isi=std_isi,\
         first_spike_times=first_spike_times, all_wave_origins=all_wave_origins, all_wave_timings = all_wave_timings, phase_diff=phase_diff)

if bool(args.write_video):
    vid = prep_visualize(n, stride=int((1/args.video_sampling_rate)//sampling_rate))
    print(vid.shape)
    skio.imsave(os.path.join(subfolder_path, "%s_vid.tif" % args.bif), vid)
    if len(all_watersheds) > 0:
        all_watersheds = np.array(all_watersheds)
        print(all_watersheds.shape)
        indices = np.sort(np.random.choice(all_watersheds.shape[0], size=min(500, all_watersheds.shape[0]), replace=False))
        skio.imsave(os.path.join(subfolder_path, "%s_watersheds.tif" % args.bif), all_watersheds[indices])
