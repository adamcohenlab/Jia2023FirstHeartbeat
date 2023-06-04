import argparse
import skimage.io as skio
import os
from spikecounter.analysis import images, traces
from spikecounter import utils
import numpy as np
import itertools
import pickle
import matplotlib.pyplot as plt
import colorcet as cc

parser = argparse.ArgumentParser()
parser.add_argument("rootdir", help="Root directory")
parser.add_argument("exptname", help="Experiment Name")
parser.add_argument("--subfolder", default="None", type=str)
parser.add_argument("--stim_channel", default="None", type=str)
parser.add_argument("--s", type=float, default = 0.05)
parser.add_argument("--n_knots", type=int, default=15)
parser.add_argument("--sta_before_s", type=float, default=1)
parser.add_argument("--sta_after_s", type=float, default=1)
parser.add_argument("--normalize_height", type=int, default=1)
parser.add_argument("--bootstrap_n", type=int, default=0)

args = parser.parse_args()

rootdir = args.rootdir
exptname = args.exptname
subfolder = args.subfolder

img, expt_data = images.load_image(rootdir, exptname, subfolder=subfolder)
dt_dict, t = utils.traces_to_dict(expt_data, trim=True)
dt = np.mean(np.diff(t))

if args.sta_before_s < 0 or args.sta_after_s < 0:
    sta_bounds = "auto"
else:
    sta_bounds = (int(np.ceil(args.sta_before_s/dt)),int(np.ceil(args.sta_after_s/dt)))

os.makedirs(os.path.join(rootdir, "analysis", subfolder, exptname), exist_ok=True)
if args.stim_channel != "None":
    stims = dt_dict[args.stim_channel]
    stim_indices = np.argwhere(np.diff(stims)==1).ravel()
    if len(stim_indices) > 0:
        dFF_img = images.get_image_dFF(img)
        mean_trace = dFF_img.mean(axis=(1,2))
        # Exclude stims with image-level artifacts
        window_std = np.array([np.nanstd(mean_trace[st-sta_bounds[0]:st+sta_bounds[1]]) for st in stim_indices])
        include_mask = window_std < np.mean(window_std)*2
        fig1, ax1 = plt.subplots(figsize=(8,4))
        ax1.plot(mean_trace)
        ax1.vlines(stim_indices, *ax1.get_ylim())
        ax1.vlines(stim_indices[~include_mask], *ax1.get_ylim(), color="red")
        plt.savefig(os.path.join(rootdir, "analysis", subfolder, exptname, "mean_trace.svg"))
        sta, spike_triggered_images = images.spike_triggered_average_video(dFF_img, stim_indices, sta_bounds,
                                                                           full_output=True, normalize_height=bool(args.normalize_height), include_mask=include_mask)
        skio.imsave(os.path.join(rootdir, "analysis", subfolder, exptname, "sta.tif"), sta)
    else:
        sta, spike_triggered_images = images.image_to_sta(img, fs=1/dt, plot=True, prom_pct=80, \
                   savedir=os.path.join(rootdir, "analysis", subfolder, exptname), offset=100, \
                sta_bounds=sta_bounds, normalize_height=bool(args.normalize_height),\
                full_output=True)
else:
    sta, spike_triggered_images = images.image_to_sta(img, fs=1/dt, plot=True, prom_pct=80, \
                       savedir=os.path.join(rootdir, "analysis", subfolder, exptname), offset=100, \
                    sta_bounds=sta_bounds, normalize_height=bool(args.normalize_height),\
                    full_output=True)


beta, smoothed_vid = images.spline_timing(sta, s=args.s, n_knots=args.n_knots)
skio.imsave(os.path.join(rootdir, "analysis", subfolder, exptname, "spline_smoothed_vid.tif"), smoothed_vid)
np.savez(os.path.join(rootdir, "analysis", subfolder, "%s_snapt.npz") % exptname,\
                     beta=beta)

if args.bootstrap_n > 0:
    os.makedirs(os.path.join(rootdir, "analysis", subfolder, exptname, "bootstrap"), exist_ok=True)
    if spike_triggered_images.shape[0] <= args.bootstrap_n:
        print("Insufficient spikes detected to do leave-%d out analysis" % args.bootstrap_n)
    else:
        all_combinations_bootstrap = itertools.combinations(range(spike_triggered_images.shape[0])\
                                                          ,args.bootstrap_n)
        with open(os.path.join(rootdir, "analysis", subfolder, exptname, "bootstrap", "all_combinations.pickle"), "wb") as f:
            pickle.dump(all_combinations_bootstrap, f)
        for idx, combo in enumerate(all_combinations_bootstrap):
            mask = np.zeros(spike_triggered_images.shape[0], dtype=bool)
            mask[combo] = True
            masked_sta = np.nanmean(spike_triggered_images[~mask], axis=0)
            beta, smoothed_vid = images.spline_timing(masked_sta, s=args.s, n_knots=args.n_knots)
            skio.imsave(os.path.join(rootdir, "analysis", subfolder, exptname, "bootstrap",\
                                     "spline_smoothed_vid_bs%d.tif" % idx), smoothed_vid)
            np.savez(os.path.join(rootdir, "analysis", subfolder, exptname, "bootstrap", \
                            "%s_snapt_bs%d.npz") % (exptname, idx),\
                     beta=beta)
