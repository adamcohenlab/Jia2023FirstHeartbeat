import argparse
import skimage.io as skio
import os
from spikecounter.analysis import images, traces
from spikecounter import utils
from skimage import exposure, draw
import numpy as np
from scipy import ndimage, signal
import itertools
import pickle
import pandas as pd
import time
import matplotlib.pyplot as plt
import colorcet as cc


parser = argparse.ArgumentParser()
parser.add_argument("rootdir", help="Root directory")
parser.add_argument("exptname", help="Experiment Name")
parser.add_argument("--subfolder", default="None", type=str)
parser.add_argument("--s", type=float, default = 0.05)
parser.add_argument("--n_knots", type=int, default=15)
parser.add_argument("--sta_before_s", type=float, default=0.75)
parser.add_argument("--sta_after_s", type=float, default=0.7)
parser.add_argument("--um_per_px", type=float, default=1.06)

args = parser.parse_args()

rootdir = args.rootdir
filename = args.exptname
subfolder = args.subfolder
um_per_px = args.um_per_px

os.makedirs(os.path.join(rootdir, "analysis"), exist_ok=True)

tic = time.perf_counter()

img, expt_data = images.load_image(rootdir, filename, subfolder=subfolder)
traces_dict, t = utils.traces_to_dict(expt_data)
dt = np.mean(np.diff(t))
t = t[:img.shape[0]]
rising_edges = np.argwhere(np.diff(traces_dict["enable488"]) > 0).ravel()
n_stimuli = np.sum(np.diff(traces_dict["enable488"]) > 0)
os.makedirs(os.path.join(rootdir, "analysis", subfolder, filename, "individual_spikes"), exist_ok=True)
    

factor = 4
target_img_space = expt_data["dmd_lightcrafter"]['target_image_space']
# plt.imshow(target_img_space)
target_img_space = target_img_space[::factor,::factor]
offset = [(target_img_space.shape[0]-img.shape[1])//2, (target_img_space.shape[1]-img.shape[2])//2]
target_img_space = target_img_space[offset[0]:-offset[0],offset[1]:-offset[1]]
target_img_space = target_img_space.astype(bool)

dFF_img = images.get_image_dFF(img)

pks, dFF_mean = images.image_to_peaks(dFF_img, fs = 1/dt, prom_pct=25, f_c=3, min_width_s=0.5)
pctile = np.percentile(dFF_mean, [5,95])


valleys, _ = signal.find_peaks(np.mean(dFF_mean[pks]) - dFF_mean, prominence = np.diff(pctile)[0]*0.2)
stim_valleys, mindist, associated_stims = traces.calculate_lags(valleys, rising_edges[0]-10, rising_edges[-1]+10, rising_edges,\
                                                          forward_only=False)
evoked_valleys = stim_valleys[np.abs(mindist) < 10]
first_pks_after = np.diff(np.greater.outer(pks, evoked_valleys), axis=0, prepend=False)
associated_pks = pks[np.argwhere(first_pks_after)[:,0]]

all_stim_pks, mindist, _ = traces.calculate_lags(pks, rising_edges[0]-10, rising_edges[-1]+10, rising_edges,\
                                                          forward_only=False)
pk_distances = dict(zip(all_stim_pks, mindist*dt))

# Plot results

fig1, ax1 = plt.subplots(figsize=(12,4))
ax1.plot(t, dFF_mean)
ax1.plot(t[pks], dFF_mean[pks], "rx")
ax1.plot(t[valleys], dFF_mean[valleys], "x", color="C1")
ax1.plot(t[evoked_valleys], dFF_mean[evoked_valleys], "ko")
ax1.plot(t[associated_pks], dFF_mean[associated_pks], "kx")
ax1.set_title(filename)

ax2 = ax1.twinx()
ax2.plot(t,traces_dict["enable488"][:img.shape[0]], color="C2")
plt.savefig(os.path.join(rootdir, "analysis", subfolder, filename, "peak_detect.svg"))
# exit()

# Separate out remaining peaks
associated_pk_mask = np.sum(first_pks_after, axis=1)
pre_stim_pks = pks[pks<rising_edges[0]][1:]
post_stim_pks = pks[pks>rising_edges[-1]][1:]
during_stim_pks = pks[((pks>=rising_edges[0])&(pks<rising_edges[-1])&(~associated_pk_mask)).astype(bool)]


# Measure ISIs

isi_before = np.mean(np.diff(t[pre_stim_pks]))
isi_after = np.mean(np.diff(t[post_stim_pks]))
isi_during = np.mean(np.diff(t[pks[((pks>=rising_edges[0])&(pks<rising_edges[-1])).astype(bool)]]))
stim_interval = np.mean(np.diff(rising_edges))*dt
sta_before_s = min(args.sta_before_s, stim_interval/2)
sta_after_s = min(args.sta_after_s, stim_interval/2)
sta_bounds = (int(np.ceil(sta_before_s/dt)),int(np.ceil(sta_after_s/dt)))

# Generate evoked sta
evoked_sta, evoked_imgs = images.spike_triggered_average_video(dFF_img, associated_pks, *sta_bounds, normalize_height=True,\
                                                              full_output=True)

skio.imsave(os.path.join(rootdir, "analysis", subfolder, filename, "evoked_sta.tif"),\
           exposure.rescale_intensity(evoked_sta, out_range=(0,255)).astype(np.uint8))

# Generate pre-stim sta

pre_sta, pre_imgs = images.spike_triggered_average_video(dFF_img, pre_stim_pks, *sta_bounds, normalize_height=True, full_output=True)
skio.imsave(os.path.join(rootdir, "analysis", subfolder, filename, "pre_stim_sta.tif"),\
           exposure.rescale_intensity(pre_sta, out_range=(0,255)).astype(np.uint8))

# Generate STA for non-stim values

# SNAPT

beta_stim, smoothed_vid_stim = images.spline_timing(evoked_sta, s=args.s, n_knots=args.n_knots)
skio.imsave(os.path.join(rootdir, "analysis", subfolder, filename, "spline_smoothed_vid_evoked.tif"), smoothed_vid_stim)
np.savez(os.path.join(rootdir, "analysis", subfolder, "%s_snapt_evoked.npz") % filename,\
                     beta=beta_stim)

beta_pre, smoothed_vid_pre = images.spline_timing(pre_sta, s=args.s, n_knots=args.n_knots)
skio.imsave(os.path.join(rootdir, "analysis", subfolder, filename, "spline_smoothed_vid_pre_stim.tif"), smoothed_vid_pre)
np.savez(os.path.join(rootdir, "analysis", subfolder, "%s_snapt_pre_stim.npz") % filename,\
                     beta=beta_pre)

# Generate activation map
q_stim = images.analyze_wave_dynamics(beta_stim, dt, um_per_px, deltax=13)
q_pre = images.analyze_wave_dynamics(beta_pre, dt, um_per_px, deltax=13)

tsmoothed_stim = q_stim[2] - np.nanpercentile(q_stim[2], 2)
tsmoothed_pre = q_pre[2]- np.nanpercentile(q_pre[2], 2)

# Normalize
clamped_sta_stim = images.clamp_intensity(tsmoothed_stim)
clamped_sta_pre = images.clamp_intensity(tsmoothed_pre)

normalized_sta_stim = clamped_sta_stim/np.nanmax(clamped_sta_stim)
normalized_sta_pre = clamped_sta_pre/np.nanmax(clamped_sta_pre)


# Measure distance of LOI from DMD target

y_com, x_com = ndimage.center_of_mass(target_img_space)
y_com *= um_per_px
x_com *= um_per_px
x_loi_pre, y_loi_pre = q_pre[0][4:6]
x_loi_stim, y_loi_stim = q_stim[0][4:6]
x_loi_pre *= um_per_px
y_loi_pre *= um_per_px
x_loi_stim *= um_per_px
y_loi_stim *= um_per_px

loi_dist_pre = np.linalg.norm(np.array([x_loi_pre-x_com, y_loi_pre-y_com]))
loi_dist_stim = np.linalg.norm(np.array([x_loi_stim-x_com, y_loi_stim-y_com]))

# Measure activation time at LOI and target locations
pre_loi_spot_coords = draw.disk((y_loi_pre, x_loi_pre), 5)
target_spot_coords = draw.disk((y_com, x_com), 5)

pre_loi_spot = np.zeros_like(tsmoothed_pre, dtype=bool)
target_spot = np.zeros_like(tsmoothed_pre, dtype=bool)
pre_loi_spot[pre_loi_spot_coords] = True
target_spot[target_spot_coords] = True

t_target = np.nanmean(clamped_sta_pre[target_spot])
t_loi = np.nanmean(clamped_sta_pre[pre_loi_spot])


# Collect remaining individual spike images

post_imgs = images.get_spike_videos(dFF_img, post_stim_pks, *sta_bounds, normalize_height=True)
during_imgs = images.get_spike_videos(dFF_img, during_stim_pks, *sta_bounds, normalize_height=True)

# Compare individual spikes to STAs

valid_mask_pre = np.isfinite(tsmoothed_pre)
valid_mask_stim = np.isfinite(tsmoothed_stim)

spike_imgs = [pre_imgs, during_imgs, evoked_imgs, post_imgs]
spike_pks = [pre_stim_pks, during_stim_pks, associated_pks, post_stim_pks]
labels = ["pre", "during", "stim", "post"]

all_data_entries = []

# for i in [0]:
for i in range(len(spike_imgs)):
    imgs = spike_imgs[i]
    pks = spike_pks[i]
    label = labels[i]
    # for j in [0]:
    for j in range(imgs.shape[0]):
        beta_test, smoothed_vid = images.spline_timing(imgs[j], s=args.s, n_knots=args.n_knots)
        skio.imsave(os.path.join(rootdir, "analysis", subfolder, filename, "individual_spikes", "%s_spike_%d.tif" % (label, pks[j])),\
                   exposure.rescale_intensity(smoothed_vid, out_range=(0,255)).astype(np.uint8))
        np.savez(os.path.join(rootdir, "analysis", subfolder, filename, "individual_spikes", "%s_spike_%d.npz"% (label, pks[j])) ,\
                         beta=beta_pre)
        q_test_pre = images.analyze_wave_dynamics(beta_test, dt, um_per_px, deltax=13, valid_mask=valid_mask_pre)
        q_test_stim = images.analyze_wave_dynamics(beta_test, dt, um_per_px, deltax=13, valid_mask=valid_mask_stim)
        x_loi_test, y_loi_test = q_test_stim[0][4:6]
        x_loi_test *= um_per_px
        y_loi_test *= um_per_px
        dist_to_pre_loi = np.linalg.norm(np.array([x_loi_pre-x_loi_test, y_loi_pre-y_loi_test]))
        dist_to_stim_loi = np.linalg.norm(np.array([x_loi_stim-x_loi_test, y_loi_stim-y_loi_test]))
        dist_to_target = np.linalg.norm(np.array([x_com-x_loi_test, y_com-y_loi_test]))


        
        tsmoothed_test_stim = q_test_stim[2] - np.nanpercentile(q_test_stim[2], 2)
        tsmoothed_test_pre = q_test_pre[2]- np.nanpercentile(q_test_pre[2], 2)
        clamped_test_stim = images.clamp_intensity(tsmoothed_test_stim)
        clamped_test_pre = images.clamp_intensity(tsmoothed_test_pre)
        normalized_test_stim = clamped_test_stim/np.nanmax(clamped_test_stim)
        normalized_test_pre = clamped_test_pre/np.nanmax(clamped_test_pre)
        fig1, axes = plt.subplots(1,2,figsize=(6,3))
        axes[0].imshow(normalized_test_pre, cmap="cet_CET_R1")
        axes[1].imshow(normalized_test_stim, cmap="cet_CET_R1")
        
        t_test_loi = np.nanmean(clamped_test_pre[pre_loi_spot])
        t_test_target = np.nanmean(clamped_test_stim[target_spot])
        
        plt.savefig(os.path.join(rootdir, "analysis", subfolder, filename, "individual_spikes", "%s_spike_%d_activation_map.tif" % (label, pks[j])), dpi=150)

        diff_pre = np.nansum((normalized_test_pre-normalized_sta_pre)**2)/np.nansum(valid_mask_pre)
        diff_stim = np.nansum((normalized_test_stim-normalized_sta_stim)**2)/np.nansum(valid_mask_stim)
        
        print(label, diff_pre, diff_stim)
                   
        if pks[j] in pk_distances:
            offset = pk_distances[pks[j]]
        else:
            offset = np.nan
        
        all_data_entries.append((label, t[pks[j]], offset, diff_pre, diff_stim, x_loi_test, y_loi_test, dist_to_pre_loi, dist_to_stim_loi, dist_to_target, t_test_loi, t_test_target, filename, \
                        stim_interval, n_stimuli, isi_before, isi_during, isi_after,x_loi_pre, y_loi_pre, x_loi_stim, y_loi_stim, loi_dist_pre, loi_dist_stim, t_loi, t_target, x_com, y_com))
        print(all_data_entries[-1])
        
all_data_entries = pd.DataFrame(all_data_entries, columns=["condition", "time", "offset_nearest_stim", "diff_pre", "diff_stim", "x_loi_test", "y_loi_test", "dist_to_pre_loi", "dist_to_stim_loi", "dist_to_target", "t_test_loi", "t_test_target",\
                                                    "file_name",\
                                        "stim_interval", "n_stimuli", "isi_before", "isi_during", "isi_after", "x_loi_pre", "y_loi_pre", "x_loi_stim", "y_loi_stim", "loi_dist_pre", "loi_dist_stim", "t_loi", "t_target",\
                                                          "x_target", "y_target"])
all_data_entries.to_csv(os.path.join(rootdir, "analysis", subfolder, filename, "wave_data.csv"), index=False)

toc = time.perf_counter()
print(toc-tic)