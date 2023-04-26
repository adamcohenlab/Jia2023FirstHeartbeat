""" Analyze experiments involving simultaneous voltage and calcium imaging.

"""
import sys
from pathlib import Path
import os
import warnings
import argparse
import pickle

import numpy as np
import skimage.io as skio
from scipy import ndimage, signal
from skimage import filters

import matplotlib
import matplotlib.pyplot as plt
import colorcet as cc

from sklearn.utils.extmath import randomized_svd

SPIKECOUNTER_PATH = os.getenv("SPIKECOUNTER_PATH")
ANAYLSIS_OUTPUT_ROOTDIR = os.getenv("ANALYSIS_OUTPUT_ROOTDIR")
assert SPIKECOUNTER_PATH is not None
assert ANAYLSIS_OUTPUT_ROOTDIR is not None

sys.path.append(SPIKECOUNTER_PATH)
from spikecounter.analysis import images
from spikecounter.analysis import stats as sstats
from spikecounter.ui import visualize
from spikecounter import utils


parser = argparse.ArgumentParser()
parser.add_argument("rootdir", type=str)
parser.add_argument("file_name", type=str)
parser.add_argument("--plot", default=True, type=utils.str2bool)
parser.add_argument("--um_per_px", default=0.265 * 4, type=float)
parser.add_argument("--hard_cutoff", default=0.005, type=float)
parser.add_argument("--downsample_factor", default=16, type=int)
parser.add_argument("--window_size", default=111, type=int)
parser.add_argument("--sta_before", default=40, type=int)
parser.add_argument("--sta_after", default=100, type=int)
args = parser.parse_args()

rootdir = Path(args.rootdir)
file_name = args.file_name
um_per_px = args.um_per_px
factor = args.downsample_factor
window_size = args.window_size
warnings.filterwarnings("once")
matplotlib.use('Agg')
plt.style.use(Path(SPIKECOUNTER_PATH, "config", "bio_publications.mplstyle"))

logger = utils.initialize_logging(rootdir / file_name)
logger.info(
    "PERFORM ANALYSIS OF SIMULTANEOUS VOLTAGE AND CALCIUM IMAGING DATA USING \
    analyze_simultaneous_voltage_calcium.py"
)
logger.info(f"Arguments: \n {str(args)}")
expt_name = rootdir.stem
output_datadir = Path(
    ANAYLSIS_OUTPUT_ROOTDIR,
    "2022 First Heartbeat_Submitted",
    "Figures/Data/Figure3/", expt_name, file_name
)
os.makedirs(output_datadir, exist_ok=True)
analysis_dir = rootdir / "analysis" / file_name
os.makedirs(analysis_dir, exist_ok=True)

logger.info("Loading voltage imaging data")
subfolder_voltage = "cam1_registered_downsampled"
voltage_image, metadata = images.load_image(
    rootdir, file_name, subfolder=subfolder_voltage
)
assert metadata is not None
mean_dt = np.mean(np.diff(utils.get_frame_times(metadata)))
t = np.arange(voltage_image.shape[0]) * mean_dt

logger.info("Regressing background from voltage imaging data")
potential_bg_traces_v = images.extract_background_traces(
    voltage_image, mode=["linear", "dark", "corners", "exp", "biexp"]
).astype(np.float32)
multi_regressed_v = images.regress_video(voltage_image, potential_bg_traces_v.T)

skio.imsave(
    analysis_dir / f"{file_name}_multi_regressed_v.tif",
    multi_regressed_v.astype(np.uint16),
)

if args.plot:
    logger.info("Plotting background regression of voltage imaging data")
    fig1, axs = plt.subplots(1, 2, figsize=(10, 6))
    _ = visualize.stackplot(potential_bg_traces_v.T, ax=axs[0])
    axs[1].plot(t, voltage_image.mean(axis=(1, 2)))
    ax2 = axs[1].twinx()
    ax2.plot(t, multi_regressed_v.mean(axis=(1, 2)), color="C1")
    axs[1].set_xlabel("Time (s)")
    plt.savefig(analysis_dir / f"{file_name}_bg_regression_v.svg")
    plt.savefig(output_datadir / f"{file_name}_bg_regression_v.svg")


logger.info("Calculating dFF from voltage imaging data")
v_dFF = (
    images.get_image_dFF(
        ndimage.gaussian_filter(multi_regressed_v, (2, 3, 3)).astype(np.float32), invert=True
    )
    - 1
)
del multi_regressed_v
skio.imsave(
    analysis_dir / f"{file_name}_v_dFF_smoothed.tif",
    v_dFF.astype(np.float32),
)

logger.info("Performing PCA denoising on voltage imaging data")
rd = v_dFF.reshape(v_dFF.shape[0], -1).T
u, s, v = randomized_svd(rd, n_components=15)

if args.plot:
    plt.close("all")
    fig1, axs = visualize.plot_pca_data(
        v,
        rd,
        v_dFF.shape[-2:],
        mode="spatial",
        single_figsize=(12, 3),
        n_components=15,
    )
    plt.savefig(analysis_dir / f"{file_name}_pca_denoising.tif", dpi=300)
    plt.savefig(output_datadir / f"{file_name}_pca_denoising.tif", dpi=300)

    plt.close("all")

# Single PC looks weird, because it will just be a linear modulation of a fixed image. With multiple
# PCs you can get something that looks more like a wave.
indices = np.zeros_like(s, dtype=bool)
indices[[0, 1, 2]] = True
indices = np.ones_like(s, dtype=bool)
comp_vid = sstats.reconstruct_svd(u, s, v, indices).T.reshape(v_dFF.shape)
skio.imsave(analysis_dir / f"{file_name}_multiPC.tif", comp_vid.astype(np.float32))
del comp_vid

# Perform the same preprocessing on the calcium imaging data
logger.info("Loading calcium imaging data")
subfolder_calcium = "cam2_registered_downsampled"
calcium_image, _ = images.load_image(rootdir, file_name, subfolder=subfolder_calcium)

logger.info("Regressing background from calcium imaging data")
potential_bg_traces_ca = images.extract_background_traces(
    calcium_image, mode=["linear", "dark", "corners", "exp", "biexp"]
).astype(np.float32)
multi_regressed_ca = images.regress_video(
    calcium_image, potential_bg_traces_ca[:, [0, 3]].T
)

if args.plot:
    logger.info("Plotting background regression of calcium imaging data")
    fig1, axs = plt.subplots(1, 2, figsize=(10, 6))
    _ = visualize.stackplot(potential_bg_traces_ca.T, ax=axs[0])
    axs[1].plot(t, calcium_image.mean(axis=(1, 2)))
    ax2 = axs[1].twinx()
    ax2.plot(t, multi_regressed_ca.mean(axis=(1, 2)), color="C1")
    axs[1].set_xlabel("Time (s)")
    plt.savefig(analysis_dir / f"{file_name}_bg_regression_ca.svg")
    plt.savefig(output_datadir / f"{file_name}_bg_regression_ca.svg")

logger.info("Calculating dFF from calcium imaging data")
ca_dFF = (
    images.get_image_dFF(
        ndimage.gaussian_filter(multi_regressed_ca, (2, 3, 3)).astype(np.float32), invert=False
    )
    - 1
)
skio.imsave(
    analysis_dir / f"{file_name}_ca_dFF_smoothed.tif", ca_dFF.astype(np.float32)
)
del multi_regressed_ca

logger.info("Identifying regions of calcium activity")
rd = ca_dFF.reshape(ca_dFF.shape[0], -1).T
u, s, v = randomized_svd(rd, n_components=15)
pc1 = np.abs(u[:, 0].reshape(ca_dFF.shape[1:]))
pc1_mask = pc1 > filters.threshold_otsu(pc1)
ca_trace = images.extract_mask_trace(ca_dFF, pc1_mask)
ca_trace_smoothed = ndimage.gaussian_filter(ca_trace, 2)

logger.info("Detecting peaks in calcium trace for triggered averages")
pks, _ = signal.find_peaks(
    ca_trace_smoothed,
    prominence=max(
        0.3 * (ca_trace_smoothed.max() - ca_trace_smoothed.min()), args.hard_cutoff
    ),
)

if args.plot:
    logger.info("Plotting spike detection")
    plt.close("all")
    visualize.plot_pca_data(
        v,
        rd,
        ca_dFF.shape[-2:],
        mode="spatial",
        single_figsize=(12, 3),
        n_components=15,
    )
    plt.savefig(analysis_dir / f"{file_name}_calcium_pca.svg")
    plt.savefig(output_datadir / f"{file_name}_calcium_pca.svg")
    plt.close("all")


    fig1, ax1 = plt.subplots(figsize=(4, 4))
    visualize.display_roi_overlay(pc1, pc1_mask, alpha=0.5, ax=ax1)
    ax1.axis("off")
    plt.savefig(analysis_dir / f"{file_name}_pc1_mask.svg")
    plt.savefig(output_datadir / f"{file_name}_pc1_mask.svg")

    fig1, ax1 = plt.subplots(figsize=(6, 3))
    ax1.plot(t, ca_trace)
    ax1.plot(t, ca_trace_smoothed)
    ax1.plot(t[pks], ca_trace_smoothed[pks], "o", color="red")
    ax1.axhline(np.median(ca_trace_smoothed), color="black", linestyle="--")
    ax1.set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(analysis_dir / f"{file_name}_ca_spike_trace.svg")
    plt.savefig(output_datadir / f"{file_name}_ca_spike_trace.svg")

logger.info("Generating triggered averages of videos and mean traces of active regions")
ca_triggered_sta_ca, _ = images.spike_triggered_average_video(
    ca_dFF, pks, (args.sta_before, args.sta_after)
)
ca_triggered_sta_v, _ = images.spike_triggered_average_video(
    v_dFF, pks, (args.sta_before, args.sta_after)
)

logger.info("Saving STA videos")
combined_sta_vid = np.stack([ca_triggered_sta_ca, ca_triggered_sta_v], axis=1)
skio.imsave(
    analysis_dir / f"{file_name}_ca_triggered_sta.tif",
    combined_sta_vid.astype(np.float32),
)
skio.imsave(
    output_datadir / f"{file_name}_ca_triggered_sta.tif",
    combined_sta_vid.astype(np.float32),
)

mask_trace_sta_v = images.extract_mask_trace(ca_triggered_sta_v, pc1_mask).astype(float)
mask_trace_sta_c = images.extract_mask_trace(ca_triggered_sta_ca, pc1_mask).astype(
    float
)
mask_trace_sta_v_padded = np.pad(
    mask_trace_sta_v, (window_size // 2, window_size // 2), mode="reflect"
)
mask_trace_sta_c_padded = np.pad(
    mask_trace_sta_c, (window_size // 2, window_size // 2), mode="reflect"
)
mask_trace_sta_v_ma = np.convolve(
    mask_trace_sta_v_padded, np.ones(window_size) / window_size, mode="valid"
)
mask_trace_sta_c_ma = np.convolve(
    mask_trace_sta_c_padded, np.ones(window_size) / window_size, mode="valid"
)
# Identify the lag between spike events of calcium and voltage traces
if len(pks) < 1:
    logger.warning("No spikes detected in calcium trace")
    ca_lag = 0
else:
    sta_corr = signal.correlate(mask_trace_sta_v, mask_trace_sta_c)
    ca_lag = np.argmax(sta_corr) - mask_trace_sta_v.size + 1

if args.plot:
    logger.info("Plotting results of triggered averaging analysis")
    t_sta = np.arange(-args.sta_before, args.sta_after) * mean_dt
    fig1, axs = plt.subplots(3, 1, figsize=(5, 6))
    axs[0].plot(t_sta, mask_trace_sta_v)
    ax2 = axs[0].twinx()
    ax2.plot(t_sta, mask_trace_sta_c, color="C1")
    axs[0].set_ylabel("Voltage dFF")
    ax2.set_ylabel("Calcium dFF")
    axs[0].set_title("STA dFF")
    axs[1].plot(t_sta, mask_trace_sta_v_ma)
    ax2 = axs[1].twinx()
    ax2.plot(t_sta, mask_trace_sta_c_ma, color="C1")
    axs[1].set_ylabel("Voltage dFF")
    ax2.set_ylabel("Calcium dFF")
    axs[1].set_title("Moving average")
    axs[2].plot(t_sta, mask_trace_sta_v - mask_trace_sta_v_ma)
    ax2 = axs[2].twinx()
    ax2.plot(t_sta, mask_trace_sta_c - mask_trace_sta_c_ma, color="C1")
    axs[2].set_ylabel("Voltage dFF")
    ax2.set_ylabel("Calcium dFF")
    axs[2].set_title("STA dFF - moving average")
    plt.tight_layout()
    plt.savefig(analysis_dir / f"{file_name}_sta_traces.svg")
    plt.savefig(output_datadir / f"{file_name}_sta_traces.svg")

# Perform correlation analysis
logger.info("Downsampling videos for correlation analysis")
mask_downsampled = np.round(
    images.downsample_video(pc1_mask[None, :, :], factor)
).astype(bool)
v_downsampled = images.downsample_video(v_dFF, factor)
ca_downsampled = images.downsample_video(ca_dFF, factor)


# Calculate the Pearson R coefficient between (two time x height x width) videos, `v_downsampled`
# and `ca_downsampled` at each pixel, over a moving time window `window size` centered around
# each timepoint
def calculate_local_corrs(v_ravelled, ca_ravelled, ws):
    # Define convolution kernels
    ma_kernel = np.ones((1, ws)) / ws
    sum_kernel = np.ones((1, ws))

    # Pad the arrays and perform moving average
    v_padded = np.pad(v_ravelled, ((0, 0), (ws // 2, ws // 2)), mode="reflect")
    ca_padded = np.pad(ca_ravelled, ((0, 0), (ws // 2, ws // 2)), mode="reflect")
    v_ma = signal.convolve(v_padded, ma_kernel, mode="valid")
    ca_ma = signal.convolve(ca_padded, ma_kernel, mode="valid")
    v_sum = signal.convolve(v_padded, sum_kernel, mode="valid")
    ca_sum = signal.convolve(ca_padded, sum_kernel, mode="valid")

    # Calculate the numerator and denominator of Pearson R coefficient
    num = (
        signal.convolve(v_padded * ca_padded, sum_kernel, mode="valid")
        + v_ma * ca_ma * ws
        - v_ma * ca_sum
        - ca_ma * v_sum
    )
    v_m_moment2 = (
        v_ma**2 * ws
        - 2 * v_sum * v_ma
        + signal.convolve(v_padded**2, sum_kernel, mode="valid")
    )
    ca_m_moment2 = (
        ca_ma**2 * ws
        - 2 * ca_sum * ca_ma
        + signal.convolve(ca_padded**2, sum_kernel, mode="valid")
    )
    den = np.sqrt(v_m_moment2 * ca_m_moment2)

    # Calculate the local correlation coefficients
    return num / den


def final_analysis_plotting(
    output_path, local_corrs, pks, mask_downsampled
):
    # Plot an image (pixels x time) of the local correlation coefficients. Regions of putative
    # calcium activity are masked in gray. Detected peaks are marked with black lines.
    fig1, axs = plt.subplots(2,1, figsize=(12, 8))
    has_calcium_activity = np.zeros_like(local_corrs, dtype=bool)
    has_calcium_activity[mask_downsampled.ravel(), :] = True
    q = axs[0].imshow(local_corrs, aspect="auto", cmap="cet_CET_D1", vmin=-1, vmax=1)
    axs[0].imshow(
        np.ma.masked_array(has_calcium_activity, ~has_calcium_activity),
        aspect="auto",
        cmap="gray",
        alpha=0.3,
    )
    ymin, ymax = axs[0].get_ylim()
    cb = fig1.colorbar(q, ax=axs[0])
    cb.ax.set_title(r"$\rho$")
    axs[0].vlines(pks, ymin, ymax, color="black")
    axs[0].set_xticks(axs[0].get_xticks(), labels=(axs[0].get_xticks() * mean_dt).astype(int))
    axs[0].set_xlim(0, local_corrs.shape[1] - 1)
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Pixel")

    q = axs[1].imshow(np.abs(local_corrs), aspect="auto", vmin=0, vmax=1)
    axs[1].imshow(
        np.ma.masked_array(has_calcium_activity, ~has_calcium_activity),
        aspect="auto",
        cmap="gray",
        alpha=0.3,
    )
    ymin, ymax = axs[1].get_ylim()
    cb = fig1.colorbar(q, ax=axs[1])
    cb.ax.set_title(r"$|\rho|$")
    axs[1].vlines(pks, ymin, ymax, color="black")
    axs[1].set_xticks(axs[1].get_xticks(), labels=(axs[1].get_xticks() * mean_dt).astype(int))
    axs[1].set_xlim(0, local_corrs.shape[1] - 1)
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Pixel")


    plt.savefig(output_path / "pixelwise_raster.tif", dpi=300)

    # Plot an image (pixels x time) of the local correlation coefficients, sorted by whether there 
    # is calcium activity. Detected peaks are marked with black lines.
    local_corrs_sorted = np.concatenate([local_corrs[has_calcium_activity[:,0]],
                                  local_corrs[~has_calcium_activity[:,0]]], axis=0
                                  )
    
    fig1, axs = plt.subplots(2,1, figsize=(12, 8))
    q = axs[0].imshow(local_corrs_sorted, aspect="auto", cmap="cet_CET_D1", vmin=-1, vmax=1)
    ymin, ymax = axs[0].get_ylim()
    cb = fig1.colorbar(q, ax=axs[0])
    cb.ax.set_title(r"$\rho$")
    axs[0].vlines(pks, ymin, ymax, color="black")
    axs[0].axhline(np.sum(has_calcium_activity[:,0]) - 0.5, color="black", linestyle="--")
    axs[0].set_xticks(axs[0].get_xticks(), labels=(axs[0].get_xticks() * mean_dt).astype(int))
    axs[0].set_xlim(0, local_corrs.shape[1] - 1)
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Pixel")


    q = axs[1].imshow(np.abs(local_corrs_sorted), aspect="auto", vmin=0, vmax=1)

    ymin, ymax = axs[1].get_ylim()
    cb = fig1.colorbar(q, ax=axs[1])
    cb.ax.set_title(r"$|\rho|$")
    axs[1].vlines(pks, ymin, ymax, color="black")
    axs[1].axhline(np.sum(has_calcium_activity[:,0]) - 0.5, color="black", linestyle="--")
    axs[1].set_xticks(axs[1].get_xticks(), labels=(axs[1].get_xticks() * mean_dt).astype(int))
    axs[1].set_xlim(0, local_corrs.shape[1] - 1)
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Pixel")
    plt.savefig(output_path / "pixelwise_raster_sorted.tif", dpi=300)

    # Plot images of the standard deviation, mean, and maximum of the local correlation coefficients
    # at each pixel over time
    std_corr_img = np.std(local_corrs, axis=1).reshape(mask_downsampled.shape[1:])
    mean_corr_img = np.mean(local_corrs, axis=1).reshape(mask_downsampled.shape[1:])
    max_corr_img = np.max(local_corrs, axis=1).reshape(mask_downsampled.shape[1:])
    fig1, axs = plt.subplots(1, 3, figsize=(12, 4))
    for ax in axs:
        ax.axis("off")
    for ax, img, title in zip(
        axs, [std_corr_img, mean_corr_img, max_corr_img], ["std", "mean", "max"]
    ):
        q = ax.imshow(img, vmin=-1, vmax=1)
        ax.set_title(title)
        fig1.colorbar(q, ax=ax)
    plt.savefig(output_path / "local_corr_temporal_stats_img.tif", dpi=300)

    # Plot the mean local correlation coefficient over time for pixels with and without calcium, and
    # compare to an equal number of randomly selected pixels
    active_calcium_mean_corrs = local_corrs[has_calcium_activity[:, 0], :].mean(axis=0)
    inactive_calcium_mean_corrs = np.nanmean(
        local_corrs[~has_calcium_activity[:, 0], :], axis=0
    )
    random_mask = np.zeros(has_calcium_activity.shape[0], dtype=bool)
    random_mask[
        np.random.choice(
            np.arange(has_calcium_activity.shape[0]),
            size=np.sum(has_calcium_activity[:, 0]).astype(int),
            replace=False,
        )
    ] = True
    random_px_mean_corrs = local_corrs[random_mask].mean(axis=0)
    with open(output_path / "local_corr_traces.pickle", "wb") as f:
        # Save the correlation traces for each pixel type
        pickle.dump(
            {
                "active_calcium_corrs": local_corrs[has_calcium_activity[:, 0], :],
                "inactive_calcium_corrs": local_corrs[~has_calcium_activity[:, 0], :],
                "random_px_corrs": local_corrs[random_mask],
            },
            f,
        )
    fig1, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(t, active_calcium_mean_corrs, color="C0", label="Active calcium")
    ax1.plot(t, inactive_calcium_mean_corrs, color="C1", label="Inactive calcium")
    ax1.plot(t, random_px_mean_corrs, color="C2", label="Random pixels")

    ymin, ymax = ax1.get_ylim()
    ax1.vlines(t[pks], ymin, ymax, color="black")
    for pk in pks:
        ax1.fill_between(
            np.arange(pk - 20, pk + 100) * mean_dt, ymin, ymax, color="gray", alpha=0.2
        )
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel(r"Mean $\rho$" + f" ({window_size} frames)")
    ax1.legend()
    plt.tight_layout()
    plt.savefig(output_path / "mean_local_corr_over_time.svg")


logger.info("Calculating local Pearson Rs using moving means")
# Reshape input videos to 2D arrays
v_downsampled_ravelled = v_downsampled.reshape(v_downsampled.shape[0], -1).T
ca_downsampled_ravelled = ca_downsampled.reshape(ca_downsampled.shape[0], -1).T
# Calculate local correlation coefficients
local_corrs = calculate_local_corrs(
    v_downsampled_ravelled, ca_downsampled_ravelled, window_size
)
corr_video = local_corrs.T.reshape(v_downsampled.shape).astype(np.float32)
skio.imsave(output_datadir / f"{file_name}_corr_video.tif", corr_video)
os.makedirs(output_datadir / "final_plots", exist_ok=True)
final_analysis_plotting(
    output_datadir / "final_plots", local_corrs, pks, mask_downsampled
)


# Plot local correlation coefficients for a few random pixels
# randpx = np.random.choice(
#     np.arange(v_downsampled_ravelled.shape[0]), size=5, replace=False
# )
# fig1, ax1 = plt.subplots(figsize=(12, 4))
# ax1.set_ylim(-1, 1)
# for i, px in enumerate(randpx):
#     color = "red" if has_calcium_activity[px, 0] else f"C{i}"
#     alpha = 0.2 if has_calcium_activity[px, 0] else 1
#     ax1.plot(local_corrs[px, :], color=color, alpha=alpha)
# ymin, ymax = ax1.get_ylim()
# ax1.vlines(pks, ymin, ymax, color="black")


# Try with optimal lag calculated from STA
logger.info("Calculating local correlations with optimal lag")
v_downsampled_ravelled = v_downsampled.reshape(v_downsampled.shape[0], -1).T
ca_downsampled_ravelled = np.roll(
    ca_downsampled.reshape(ca_downsampled.shape[0], -1).T, ca_lag, axis=1
)
local_corrs = calculate_local_corrs(
    v_downsampled_ravelled, ca_downsampled_ravelled, window_size
)
corr_video = local_corrs.T.reshape(v_downsampled.shape).astype(np.float32)
skio.imsave(output_datadir / f"{file_name}_corr_video_lag_corrected.tif", corr_video)
os.makedirs(output_datadir / "final_plots_lag_corrected", exist_ok=True)
final_analysis_plotting(
    output_datadir / "final_plots_lag_corrected",
    local_corrs,
    pks,
    mask_downsampled
)
