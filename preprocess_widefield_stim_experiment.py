### Downsample and denoise videos
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import argparse
import skimage.io as skio
from skimage import exposure
import numpy as np
from spikecounter.analysis import images, traces
from spikecounter.analysis import stats as sstats
from spikecounter import utils
import pickle

matplotlib.use("Agg")

parser = argparse.ArgumentParser()
parser.add_argument("rootdir", help="rootdir")
parser.add_argument("expt_name", help="Input file")
parser.add_argument("n_expected_stims", type=int)
parser.add_argument(
    "--output_folder", help="Output folder for results", default="None", type=str
)
parser.add_argument(
    "--scale_factor", help="Scale factor for downsampling", default=2, type=float
)
parser.add_argument("--n_pcs", help="Number of PCs to keep", default=50, type=int)
parser.add_argument("--pb_correct_method", default="localmin", type=str)
parser.add_argument("--pb_correct_mask", default="None", type=str)
parser.add_argument(
    "--remove_from_start", help="Time indices to trim from start", default=0, type=int
)
parser.add_argument(
    "--remove_from_end", help="Time indices to trim from end", default=0, type=int
)
parser.add_argument(
    "--zsc_threshold",
    help="Threshold for removing blue illumination artifacts",
    default=2,
    type=float,
)
parser.add_argument("--fs", default=10.2, type=float)
parser.add_argument("--start_from_downsampled", default=False, type=utils.str2bool)
parser.add_argument("--upper", default="0", type=utils.str2bool)
parser.add_argument("--expected_stim_width", default=3, type=int)
parser.add_argument("--fallback_mask_path", default="None", type=str)
parser.add_argument("--skewness_threshold", default=0, type=float)
parser.add_argument("--decorrelate", default=False, type=utils.str2bool)
parser.add_argument("--crosstalk_mask", type=str, default="None")
parser.add_argument("--denoise", default=False, type=utils.str2bool)
parser.add_argument("--decorr_pct", type=str, default="None")
parser.add_argument("--invert", default=False, type=utils.str2bool)

args = parser.parse_args()
rootdir = Path(args.rootdir)
expt_name = args.expt_name
output_folder = args.output_folder
scale_factor = args.scale_factor
n_pcs = args.n_pcs
remove_from_start = args.remove_from_start
remove_from_end = args.remove_from_end
fs = args.fs
zsc_threshold = args.zsc_threshold

logger = utils.initialize_logging(rootdir / expt_name)
logger.info(f"STARTING PREPROCESSING USING preprocess_widefield_stim_experiment.py")
logger.info(f"Arguments: \n {str(args)}")


expt_data = utils.load_video_metadata(rootdir, expt_name)
if expt_data and "frame_counter" in expt_data.keys():
    trace_dict, t = utils.traces_to_dict(expt_data)
    dt_frame = np.mean(np.diff(t))
    fs = 1 / dt_frame
else:
    logger.warning("No video metadata found, using default parameters")
    fs = args.fs


if args.crosstalk_mask == "None":
    crosstalk_mask = None
else:
    crosstalk_mask = skio.imread(args.crosstalk_mask) > 0
    crosstalk_mask = crosstalk_mask[:: int(scale_factor), :: int(scale_factor)]
    print(crosstalk_mask.shape)

if output_folder == "None":
    output_folder = rootdir
else:
    output_folder = Path(output_folder)
if args.upper:
    direction = "upper"
else:
    direction = "lower"

for subfolder in [
    "downsampled",
    "denoised",
    "corrected",
    "stim_frames_removed/trace_plots",
]:
    os.makedirs(output_folder / subfolder, exist_ok=True)

if args.start_from_downsampled != 1:
    logger.info(f"Loading raw image")
    img, _ = images.load_image(rootdir, expt_name)
    logger.info(f"Loaded image of dimensions {img.shape}")
    trimmed = img[remove_from_start : img.shape[0] - remove_from_end]
    # LP filter then downsample
    logger.info(f"Downsampling by factor {scale_factor}")
    if scale_factor > 1:
        downsampled = images.downsample_video(trimmed, scale_factor)
        print(downsampled.shape)
        skio.imsave(
            output_folder / "downsampled" / f"{expt_name}.tif",
            np.round(downsampled).astype(np.uint16),
        )
    else:
        downsampled = trimmed
else:
    logger.info("Starting from downsampled image")
    downsampled = skio.imread(output_folder / "downsampled" / f"{expt_name}.tif")


trace = images.extract_mask_trace(downsampled, mask=crosstalk_mask)
logger.info("Performing photobleaching correction")
if direction == "upper":
    pb_corrected_trace, _, _ = traces.correct_photobleach(
        trace, method="localmin", nsamps=int(((fs * 2) // 2) * 2 + 1)
    )
else:
    pb_corrected_trace, _, _ = traces.correct_photobleach(
        -trace, method="localmin", nsamps=int(((fs * 2) // 2) * 2 + 1)
    )
logger.info("Identifying stim frames in mask trace")
crosstalk_removed, t_mask = traces.remove_stim_crosstalk(
    pb_corrected_trace,
    threshold=zsc_threshold,
    fs=fs,
    side=direction,
    mode="interpolate",
    method="peak_detect",
    lpad=-1,
    rpad=0,
    fixed_width_remove=False,
    expected_stim_width=args.expected_stim_width,
    plot=False,
)
n_stims_detected = np.sum((np.diff(t_mask.astype(int)) == 1).astype(int))
logger.info(f"n_stims_detected: {n_stims_detected}")
if n_stims_detected < args.n_expected_stims and args.fallback_mask_path != "None":
    logger.info("Stims failed to be detected, opening manual stim timepoints")
    with open(args.fallback_mask_path, "rb") as f:
        t_mask = pickle.load(f)

ts = np.arange(len(trace)) / fs
fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(ts, trace)
ax2 = ax1.twinx()
ax2.plot(ts, pb_corrected_trace, color="C1")
ax1.plot(ts[t_mask], trace[t_mask], "rx")
logger.info("Interpolating stim frames in video")
stim_frames_removed = utils.interpolate_invalid_values(downsampled, t_mask)
stim_removed_trace = images.extract_mask_trace(stim_frames_removed)
ax1.plot(ts, stim_removed_trace - np.nanmean(stim_removed_trace) / 2, color="C2")
mean_img = stim_frames_removed.mean(axis=0)
if args.decorrelate:
    if args.decorr_pct == "None":
        decorr_trace = stim_frames_removed.mean(axis=(1, 2))
    else:
        cutoff = np.percentile(mean_img, float(args.decorr_pct))
        decorr_trace = stim_frames_removed[:, mean_img < cutoff].mean(axis=1)
    stim_frames_removed = (
        images.regress_video(stim_frames_removed, decorr_trace, regress_dc=False)
        + mean_img
    )
else:
    pass

plt.savefig(
    output_folder / "stim_frames_removed/trace_plots" / f"{expt_name}_plot.svg",
    bbox_inches="tight",
    dpi=300,
)

skio.imsave(
    output_folder / "stim_frames_removed" / f"{expt_name}.tif",
    np.round(stim_frames_removed).astype(np.uint16),
)

logger.info("Correcting for photobleaching in video")
nsamps = (int(2 * fs) // 2) * 2 + 1
if args.pb_correct_method == "monoexp" or args.pb_correct_mask == "dynamic":
    mask = mean_img > np.percentile(mean_img, 70)
else:
    mask = None

pb_corrected_img = images.correct_photobleach(
    stim_frames_removed,
    mask=mask,
    method=args.pb_correct_method,
    nsamps=nsamps,
    invert=args.invert,
)
assert isinstance(pb_corrected_img, np.ndarray)
skio.imsave(
    output_folder / "corrected" / f"{expt_name}.tif",
    pb_corrected_img.astype(np.float32),
)

if args.denoise == 1:
    logger.info("Denoising video using SVD")
    mean_img = pb_corrected_img.mean(axis=0)
    # Zero the mean over time
    t_zeroed = pb_corrected_img - mean_img
    data_matrix = t_zeroed.reshape((t_zeroed.shape[0], -1))

    # correlate to mean trace
    mean_trace = data_matrix.mean(axis=1)
    corr = np.matmul(data_matrix.T, mean_trace) / np.dot(mean_trace, mean_trace)
    resids = data_matrix - np.outer(mean_trace, corr)

    denoised = sstats.denoise_svd(
        resids, n_pcs, skewness_threshold=args.skewness_threshold
    )
    denoised = denoised.reshape(downsampled.shape)

    # Add back DC offset for the purposes of comparing noise to mean intensity
    denoised += mean_img
    denoised = exposure.rescale_intensity(denoised, out_range=np.uint16)
    skio.imsave(output_folder / "denoised" / f"{expt_name}.tif", denoised)

for subfolder in ["stim_frames_removed", "denoised", "corrected"]:
    os.makedirs(
        output_folder / f"{subfolder}/analysis/stim_indices",
        exist_ok=True,
    )
    with open(
        output_folder
        / "{subfolder}/analysis/stim_indices/{expt_name}_stim_indices.pickle",
        "wb",
    ) as f:
        pickle.dump(t_mask, f)
