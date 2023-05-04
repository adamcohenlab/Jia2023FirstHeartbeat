"""
Downsample and denoise Optopatch videos with crosstalk from blue and available metadata from
2020RigControl
"""
import argparse
import os
import sys
from pathlib import Path

import skimage.io as skio
import numpy as np
from skimage import exposure
from scipy import signal
import pickle
from spikecounter.analysis import images, traces
from spikecounter.analysis import stats as sstats
from spikecounter import utils


parser = argparse.ArgumentParser()
parser.add_argument("rootdir", help="Root directory")
parser.add_argument("exptname", help="Experiment Name")
parser.add_argument("crosstalk_channel", help="Channel of crosstalk source")
parser.add_argument("--output_folder", help="Output folder for results", default=None)
parser.add_argument(
    "--initial_subfolder", help="Subfolder for initial data", default="None"
)
parser.add_argument(
    "--scale_factor", help="Scale factor for downsampling", default=2, type=int
)
parser.add_argument("--n_pcs", help="Number of PCs to keep", default=50, type=int)
parser.add_argument(
    "--remove_from_start", help="Time indices to trim from start", default=0, type=int
)
parser.add_argument(
    "--remove_from_end", help="Time indices to trim from end", default=0, type=int
)
parser.add_argument("--start_from_downsampled", default=0, type=int)
parser.add_argument("--pb_correct_method", default="localmin", type=str)
parser.add_argument("--pb_correct_mask", default="None", type=str)
parser.add_argument("--filter_skewness", default=True, type=utils.str2bool)
parser.add_argument("--skewness_threshold", default=2, type=float)
parser.add_argument("--left_shoulder", default=16, type=float)
parser.add_argument("--right_shoulder", default=19, type=float)
parser.add_argument("--invert", default=0, type=int)
parser.add_argument("--lpad", default=0, type=int)
parser.add_argument("--rpad", default=0, type=int)
parser.add_argument("--fs", default=10.2, type=float)
parser.add_argument("--decorrelate", default=True, type=utils.str2bool)
parser.add_argument("--decorr_pct", default="None")
parser.add_argument("--denoise", default=True, type=utils.str2bool)
parser.add_argument("--bg_const", default=100, type=float)

args = parser.parse_args()
expt_name = args.exptname
rootdir = Path(args.rootdir)
crosstalk_channel = args.crosstalk_channel
output_folder = args.output_folder
scale_factor = args.scale_factor
n_pcs = args.n_pcs
remove_from_start = args.remove_from_start
remove_from_end = args.remove_from_end

logger = utils.initialize_logging(rootdir/expt_name)
logger.info(f"STARTING PREPROCESSING USING preprocess_stim_experiment2.py")
logger.info(f"Arguments: \n {str(args)}")

if output_folder is None:
    output_folder = rootdir
output_folder = Path(output_folder)


expt_data = utils.load_video_metadata(rootdir, expt_name)
expt_data["remove_from_start"] = remove_from_start
expt_data["remove_from_end"] = remove_from_end
np.savez_compressed(rootdir/expt_name/"output_data_py.npz", dd_compat_py=expt_data)
if expt_data is None or "frame_counter" not in expt_data:
    logger.warning("No video metadata found, using default parameters")
    fs = args.fs
    trace_dict = None
else:
    trace_dict, t = utils.traces_to_dict(expt_data)
    dt_frame = np.mean(np.diff(t))
    fs = 1 / dt_frame

# Make new subfolders
output_folders = {}
for d in ["downsampled", "stim_frames_removed", "corrected", "denoised"]:
    if args.initial_subfolder: # pylint: disable=no-member
        output_folders[d] = output_folder/f"{args.initial_subfolder}_{d}"
    else:
        output_folders[d] = output_folder/d
    os.makedirs(output_folders[d], exist_ok=True)

if args.start_from_downsampled != 1:
    logger.info(f"Loading image from subfolder {args.initial_subfolder}")
    if args.initial_subfolder != "None":
        img, _ = images.load_image(rootdir, expt_name, subfolder=args.initial_subfolder, cam_indices=0)
    else:
        img, _ = images.load_image(rootdir, expt_name, cam_indices=0)
    logger.info(f"Loaded image of dimensions {img.shape}")

    trimmed = img[remove_from_start : img.shape[0] - remove_from_end]
    # LP filter then downsample
    logger.info(f"Downsampling by factor {scale_factor}")
    if scale_factor > 1:
        downsampled = images.downsample_video(trimmed, scale_factor)
        skio.imsave(
            output_folders["downsampled"]/f"{expt_name}.tif",
            np.round(downsampled).astype(np.uint16),
        )
    else:
        downsampled = trimmed
else:
    logger.info("Starting from downsampled")
    downsampled = skio.imread(
        output_folders["downsampled"]/f"{expt_name}.tif"
    )

downsampled = downsampled.astype(np.float32) - args.bg_const

if crosstalk_channel != "None":
    if trace_dict is None:
        logger.warning("Could not load DAQ data, cannot correct for crosstalk")
        stim_frames_removed = downsampled
    else:
        logger.info(f"Correcting for crosstalk using channel {crosstalk_channel}")
        invalid_indices = (
            traces.generate_invalid_frame_indices(trace_dict[crosstalk_channel])
            - remove_from_start
        )
        # print(invalid_indices)
        invalid_mask = np.zeros(downsampled.shape[0], dtype=bool)
        invalid_indices = invalid_indices[invalid_indices < downsampled.shape[0]]
        invalid_mask[invalid_indices] = True
        if args.lpad > 0:
            for i in range(args.lpad + 1):
                print("lpad", i)
                invalid_mask[invalid_indices - i] = True
        if args.rpad > 0:
            for i in range(args.rpad + 1):
                invalid_mask[invalid_indices + i] = True

        stim_frames_removed = utils.interpolate_invalid_values(
            downsampled, invalid_mask
        )
else:
    stim_frames_removed = downsampled

mean_img = stim_frames_removed.mean(axis=0)
if args.decorrelate:
    logger.info(f"Decorrelating from image average, pct={args.decorr_pct}")
    if args.decorr_pct == "None":
        decorr_trace = stim_frames_removed.mean(axis=(1, 2))
    else:
        cutoff = np.percentile(mean_img, float(args.decorr_pct))
        decorr_trace = stim_frames_removed[:, mean_img < cutoff].mean(axis=1)
    stim_frames_removed = (
        images.regress_video(stim_frames_removed, decorr_trace, regress_dc=False) + mean_img
    )
skio.imsave(
    output_folders["stim_frames_removed"]/f"{expt_name}.tif",
    stim_frames_removed,
)
if args.pb_correct_method == "None":
    logger.info("No photobleaching correction, downsampling only")
    sys.exit()

logger.info(f"Correcting for photobleaching using method {args.pb_correct_method}")
nsamps = (int(2 * fs) // 2) * 2 + 1
if args.pb_correct_method == "monoexp" or args.pb_correct_mask == "dynamic":
    mask = mean_img > np.percentile(mean_img, 80)
else:
    mask = None
pb_corrected_img = exposure.rescale_intensity(
    images.correct_photobleach(
        stim_frames_removed,
        mask=mask,
        method=args.pb_correct_method,
        nsamps=nsamps,
        invert=args.invert,
        amplitude_window=2,
    ),
    out_range=np.uint16,
)
skio.imsave(
    output_folders["corrected"]/f"{expt_name}.tif", pb_corrected_img
)

if args.denoise == 0:
    logger.info("No denoising, exiting")
    sys.exit()


# Zero the mean over time
mean_img = pb_corrected_img.mean(axis=0)
t_zeroed = pb_corrected_img - mean_img
data_matrix = t_zeroed.reshape((t_zeroed.shape[0], -1))

# bandpass to get rid of BU noise
if args.left_shoulder < fs / 2:
    if args.right_shoulder == -1:
        sos = signal.butter(5, [args.left_shoulder], output="sos", fs=fs)
    else:
        sos = signal.butter(
            5,
            [args.left_shoulder, args.right_shoulder],
            btype="bandstop",
            output="sos",
            fs=fs,
        )
    data_matrix_filtered = signal.sosfiltfilt(sos, data_matrix, axis=0)
else:
    data_matrix_filtered = data_matrix


# SVD
denoised = sstats.denoise_svd(
    data_matrix_filtered, n_pcs, skewness_threshold=args.skewness_threshold
)
denoised = denoised.reshape(downsampled.shape)

# Add back DC offset for the purposes of comparing noise to mean intensity
denoised += mean_img

skio.imsave(
    output_folders["denoised"]/f"{expt_name}.tif",
    exposure.rescale_intensity(denoised, out_range=np.uint16),
)
