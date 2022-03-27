### Downsample and denoise videos
import matplotlib
import matplotlib.pyplot as plt
import argparse
import os
import skimage.io as skio
import numpy as np
from sklearn.utils.extmath import randomized_svd
from skimage import transform, morphology
from scipy import ndimage
from spikecounter.analysis import images, traces
from spikecounter.analysis import stats as sstats
import pickle
matplotlib.use("Agg")

parser = argparse.ArgumentParser()
parser.add_argument("input_file", help="Input file")
parser.add_argument("n_expected_stims", type=int)
parser.add_argument("--output_folder", help="Output folder for results", default=None)
parser.add_argument("--scale_factor", help="Scale factor for downsampling", default=2, type=float)
parser.add_argument("--n_pcs", help="Number of PCs to keep", default=50, type=int)
parser.add_argument("--remove_from_start", help="Time indices to trim from start", default=0, type=int)
parser.add_argument("--remove_from_end", help="Time indices to trim from end", default=0, type=int)
parser.add_argument("--zsc_threshold", help="Threshold for removing blue illumination artifacts", default=2, type=float)
parser.add_argument("--fs", default=10.2, type=float)
parser.add_argument("--start_from_downsampled", default=0, type=int)
parser.add_argument("--upper", default=0, type=int)
parser.add_argument("--expected_stim_width", default=3, type=int)
parser.add_argument("--fallback_mask_path", default="0", type=str)
parser.add_argument("--skewness_threshold", default=0, type=float)
parser.add_argument("--crosstalk_mask", type=str, default="0")

args = parser.parse_args()
input_file = args.input_file
output_folder = args.output_folder
scale_factor = args.scale_factor
n_pcs = args.n_pcs
remove_from_start = args.remove_from_start
remove_from_end = args.remove_from_end
fs = args.fs
zsc_threshold = args.zsc_threshold

if args.crosstalk_mask == "0":
    crosstalk_mask = None
else:
    crosstalk_mask = skio.imread(args.crosstalk_mask) > 0
    crosstalk_mask = crosstalk_mask[::int(scale_factor),::int(scale_factor)]
    print(crosstalk_mask.shape)

if output_folder is None:
    output_folder = os.path.dirname(input_file)
if args.upper ==1:
    direction= "upper"
else:
    direction = "lower"
print(direction)

filename = os.path.splitext(os.path.basename(input_file))[0]


os.makedirs(os.path.join(output_folder, "denoised"), exist_ok=True)
if args.start_from_downsampled != 1:
    img = skio.imread(input_file)
    print(img.shape)

    trimmed = img[remove_from_start:img.shape[0]-remove_from_end]
    print(trimmed.shape)

    # LP filter then downsample
    sigma = scale_factor
    if scale_factor > 1:
        os.makedirs(os.path.join(output_folder, "downsampled"), exist_ok=True)
        smoothed = ndimage.gaussian_filter(trimmed, [0,sigma,sigma])
        print(smoothed.shape)
        downsampled = smoothed[:,np.arange(smoothed.shape[1], step=sigma, dtype=int),:]
        print(downsampled.shape)
        downsampled = downsampled[:,:,np.arange(downsampled.shape[2], step=sigma, dtype=int)]
        print(downsampled.shape)

        skio.imsave(os.path.join(output_folder, "downsampled", os.path.basename(input_file)), np.round(downsampled).astype(np.uint16))
    else:
        downsampled = trimmed
else:
    downsampled = skio.imread(os.path.join(output_folder, "downsampled", os.path.basename(input_file)))

trace = images.image_to_trace(downsampled, mask=crosstalk_mask)
print(trace.shape)
if direction == "upper":
    pb_corrected_trace, _ = traces.correct_photobleach(trace, method="localmin", nsamps=int(((fs*2)//2)*2+1))
else:
    pb_corrected_trace, _ = traces.correct_photobleach(-trace, method="localmin", nsamps=int(((fs*2)//2)*2+1))
crosstalk_removed, t_mask = traces.remove_stim_crosstalk(pb_corrected_trace, threshold=zsc_threshold, fs=fs, side=direction, mode="interpolate", method="peak_detect",lpad=-1, rpad=0, fixed_width_remove=False, expected_stim_width=args.expected_stim_width, plot=False)
n_stims_detected = np.sum((np.diff(t_mask.astype(int))==1).astype(int))
print("n_stims_detected: ", n_stims_detected)
if n_stims_detected < args.n_expected_stims and args.fallback_mask_path != "0":
    print("Stims failed to be detected")
    with open(args.fallback_mask_path, "rb") as f:
        t_mask = pickle.load(f)

os.makedirs(os.path.join(output_folder, "stim_frames_removed/trace_plots"), exist_ok=True)
ts = np.arange(len(trace))/fs
fig1, ax1 = plt.subplots(figsize=(10,4))
ax1.plot(ts, trace)
ax2 = ax1.twinx()
ax2.plot(ts, pb_corrected_trace, color="C1")
ax1.plot(ts[t_mask], trace[t_mask], "rx")

stim_frames_removed = images.interpolate_invalid_values(downsampled, t_mask)
stim_removed_trace = images.image_to_trace(stim_frames_removed)
ax1.plot(ts, stim_removed_trace - np.nanmean(stim_removed_trace)/2, color="C2")

plt.savefig(os.path.join(output_folder, "stim_frames_removed/trace_plots", "%s_plot.svg" \
                        % filename))

skio.imsave(os.path.join(output_folder, "stim_frames_removed", os.path.basename(input_file)), np.round(stim_frames_removed).astype(np.uint16))


mean_img = stim_frames_removed.mean(axis=0)

# Zero the mean over time
t_zeroed = stim_frames_removed - mean_img
data_matrix = t_zeroed.reshape((t_zeroed.shape[0], -1))

# correlate to mean trace
mean_trace = data_matrix.mean(axis=1)
corr = np.matmul(data_matrix.T, mean_trace)/np.dot(mean_trace, mean_trace)
resids = data_matrix - np.outer(mean_trace, corr)

denoised = sstats.denoise_svd(resids, n_pcs, skewness_threshold=args.skewness_threshold)
denoised = denoised.reshape(downsampled.shape)

# Add back DC offset for the purposes of comparing noise to mean intensity
denoised += mean_img
skio.imsave(os.path.join(output_folder, "denoised", os.path.basename(input_file)), np.round(denoised).astype(np.uint16))
os.makedirs(os.path.join(output_folder, "denoised/analysis/stim_indices"), exist_ok=True)
with open(os.path.join(output_folder, "denoised/analysis/stim_indices/%s_stim_indices.pickle" % filename), "wb") as f:
    pickle.dump(t_mask, f)
