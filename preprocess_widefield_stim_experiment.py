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
import pickle
matplotlib.use("Agg")

parser = argparse.ArgumentParser()
parser.add_argument("input_file", help="Input file")
parser.add_argument("--output_folder", help="Output folder for results", default=None)
parser.add_argument("--scale_factor", help="Scale factor for downsampling", default=2, type=float)
parser.add_argument("--n_pcs", help="Number of PCs to keep", default=50, type=int)
parser.add_argument("--remove_from_start", help="Time indices to trim from start", default=0, type=int)
parser.add_argument("--remove_from_end", help="Time indices to trim from end", default=0, type=int)
parser.add_argument("--zsc_threshold", help="Threshold for removing blue illumination artifacts", default=2, type=float)
parser.add_argument("--fs", default=10.2, type=float)
parser.add_argument("--start_from_downsampled", default=0, type=int)
parser.add_argument("--upper", default=0, type=int)

args = parser.parse_args()
input_file = args.input_file
output_folder = args.output_folder
scale_factor = args.scale_factor
n_pcs = args.n_pcs
remove_from_start = args.remove_from_start
remove_from_end = args.remove_from_end
fs = args.fs
zsc_threshold = args.zsc_threshold

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
        smoothed = ndimage.gaussian_filter(trimmed, [1,sigma,sigma])
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

mean_img = downsampled.mean(axis=0)
img_mask = mean_img > np.percentile(mean_img, 97)
img_mask = morphology.binary_closing(img_mask, selem = np.ones((8,8)))

trace = images.image_to_trace(downsampled, mask=img_mask)
# print(trace.shape)
pb_corrected_trace, _ = traces.correct_photobleach(trace, method="localmin", nsamps=int(((fs*2)//2)*2+1))
crosstalk_removed, t_mask = traces.remove_stim_crosstalk(pb_corrected_trace, threshold=zsc_threshold, fs=fs, side=direction, mode="interpolate", method="peak_detect",lpad=-1, rpad=0, fixed_width_remove=True, max_width=15, plot=False)

os.makedirs(os.path.join(output_folder, "stim_frames_removed/trace_plots"), exist_ok=True)
ts = np.arange(len(trace))/fs
fig1, ax1 = plt.subplots(figsize=(10,4))
ax1.plot(ts, trace)
ax2 = ax1.twinx()
ax2.plot(ts, pb_corrected_trace, color="C1")
ax1.plot(ts[t_mask], trace[t_mask], "rx")
plt.savefig(os.path.join(output_folder, "stim_frames_removed/trace_plots", "%s_plot.svg" \
                        % filename))

stim_frames_removed = images.interpolate_invalid_values(downsampled, t_mask)

skio.imsave(os.path.join(output_folder, "stim_frames_removed", os.path.basename(input_file)), np.round(stim_frames_removed).astype(np.uint16))


mean_img = stim_frames_removed.mean(axis=0)

# Zero the mean over time
t_zeroed = stim_frames_removed - mean_img
data_matrix = t_zeroed.reshape((t_zeroed.shape[0], -1))

# correlate to mean trace
mean_trace = data_matrix.mean(axis=1)
corr = np.matmul(data_matrix.T, mean_trace)/np.dot(mean_trace, mean_trace)
resids = data_matrix - np.outer(mean_trace, corr)

# SVD
u, s, v = randomized_svd(resids, n_components=100)

denoised = u[:,:n_pcs]@ np.diag(s[:n_pcs]) @ v[:n_pcs,:]
denoised = denoised.reshape(downsampled.shape)

# Add back DC offset for the purposes of comparing noise to mean intensity
denoised += mean_img
skio.imsave(os.path.join(output_folder, "denoised", os.path.basename(input_file)), np.round(denoised).astype(np.uint16))
os.makedirs(os.path.join(output_folder, "denoised/analysis/stim_indices"), exist_ok=True)
with open(os.path.join(output_folder, "denoised/analysis/stim_indices/%s_stim_indices.pickle" % filename), "wb") as f:
    pickle.dump(t_mask, f)
