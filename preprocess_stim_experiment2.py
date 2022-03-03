### Downsample and denoise Optopatch videos with crosstalk from blue and available metadata from 2020RigControl
import matplotlib
import matplotlib.pyplot as plt
import argparse
import os
import skimage.io as skio
import numpy as np
from sklearn.utils.extmath import randomized_svd
from skimage import transform, morphology
from scipy import ndimage, signal, stats
from spikecounter.analysis import images, traces
from spikecounter import utils
import pickle
import mat73
matplotlib.use("Agg")

parser = argparse.ArgumentParser()
parser.add_argument("rootdir", help="Root directory")
parser.add_argument("exptname", help="Experiment Name")
parser.add_argument("crosstalk_channel", help="Channel of crosstalk source")
parser.add_argument("--output_folder", help="Output folder for results", default=None)
parser.add_argument("--scale_factor", help="Scale factor for downsampling", default=2, type=float)
parser.add_argument("--n_pcs", help="Number of PCs to keep", default=50, type=int)
parser.add_argument("--remove_from_start", help="Time indices to trim from start", default=0, type=int)
parser.add_argument("--remove_from_end", help="Time indices to trim from end", default=0, type=int)
parser.add_argument("--start_from_downsampled", default=0, type=int)
parser.add_argument("--filter_skewness", default=1, type=int)
parser.add_argument("--skewness_threshold", default=2, type=float)
parser.add_argument("--left_shoulder", default=16,type=float)
parser.add_argument("--right_shoulder", default=19, type=float)

def generate_invalid_frame_indices(stim_trace):
    invalid_indices_daq = np.argwhere(stim_trace > 0).ravel()
    invalid_indices_camera = np.concatenate((invalid_indices_daq, invalid_indices_daq+1))
    return invalid_indices_camera
    

args = parser.parse_args()
expt_name = args.exptname
rootdir = args.rootdir
crosstalk_channel = args.crosstalk_channel
output_folder = args.output_folder
scale_factor = args.scale_factor
n_pcs = args.n_pcs
remove_from_start = args.remove_from_start
remove_from_end = args.remove_from_end

if output_folder is None:
    output_folder = rootdir

expt_data = mat73.loadmat(os.path.join(rootdir, expt_name, "output_data_py.mat"))["dd_compat_py"]

trace_dict, dt_frame = utils.traces_to_dict(expt_data)
fs = 1/dt_frame

os.makedirs(os.path.join(output_folder, "denoised"), exist_ok=True)
os.makedirs(os.path.join(output_folder, "downsampled"), exist_ok=True)
os.makedirs(os.path.join(output_folder, "corrected"), exist_ok=True)


if args.start_from_downsampled != 1:
    img, _ = images.load_image(rootdir, expt_name)
    print(img.shape)

    trimmed = img[remove_from_start:img.shape[0]-remove_from_end]

    # LP filter then downsample
    sigma = scale_factor
    if scale_factor > 1:
        smoothed = ndimage.gaussian_filter(trimmed, [0,sigma,sigma])
        print(smoothed.shape)
        downsampled = smoothed[:,np.arange(smoothed.shape[1], step=sigma, dtype=int),:]
        print(downsampled.shape)
        downsampled = downsampled[:,:,np.arange(downsampled.shape[2], step=sigma, dtype=int)]
        print(downsampled.shape)

        skio.imsave(os.path.join(output_folder, "downsampled", "%s.tif" % expt_name), np.round(downsampled).astype(np.uint16))
    else:
        downsampled = trimmed
else:
    downsampled = skio.imread(os.path.join(output_folder, "downsampled", "%s.tif" % expt_name))

if crosstalk_channel =="None":
    stim_frames_removed = downsampled
else:
    invalid_indices = generate_invalid_frame_indices(trace_dict[crosstalk_channel]) - remove_from_start
    invalid_mask = np.zeros(downsampled.shape[0], dtype=bool)
    invalid_mask[invalid_indices] = True
    stim_frames_removed = images.interpolate_invalid_values(downsampled, invalid_mask)

# Correct photobleach
nsamps = (int(2*fs)//2)*2 +1
pb_corrected_img = images.correct_photobleach(stim_frames_removed, mask=None, method="localmin", nsamps=nsamps)

skio.imsave(os.path.join(output_folder, "corrected", "%s.tif" % expt_name), pb_corrected_img)


mean_img = pb_corrected_img.mean(axis=0)

# Zero the mean over time
t_zeroed = pb_corrected_img - mean_img
data_matrix = t_zeroed.reshape((t_zeroed.shape[0], -1))

# bandpass to get rid of BU noise
if args.left_shoulder < fs/2:
    if args.right_shoulder == -1:
        sos = signal.butter(5,[args.left_shoulder], output="sos", fs=fs)
    else:
        sos = signal.butter(5,[args.left_shoulder,args.right_shoulder], btype="bandstop", output="sos", fs=fs)
    data_matrix_filtered = np.apply_along_axis(lambda x: signal.sosfiltfilt(sos, x), 0, data_matrix)
else:
    data_matrix_filtered = data_matrix

# SVD
u, s, v = randomized_svd(data_matrix_filtered, n_components=60)

use_pcs = np.zeros_like(s,dtype=bool)
use_pcs[:n_pcs] = True
if args.filter_skewness:
    skw = np.apply_along_axis(lambda x: stats.skew(np.abs(x)), 1, v)
    use_pcs = use_pcs & (skw > args.skewness_threshold)
    

denoised = u[:,use_pcs]@ np.diag(s[use_pcs]) @ v[use_pcs,:]
denoised = denoised.reshape(pb_corrected_img.shape)

# Add back DC offset for the purposes of comparing noise to mean intensity
denoised += mean_img
skio.imsave(os.path.join(output_folder, "denoised", "%s.tif" % expt_name), denoised.astype(np.float32))
