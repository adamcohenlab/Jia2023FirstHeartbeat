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
from spikecounter.analysis import stats as sstats
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
parser.add_argument("--invert", default=0, type=int)

def generate_invalid_frame_indices(stim_trace, dt_frame):
    invalid_indices_daq = np.argwhere(stim_trace > 0).ravel()
    if np.sum(np.diff(invalid_indices_daq)<2) == 0:
        invalid_indices_camera = np.unique(np.concatenate((invalid_indices_daq, invalid_indices_daq+1)))
    else:
        invalid_indices_camera = invalid_indices_daq
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

trace_dict, t = utils.traces_to_dict(expt_data)
dt_frame = np.mean(np.diff(t))
fs = 1/dt_frame

os.makedirs(os.path.join(output_folder, "denoised"), exist_ok=True)
os.makedirs(os.path.join(output_folder, "downsampled"), exist_ok=True)
os.makedirs(os.path.join(output_folder, "stim_frames_removed"), exist_ok=True)
os.makedirs(os.path.join(output_folder, "corrected"), exist_ok=True)


if args.start_from_downsampled != 1:
    img, _ = images.load_image(rootdir, expt_name)
    print(img.shape)

    trimmed = img[remove_from_start:img.shape[0]-remove_from_end]
    # LP filter then downsample
    print(scale_factor)
    if scale_factor > 1:
        downsampled = images.downsample_video(trimmed, scale_factor)
        print(downsampled.shape)
        skio.imsave(os.path.join(output_folder, "downsampled", "%s.tif" % expt_name), np.round(downsampled).astype(np.uint16))
    else:
        downsampled = trimmed
else:
    downsampled = skio.imread(os.path.join(output_folder, "downsampled", "%s.tif" % expt_name))

    


if crosstalk_channel =="None":
    mean_img = downsampled.mean(axis=0)
    data_matrix = downsampled.reshape(downsampled.shape[0], -1)
    mean_trace = data_matrix.mean(axis=1)
    corr = np.matmul(data_matrix.T, mean_trace)/np.dot(mean_trace, mean_trace)
    resids = data_matrix - np.outer(mean_trace, corr)
    stim_frames_removed = mean_img + resids.reshape(downsampled.shape)
    skio.imsave(os.path.join(output_folder, "stim_frames_removed", "%s.tif" % expt_name), stim_frames_removed)
else:
    try:
        invalid_indices = generate_invalid_frame_indices(trace_dict[crosstalk_channel], dt_frame) - remove_from_start
        invalid_mask = np.zeros(downsampled.shape[0], dtype=bool)
        invalid_indices = invalid_indices[invalid_indices < downsampled.shape[0]]
        invalid_mask[invalid_indices] = True

        mean_img = downsampled[~invalid_mask].mean(axis=0)
        data_matrix = downsampled.reshape(downsampled.shape[0], -1)
        mean_trace = data_matrix.mean(axis=1)
        corr = np.matmul(data_matrix.T, mean_trace)/np.dot(mean_trace, mean_trace)
        resids = data_matrix - np.outer(mean_trace, corr)
        
        stim_frames_removed = images.interpolate_invalid_values(mean_img + resids.reshape(downsampled.shape), invalid_mask)
        skio.imsave(os.path.join(output_folder, "stim_frames_removed", "%s.tif" % expt_name), stim_frames_removed)
    except Exception:
        stim_frames_removed = downsampled
        skio.imsave(os.path.join(output_folder, "stim_frames_removed", "%s.tif" % expt_name), stim_frames_removed)



# Correct photobleach
nsamps = (int(2*fs)//2)*2 +1
print(args.invert)
pb_corrected_img = images.correct_photobleach(stim_frames_removed, mask=None, method="localmin", nsamps=nsamps, invert=args.invert)
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
denoised = sstats.denoise_svd(data_matrix_filtered, n_pcs, skewness_threshold=args.skewness_threshold)
denoised = denoised.reshape(downsampled.shape)

# Add back DC offset for the purposes of comparing noise to mean intensity
denoised += mean_img
skio.imsave(os.path.join(output_folder, "denoised", "%s.tif" % expt_name), denoised.astype(np.float32))
