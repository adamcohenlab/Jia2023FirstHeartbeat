#! /usr/bin/python3
import numpy as np
from skimage.io import imread, imsave
from skimage.color import rgb2gray
from skimage import img_as_ubyte
import skimage.filters as filters
import skimage.morphology as morph
import scipy.stats as stats
import scipy.signal as signal
import scipy.ndimage as ndimage
import argparse
import os
import matplotlib.pyplot as plt
import pandas as pd

def calculate_cell_intensity_stats(data):
    min_pixel_value = 1
    nonzero_data = data[data >= min_pixel_value]
    if len(nonzero_data) == 0:
        nonzero_data = [0]
    return np.mean(data), np.median(data), np.std(data), np.max(data), np.mean(nonzero_data), np.median(nonzero_data), np.std(nonzero_data), np.max(nonzero_data)

## Get file information
parser = argparse.ArgumentParser()
parser.add_argument("input", help="Input file or folder")
parser.add_argument("--output_folder", help="Output folder for results", default=None)
parser.add_argument("--is_folder", help="Is the input a folder", default=False)
parser.add_argument("--x_um", help="X spacing", default=1)
parser.add_argument("--y_um", help="Y spacing", default=1)
parser.add_argument("--z_um", help="Z spacing", default=1)

args = parser.parse_args()
input_path = args.input
filename = os.path.splitext(os.path.basename(input_path))[0]
if args.output_folder is None:
    output_folder = os.path.dirname(input_path)
else:
    output_folder = args.output_folder
if not os.path.exists(output_folder):
    os.mkdir(output_folder)
x_um = args.x_um
y_um = args.y_um
z_um = args.z_um

# Dimensions are t, z, x, y, c
img = imread(input_path)
n_timepoints = img.shape[0]

## (Segment cells for long timescales)
## Two ways of doing spike detection - peak detection from MIT by pixel or by segment, or just find a baseline for short timescales
    ## Determine baseline over time - over short timescales a median over time should be sufficient but over long timescales need to do segmentation first to account for cell motion

## simple way
mean_values_over_time = np.tile(np.mean(img, axis=0), (n_timepoints,1,1,1))
# imsave("mean_time.tif", mean_values_over_time.astype(np.uint8))
# print(mean_values_over_time.shape)

flatdisk = np.zeros((1,1,9,9))
flatdisk[0,0,:,:] = morph.disk(4)

# mean_threshold = 1
# for spike_threshold in np.linspace(1,1.5,5):
#     spikes = img > spike_threshold*mean_values_over_time
#     spikes = spikes * (mean_values_over_time > mean_threshold)

#     imsave("spikes_%.2f.tif" % spike_threshold, img_as_ubyte(spikes))
#     spikes = ndimage.binary_opening(spikes, flatdisk)
#     imsave("spikes_opened%.2f.tif" % spike_threshold, img_as_ubyte(spikes))
#     spike_detected = np.argwhere(spikes)
#     print(spike_detected.shape[0])

mean_threshold = 1
spike_threshold = 1.25
spikes = img > spike_threshold*mean_values_over_time
spikes = spikes * (mean_values_over_time > mean_threshold)
imsave(os.path.join(output_folder, "%s_spikes_%.2f.tif" % (filename, spike_threshold)), img_as_ubyte(spikes))
## Merge spikes in space if detection was done by pixel
spikes = ndimage.binary_opening(spikes, flatdisk)
## Output detected spike mask
imsave(os.path.join(output_folder, "%s_spikes_opened_%.2f.tif" % (filename, spike_threshold)), img_as_ubyte(spikes))
print(spikes.shape)
struct = np.zeros((1,3,3,3))

spike_detected = np.argwhere(spikes)
print(spike_detected.shape)

# n_traces_to_sample = 10
# for i in range(n_traces_to_sample):
#     idx_to_sample = np.random.randint(spike_detected.shape[0])
#     fig1, ax1 = plt.subplots(figsize=(10,6))
#     coords = spike_detected[idx_to_sample,:]
#     trace = img[:,coords[1], coords[2], coords[3]]
#     ax1.plot(trace)
#     ax1.set_title(str(coords))
# plt.show()

dF = (img-mean_values_over_time)/mean_values_over_time*spikes
dF[np.isnan(dF)] = 0
dF[np.isinf(dF)] = 0
dF = np.maximum(np.zeros_like(dF), dF)
imsave(os.path.join(output_folder, "%s_dF_%.2f.tif" % (filename, spike_threshold)), dF.astype(np.float32))

spikes_labeled_all_times = np.zeros_like(spikes, dtype=np.uint32)
total_spike_count = 0
all_data = []
for t in range(n_timepoints):
    spikes_labeled, _ = ndimage.label(spikes[t,:,:,:])
    spikes_labeled_copy = np.copy(spikes_labeled)
    spikes_labeled_copy[spikes_labeled_copy > 0] += total_spike_count
    spikes_labeled_all_times[t,:,:,:] = spikes_labeled_copy
    n_spikes = np.max(spikes_labeled)
    total_spike_count += n_spikes
    # print(n_spikes)

    ## Output spike statistics - size, centroid, magnitude
    spike_coords = ndimage.center_of_mass(spikes[t,:,:,:], labels=spikes_labeled, index = np.arange(1, n_spikes+1))
    # print(spike_coords)
    spike_coords = pd.DataFrame(spike_coords, columns=["z", "y", "x"])
    spike_coords["t"] = t
    spike_volumes = ndimage.labeled_comprehension(spikes_labeled, spikes_labeled, np.arange(1, n_spikes+1), lambda x: x.size*x_um*y_um*z_um, int, 0)
    # print(spike_volumes)
    spike_volumes = pd.DataFrame(spike_volumes, columns=["volume"])
    spike_intensity_statistics = ndimage.labeled_comprehension(dF[t,:,:,:], spikes_labeled, np.arange(1, n_spikes+1), calculate_cell_intensity_stats, (float,8) , None)
    gene1 = "DF"
    spike_intensity_statistics = pd.DataFrame(spike_intensity_statistics, columns=["%s_mean" %gene1, "%s_median" %gene1, "%s_std" %gene1, "%s_max" %gene1, "%s_mean_nonzero" %gene1, "%s_median_nonzero" %gene1, "%s_std_nonzero" %gene1, "%s_max_nonzero" %gene1])
    timepoint_data = pd.concat([spike_coords, spike_volumes, spike_intensity_statistics], axis=1)
    all_data.append(timepoint_data)

all_data = pd.concat(all_data, axis=0)
all_data.to_csv(os.path.join(output_folder, "%s_data.csv" % (filename)), index=False)
print(np.max(spikes_labeled_all_times))
print(total_spike_count)
imsave(os.path.join(output_folder, "%s_spikes_labeled_%.2f.tif" % (filename, spike_threshold)), spikes_labeled_all_times)
