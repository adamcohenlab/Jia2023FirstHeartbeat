#! /usr/bin/python3
import numpy as np
from tifffile import imread, imsave
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
from spikecounter import utils

def calculate_cell_intensity_stats(data):
    min_pixel_value = 1
    nonzero_data = data[data >= min_pixel_value]
    if len(nonzero_data) == 0:
        nonzero_data = [0]
    return np.mean(data), np.median(data), np.std(data), np.max(data), np.mean(nonzero_data), np.median(nonzero_data), np.std(nonzero_data), np.max(nonzero_data)

## Get file information
parser = argparse.ArgumentParser()
parser.add_argument("input", help="Input file")
parser.add_argument("channel", help="Spike channel", type=int)
parser.add_argument("--mask", help="Use mask specified in file to select spikes", default=None, type=str)
parser.add_argument("--invert_mask", help="Use mask to exclude regions", default=False, type=bool)
parser.add_argument("--output_folder", help="Output folder for results", default=None)
parser.add_argument("--is_folder", help="Is the input a folder", default=False)
parser.add_argument("--x_um", help="X spacing", default=1)
parser.add_argument("--y_um", help="Y spacing", default=1)
parser.add_argument("--z_um", help="Z spacing", default=1)

args = parser.parse_args()
input_path = args.input
files = utils.generate_file_list(input_path)

x_um = args.x_um
y_um = args.y_um
z_um = args.z_um
output_folder = utils.make_output_folder(input_path=input_path, output_path=args.output_folder, make_folder_from_file=True)

print(input_path)
print(files)
print(output_folder)

for file_path in files:
    filename = utils.extract_experiment_name(file_path)

    # Dimensions are t, z, c, x, y
    img = imread(file_path)[:,:,args.channel,:,:]
    print(img.shape)
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
    imsave(os.path.join(output_folder, "%s_spikes_%.2f.tif" % (filename, spike_threshold)), utils.img_to_8bit(np.expand_dims(spikes, 2)), imagej=True)
    ## Merge spikes in space if detection was done by pixel
    spikes = ndimage.binary_opening(spikes, flatdisk)
    ## Output detected spike mask
    imsave(os.path.join(output_folder, "%s_spikes_opened_%.2f.tif" % (filename, spike_threshold)), utils.img_to_8bit(np.expand_dims(spikes, 2)), imagej=True)
    print(spikes.shape)
    struct = np.zeros((1,3,3,3))

    spike_detected = np.argwhere(spikes)
    print(spike_detected.shape)



    dF = (img-mean_values_over_time)/mean_values_over_time*spikes
    dF[np.isnan(dF)] = 0
    dF[np.isinf(dF)] = 0
    dF = np.maximum(np.zeros_like(dF), dF)
    imsave(os.path.join(output_folder, "%s_dF_%.2f.tif" % (filename, spike_threshold)), np.expand_dims(dF.astype(np.float32), 2), imagej=True)

    spikes_labeled_all_times = np.zeros_like(spikes, dtype=np.uint16)
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
        try:
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
        except Exception:
            pass

    all_data = pd.concat(all_data, axis=0)
    all_data["experiment"] = filename
    all_data.to_csv(os.path.join(output_folder, "%s_data.csv" % (filename)), index=False)
    print(np.max(spikes_labeled_all_times))
    print(total_spike_count)
    print(spikes_labeled_all_times.dtype)
    imsave(os.path.join(output_folder, "%s_spikes_labeled_%.2f.tif" % (filename, spike_threshold)), spikes_labeled_all_times, imagej=True)
