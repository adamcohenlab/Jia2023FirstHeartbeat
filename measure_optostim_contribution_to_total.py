#! /usr/bin/python3

### The purpose of this script is to check how much OptoSTIM-activated cells are contributing to total calcium elevation
import numpy as np
from tifffile import imread, imsave
import pandas as pd
from skimage.color import rgb2gray
from skimage import img_as_ubyte
import skimage.filters as filters
import skimage.morphology as morph
from skimage.measure import regionprops
import scipy.stats as stats
import scipy.signal as signal
import spikecounter.utils as utils
import argparse
import os
from spikecounter.ui import HyperStackViewer
import spikecounter.segmentation.preprocess as preprocess
import spikecounter.measurement.correlations as correlations
import matplotlib.pyplot as plt
plt.style.use("/mnt/d/Documents/Cohen_Lab/Data/report.mplstyle")

## Get file information
parser = argparse.ArgumentParser()
parser.add_argument("input", help="Input file or folder")
parser.add_argument("channel", type=int, help="Channel of calcium spikes")
parser.add_argument("--optostim", default=1, type=int)
parser.add_argument("--output_folder", help="Output folder for results", default=None)
parser.add_argument("--t_max", type=int, default=None)
parser.add_argument("--x_um", help="X spacing", default=1)
parser.add_argument("--y_um", help="Y spacing", default=1)
parser.add_argument("--z_um", help="Z spacing", default=1)

args = parser.parse_args()

input_path = args.input
files = utils.generate_file_list(input_path)

print(files)

output_folder = utils.make_output_folder(input_path=input_path, output_path=args.output_folder, make_folder_from_file=True)


def stack_traces_to_pandas(index, traces):
    slice_arrays = []
    for z in range(traces.shape[1]):
        arr = np.concatenate([np.ones((traces.shape[0],2))*np.array([index, z]), np.arange(traces.shape[0])[:,np.newaxis], traces[:, z][:,np.newaxis]], axis=1)
        slice_arrays.append(arr)
    return np.concatenate(slice_arrays, axis=0)

for file_path in files:
    filename = os.path.splitext(os.path.basename(file_path))[0]
    img = utils.standardize_n_dims(imread(file_path))
    if args.t_max is not None:
        img = img[:args.t_max, :, :, :, :]
    print(img.shape)
    # Segment OptoSTIM
    optostim_channel = img[:,:,0,:,:]
    mask = np.zeros_like(optostim_channel, dtype=bool)
    no_optostim_flag = False
    multiplier = 0.8
    if args.optostim == 1:
        for t in range(img.shape[0]):
            try:
                mask[t,:,:,:] = optostim_channel[t,:,:,:] > filters.threshold_otsu(optostim_channel[t,:,:,:])*multiplier
            except Exception:
                mask = np.zeros_like(optostim_channel, dtype=bool)
                no_optostim_flag = True
                break
    # Display max projection
    random_timepoint = np.random.randint(img.shape[0])
    plt.imshow(img[random_timepoint,:,0,:,:].max(axis=0))
    plt.imshow(mask[random_timepoint].max(axis=0), alpha=0.5)
    plt.show()
    # Count total fluorescence increase in whole embryo 
    total_fluorescence = []
    total_fluorescence_optoSTIM = []
    mean_fluorescence_optoSTIM = []
    for t in range(img.shape[0]):
        timepoint = img[t,:,args.channel,:,:]
        total_fluorescence.append(np.sum(timepoint))
        if args.optostim:
            total_fluorescence_optoSTIM.append(np.sum(timepoint[mask[t]]))
            mean_fluorescence_optoSTIM.append(np.mean(timepoint[mask[t]]))
    total_fluorescence = np.array(total_fluorescence)
    total_fluorescence_optoSTIM = np.array(total_fluorescence_optoSTIM)
    fig1, ax1 = plt.subplots(figsize=(10,6))
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Total fluorescence")

    l1 = ax1.plot(np.arange(img.shape[0])*6, total_fluorescence, label="Total")
    if not no_optostim_flag and args.optostim==1:
        ax2 = ax1.twinx()
        ax2.set_ylabel("Total OptoSTIM fluorescence")
        l2 = ax1.plot(np.arange(img.shape[0])*6, total_fluorescence - total_fluorescence_optoSTIM, label="OptoSTIM Subtracted")
        l3 = ax2.plot(np.arange(img.shape[0])*6, total_fluorescence_optoSTIM, label="OptoSTIM Cells", color="C2")
        lines = l1 + l2 + l3
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "%s_total_fluorescence.svg" % filename))
    plt.show()
    if args.optostim==1:
        # Generate image of correlation to OptoSTIM
        fig1, ax1 = plt.subplots(figsize=(10,10))
        ax1.imshow(img[0,:,args.channel,:,:].max(axis=0))
        plt.show()

        correlation = correlations.pearson_image_to_trace(img[:,:,args.channel,:,:], np.array(mean_fluorescence_optoSTIM))
        # print(correlation)
        fig1, ax1 = plt.subplots(figsize=(12,10))
        im = ax1.imshow(correlation.max(axis=0), cmap=plt.get_cmap("coolwarm"))
        cb = plt.colorbar(im)
        cb.set_label(r"$f_{opto} \cdot f_{pixel}$")
        mask_maxproj = mask[0].max(axis=0)
        x, y = np.meshgrid(np.arange(512), np.arange(512))
        ax1.contour(x, y, mask_maxproj, [0.9], linewidths=0.8, colors="k")
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "%s_pearson_correlation.svg" % filename))
        plt.show()
        imsave(os.path.join(output_folder, "%s_pearson_correlation.tif" % filename), (correlation+1).astype(np.float32), imagej=True)
        imsave(os.path.join(output_folder, "%s_mask.tif" % filename), mask.astype(np.uint8), imagej=True)