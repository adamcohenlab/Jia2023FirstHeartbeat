#! /usr/bin/python3
import argparse
import os
import skimage.io as skio
import skimage.filters as filters
import skimage.morphology as morph
from spikecounter import utils
from spikecounter.segmentation import membranes
from skimage import img_as_ubyte
import numpy as np
import json
import scipy.ndimage as ndimage
import gc
import pandas as pd

# axes are T, (Z), (C), (Y), (X) 

parser = argparse.ArgumentParser()
parser.add_argument("input", help="Input file or folder")
parser.add_argument("n_timepoints", help="Number of timepoints", type=int)
parser.add_argument("--expt_name", help="Experiment name", default=None)
parser.add_argument("--reuse_masks", default=False, type=bool)
parser.add_argument("--output_folder", help="Output folder for results", default=None)

args = parser.parse_args()
input_path = args.input
n_timepoints = args.n_timepoints
reuse_masks = args.reuse_masks

expt_name = utils.extract_experiment_name(input_path)

if args.output_folder is None:
    output_folder = input_path
else:
    output_folder = args.output_folder
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

print(expt_name)
print(output_folder)

utils.write_subfolders(output_folder, ["binary_masks"])

n_binary_masks_made = len(os.listdir(os.path.join(output_folder, "binary_masks")))

strel = morph.disk(3)
strel = strel.reshape([1] + list(strel.shape))

rows = []

for timepoint in range(n_timepoints):
    ratio_raw = skio.imread(os.path.join(input_path, "raw_ratios_medfiltered", "%s_ratio_raw_t%d.tif" % (expt_name, timepoint)))
    if  n_binary_masks_made == n_timepoints and reuse_masks:
        binary_mask = skio.imread(os.path.join(output_folder, "binary_masks", "%s_binary_mask_t%d.tif" % (expt_name, timepoint)))/255
    else:
        raw_membranes = skio.imread(os.path.join(input_path, "C2", "%s_C2_t%d.tif" % (expt_name, timepoint)))
        binary_mask = membranes.raw_membrane_to_mask(raw_membranes, erode=False)
        skio.imsave(os.path.join(output_folder, "binary_masks", "%s_binary_mask_t%d.tif" % (expt_name, timepoint)), img_as_ubyte(binary_mask))
    
    binary_mask = binary_mask.astype(bool)
    fp_data = skio.imread(os.path.join(input_path, "C1", "%s_C1_t%d.tif") % (expt_name, timepoint))
    # Currently this method does not account for the surface of the embryo
    expanded_mask = ndimage.binary_dilation(binary_mask, strel)
    cytosol = np.bitwise_and(expanded_mask, np.invert(binary_mask))

    mean_mem_intensity = np.mean(fp_data[binary_mask])
    mean_cytosol_intensity = np.mean(fp_data[cytosol])
    mem_cytosol_ratio = mean_mem_intensity/mean_cytosol_intensity
    rows.append([expt_name, timepoint, "Cytosol", mean_cytosol_intensity, mean_mem_intensity/mean_cytosol_intensity])
    rows.append([expt_name, timepoint, "Membrane", mean_mem_intensity, mean_mem_intensity/mean_cytosol_intensity])
df = pd.DataFrame(rows, columns=["Experiment", "Timepoint", "Location", "Mean_Intensity", "MC_Ratio"])
df.to_csv(os.path.join(output_folder, "results.csv"), index=False)