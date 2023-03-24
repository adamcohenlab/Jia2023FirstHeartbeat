#! /usr/bin/python3
import argparse
import os
import skimage.io as skio
import skimage.filters as filters
from spikecounter.segmentation import membranes
from skimage import img_as_ubyte
import spikecounter.utils as utils
import numpy as np
import json
import gc

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

if args.expt_name is None:
    expt_name = utils.extract_experiment_name(input_path)
else:
    expt_name = args.expt_name
print(expt_name)

if args.output_folder is None:
    output_folder = os.path.dirname(input_path)
else:
    output_folder = args.output_folder
utils.write_output_subfolders(output_folder, ["binary_masks"])

n_binary_masks_made = len(os.listdir(os.path.join(output_folder, "binary_masks")))


for timepoint in range(n_timepoints):
    # ratio_raw = skio.imread(os.path.join(input_path, "raw_ratios_medfiltered", "%s_ratio_raw_t%d.tif" % (expt_name, timepoint)))
    if  n_binary_masks_made == n_timepoints and reuse_masks:
        binary_mask = skio.imread(os.path.join(output_folder, "binary_masks", "%s_binary_mask_t%d.tif" % (expt_name, timepoint)))/255
    else:
        raw_membranes = skio.imread(os.path.join(input_path, "C1", "%s_C1_t%d.tif" % (expt_name, timepoint)))
        binary_mask = membranes.raw_membrane_to_mask(raw_membranes, erode=False)
        skio.imsave(os.path.join(output_folder, "binary_masks", "%s_binary_mask_t%d.tif" % (expt_name, timepoint)), img_as_ubyte(binary_mask))

