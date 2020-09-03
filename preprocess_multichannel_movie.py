#! /usr/bin/python3
import argparse
import os
import skimage.io as skio
import numpy as np
import json
import gc
from spikecounter import utils
import skimage.morphology as morph
import skimage.filters as filters
# axes are T, (Z), (C), (Y), (X) 

parser = argparse.ArgumentParser()
parser.add_argument("input", help="Input file")
parser.add_argument("--output_folder", help="Output folder for results", default=None)

args = parser.parse_args()
input_path = args.input

expt_name = utils.extract_experiment_name(input_path)

if args.output_folder is None:
    output_folder = os.path.join(os.path.dirname(input_path), expt_name)
else:
    output_folder = args.output_folder
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

utils.write_subfolders(output_folder, ["raw_ratios_medfiltered", "C1_medfiltered", "C2_medfiltered", "C1", "C2"])
stack = utils.standardize_n_dims(skio.imread(input_path))
timepoints = stack.shape[0]


strel = morph.disk(2)

for timepoint in range(timepoints):
    C1 = stack[timepoint,:,0,:,:]
    C2 = stack[timepoint,:,1,:,:]

    C1_medfiltered = np.zeros_like(C1)
    C2_medfiltered = np.zeros_like(C2)
    for z in range(C1.shape[0]):
        C1_medfiltered[z,:,:] = filters.median(C1[z,:,:], strel)
        C2_medfiltered[z,:,:] = filters.median(C2[z,:,:], strel)

    ratio = C1_medfiltered/C2_medfiltered
    ratio[np.isnan(ratio)] = 0
    ratio[np.isinf(ratio)] = 0
    ratio = ratio.astype(np.float32)

    skio.imsave(os.path.join(output_folder, "raw_ratios_medfiltered", "%s_ratio_raw_t%d.tif") % (expt_name, timepoint), ratio)

    
    skio.imsave(os.path.join(output_folder, "C1", "%s_C1_t%d.tif") % (expt_name, timepoint), C1)
    skio.imsave(os.path.join(output_folder, "C2", "%s_C2_t%d.tif")  % (expt_name, timepoint), C2)

        
    skio.imsave(os.path.join(output_folder, "C1_medfiltered", "%s_C1_medfiltered_t%d.tif") % (expt_name, timepoint), C1_medfiltered)
    skio.imsave(os.path.join(output_folder, "C2_medfiltered", "%s_C2_medfiltered_t%d.tif")  % (expt_name, timepoint), C2_medfiltered)