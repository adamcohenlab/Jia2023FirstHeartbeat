#! /usr/bin/python3
import argparse
import os
import skimage.io as skio
import numpy as np
import json
import gc
import skimage.morphology as morph
import skimage.filters as filters
# axes are T, (Z), (C), (Y), (X) 

parser = argparse.ArgumentParser()
parser.add_argument("input", help="Input file or folder")
parser.add_argument("--output_folder", help="Output folder for results", default=None)

args = parser.parse_args()
# Locate the XML file
input_path = args.input

folder_names = input_path.split("/")
if folder_names[-1] == "":
    expt_name = folder_names[-2]
else:
    expt_name = folder_names[-1]
expt_name = expt_name.split(".tif")[0]
print(expt_name)

if args.output_folder is None:
    output_folder = os.path.dirname(input_path)
else:
    output_folder = args.output_folder
if not os.path.exists(output_folder):
    os.mkdir(output_folder)
try:
    os.mkdir(os.path.join(output_folder, "raw_ratios_medfiltered"))
    os.mkdir(os.path.join(output_folder, "C1_medfiltered"))
    os.mkdir(os.path.join(output_folder, "C2_medfiltered"))
    os.mkdir(os.path.join(output_folder, "C1"))
    os.mkdir(os.path.join(output_folder, "C2"))

except Exception:
    pass
stack = skio.imread(input_path)
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