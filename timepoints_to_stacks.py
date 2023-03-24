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
parser.add_argument("input", help="Input folder")
parser.add_argument("n_timepoints", help="Number of timepoints", type=int)
parser.add_argument("--output_folder", help="Output folder for results", default=None)
parser.add_argument("--expt_name", help="Experiment name", default=None)

args = parser.parse_args()
input_path = args.input
n_timepoints = args.n_timepoints
output_folder = args.output_folder

folder_names = input_path.split("/")
if args.expt_name is None:
    if folder_names[-1] == "":
        expt_name = folder_names[-2]
    else:
        expt_name = folder_names[-1]
    expt_name = expt_name.split(".tif")[0]
else:
    expt_name = args.expt_name
print(expt_name)

for t in range(n_timepoints):
    curr_timepoint = skio.imread(os.path.join(input_path, "%s_t%d.tif" %(expt_name, t)))
    if t == 0:
        stack = np.zeros([n_timepoints] + list(curr_timepoint.shape), dtype=curr_timepoint.dtype)
    stack[t,:,:,:] = curr_timepoint

skio.imsave(os.path.join(output_folder, "%s_stack.tif" % expt_name), stack)