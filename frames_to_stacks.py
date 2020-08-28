#! /usr/bin/python3
import argparse
import os
import skimage.io as skio
import numpy as np
import json
import gc

# axes are T, (Z), (C), (Y), (X) 

parser = argparse.ArgumentParser()
parser.add_argument("input", help="Input file or folder")
parser.add_argument("--output_folder", help="Output folder for results", default=None)
parser.add_argument("--expt_name", help="Experiment name", default=None)

args = parser.parse_args()
# Locate the XML file
input_path = args.input
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

if args.output_folder is None:
    output_folder = os.path.dirname(input_path)
else:
    output_folder = args.output_folder
if not os.path.exists(output_folder):
    os.mkdir(output_folder)



def frames_to_stacks(input_folder, output_folder, expt_name, n_zs=None, n_channels=2, n_timepoints=None, width=512, height=512,block_size=200):
    if n_timepoints is None:
        ## Find dimensions from filenames here
        x = 1
    for block in range(int(n_timepoints/block_size)):
        curr_stack = np.zeros((block_size,  n_zs, n_channels, height, width), dtype=np.uint8)
        print(curr_stack.shape)
        for t in range(block_size):
            total_time = block*block_size+1 + t
            print(total_time)
            for z in range(n_zs):
                for c in range(n_channels):
                    curr_path = os.path.join(input_folder, "%s_t%03dz%02dc%d_ORG.tif" %(expt_name, total_time, z+1, c+1))
                    img = skio.imread(curr_path)
                    curr_stack[t,z,c,:,:] = img
        output_path = os.path.join(output_folder, "%s_block%d.tif" % (expt_name, block))
        skio.imsave(output_path, curr_stack)
        print(output_path)
        del curr_stack
        gc.collect()
    
    meta_dict = {"n_timepoints": n_timepoints, "n_channels": n_channels, "n_zs": n_zs, "height":height, "width":width, "block_size": block_size}

    with open(os.path.join(output_folder, "dimensions.json"), "w+") as outfile:
        json.dump(meta_dict, outfile)

    return None

frames_to_stacks(input_path, output_folder, expt_name, n_zs=31, n_channels=2, n_timepoints=300, block_size=300)