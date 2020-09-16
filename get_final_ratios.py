#! /usr/bin/python3
import argparse
import os
from tifffile import imread, imsave
import skimage.filters as filters
from spikecounter.segmentation import membranes
from skimage import img_as_ubyte
import numpy as np
from spikecounter import utils
import json
import gc

# axes are T, (Z), (C), (Y), (X) 

parser = argparse.ArgumentParser()
parser.add_argument("input", help="Input file or folder")
parser.add_argument("--reuse_masks", default=False, type=bool)
parser.add_argument("--output_folder", help="Output folder for results", default=None)
args = parser.parse_args()
input_path = args.input
reuse_masks = args.reuse_masks

output_folder = utils.make_output_folder(input_path=input_path, output_path=args.output_folder)
utils.write_subfolders(output_folder, ["binary_masks", "final_ratios"])

folder_names = input_path.split("/")

files = utils.generate_file_list(input_path)

for file_path in files:
    filename = os.path.splitext(os.path.basename(file_path))[0]
    ratio_raw = imread(os.path.join(input_path, "ratios_raw", "%s_ratio_raw.tif" % filename))
    mask_found = False
    if reuse_masks:
        try:
            binary_mask = imread(os.path.join(input_path, "binary_masks", "%s_binary_mask.tif" % (filename)))/255
            mask_found = True
        except FileNotFoundError:
            pass
    if not mask_found:
        raw_membranes = utils.standardize_n_dims(imread(os.path.join(input_path, "medfiltered", "%s_medfiltered.tif" % filename)))[:,:,1,:,:]
        binary_mask = membranes.raw_membrane_to_mask(raw_membranes, erode=False)
        imsave(os.path.join(output_folder, "binary_masks", "%s_binary_mask.tif" % (filename)), img_as_ubyte(binary_mask))
        print("Mask generated for %s" % filename)
    final_ratio = ratio_raw*binary_mask
    imsave(os.path.join(output_folder, "final_ratios", "%s_ratio_final.tif"% (filename)), np.expand_dims(final_ratio, 2).astype(np.float32), imagej=True)
    