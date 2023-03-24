#! /usr/bin/python3
import argparse
import os
from tifffile import imread, imsave
import numpy as np
import json
import gc
import scipy.ndimage as ndimage
from spikecounter import utils
import skimage.morphology as morph
import skimage.filters as filters
import spikecounter.segmentation.preprocess as preprocess
# axes are T, (Z), (C), (Y), (X) 

parser = argparse.ArgumentParser()
parser.add_argument("input", help="Input file")
parser.add_argument("--output_folder", help="Output folder for results", default=None)
args = parser.parse_args()

input_path = args.input
files = utils.generate_file_list(input_path)
output_folder = utils.make_output_folder(input_path=input_path, output_path=args.output_folder)
utils.write_subfolders(output_folder, ["medfiltered", "ratios_raw"])



strel = morph.disk(2)
strel = strel.reshape([1,1,1] + list(strel.shape))

for file_path in files:
    filename = os.path.splitext(os.path.basename(file_path))[0]
    stack = utils.standardize_n_dims(imread(file_path))
    # Subtract background
    background_subtracted = preprocess.subtract_background(stack.astype(np.float32))
    print("Background subtracted")
    
    #Median filter to remove shot noise
    medfiltered = ndimage.median_filter(background_subtracted, footprint=strel).astype(np.uint8)
    print("Median filtered")

    imsave(os.path.join(output_folder, "medfiltered", "%s_medfiltered.tif" % (filename)), medfiltered, imagej=True)
    # Take ratio of green to red
    ratio = medfiltered[:,:,0,:,:]/medfiltered[:,:,1,:,:]
    del medfiltered
    gc.collect()

    ratio[np.isnan(ratio)] = 0
    ratio[np.isinf(ratio)] = 0
    ratio = ratio.astype(np.float32)
    imsave(os.path.join(output_folder, "ratios_raw", "%s_ratio_raw.tif" % (filename)), ratio, imagej=True)
