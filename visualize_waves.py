#! /usr/bin/python3
import numpy as np
from skimage.io import imread, imsave
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

def calculate_cell_intensity_stats(data):
    min_pixel_value = 1
    nonzero_data = data[data >= min_pixel_value]
    if len(nonzero_data) == 0:
        nonzero_data = [0]
    return np.mean(data), np.median(data), np.std(data), np.max(data), np.mean(nonzero_data), np.median(nonzero_data), np.std(nonzero_data), np.max(nonzero_data)

## Get file information
parser = argparse.ArgumentParser()
parser.add_argument("input", help="Input file or folder")
parser.add_argument("--output_folder", help="Output folder for results", default=None)
parser.add_argument("--is_folder", help="Is the input a folder", default=False)
parser.add_argument("--x_um", help="X spacing", default=1)
parser.add_argument("--y_um", help="Y spacing", default=1)
parser.add_argument("--z_um", help="Z spacing", default=1)

args = parser.parse_args()
input_path = args.input
filename = os.path.splitext(os.path.basename(input_path))[0]
if args.output_folder is None:
    output_folder = os.path.dirname(input_path)
else:
    output_folder = args.output_folder
if not os.path.exists(output_folder):
    os.mkdir(output_folder)
x_um = args.x_um
y_um = args.y_um
z_um = args.z_um

img = imread(input_path).astype(np.int16)
n_timepoints = img.shape[0]
print(n_timepoints)
print(np.min(img))
print(np.max(img))

for t in range(n_timepoints-1):
    diff = np.zeros([n_timepoints-1 - t] + list(img.shape[1:]))
    for t2 in range(t+1, n_timepoints):
        diff[t2-t-1,:,:,:] = img[t2,:,:,:] - img[t,:,:,:]
        # print("%d, %d" %(t, t2))
    diff = np.maximum(diff, 0)
    print(np.min(diff))
    print(np.max(diff))
    imsave(os.path.join(output_folder, "%s_difference_%d.tif" % (filename, t)), diff.astype(np.uint8))

diff = np.zeros([n_timepoints-1] + list(img.shape[1:]))
for t in range(n_timepoints-1):
    diff[t,:,:,:] = img[t+1,:,:,:] - img[t,:,:,:]
diff = np.maximum(diff, 0)
imsave(os.path.join(output_folder, "%s_difference_adj.tif" % (filename)), diff.astype(np.uint8))
