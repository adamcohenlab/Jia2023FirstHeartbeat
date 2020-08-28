#! /usr/bin/python3
import numpy as np
from skimage.io import imread, imsave
from skimage.color import rgb2gray
from skimage import img_as_ubyte
import skimage.filters as filters
import skimage.morphology as morph
import scipy.stats as stats
import scipy.signal as signal
import argparse
import os

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

# Dimensions are t, z, x, y, c
c_img = imread(input_path)
img = np.zeros(c_img.shape[:-1])
n_timepoints = c_img.shape[0]
for i in range(n_timepoints):
    img[i,:,:,:] = rgb2gray(c_img[i,:,:,:,:])
# Scale to 1
img = img/np.max(img.ravel())

## Subtract background for each time frame

# define background as mean of minimum 5% of non-zero pixels (Should this be by slice?)
greater_than_zero = np.sum(img == 0, axis=(1,2,3))
flatdisk = np.zeros((1,5,5))
flatdisk[0,:,:] = 1
pct_zeros = greater_than_zero/img[0].size*100
# Subtract cutoffs
for t in range(n_timepoints):
    cutoff = np.percentile(img[t,:,:,:], pct_zeros[t] + 5*(100-pct_zeros[t])/100)
    curr_timepoint = img[t,:,:,:]
    bg_values = curr_timepoint[curr_timepoint <= cutoff]
    bg_values = bg_values[bg_values > 0]
    if len(bg_values) > 0:
        cutoff = np.mean(bg_values)

    img[t,:,:,:] = np.maximum(np.zeros(img.shape[1:]), img[t,:,:,:] - cutoff)

    ## Gaussian blur? Median filter?
    img[t,:,:,:] = filters.median(img[t,:,:,:], selem=flatdisk)

# img = signal.wiener(img, mysize=(5,1,1,1))
imsave(os.path.join(output_folder, ("%s_preprocessed.tif" % filename)), img_as_ubyte(img))
