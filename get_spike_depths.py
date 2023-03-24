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
import skfmm

parser = argparse.ArgumentParser()
parser.add_argument("spikes", help="Spike data")
parser.add_argument("img_input", help="Input file or folder")
parser.add_argument("--output_folder", help="Output folder for results", default=None)
parser.add_argument("--is_folder", help="Is the input a folder", default=False)
parser.add_argument("--x_um", help="X spacing", type=float, default=1)
parser.add_argument("--y_um", help="Y spacing", type=float, default=1)
parser.add_argument("--z_um", help="Z spacing", type=float, default=1)

args = parser.parse_args()
spikes = pd.read_csv(args.spikes, index_col=False)
input_path = args.img_input
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

c_img = imread(input_path)
img = np.zeros(c_img.shape[:-1])
n_timepoints = c_img.shape[0]
for i in range(n_timepoints):
    img[i,:,:,:] = rgb2gray(c_img[i,:,:,:,:])
binary_mask = img > 0
binary_mask = binary_mask.astype(bool)
print(binary_mask.shape)

flatdisk = np.zeros((1,3,5,5))
flatdisk[0,1,:,:] = morph.disk(2)
flatdisk[0,0,1:4,1:4] = 1
flatdisk[0,1,1:4,1:4] = 1 
binary_mask = ndimage.binary_closing(binary_mask, flatdisk, border_value=1)

flatdisk2 = np.zeros((1,1,7,7))
flatdisk2[0,0,:,:] = morph.disk(3)
binary_mask = ndimage.binary_opening(binary_mask, flatdisk2)

strel3 = np.zeros((1,3,7,7))
strel3[0,1,:,:] = morph.disk(3)
strel3[0,0,1:6,1:6] = 1
strel3[0,1,1:6,1:6] = 1
binary_mask = ndimage.binary_closing(binary_mask, strel3, iterations=2)
binary_mask = ndimage.binary_closing(binary_mask, strel3, border_value=1)
binary_mask[:,0,:,:] = binary_mask[:,2,:,:]
binary_mask[:,1,:,:] = binary_mask[:,2,:,:]
# binary_mask[23,0,:,:] = binary_mask[23,3,:,:]
# binary_mask[23,1,:,:] = binary_mask[23,3,:,:]
# binary_mask[23,2,:,:] = binary_mask[23,3,:,:]
imsave(os.path.join(output_folder, "%s_mask_closed.tif" % filename), img_as_ubyte(binary_mask))
binary_mask = binary_mask.astype(np.uint8)
for t in range(binary_mask.shape[0]):
    for z in range(binary_mask.shape[1]):
        binary_mask[t,z,:,:] = morph.area_closing(binary_mask[t,z,:,:], area_threshold=600)
        binary_mask[t,z,:,:] = morph.flood_fill(binary_mask[t,z,:,:], (256, 256), True)
        binary_mask[t,z,:,:] = morph.area_closing(binary_mask[t,z,:,:], area_threshold=1000)
binary_mask = binary_mask.astype(bool)
imsave(os.path.join(output_folder, "%s_embryo_filled.tif" % filename), img_as_ubyte(binary_mask))

distance_over_time = np.zeros_like(binary_mask).astype(np.float32)
for t in range(binary_mask.shape[0]):
    distance_over_time[t,:,:,:] = skfmm.distance(binary_mask[t,:,:,:], dx = [z_um, y_um, x_um])
imsave(os.path.join(output_folder, "%s_distance_to_surface.tif" % filename), distance_over_time)

distance_to_surface = []

for _, spike in spikes.iterrows():
    distance_to_surface.append(distance_over_time[int(spike["t"]), int(np.round(spike["z"])), int(np.round(spike["y"])), int(np.round(spike["x"]))])
spikes["distance_to_surface"] = distance_to_surface
spikes.to_csv(args.spikes, index=False)