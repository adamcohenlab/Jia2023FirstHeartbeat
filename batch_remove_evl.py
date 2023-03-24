#! /usr/bin/python3
import numpy as np
from tifffile import imread, imsave
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
from spikecounter.ui import stackViewer
from spikecounter import utils
from spikecounter.segmentation import cells

def calculate_cell_intensity_stats(data):
    min_pixel_value = 1
    nonzero_data = data[data >= min_pixel_value]
    if len(nonzero_data) == 0:
        nonzero_data = [0]
    return np.mean(data), np.median(data), np.std(data), np.max(data), np.mean(nonzero_data), np.median(nonzero_data), np.std(nonzero_data), np.max(nonzero_data)


def fill_mask_clicky(img, mask):
    viewer = stackViewer.HyperStackViewer(img, overlay=np.expand_dims(mask,2))
    clicky_mask = viewer.select_region_clicky(snap_to_edge=True)
    if clicky_mask is None:
        return mask
    else:
        return np.logical_or(mask, clicky_mask[:,:,0,:,:])

## Get file information
parser = argparse.ArgumentParser()
parser.add_argument("input", help="Input file")
parser.add_argument("channel", help="Spike channel", type=int)
parser.add_argument("--mask", help="Use mask specified in file to select spikes", default=None, type=str)
parser.add_argument("--invert_mask", help="Use mask to exclude regions", default=False, type=bool)
parser.add_argument("--output_folder", help="Output folder for results", default=None)
parser.add_argument("--start_index", default=0, type=int)
parser.add_argument("--x_um", help="X spacing", default=1, type=float)
parser.add_argument("--y_um", help="Y spacing", default=1, type=float)
parser.add_argument("--z_um", help="Z spacing", default=1, type=float)

args = parser.parse_args()
input_path = args.input
files = utils.generate_file_list(input_path)

x_um = args.x_um
y_um = args.y_um
z_um = args.z_um
output_folder = utils.make_output_folder(input_path=input_path, output_path=args.output_folder, make_folder_from_file=True)
utils.write_subfolders(output_folder, ["embryo_masks", "evl_removed"])

for file_path in files[args.start_index:]:
    filename = utils.extract_experiment_name(file_path)

    # Dimensions are t, z, c, x, y
    img = imread(file_path)
    print(img.shape)
    n_timepoints = img.shape[0]
    mask = cells.get_whole_embryo_mask(img, close_size=3, multiplier=0.28)
    mask = fill_mask_clicky(img, mask)
    imsave(os.path.join(output_folder, "embryo_masks", "%s.tif" % filename), mask.astype(np.uint8)*255, imagej=True)
    img = cells.remove_evl(img, mask, channels=[args.channel], z_scale=z_um/x_um, depth=17)
    imsave(os.path.join(output_folder, "evl_removed", "%s.tif" % filename), img, imagej=True)
