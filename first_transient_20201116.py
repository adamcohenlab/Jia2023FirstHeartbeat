#! /usr/bin/python3
import numpy as np
from tifffile import imread, imsave
from skimage.color import rgb2gray
from skimage import img_as_ubyte
import skimage.filters as filters
import skimage.morphology as morph
import skimage.feature as feature
from skimage.segmentation import watershed
import scipy.stats as stats
import scipy.signal as signal
import scipy.ndimage as ndimage
import argparse
import os
import matplotlib.pyplot as plt
import pandas as pd
from spikecounter import utils
from spikecounter.ui import stackViewer
from spikecounter.segmentation import detect_peaks, cells

def calculate_cell_intensity_stats(data):
    min_pixel_value = 1
    nonzero_data = data[data >= min_pixel_value]
    if len(nonzero_data) == 0:
        nonzero_data = [0]
    return np.mean(data), np.median(data), np.std(data), np.max(data), np.mean(nonzero_data), np.median(nonzero_data), np.std(nonzero_data), np.max(nonzero_data)

def calculate_dF(trace, t):
    return trace[t]/np.mean(trace[trace < np.percentile(trace,20)])

## Get file information
parser = argparse.ArgumentParser()
parser.add_argument("input", help="Input file")
parser.add_argument("channel", help="Spike channel", type=int)
parser.add_argument("--mask", help="Use mask specified in file to select spikes", default=None, type=str)
parser.add_argument("--invert_mask", help="Use mask to exclude regions", default=False, type=bool)
parser.add_argument("--output_folder", help="Output folder for results", default=None)
parser.add_argument("--is_folder", help="Is the input a folder", default=False)
parser.add_argument("--analyze_max_proj", default=False, type=bool)
parser.add_argument("--dt", help="t spacing (s)", type=float, default=1)
parser.add_argument("--x_um", help="X spacing", type=float, default=1)
parser.add_argument("--y_um", help="Y spacing", type=float, default=1)
parser.add_argument("--z_um", help="Z spacing", type=float, default=1)

args = parser.parse_args()
input_path = args.input
files = utils.generate_file_list(input_path)

x_um = args.x_um
y_um = args.y_um
z_um = args.z_um
dt = args.dt
output_folder = utils.make_output_folder(input_path=input_path, output_path=args.output_folder, make_folder_from_file=True)
utils.write_subfolders(output_folder, ["traces", "datatables"])

thresholds = [0.4, 0.3, 0.15]

for idx, file_path in enumerate(files):
    filename = utils.extract_experiment_name(file_path)
    img = utils.standardize_n_dims(imread(file_path), missing_dims=[1])[:,:,args.channel,:,:]
    # img = imread(file_path)[:,:,args.channel,:,:]
    img = np.expand_dims(img, 2)
    n_timepoints = img.shape[0]

    ## (Segment cells for long timescales)
    ## Two ways of doing spike detection - peak detection from MIT by pixel or by segment, or just find a baseline for short timescales
        ## Determine baseline over time - over short timescales a median over time should be sufficient but over long timescales need to do segmentation first to account for cell motion

    # ## simple way
    # mean_values_over_time = np.tile(np.mean(img, axis=0), (n_timepoints,1,1,1))

    # Select region to analyze

    # viewer = stackViewer.HyperStackViewer(img)
    # mask = viewer.select_region_clicky()
    # if args.analyze_max_proj:
    #     img = np.expand_dims(img.max(axis=1), 1)
    # else:
    #     mask = np.tile(mask, (1, img.shape[1], 1, 1, 1))
    # imsave(os.path.join(output_folder, "%s_regions.tif" % (filename)), mask[0,0,0,:,:].astype(np.uint8), imagej=True)
    
    mask = None

    if mask is None:
        max_value = np.percentile(img, 99.999)
    else:
        max_value = np.percentile(img[mask], 99.999)
    
    # Normalize
    img = np.minimum(img.astype(np.float32)/float(max_value), 1)

    # threshold

    spikes = img > thresholds[idx]
    spike_areas = np.sum(spikes, axis=(1,2,3,4))
    masked_img = np.ma.array(img, mask=~spikes)
    mean_spike_intensity = masked_img.mean(axis=(1,2,3,4))
    # plt.plot(spike_areas)

    # viewer = stackViewer.HyperStackViewer(img)
    # viewer.view_stack()
    df = pd.DataFrame(np.concatenate([spike_areas[:,np.newaxis], mean_spike_intensity[:,np.newaxis]], axis=1), columns=["area_px", "mean_intensity"])
    df.to_csv(os.path.join(output_folder, "%s_areas.csv" % filename), index=False)