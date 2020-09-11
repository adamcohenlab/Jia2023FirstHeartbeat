#! /usr/bin/python3
import numpy as np
from tifffile import imread, imsave
import pandas as pd
from skimage.color import rgb2gray
from skimage import img_as_ubyte
import skimage.filters as filters
import skimage.morphology as morph
from skimage.measure import regionprops
import scipy.stats as stats
import scipy.signal as signal
import spikecounter.utils as utils
import argparse
import os
from spikecounter.ui import HyperStackViewer
import spikecounter.segmentation.preprocess as preprocess

## Get file information
parser = argparse.ArgumentParser()
parser.add_argument("input", help="Input file or folder")
parser.add_argument("channel", type=int, help="Channel of calcium spikes", default=0)
parser.add_argument("--path_to_regions", type=str, help="path to already made clicky mask", default=None)
parser.add_argument("--n_traces", type=int, help="Number of traces to clicky", default=1)
parser.add_argument("--output_folder", help="Output folder for results", default=None)
parser.add_argument("--x_um", help="X spacing", default=1)
parser.add_argument("--y_um", help="Y spacing", default=1)
parser.add_argument("--z_um", help="Z spacing", default=1)

args = parser.parse_args()

input_path = args.input
files = utils.generate_file_list(input_path)

print(files)

output_folder = utils.make_output_folder(input_path=input_path, output_path=args.output_folder, make_folder_from_file=True)


def stack_traces_to_pandas(index, traces):
    slice_arrays = []
    for z in range(traces.shape[1]):
        arr = np.concatenate([np.ones((traces.shape[0],2))*np.array([index, z]), np.arange(traces.shape[0])[:,np.newaxis], traces[:, z][:,np.newaxis]], axis=1)
        slice_arrays.append(arr)
    return np.concatenate(slice_arrays, axis=0)

for file_path in files:
    filename = os.path.splitext(os.path.basename(file_path))[0]
    img = utils.standardize_n_dims(imread(file_path))
    mean_fluorescence = []
    for z in range(img.shape[1]):
        for t in range(img.shape[0]):
            sl = img[t,z,args.channel,:,:]
            mean_fluorescence.append((z, t, np.mean(sl[sl > 0])))
    mean_fluorescence = pd.DataFrame(mean_fluorescence, columns=["z", "t", "mean_intensity"]).astype({"z":int, "t":float, "mean_intensity":float})
    mean_fluorescence.to_csv(os.path.join(output_folder, "%s_whole_sample.csv" % (filename)), index=False)
    
    region_data = []
    trace_data = []
    mask_map = np.zeros((img.shape[3], img.shape[4]), dtype=np.uint8)
    if args.path_to_regions is None:
        print(img.shape)
        for i in range(args.n_traces):
            h = HyperStackViewer(img, width=10, height=10, overlay=mask_map)
            mask = h.select_region_clicky()
            props = regionprops(mask[0,0,0,:,:].astype(np.uint8))[0]
            mask_map += (mask[0,0,0,:,:].astype(np.uint8)*(i+1))
            region_data.append([props.centroid[1], props.centroid[0], props.area, props.eccentricity])
            print(region_data)
            masked_img = np.ma.array(img, mask=~mask)
            masked_img = masked_img[:,:,args.channel,:,:]
            stack_traces = stack_traces_to_pandas(i, masked_img.mean(axis=(2,3)))
            trace_data.append(stack_traces)
    else:
        mask_map = imread(args.path_to_regions)
        for i in range(1,args.n_traces+1):
            mask = mask_map == i
            mask = np.tile(mask, (img.shape[0], img.shape[1], img.shape[2], 1, 1))
            masked_img = np.ma.array(img, mask=~mask)
            masked_img = masked_img[:,:,args.channel,:,:]
            stack_traces = stack_traces_to_pandas(i-1, masked_img.mean(axis=(2,3)))
            trace_data.append(stack_traces)

    region_data = pd.DataFrame(region_data, columns=["cent_x", "cent_y", "area", "eccentricity"])
    trace_data = pd.DataFrame(np.concatenate(trace_data, axis=0), columns=["region", "z", "t", "mean_intensity"]).astype({"region": int, "z":int, "t":float, "mean_intensity":float})
    imsave(os.path.join(output_folder, "%s_selected_regions.tif" % (filename)), mask_map, imagej=True)
    region_data.to_csv(os.path.join(output_folder, "%s_regions.csv" % (filename)), index=False)
    trace_data.to_csv(os.path.join(output_folder, "%s_traces.csv" % (filename)), index=False)
        