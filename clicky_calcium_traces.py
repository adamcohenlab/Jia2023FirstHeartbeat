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
import bioformats
import javabridge

## Get file information
parser = argparse.ArgumentParser()
parser.add_argument("input", help="Input file or folder")
parser.add_argument("channel", type=int, help="Channel of calcium spikes", default=0)
parser.add_argument("--path_to_regions", type=str, help="path to already made clicky mask", default=None)
parser.add_argument("--n_traces", type=int, help="Number of traces to clicky", default=1)
parser.add_argument("--output_folder", help="Output folder for results", default=None)
parser.add_argument("--global_delta_t", type=bool, default=False)
parser.add_argument("--x_um", help="X spacing", default=1)
parser.add_argument("--y_um", help="Y spacing", default=1)
parser.add_argument("--z_um", help="Z spacing", default=1)

args = parser.parse_args()

input_path = args.input
files = utils.generate_file_list(input_path)
if not os.path.isdir(input_path):
    input_path = os.path.split(input_path)[0]

print(input_path)
print(files)

output_folder = utils.make_output_folder(input_path=input_path, output_path=args.output_folder, make_folder_from_file=True)


javabridge.start_vm(class_path=bioformats.JARS, run_headless=True)


def stack_traces_to_pandas(index, traces):
    slice_arrays = []
    for z in range(traces.shape[1]):
        arr = np.concatenate([np.ones((traces.shape[0],2))*np.array([index, z]), np.arange(traces.shape[0])[:,np.newaxis], traces[:, z][:,np.newaxis]], axis=1)
        slice_arrays.append(arr)
    return np.concatenate(slice_arrays, axis=0)

for file_path in files:
    n_traces = 0
    if isinstance(file_path, list):
        filename = os.path.splitext(os.path.basename(file_path[0]))[0].split("_block")[0]
        print(filename)
        file_path_1 = file_path[0]
    else:
        filename = os.path.splitext(os.path.basename(file_path))[0]
        file_path_1 = file_path
        
    try:
        with open(os.path.join(input_path, "%s_meta.xml" % filename)) as f:
            xml = f.read()
            
        metadata = bioformats.OMEXML(xml=xml)
        pixel_data = metadata.image(index=0).Pixels
        time_array = []
        for pidx in range(pixel_data.get_plane_count()):
            plane = pixel_data.Plane(index=pidx)
            if int(plane.get_TheC()) == 0:
                if args.global_delta_t:
                    time_array.append(float(plane.get_DeltaT()))
                else:
                    starttime = utils.datestring_to_epoch(metadata.image(index=0).get_AcquisitionDate())
                    time_array.append(starttime+float(plane.get_DeltaT()))
        img = utils.standardize_n_dims(imread(file_path_1), missing_dims=[1,2])
    except Exception:
        img = utils.standardize_n_dims(imread(file_path_1), missing_dims=[1,2])
        n_timepoints = img.shape[0]


    mean_fluorescence = []
    for z in range(img.shape[1]):
        for t in range(img.shape[0]):
            sl = img[t,z,args.channel,:,:]
            mean_fluorescence.append((z, t, np.mean(sl[sl > 0])))
    mean_fluorescence = pd.DataFrame(mean_fluorescence, columns=["z", "t", "mean_intensity"]).astype({"z":int, "t":float, "mean_intensity":float})
    mean_fluorescence
    # mean_fluorescence["t"] = time_array
    mean_fluorescence.to_csv(os.path.join(output_folder, "%s_whole_sample.csv" % (filename)), index=False)
    
    region_data = []
    trace_data = []
    mask_map = np.zeros((img.shape[0],img.shape[1], 1, img.shape[3], img.shape[4]), dtype=np.uint8)
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
        if os.path.isdir(args.path_to_regions):
            map_path = os.path.join(args.path_to_regions, filename + "_ROIs.tif")
        else:
            map_path = args.path_to_regions
        mask_map = imread(map_path)
        for i in range(1,np.max(mask_map)+1):
            mask = mask_map == i
            mask = np.tile(mask, (img.shape[0], img.shape[1], img.shape[2], 1, 1))
            masked_img = np.ma.array(img, mask=~mask)
            masked_img = masked_img[:,:,args.channel,:,:]
            stack_traces = stack_traces_to_pandas(i-1, masked_img.mean(axis=(2,3)))
            trace_data.append(stack_traces)
    region_data = pd.DataFrame(region_data, columns=["cent_x", "cent_y", "area", "eccentricity"])

    if isinstance(file_path, list):
        for block_path in file_path[1:]:
            print(block_path)
            img = utils.standardize_n_dims(imread(block_path), missing_dims=[1,2])
            
            for i in range(1,np.max(mask_map)+1):
                mask = mask_map == i
                mask = np.tile(mask, (img.shape[0], img.shape[1], img.shape[2], 1, 1))
                masked_img = np.ma.array(img, mask=~mask)
                masked_img = masked_img[:,:,args.channel,:,:]
                stack_traces = stack_traces_to_pandas(i-1, masked_img.mean(axis=(2,3)))
                stack_traces[:,2] += n_timepoints
                trace_data.append(stack_traces)
            n_timepoints += img.shape[0]
    
    trace_data = pd.DataFrame(np.concatenate(trace_data, axis=0), columns=["region", "z", "t", "mean_intensity"]).astype({"region": int, "z":int, "t":float, "mean_intensity":float})
    # trace_data = df.
    # print(filename)
    # trace_data["t"] = time_array*list(np.arange(n_timepoints))
    

    imsave(os.path.join(output_folder, "%s_selected_regions.tif" % (filename)), mask_map, imagej=True)
    region_data.to_csv(os.path.join(output_folder, "%s_regions.csv" % (filename)), index=False)
    trace_data.to_csv(os.path.join(output_folder, "%s_traces.csv" % (filename)), index=False)
javabridge.kill_vm()