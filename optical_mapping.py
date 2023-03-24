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
parser.add_argument("--path_to_regions", type=str, help="path to already made clicky mask", default=None)
parser.add_argument("--output_folder", help="Output folder for results", default=None)
parser.add_argument("--global_delta_t", type=bool, default=False)
parser.add_argument("--x_um", help="X spacing", default=1)
parser.add_argument("--y_um", help="Y spacing", default=1)
parser.add_argument("--z_um", help="Z spacing", default=1)

args = parser.parse_args()
args.n_traces=1

input_path = args.input
files = utils.generate_file_list(input_path)
if not os.path.isdir(input_path):
    input_path = os.path.split(input_path)[0]

print(files)

output_folder = utils.make_output_folder(input_path=input_path, output_path=args.output_folder, make_folder_from_file=True)
javabridge.start_vm(class_path=bioformats.JARS, run_headless=True)


def stack_traces_to_pandas(index, traces):
    slice_arrays = []
    for z in range(traces.shape[1]):
        arr = np.concatenate([np.ones((traces.shape[0],2))*np.array([index, z]), np.arange(traces.shape[0])[:,np.newaxis], traces[:, z][:,np.newaxis]], axis=1)
        slice_arrays.append(arr)
    return np.concatenate(slice_arrays, axis=0)

map_channel = 0
illum_channel = 0

for file_path in files:
    filename = os.path.splitext(os.path.basename(file_path))[0]
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
    except Exception:
        df = pd.read_csv(os.path.join(input_path, "E1_times.csv"))
        time_array = list(df["t"])


    img = utils.standardize_n_dims(imread(file_path), missing_dims=[1,2])
    img = np.round((img/(np.iinfo(img.dtype).max))*255).astype(np.uint8)
    print(img.dtype)
    mean_fluorescence = []
    for z in range(img.shape[1]):
        for t in range(img.shape[0]):
            sl = img[t,z,illum_channel,:,:]
            mean_fluorescence.append((z, t, np.mean(sl[sl > 0])))
    mean_fluorescence = pd.DataFrame(mean_fluorescence, columns=["z", "t", "mean_intensity"]).astype({"z":int, "t":float, "mean_intensity":float})
    mean_fluorescence
    mean_fluorescence["t"] = time_array
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
            masked_img = masked_img[:,:,map_channel,:,:]
            masked_img = masked_img.reshape(masked_img.shape[0], masked_img.shape[1], -1)
            mean = np.mean(masked_img, axis=-1)
            var = np.var(masked_img, axis=-1)
            entropy = np.zeros((masked_img.shape[0], masked_img.shape[1]))
            for t in range(masked_img.shape[0]):
                for z in range(masked_img.shape[1]):
                    hist, _ = np.histogram(masked_img[t,z,:].compressed(), density=True)
                    entropy[t,z] = stats.entropy(hist)
            mean = mean[:,0]
            var = var[:,0]
            entropy = entropy[:,0]
            df = pd.DataFrame()
            df["mean"] = mean
            df["var"] = var
            df["entropy"] = entropy
            df["t"] = time_array
            df.to_csv(os.path.join(output_folder, "%s_traces.csv" % (filename)), index=False)
    else:
        mask_map = imread(args.path_to_regions)
        for i in range(1,args.n_traces+1):
            mask = mask_map == i
            mask = np.tile(mask, (img.shape[0], img.shape[1], img.shape[2], 1, 1))
            masked_img = np.ma.array(img, mask=~mask)
            masked_img = masked_img[:,:,map_channel,:,:]
            stack_traces = stack_traces_to_pandas(i-1, masked_img.mean(axis=(2,3)))
            trace_data.append(stack_traces)

    # region_data = pd.DataFrame(region_data, columns=["cent_x", "cent_y", "area", "eccentricity"])
    # trace_data = pd.DataFrame(np.concatenate(trace_data, axis=0), columns=["region", "z", "t", "mean_intensity"]).astype({"region": int, "z":int, "t":float, "mean_intensity":float})
    # trace_data["t"] = time_array*args.n_traces

    # imsave(os.path.join(output_folder, "%s_selected_regions.tif" % (filename)), mask_map, imagej=True)
    # region_data.to_csv(os.path.join(output_folder, "%s_regions.csv" % (filename)), index=False)
    # trace_data.to_csv(os.path.join(output_folder, "%s_traces.csv" % (filename)), index=False)
javabridge.kill_vm()