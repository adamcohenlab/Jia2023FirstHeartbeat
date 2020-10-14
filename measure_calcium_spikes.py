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


for file_path in files:
    filename = utils.extract_experiment_name(file_path)

    # Dimensions are t, z, c, x, y
    img = np.expand_dims(imread(file_path)[:,:,args.channel,:,:], 2)
    n_timepoints = img.shape[0]

    ## (Segment cells for long timescales)
    ## Two ways of doing spike detection - peak detection from MIT by pixel or by segment, or just find a baseline for short timescales
        ## Determine baseline over time - over short timescales a median over time should be sufficient but over long timescales need to do segmentation first to account for cell motion

    # ## simple way
    # mean_values_over_time = np.tile(np.mean(img, axis=0), (n_timepoints,1,1,1))

    # Select region to analyze

    viewer = stackViewer.HyperStackViewer(np.expand_dims(img.max(axis=1), 1))
    mask = viewer.select_region_clicky()
    if args.analyze_max_proj:
        img = np.expand_dims(img.max(axis=1), 1)
    else:
        mask = np.tile(mask, (1, img.shape[1], 1, 1, 1))
    imsave(os.path.join(output_folder, "%s_regions.tif" % (filename)), mask[0,0,0,:,:].astype(np.uint8), imagej=True)
    

    if mask is None:
        max_value = np.percentile(img, 99.999)
    else:
        max_value = np.percentile(img[mask], 99.999)
    
    # Normalize
    img = np.minimum(img.astype(np.float32)/float(max_value), 1)

    # Detect peaks
    peak_mask = np.zeros_like(img, dtype=bool)
    for z in range(img.shape[1]):
        for y in range(img.shape[3]):
            for x in range(img.shape[4]):
                if mask[0,z,0,y,x]:
                    trace = img[:,z, 0, y, x]
                    trace = signal.savgol_filter(trace, 5, 3)
                    padding=5
                    med = np.median(trace)
                    thresh = 1.2*med
                    trace_padded = np.concatenate((np.ones(padding)*med, trace, np.ones(padding)*med))
                    # trace_peaks = detect_peaks.detect_peaks(trace, threshold=thresh, max_half_width=2, edge="both")
                    trace_peaks, peak_properties = signal.find_peaks(trace_padded, height=thresh, prominence=0.12, distance=1, width=(0,10))
                    if len(trace_peaks) > 0:
                        # prob_show = 0.001
                        # if np.random.rand() < prob_show:
                        #     print(trace_peaks)
                        #     print(peak_properties["left_ips"])
                        #     print(peak_properties["right_ips"])
                        #     fig1, ax1 = plt.subplots(figsize=(10,6))
                        #     ax1.set_title("%d, %d" % (x, y))
                        #     ax1.plot(trace)
                        #     ax1.scatter(trace_peaks, trace[trace_peaks], color="red")
                        #     ax1.axhline(thresh)
                        #     fig2, ax2 = plt.subplots(figsize=(10, 10))
                        #     t = trace_peaks[0]
                        #     ax2.imshow(img[t,z,0,:,:])
                        #     ax2.scatter(x,y, color="red")
                        #     plt.show()
                        for idx in range(len(trace_peaks)):
                            left = int(np.ceil(peak_properties["left_ips"][idx]-padding))
                            right = int(np.ceil(peak_properties["right_ips"][idx]-padding)) 
                            peak_mask[left:right,z,0,y,x] = 1
    
    ## Merge peaks in space

    disk_radius = 4
    flatdisk = np.zeros((1,1,1,disk_radius*2+1, disk_radius*2+1))
    flatdisk[0,0,0,:,:] = morph.disk(disk_radius)
    peak_mask = ndimage.binary_closing(peak_mask, flatdisk)
    disk_radius = 2
    flatdisk = np.zeros((1,1,1,disk_radius*2+1, disk_radius*2+1))
    flatdisk[0,0,0,:,:] = morph.disk(disk_radius)
    peak_mask = ndimage.binary_opening(peak_mask, flatdisk)
    peak_mask = ndimage.binary_closing(peak_mask, flatdisk)


    viewer = stackViewer.HyperStackViewer(img, overlay=peak_mask)
    test_point = viewer.select_points_clicky(1)
    # test_point = test_point.ravel()
    # trace = img[:,0, 0, int(np.round(test_point[1])), int(np.round(test_point[0]))]
    # trace = signal.savgol_filter(trace, 5, 3)
    # padding=5
    # med = np.median(trace)
    # thresh = 1.2*med
    # trace_padded = np.concatenate((np.ones(padding)*med, trace, np.ones(padding)*med))
    # trace_peaks, peak_properties = signal.find_peaks(trace_padded, height=thresh, prominence=0.1, distance=1, width=(0,10))
    # plt.plot(trace_padded)
    # plt.scatter(trace_peaks, trace_padded[trace_peaks], color="red")
    # plt.show()

    # Label regions
    peak_labels, n_features = cells.label_5d(peak_mask)
    imsave(os.path.join(output_folder, "%s_peaks.tif" % (filename)), peak_labels.astype(np.uint16), imagej=True)

    # plt.imshow(peak_labels.max(axis=0)[0,0,:,:], cmap=plt.cm.nipy_spectral)
    # plt.show()

    traces = {}

    # Get mean traces of each spiking region
    utils.write_subfolders(os.path.join(output_folder, "traces"), [filename])
    for peak_idx in range(1, n_features+1):
        mask = peak_labels == peak_idx
        max_t = mask.argmax(axis=0)
        spike_timepoints = np.unique(max_t)
        if len(spike_timepoints) > 2:
            print(spike_timepoints)
            quit()
        t = spike_timepoints.max()
        mask = mask[t,:,:,:,:]
        mask = np.tile(~mask, (n_timepoints, 1, 1, 1, 1))
        masked_img = np.ma.array(img, mask=mask, copy=True)
        trace = masked_img.mean(axis=(1,2,3,4))
        traces[peak_idx] = (t, trace)
        df = pd.DataFrame(np.array([np.arange(n_timepoints)*dt, trace]).T, columns=["time", "intensity"])
        df.to_csv(os.path.join(output_folder, "traces", filename, "trace%d.csv" % peak_idx), index=False)

    ## Output spike statistics - size, centroid, magnitude
    spike_coords = ndimage.center_of_mass(peak_mask, labels=peak_labels, index = np.arange(1, n_features+1))
    # print(spike_coords)
    spike_coords = pd.DataFrame(spike_coords, columns=["t", "z", "c", "y", "x"])
    spike_coords["t_index"] = spike_coords["t"]
    spike_coords["t"] = spike_coords["t"] * dt
    spike_coords["z"] = spike_coords["z"] * z_um
    spike_coords["x"] = spike_coords["x"] * x_um
    spike_coords["y"] = spike_coords["y"] * y_um
    del spike_coords["c"]

    dF = []
    for peak_idx in range(1, n_features+1):
        t, trace = traces[peak_idx]
        dF.append(calculate_dF(trace, t))
        print(dF[-1])
    dF = pd.DataFrame(np.array([np.arange(1, n_features+1), np.array(dF)]).T, columns=["spike", "dF"])

    spike_volumes = ndimage.labeled_comprehension(peak_labels, peak_labels, np.arange(1, n_features+1), lambda x: x.size*x_um*y_um*z_um, float, 0)
    # print(spike_volumes)
    spike_volumes = pd.DataFrame(spike_volumes, columns=["volume"])
    spike_intensity_statistics = ndimage.labeled_comprehension(img, peak_labels, np.arange(1, n_features+1), calculate_cell_intensity_stats, (float,8) , None)
    param = "intensity"
    spike_intensity_statistics = pd.DataFrame(spike_intensity_statistics, columns=["%s_mean" %param, "%s_median" %param, "%s_std" %param, "%s_max" %param, "%s_mean_nonzero" %param, "%s_median_nonzero" %param, "%s_std_nonzero" %param, "%s_max_nonzero" %param])


    all_data = pd.concat([dF, spike_coords, spike_volumes, spike_intensity_statistics], axis=1)
    all_data["experiment"] = filename
    all_data.to_csv(os.path.join(output_folder, "datatables", "%s_data.csv" % (filename)), index=False)
