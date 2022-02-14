import argparse
from spikecounter.analysis import images
import skimage.io as skio
import pandas as pd
import numpy as np
import os
import pickle
import scipy.ndimage as ndi

parser = argparse.ArgumentParser()
parser.add_argument("expt_info_path")
parser.add_argument("data_folder")
parser.add_argument("n_embryos", type=int)
parser.add_argument("--start_idx", default=0, type=int)
parser.add_argument("--end_idx", default=0, type=int)
parser.add_argument("--time_remove_from_start", default=0, type=int)
parser.add_argument("--time_remove_from_end", default=0, type=int)
parser.add_argument("--f_s", default=10.2, type=float)
parser.add_argument("--opening_size", default=5, type=int)
parser.add_argument("--dilation_size", default=5, type=int)
parser.add_argument("--intensity_threshold", default=0.5, type=float)
parser.add_argument("--band_threshold", default=0.45, type=float)
parser.add_argument("--corr_threshold", default=0.9, type=float)
parser.add_argument("--band_min", default = 0.1, type=float)
parser.add_argument("--band_max", default = 2, type=float)

def link_frames(curr_labels, prev_labels, prev_coms, radius=15):
    curr_mask = curr_labels > 0
    all_curr_labels = np.arange(1,np.max(curr_labels)+1)
    curr_coms = ndi.center_of_mass(curr_mask, labels=curr_labels, index=all_curr_labels)
    curr_coms = np.array(curr_coms)
    if len(curr_coms.shape) == 2:
        curr_coms = curr_coms[:,0] + 1j*curr_coms[:,1]
        pairwise_dist = np.abs(np.subtract.outer(curr_coms, prev_coms))

        mindist_indices = np.argmin(pairwise_dist, axis=1)

        mindist = pairwise_dist[np.arange(curr_coms.shape[0]), mindist_indices]

        link_curr = np.argwhere(mindist < radius).ravel()

        link_prev = set(mindist_indices[link_curr]+1)

        link_curr = set(link_curr+1)
    elif len(curr_coms.shape) ==1:
        link_curr = set([])
        link_prev = set([])
    all_curr_labels = set(all_curr_labels)
    all_prev_labels = set(np.arange(1,prev_coms.shape[0]+1))

    new_labels = np.zeros_like(curr_labels)
    
    for label in link_curr:
        new_labels[curr_labels == label] = mindist_indices[label-1]+1
    
    unassigned_prev_labels = all_prev_labels - link_prev
    for label in unassigned_prev_labels:
        new_labels[prev_labels == label] = label
    
    unassigned_curr_labels = all_curr_labels - link_curr
    new_rois_counter = 0
    starting_idx = prev_coms.shape[0]+1
    for label in unassigned_curr_labels:
        if np.all(new_labels[curr_labels==label]==0):
            new_labels[curr_labels == label] = starting_idx + new_rois_counter
            new_rois_counter += 1
    new_mask = new_labels > 0
    new_coms =  ndi.center_of_mass(new_mask, labels=new_labels, index=np.arange(1,np.max(new_labels)+1))
    new_coms = np.array(new_coms)
    new_coms = new_coms[:,0] + 1j*new_coms[:,1]
    return new_labels, new_coms
    
    

def link_stack(stack, step=-1, radius=15):
    if step <0:
        curr_t = 1
    else:
        curr_t = stack.shape[0]-1
    
    prev_labels = stack[curr_t+step]
    prev_mask = prev_labels > 0
    
    prev_coms = ndi.center_of_mass(prev_mask, labels=prev_labels, index=np.arange(1,np.max(prev_labels)+1))
    prev_coms = np.array(prev_coms)
    prev_coms = prev_coms[:,0] + 1j*prev_coms[:,1]
    new_labels = [prev_labels]
    while curr_t >=0 and curr_t < stack.shape[0]:
        curr_labels = stack[curr_t]
        curr_labels, curr_coms = link_frames(curr_labels, prev_labels, prev_coms, radius=radius)
        prev_labels = curr_labels
        prev_coms = curr_coms
        curr_t -= step
        new_labels.append(curr_labels)
    new_labels = np.array(new_labels)
    return new_labels

def filter_by_appearances(linked_vid, unlinked_vid, threshold=1/3):
    roi_found = []
    for roi in np.arange(1, np.max(linked_vid)+1):
        roi_linked = linked_vid==roi
        found_in_unlinked = []
        for i in range(roi_linked.shape[0]):
            detected = unlinked_vid[i][roi_linked[i]]
            found_in_unlinked.append(np.any(detected>0)*(len(detected) > 0))
        roi_found.append(found_in_unlinked)
    roi_found = np.array(roi_found)
    keep = np.argwhere(np.sum(roi_found, axis=1)> threshold*linked_vid.shape[0]).ravel()+1
    
    filtered_vid = np.zeros_like(linked_vid)
    for idx, roi in enumerate(keep):
        filtered_vid[linked_vid==roi] = idx+1
    
    return filtered_vid

def closest_non_zero(arr):
    nonzeros = np.argwhere(arr!=0)
#     print(nonzeros)
#     print(nonzeros.shape)
    distances = np.abs(np.subtract.outer(np.arange(len(arr), dtype=int), nonzeros)).reshape((len(arr), -1))
#     print(distances.shape)
    min_indices = np.argmin(distances, axis=1)
    return nonzeros[min_indices]

def fill_missing(vid, threshold=100):
    roi_sizes = []
    for roi in range(1, np.max(vid)+1):
        roi_sizes.append(np.sum(vid==roi, axis=(1,2)))
    roi_sizes = np.array(roi_sizes)
    below_threshold = roi_sizes < threshold
    closest_above_threshold = np.apply_along_axis(closest_non_zero, 1, ~below_threshold).squeeze()
    filled_vid = np.zeros_like(vid)
    for i in range(1, closest_above_threshold.shape[0]+1):
        replaced_vals = vid[closest_above_threshold[i-1],:,:] == i
        filled_vid[replaced_vals] = i
    return filled_vid

args = parser.parse_args()

expt_info = pd.read_csv(args.expt_info_path).sort_values("start_time").reset_index()

filenames = expt_info["file_name"].tolist()
print(len(filenames))

if args.end_idx == 0:
    filenames = filenames[args.start_idx:len(filenames)]
else:
    filenames = filenames[args.start_idx:args.end_idx+1]

pathnames = [os.path.join(args.data_folder, "%s.tif" % f) for f in filenames]
vid, exclude_from_write = images.segment_widefield_series(pathnames, args.n_embryos, downsample_factor=1, \
                                      remove_from_start=args.time_remove_from_start, \
                                      remove_from_end=args.time_remove_from_end, \
                                      f_s=args.f_s, band_bounds=(args.band_min,args.band_max), opening_size=args.opening_size,\
                                     dilation_size=args.dilation_size, band_threshold=args.band_threshold,\
                                      intensity_threshold=args.intensity_threshold, corr_threshold=args.corr_threshold)
os.makedirs(os.path.join(args.data_folder, "analysis"), exist_ok=True)
skio.imsave(os.path.join(args.data_folder, "analysis", "unlinked_segmentation_video.tif"), vid)
linked_vid = link_stack(vid)

filtered_vid = filter_by_appearances(linked_vid, vid, threshold = 0.15)
filled_vid = fill_missing(filtered_vid)
forward_time_vid = np.flip(filled_vid, axis=0)


skio.imsave(os.path.join(args.data_folder, "analysis", "segmentation_video.tif"), forward_time_vid)
filenames = np.array(filenames, dtype=object)
filenames = filenames[~exclude_from_write]
with open(os.path.join(args.data_folder, "analysis", "usable_files.pickle"), "wb") as f:
    pickle.dump(filenames, f)

os.makedirs(os.path.join(args.data_folder, "analysis", "automasks"), exist_ok=True)
for i in range(forward_time_vid.shape[0]):
    f = filenames[i]
    skio.imsave(os.path.join(args.data_folder, "analysis", "automasks/%s_mask.tif" % f), forward_time_vid[i,:,:])