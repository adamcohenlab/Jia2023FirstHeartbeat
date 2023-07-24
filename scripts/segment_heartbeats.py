import argparse
import sys
from pathlib import Path
import os
SPIKECOUNTER_PATH = os.getenv("SPIKECOUNTER_PATH")
sys.path.append(SPIKECOUNTER_PATH)

from spikecounter.segmentation import preprocess
import skimage.io as skio
import os
from datetime import datetime
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("data_dir", type=str)
parser.add_argument("n_embryos", type=int)
parser.add_argument("output_dir", type=str)

args = parser.parse_args()

expt_info = pd.read_csv(os.path.join(args.data_dir,"experiment_data.csv")).sort_values("start_time").reset_index()
filenames = np.flip(np.array(expt_info["file_name"]))

frames = []
raw = skio.imread(os.path.join(args.data_dir, "..", "%s.tif" % filenames[0]))[:-20,:,:]
curr_labels, curr_coms = preprocess.segment_hearts(raw, args.n_embryos, prev_coms=None)
frames.append(np.copy(curr_labels))

for f in filenames[1:]:
    del raw
    print(f)
    raw = skio.imread(os.path.join(args.data_dir, "..", "%s.tif" % f))[:-20,:,:]
    curr_labels, curr_coms = preprocess.segment_hearts(raw, args.n_embryos, prev_coms=curr_coms,prev_mask_labels=curr_labels)
    frames.append(np.copy(curr_labels))

vid = np.array(frames)
vid_flipped = np.flip(vid, axis=0)
labels = np.unique(vid_flipped).tolist()

for i in range(1, vid_flipped.shape[0]):
    curr_frame_labels = np.unique(vid_flipped[i,:,:]).tolist()
    for label in labels:
        if label not in curr_frame_labels:
            print(i, label)
            curr_frame = vid_flipped[i,:,:]
            prev_frame = vid_flipped[i-1,:,:]
            curr_frame[prev_frame==label] = label
            vid_flipped[i,:,:] = curr_frame

s = vid_flipped.shape[0]
for i in range(1, vid_flipped.shape[0]):
    curr_frame_labels = np.unique(vid_flipped[s-i,:,:]).tolist()
    for label in labels:
        if label not in curr_frame_labels:
            print(i, label)
            curr_frame = vid_flipped[s-i,:,:]
            prev_frame = vid_flipped[s-i+1,:,:]
            curr_frame[prev_frame==label] = label
            vid_flipped[s-i,:,:] = curr_frame

full_mask =  vid_flipped.max(axis=0).astype(float)
full_mask[full_mask==0] = np.nan
vf = vid_flipped.astype(np.float32)
vf[vf==0] = np.nan

skio.imsave(os.path.join(args.output_dir,"ROIs.tif"), full_mask)
skio.imsave(os.path.join(args.output_dir,"segmentation_video_transp.tif"), vf)

os.makedirs(os.path.join(args.output_dir, "automasks"), exist_ok=True)

for i in range(vid_flipped.shape[0]):
    f = filenames[i]
    skio.imsave(os.path.join(args.output_dir, "automasks/%s_mask.tif" % f), vid_flipped[i,:,:])