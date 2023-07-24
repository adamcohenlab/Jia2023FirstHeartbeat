import sys
from pathlib import Path
import os
SPIKECOUNTER_PATH = os.getenv("SPIKECOUNTER_PATH")
sys.path.append(SPIKECOUNTER_PATH)

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
parser.add_argument("--start_idx", default=0, type=int)
parser.add_argument("--end_idx", default=0, type=int)
parser.add_argument("--f_s", default=10.2, type=float)
parser.add_argument("--opening_size", default=5, type=int)
parser.add_argument("--dilation_size", default=5, type=int)
parser.add_argument("--intensity_threshold", default=0.5, type=float)
parser.add_argument("--band_threshold", default=0.45, type=float)
parser.add_argument("--corr_threshold", default="0.9", type=str)
parser.add_argument("--band_min", default = 0.1, type=float)
parser.add_argument("--band_max", default = 2, type=float)
parser.add_argument("--block_size", default=375, type=int)
parser.add_argument("--offset", default=0.01, type=float)

args = parser.parse_args()

expt_info = pd.read_csv(args.expt_info_path).sort_values("start_time").reset_index()

filenames = expt_info["file_name"].tolist()
print(len(filenames))

if args.end_idx == 0:
    filenames = filenames[args.start_idx:len(filenames)]
else:
    filenames = filenames[args.start_idx:args.end_idx+1]

pathnames = np.flip(np.array([os.path.join(args.data_folder, "%s.tif" % f) for f in filenames]))
vid, exclude_from_write = images.segment_widefield_series(pathnames, \
                                      f_s=args.f_s, band_bounds=(args.band_min,args.band_max), opening_size=args.opening_size,\
                                     dilation_size=args.dilation_size, band_threshold=args.band_threshold,\
                                      block_size=args.block_size, offset=args.offset, corr_threshold=args.corr_threshold)
os.makedirs(os.path.join(args.data_folder, "analysis"), exist_ok=True)
skio.imsave(os.path.join(args.data_folder, "analysis", "unlinked_segmentation_video.tif"), vid)
linked_vid = images.link_stack(vid)

filtered_vid = images.filter_by_appearances(linked_vid, vid, threshold = 0.15)
filled_vid = images.fill_missing_timepoints(filtered_vid)
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