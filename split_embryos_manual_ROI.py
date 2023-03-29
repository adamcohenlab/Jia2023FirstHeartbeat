import argparse
from spikecounter.analysis import images
from spikecounter import utils
import logging
import skimage.io as skio
import pandas as pd
import numpy as np
import os
import pickle
import scipy.ndimage as ndi

parser = argparse.ArgumentParser()
parser.add_argument("expt_info_path")
parser.add_argument("data_folder")
parser.add_argument("roi_path")
parser.add_argument("--expt_index", default="None", type=str)
parser.add_argument("--block_size", default=375, type=int)
parser.add_argument("--offset", default=0.05, type=float)
parser.add_argument("--output_data_dir", default="None", type=str)



args = parser.parse_args()

output_data_dir = args.output_data_dir

if output_data_dir == "None":
    output_data_dir = args.data_folder
    
output_root = os.path.join(output_data_dir, "analysis", "individual_fish_recordings")
os.makedirs(output_root, exist_ok=True)
logging.basicConfig(filename=os.path.join(output_root, "debug.log"), level=logging.DEBUG, encoding="utf-8", filemode="w")

expt_info = pd.read_csv(args.expt_info_path).sort_values("start_time")
roi_mask = skio.imread(args.roi_path)


if args.expt_index == "None":
    expt_info["placeholder_index"] = ""
    expt_info = expt_info.reset_index().set_index("placeholder_index")


for idx in expt_info.index.unique():
    idx_string = "_".join([str(f) for f in utils.make_iterable(idx)])
    output_path = os.path.join(output_root, idx_string)
    os.makedirs(output_path, exist_ok=True)
    curr_batch_info = expt_info.loc[idx]
    segmentation_mask = []
    # for i in range(2):
    for i in range(curr_batch_info.shape[0]):
        file_name = curr_batch_info["file_name"].iloc[i]
        print(file_name)
        img = skio.imread(os.path.join(args.data_folder, "%s.tif" % file_name))
        ri = images.extract_bbox_images(img, roi_mask)
        
        for j in range(len(ri)):
            embryo = j+1
            embryo_directory = os.path.join(output_root, idx_string, "E%d" % embryo)
            os.makedirs(embryo_directory, exist_ok=True)
            skio.imsave(os.path.join(embryo_directory, "E%d_%s.tif" % (embryo, file_name)), ri[j])