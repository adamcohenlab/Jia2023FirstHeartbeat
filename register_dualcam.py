#! /usr/bin/python3
import argparse
import os
import numpy as np
from skimage import transform
import skimage.io as skio

from spikecounter.analysis import images


parser = argparse.ArgumentParser()
parser.add_argument("rootdir", help="Root directory")
parser.add_argument("expt_name", help="Experiment Name")
parser.add_argument("tform_path", type=str)
parser.add_argument("--subfolder", default="", type=str)
parser.add_argument("--output_dir", default="None", type=str)

args = parser.parse_args()
rootdir = args.rootdir
expt_name = args.expt_name
subfolder = args.subfolder
tform_path = args.tform_path
output_dir = args.output_dir

datadir = os.path.join(rootdir, subfolder)

if output_dir == "None":
    output_dir = datadir

# Load list of images
imgs, expt_data = images.load_image(datadir, expt_name)
print(imgs[0].shape, imgs[0].dtype)
print(imgs[1].shape, imgs[0].dtype)
tform = transform.SimilarityTransform(matrix=np.load(tform_path)["tform"])

# Get information about the ROI positions and sizes relative to the full frame
cam1_max_size = expt_data["cameras"][0]["max_size"]
cam1_offset = expt_data["cameras"][0]["roi"][[0,2]]

cam2_max_size = expt_data["cameras"][1]["max_size"]
cam2_offset = expt_data["cameras"][1]["roi"][[0,2]]

# Apply the transform to the video
new_vid2 = np.zeros_like(imgs[0])
for i in range(imgs[0].shape[0]):
    full_frame_cam2 = np.zeros((cam2_max_size, cam2_max_size), dtype=imgs[1].dtype)
    full_frame_cam2[cam2_offset[0]:cam2_offset[0]+imgs[1].shape[1], cam2_offset[1]:cam2_offset[1]+imgs[1].shape[2]] = imgs[1][i]
    full_frame_cam1 = transform.warp(full_frame_cam2, tform.inverse, output_shape=(cam1_max_size, cam1_max_size))
    new_vid2[i] = full_frame_cam1[cam1_offset[0]:cam1_offset[0]+imgs[0].shape[1], cam1_offset[1]:cam1_offset[1]+imgs[0].shape[2]]

registered_dualcam = np.stack((imgs[0], new_vid2), axis=1)
skio.imsave(os.path.join(output_dir, f"{expt_name}_registered_dualcam.tif"), registered_dualcam)
