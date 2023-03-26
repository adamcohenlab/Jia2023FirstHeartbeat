#! /usr/bin/python3
import argparse
import os
import numpy as np
from skimage import transform
import skimage.io as skio

from spikecounter.analysis import images

def bounds_to_bbox(bounds):
    bbox = np.array([[bounds[0,0], bounds[0,1]],\
                     [bounds[0,0], bounds[1,1]],\
                     [bounds[1,0], bounds[0,1]],\
                     [bounds[1,0], bounds[1,1]]])
    return bbox


parser = argparse.ArgumentParser()
parser.add_argument("rootdir", help="Root directory")
parser.add_argument("tform_path", type=str)
parser.add_argument("--subfolder", default="", type=str)
parser.add_argument("--output_dir", default="None", type=str)

args = parser.parse_args()
rootdir = args.rootdir
subfolder = args.subfolder
tform_path = args.tform_path
output_dir = args.output_dir

datadir = os.path.join(rootdir, subfolder)
tform = transform.SimilarityTransform(matrix=np.load(tform_path)["tform"])

for d in os.listdir(datadir):
    if os.path.isdir(os.path.join(datadir, d)) and os.path.exists(os.path.join(datadir, d, "output_data_py.mat")):
        expt_name = d
        print(expt_name)
        if output_dir == "None":
            output_dir = os.path.join(datadir, expt_name)

        # Load list of images
        imgs, expt_data = images.load_image(datadir, expt_name)
        print(imgs[0].shape, imgs[0].dtype)
        print(imgs[1].shape, imgs[0].dtype)

        # Get information about the ROI positions and sizes relative to the full frame
        cam1_max_size = int(expt_data["cameras"][0]["max_size"])
        cam1_offset = expt_data["cameras"][0]["roi"][[2,0]].astype(int)

        cam2_max_size = int(expt_data["cameras"][1]["max_size"])
        cam2_offset = expt_data["cameras"][1]["roi"][[2,0]].astype(int)

        cam1_bounds = np.array([expt_data["cameras"][0]["roi"][[2,0]], \
                    expt_data["cameras"][0]["roi"][[2,0]] + \
                    expt_data["cameras"][0]["roi"][[3,1]] ])

        cam2_bounds = np.array([expt_data["cameras"][1]["roi"][[2,0]], \
                    expt_data["cameras"][1]["roi"][[2,0]] + \
                    expt_data["cameras"][1]["roi"][[3,1]] ])

        cam2_bbox = bounds_to_bbox(cam2_bounds)
        cam1_bbox = bounds_to_bbox(cam1_bounds)

        cam2_bbox_cam1_space = tform(np.flip(cam2_bbox, axis=1)) -\
                                np.flip(cam1_offset)
        im1_minx, im1_miny = np.ceil(np.array([cam2_bbox_cam1_space.min(axis=0), \
                                    np.array([0,0])]).max(axis=0)).astype(int)
        im1_maxx, im1_maxy = np.floor(np.array([cam2_bbox_cam1_space.max(axis=0), \
                                    np.flip(imgs[0].shape[1:])]).min(axis=0)).astype(int)

        # Apply the transform to the video
        new_vid2 = np.zeros_like(imgs[0])
        for i in range(imgs[1].shape[0]):
            full_frame_cam2 = np.zeros((cam2_max_size, cam2_max_size), dtype=imgs[1].dtype)
            full_frame_cam2[cam2_offset[0]:cam2_offset[0]+imgs[1].shape[1], cam2_offset[1]:cam2_offset[1]+imgs[1].shape[2]] = imgs[1][i]
            full_frame_cam1 = transform.warp(full_frame_cam2, tform.inverse, output_shape=(cam1_max_size, cam1_max_size), preserve_range=True).astype(imgs[0].dtype)
            new_vid2[i] = full_frame_cam1[cam1_offset[0]:cam1_offset[0]+imgs[0].shape[1], cam1_offset[1]:cam1_offset[1]+imgs[0].shape[2]]
        new_vid2 = new_vid2[:,im1_miny:im1_maxy,im1_minx:im1_maxx]

        registered_dualcam = np.stack((imgs[0][:,im1_miny:im1_maxy,im1_minx:im1_maxx], new_vid2), axis=1)
        print(registered_dualcam.shape)
        skio.imsave(os.path.join(output_dir, f"{expt_name}_registered_dualcam.tif"), registered_dualcam)

