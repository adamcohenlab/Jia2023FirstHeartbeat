#! /usr/bin/python3
""" Perform registration of experiments with recordings on two cameras given a known transformation"""
import argparse
from pathlib import Path
import os
import time

import numpy as np
from skimage import transform
import skimage.io as skio

from spikecounter.analysis import images
from spikecounter import utils


def bounds_to_bbox(bounds):
    bbox = np.array(
        [
            [bounds[0, 0], bounds[0, 1]],
            [bounds[0, 0], bounds[1, 1]],
            [bounds[1, 0], bounds[0, 1]],
            [bounds[1, 0], bounds[1, 1]],
        ]
    )
    return bbox



parser = argparse.ArgumentParser()
parser.add_argument("rootdir", help="Root directory")
parser.add_argument("expt_name", help="Experiment Name")
parser.add_argument("tform_path", type=str)
parser.add_argument("--subfolder", default="", type=str)
parser.add_argument("--output_dir", default="None", type=str)

args = parser.parse_args()
rootdir = Path(args.rootdir)
expt_name = args.expt_name
subfolder = args.subfolder
tform_path = args.tform_path
output_dir = args.output_dir
logger = utils.initialize_logging(rootdir/expt_name/"debug.log", capture_warnings=True)
logger.info("PERFORMING REGISTRATION OF DUAL CAMERA VIDEOS USING register_dualcam.py.")
logger.info("Arguments:" + str(args))

datadir = rootdir / subfolder
if output_dir == "None":
    output_dir = datadir
else:
    output_dir = Path(output_dir)
os.makedirs(output_dir / "cam1_registered", exist_ok=True)
os.makedirs(output_dir / "cam2_registered", exist_ok=True)

# Load image from camera 1 and flip vertically (display error in RigControl)
start_time = time.time()
im1, expt_data = images.load_image(datadir, expt_name, cam_indices=0)
if expt_data is None:
    raise ValueError("No experiment metadata found.")

im1 = np.flip(im1, axis=1)
shape_im1 = im1.shape
logger.info(f"Took {time.time() - start_time:.2f} seconds to load image 1 {im1.shape} of type {im1.dtype}.")


# Get information about the ROI positions and sizes relative to the full frame
start_time = time.time()

tform_matrix = np.load(tform_path)["tform"]
tform = transform.SimilarityTransform(matrix=tform_matrix)
cam1_max_size = int(expt_data["cameras"][0]["max_size"])
cam2_max_size = int(expt_data["cameras"][1]["max_size"])
cam1_bounds = np.array(
    [
        expt_data["cameras"][0]["roi"][[2, 0]],
        expt_data["cameras"][0]["roi"][[2, 0]] + expt_data["cameras"][0]["roi"][[3, 1]],
    ]
)
cam2_bounds = np.array(
    [
        expt_data["cameras"][1]["roi"][[2, 0]],
        expt_data["cameras"][1]["roi"][[2, 0]] + expt_data["cameras"][1]["roi"][[3, 1]],
    ]
)
# Flip the bounds and offsets to account for the flipped video
cam1_bounds[:, 0] = cam1_max_size - cam1_bounds[:, 0]
cam1_offset = cam1_bounds.min(axis=0)

cam2_bounds[:, 0] = cam2_max_size - cam2_bounds[:, 0]
cam2_offset = cam2_bounds.min(axis=0)

cam2_bbox = bounds_to_bbox(cam2_bounds)
cam1_bbox = bounds_to_bbox(cam1_bounds)

cam2_bbox_cam1_space = tform(np.flip(cam2_bbox, axis=1)) - np.flip(cam1_offset)
im1_minx, im1_miny = np.ceil(
    np.array([cam2_bbox_cam1_space.min(axis=0), np.array([0, 0])]).max(axis=0)
).astype(int)
im1_maxx, im1_maxy = np.floor(
    np.array([cam2_bbox_cam1_space.max(axis=0), np.flip(im1.shape[1:])]).min(axis=0)
).astype(int)
new_vid1 = im1[:, im1_miny:im1_maxy, im1_minx:im1_maxx]
logger.info(f"Took {time.time() - start_time:.2f} seconds to transform bounding boxes and crop image 1.")

# Save cropped video for camera 1
start_time = time.time()
skio.imsave(output_dir / "cam1_registered" / f"{expt_name}.tif", new_vid1)
logger.info(f"Took {time.time() - start_time:.2f} s to save video 1.")


# Load raw video for camera 2 and flip vertically (display error in RigControl)
start_time = time.time()
im2, _ = images.load_image(datadir, expt_name, cam_indices=1, expt_metadata=expt_data)
im2 = np.flip(im2, axis=1)
logger.info(f"Took {time.time() - start_time:.2f} seconds to load image 2 {im2.shape} of type {im2.dtype}.")

# Determine valid point correspondences and perform transformation
start_time = time.time()
points_2d_cam1 = np.array(
    np.meshgrid(np.arange(cam1_max_size), np.arange(cam1_max_size), indexing="ij")
).T.reshape(-1, 2)
points_2d_cam2 = tform.inverse(points_2d_cam1)
points_2d_cam2 = np.round(points_2d_cam2).astype(int)
valid_mask = (points_2d_cam2 >= 0) & (points_2d_cam2 < cam2_max_size)
valid_mask = valid_mask[:, 0] & valid_mask[:, 1]
points_2d_cam2 = points_2d_cam2[valid_mask]
points_2d_cam1 = points_2d_cam1[valid_mask]
points_2d_cam1 -= np.flip(cam1_offset)
points_2d_cam2 -= np.flip(cam2_offset)
valid_mask = (
    (points_2d_cam1 >= 0)
    & (points_2d_cam1 < np.flip(shape_im1[1:]))
    & (points_2d_cam2 >= 0)
    & (points_2d_cam2 < np.flip(im2.shape[1:]))
)
valid_mask = valid_mask[:, 0] & valid_mask[:, 1]
points_2d_cam1 = points_2d_cam1[valid_mask]
points_2d_cam2 = points_2d_cam2[valid_mask]
points_2d_cam1 = np.fliplr(points_2d_cam1)
points_2d_cam2 = np.fliplr(points_2d_cam2)
logger.info(f"Took {time.time() - start_time:.2f} s to determine valid point correspondences.")

# Perform transformation on chunks of video 2 to avoid memory issues
start_time = time.time()
max_block_size_gb = 1
max_block_size = int(np.ceil(max_block_size_gb * 1e9 / 2 / (shape_im1[1] * shape_im1[2])))
n_iter = int(np.ceil(im2.shape[0] / max_block_size))
logger.info(f"Transforming video 2 in {n_iter} blocks of maximum size {max_block_size:d} frames.")
def register_time_blocks(im):
    n_iter_im = int(np.ceil(im.shape[0] / max_block_size))
    for n in range(n_iter_im):
        vid2_cam2 = np.moveaxis(
            im[
                int(n * max_block_size) : min(
                    int((n + 1) * max_block_size), im.shape[0]
                )
            ],
            0,
            -1,
        )
        real_block_size = vid2_cam2.shape[-1]
        vid2_cam2 = vid2_cam2.reshape(-1, vid2_cam2.shape[-1])
        vid2_cam1 = np.zeros(
            (np.prod(shape_im1[1:]), real_block_size), dtype=vid2_cam2.dtype
        )
        points_ravelled_cam1 = (
            points_2d_cam1[:, 1] + points_2d_cam1[:, 0] * shape_im1[2]
        )
        points_ravelled_cam2 = points_2d_cam2[:, 1] + points_2d_cam2[:, 0] * im.shape[2]
        vid2_cam1[points_ravelled_cam1, :] = vid2_cam2[points_ravelled_cam2, :]
        vid2_cam1 = np.moveaxis(
            vid2_cam1.reshape((shape_im1[1], shape_im1[2], real_block_size)), -1, 0
        )
        yield vid2_cam1


new_vid2 = np.concatenate([x for x in register_time_blocks(im2)], axis=0)
new_vid2 = new_vid2[:, im1_miny:im1_maxy, im1_minx:im1_maxx]
logger.info(f"Took {time.time() - start_time:.2f} s to transform video 2.")

starrt_time = time.time()
skio.imsave(output_dir / "cam2_registered" / f"{expt_name}.tif", new_vid2)
logger.info(f"Took {time.time() - start_time:.2f} s to save video 2.")
