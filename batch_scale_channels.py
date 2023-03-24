#! /usr/bin/python3
import os
import skimage.io as skio
import numpy as np
import tifffile


input_path = "/mnt/d/Documents/Cohen_Lab/Data/20200827_axis_duplication_hcrs/stitched"

output_dir = os.path.join(input_path, "final")
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

for f in os.listdir(input_path):
    full_path = os.path.join(input_path, f)
    if not os.path.isdir(full_path):
        print(full_path)
        img = skio.imread(full_path)
        if "uninjected" not in full_path:
            img[:,:,0] = img[:,:,0] * 0.2
        img_flipped_colors = np.zeros_like(img)
        img_flipped_colors[:,:,2] = img[:,:,0]
        img_flipped_colors[:,:,1] = img[:,:,2]
        img_flipped_colors[:,:,0] = img[:,:,1]

        pct_red = np.percentile(img_flipped_colors[:,:,0], 99.999)
        pct_green = np.percentile(img_flipped_colors[:,:,1], 99.999)

        print(255/pct_red)
        print(255/pct_green)
        print(np.max(img_flipped_colors[:,:,0]))

        img_flipped_colors[:,:,0] = img_flipped_colors[:,:,0]*(255/pct_red)
        img_flipped_colors[:,:,1] = img_flipped_colors[:,:,1]*(255/pct_green)


        skio.imsave(os.path.join(output_dir, f), img_flipped_colors.astype(np.uint8))