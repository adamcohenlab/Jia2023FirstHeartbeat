#! /usr/bin/python3
import numpy as np
from tifffile import imread, imsave
from skimage.color import rgb2gray
from skimage import img_as_ubyte
from skimage import transform
import spikecounter.utils as utils
import argparse
import subprocess
import os

## Get file information
parser = argparse.ArgumentParser()
parser.add_argument("input", help="Input file or folder")
parser.add_argument("--framerate", type=int, default=12)
parser.add_argument("--channel", help="Channel to make movie", default="RGB")
parser.add_argument("--input_is_maxproj", type=bool, default=False)
parser.add_argument("--rgb_order", default="102")
parser.add_argument("--output_folder", help="Output folder for results", default=None)

args = parser.parse_args()

input_path = args.input
files = utils.generate_file_list(input_path)

print(files)
output_folder = utils.make_output_folder(input_path=input_path, output_path=args.output_folder)
utils.write_subfolders(output_folder, ["tmp"])

scale_factors = np.array([0.25, 1,1])

for file_path in files:
    filename = os.path.splitext(os.path.basename(file_path))[0]

    if args.input_is_maxproj:
        img = utils.standardize_n_dims(np.expand_dims(imread(file_path), axis=1))
    else:
        img = utils.standardize_n_dims(imread(file_path))
    img = img.max(axis=1)
    img = np.expand_dims(img, 1)
    if args.channel != "RGB":
        channel = int(args.channel)
        img = img[:,:,channel,:,:]
        img = np.expand_dims(img, 2)
    else:
        n_channels_missing = 3 - img.shape[2]
        fill = np.zeros((img.shape[0], 1, n_channels_missing, img.shape[3], img.shape[4]), dtype=img.dtype)
        img = np.concatenate((img, fill), axis=2)
        order = [int(i) for i in list(args.rgb_order)]

        # print(img[0,: np.array(order, dtype=int), :, :])
        img[:,:, np.arange(img.shape[2]), :,:] = img[:,:, np.array(order, dtype=int), :, :]
    height = img.shape[3]
    width = img.shape[4]
    height = (height//2)*2
    width = (width//2)*2

    os.mkdir(os.path.join(output_folder, "tmp", filename))

    for t in range(img.shape[0]):
        sl = np.moveaxis(img[t,0,:,:,:], 0, 2)
        sl = transform.resize(sl, (height, width))
        sl = sl*scale_factors
        imsave(os.path.join(output_folder, "tmp", filename, "%s_%04d.tif" %(filename, t)), img_as_ubyte(np.moveaxis(sl, 2,0)))
    subprocess.run(["ffmpeg", "-y", "-r", str(args.framerate), "-f", "image2", "-s", "%dx%d" %(width, height), "-start_number", str(0), "-i", os.path.join(output_folder, "tmp", filename, "%s_%%04d.tif" % filename), "-vcodec", "libx264", "-crf", str(17), "-pix_fmt", "yuv420p", os.path.join(output_folder, "%s.mp4" % filename)])

subprocess.run(["rm", "-rf", os.path.join(output_folder, "tmp")])


