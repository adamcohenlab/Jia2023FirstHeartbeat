#! /usr/bin/python3
import numpy as np
from tifffile import imread, imsave
from skimage.color import rgb2gray
from skimage import img_as_ubyte
import skimage.filters as filters
import skimage.morphology as morph
import scipy.stats as stats
import scipy.signal as signal
import spikecounter.utils as utils
import argparse
import os
import spikecounter.segmentation.preprocess as preprocess

## Get file information
parser = argparse.ArgumentParser()
parser.add_argument("input", help="Input file or folder")
parser.add_argument("channel", type=int, help="Channel of calcium spikes", default=0)
parser.add_argument("--output_folder", help="Output folder for results", default=None)
parser.add_argument("--x_um", help="X spacing", default=1)
parser.add_argument("--y_um", help="Y spacing", default=1)
parser.add_argument("--z_um", help="Z spacing", default=1)

args = parser.parse_args()

input_path = args.input
files = utils.generate_file_list(input_path)

print(files)

output_folder = utils.make_output_folder(input_path=input_path, output_path=args.output_folder)

for file_path in files:
    filename = os.path.splitext(os.path.basename(file_path))[0]

    x_um = args.x_um
    y_um = args.y_um
    z_um = args.z_um
    # Dimensions are t, z, c, x, y
    img = utils.standardize_n_dims(imread(file_path))
    normalized_image = preprocess.normalize_intensities(img, scale=255, pct=99.999)
    imsave(os.path.join(output_folder, ("%s_preprocessed.tif" % filename)), preprocess.subtract_background(normalized_image).astype(np.uint8), imagej=True)
