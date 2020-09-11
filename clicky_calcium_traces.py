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
from spikecounter.ui import HyperStackViewer
import spikecounter.segmentation.preprocess as preprocess

## Get file information
parser = argparse.ArgumentParser()
parser.add_argument("input", help="Input file or folder")
parser.add_argument("channel", type=int, help="Channel of calcium spikes", default=0)
parser.add_argument("--n_traces", type=int, help="Number of traces to clicky", default=1)
parser.add_argument("--output_folder", help="Output folder for results", default=None)
parser.add_argument("--x_um", help="X spacing", default=1)
parser.add_argument("--y_um", help="Y spacing", default=1)
parser.add_argument("--z_um", help="Z spacing", default=1)

args = parser.parse_args()

input_path = args.input
files = utils.generate_file_list(input_path)

print(files)

# output_folder = utils.make_output_folder(input_path=input_path, output_path=args.output_folder)

for file_path in files:
    filename = os.path.splitext(os.path.basename(file_path))[0]
    img = utils.standardize_n_dims(imread(file_path))
    print(img.shape)
    for i in range(args.n_traces):
        h = HyperStackViewer(img, width=10, height=10)
        mask = h.select_region_clicky()
        masked_img = np.ma.array(img, mask=mask)
        masked_img = masked_img[:,:,args.channel,:,:]
        stack_traces = masked_img.mean(axis=(2,3))
        print(stack_traces.shape)