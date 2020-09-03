#! /usr/bin/python3
import numpy as np
import skimage.io as skio
import skimage.filters as filters
import skimage.morphology as morph
import os
from spikecounter.measurement.utils import space_average_over_time
from skimage import img_as_ubyte
import matplotlib.pyplot as plt
import pandas as pd
import argparse
plt.style.use("/mnt/d/Documents/Cohen_Lab/Code/SpikeCounter/report.mplstyle")
parser = argparse.ArgumentParser()
parser.add_argument("input", help="Input folder")
parser.add_argument("--output_folder", help="Output folder for results", default=None)


args = parser.parse_args()
# Locate the XML file
input_path = args.input

files = []
experiment_names = []
if os.path.isdir(input_path):
    for imgfile in os.listdir(os.path.join(input_path, "raw_images")):
        if os.path.splitext(imgfile)[1] == '.tif':
            files.append(os.path.join(input_path, "raw_images", imgfile))
            experiment_names.append(os.path.splitext(imgfile)[0])
else:
    raise Exception("Input path given is not a folder")

if args.output_folder is None:
    output_folder = os.path.dirname(input_path)
else:
    output_folder = args.output_folder
if not os.path.exists(output_folder):
    os.mkdir(output_folder)
if not os.path.exists(os.path.join(output_folder, "masks")):
    os.mkdir(os.path.join(output_folder, "masks"))

print(files)
print(experiment_names)

fig1, ax1 = plt.subplots(figsize=(10, 10))
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Relative intensity")

for i, experiment_name in enumerate(experiment_names):
    img = skio.imread(files[i])
    expression = img[:,0,:,:]
    print(expression.shape)
    # Generate masks
    mask = expression.max(axis=0)
    thresh = filters.threshold_otsu(mask)
    mask = mask > thresh/2
    mask = morph.binary_closing(mask, morph.disk(3))
    
    # Save masks for validation
    skio.imsave(os.path.join(output_folder, "masks", "%s_mask.tif" % experiment_name), img_as_ubyte(mask))
    response = img[:,1,:,:]
    planetable = pd.read_csv(os.path.join(input_path, "plane_tables", "%s_plane_table.csv" % experiment_name))
    times = list(planetable["TimeS"][:response.shape[0]])
    intensities = space_average_over_time(response, mask=mask.astype(int))
    intensities /= intensities[0]
    ax1.plot(times, intensities)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, "result.svg"))
plt.savefig(os.path.join(output_folder, "result.tif"))
plt.show()


# if args.output_folder is None:
#     output_folder = os.path.dirname(input_path)
# else:
#     output_folder = args.output_folder

# x = skio.imread("../../Data/20200807_optostim_io/C2-E2_stim10_during.tif", as_gray=True)
# x = x/np.max(x)
# mask = skio.imread("../../Data/20200807_optostim_io/E2_stim10_stim_mask.tif", as_gray=True)
# planetable = pd.read_csv("../../Data/20200807_optostim_io/E2_stim10_during-Image Export-05/E2_stim10_during-Image Export-05_plane_table.csv", index_col=None)
# times = list(planetable["TimeS"][:x.shape[0]])
# mask = mask/np.max(mask)
# mask = mask.astype(int)
# intensities = space_average_over_time(x, mask=mask)

# plt.plot(times, intensities)
# plt.show()