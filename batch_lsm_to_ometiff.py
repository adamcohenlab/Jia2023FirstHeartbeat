#! /usr/bin/python3

import os
import skimage.io as skio
import numpy as np
import tifffile
import xmltodict
import bioformats
import javabridge

javabridge.start_vm(class_path=bioformats.JARS, run_headless=True)


input_path = "/mnt/d/Documents/Cohen_Lab/Data/20200901_mosaic_optostim"

maxproj_dir = os.path.join(input_path, "maxproj")
if not os.path.exists(maxproj_dir):
    os.mkdir(maxproj_dir)

for f in os.listdir(input_path):
    full_path = os.path.join(input_path, f)
    if not os.path.isdir(full_path):
        filename, extname = os.path.splitext(f)
        if extname == ".lsm":
            lsm = tifffile.imread(full_path)
            print(lsm.shape)
            o = bioformats.OMEXML(bioformats.get_omexml_metadata(full_path))
            pixel_data = o.image(index=0).Pixels

            z_axis = 2 if len(lsm.shape) == 6 else 1

            tifffile.imsave(os.path.join(input_path, "%s.tif" % filename), lsm.squeeze(), imagej=True, metadata={'spacing': pixel_data.PhysicalSizeZ, 'unit': 'um'}, resolution=(1/pixel_data.PhysicalSizeX, 1/pixel_data.PhysicalSizeY))
            tifffile.imsave(os.path.join(input_path, "maxproj",  "%s_MAX.tif" % filename), lsm.max(axis=z_axis), imagej=True, resolution=(1/pixel_data.PhysicalSizeX, 1/pixel_data.PhysicalSizeY), metadata={'unit': 'um'})

javabridge.kill_vm()