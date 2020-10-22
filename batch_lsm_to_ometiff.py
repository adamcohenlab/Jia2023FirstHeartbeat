#! /usr/bin/python3

import os
import skimage.io as skio
import numpy as np
import tifffile
import xmltodict
import bioformats
import javabridge
import argparse
from xmlformatter import Formatter


parser = argparse.ArgumentParser()
parser.add_argument("input", help="Input folder")

args = parser.parse_args()
# Locate the XML file
input_path = args.input

javabridge.start_vm(class_path=bioformats.JARS, run_headless=True)

formatter = Formatter(indent="1", indent_char="\t", preserve=["literal"])


maxproj_dir = os.path.join(input_path, "maxproj")
if not os.path.exists(maxproj_dir):
    os.mkdir(maxproj_dir)

for f in os.listdir(input_path):
    full_path = os.path.join(input_path, f)
    if not os.path.isdir(full_path):
        filename, extname = os.path.splitext(f)
        if extname == ".lsm":
            try:
                lsm = tifffile.imread(full_path)
                print(lsm.shape)
                o = bioformats.OMEXML(bioformats.get_omexml_metadata(full_path))
                with open(os.path.join(input_path, "%s_meta.xml" % filename), "w+") as xml_file:
                    xml_file.write(formatter.format_string(o.to_xml()).decode("utf-8"))
                pixel_data = o.image(index=0).Pixels
                # print(dir(pixel_data))
                # print(pixel_data.get_SizeC())
                # print(pixel_data.get_SizeT())
                # print(pixel_data.get_SizeZ())
                # exit()

                if len(lsm.shape) == 6:
                    lsm = lsm[0,:,:,:,:,:]

                if pixel_data.get_SizeZ() == 1 and pixel_data.get_SizeT() > 1:
                    lsm = np.swapaxes(lsm, 0, 1)
                    print(lsm.shape)
                z_axis = 1

                tifffile.imsave(os.path.join(input_path, "%s.tif" % filename), lsm, imagej=True, metadata={'spacing': pixel_data.PhysicalSizeZ, 'unit': 'um'}, resolution=(1/pixel_data.PhysicalSizeX, 1/pixel_data.PhysicalSizeY))
                # tifffile.imsave(os.path.join(input_path, "maxproj",  "%s_MAX.tif" % filename), lsm.max(axis=z_axis), imagej=True, resolution=(1/pixel_data.PhysicalSizeX, 1/pixel_data.PhysicalSizeY), metadata={'unit': 'um'})
            except Exception:
                continue
javabridge.kill_vm()