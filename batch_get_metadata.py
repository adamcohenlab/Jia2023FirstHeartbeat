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

for f in os.listdir(input_path):
    full_path = os.path.join(input_path, f)
    if not os.path.isdir(full_path):
        filename, extname = os.path.splitext(f)
        if extname == ".lsm":
            lsm = tifffile.imread(full_path)
            print(lsm.shape)
            o = bioformats.OMEXML(bioformats.get_omexml_metadata(full_path)).to_xml()
            with open(os.path.join(input_path, "%s_meta.xml" % filename), "w+") as xml_file:
                xml_file.write(formatter.format_string(o).decode("utf-8"))
javabridge.kill_vm()