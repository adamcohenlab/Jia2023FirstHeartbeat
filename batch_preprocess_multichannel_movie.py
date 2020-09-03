#! /usr/bin/python3
import argparse
import os
import subprocess
# axes are T, (Z), (C), (Y), (X) 

parser = argparse.ArgumentParser()
parser.add_argument("input", help="Input folder")
parser.add_argument("--output_folder", help="Output folder for results", default=None)

args = parser.parse_args()
# Locate the XML file
input_path = args.input

for p in os.listdir(input_path):
    full_path = os.path.join(input_path, p)
    filename, ext = os.path.splitext(p)
    if not os.path.isdir(full_path) and ext == ".tif":
        command = ["preprocess_multichannel_movie.py", full_path]
        subprocess.run(command)
        print(command)