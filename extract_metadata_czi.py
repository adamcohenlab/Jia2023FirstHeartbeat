#! /usr/bin/python3
import argparse
import xml.etree.ElementTree as ET
import re
import os
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("input", help="Input file or folder")
parser.add_argument("--output_folder", help="Output folder for results", default=None)

args = parser.parse_args()
# Locate the XML file
input_path = args.input
if os.path.isdir(input_path):
    filelist = os.path.listdir(input_path)
    
else:
    files = [args.input]

if os.path.splitext(os.path.basename(input_path))[1] != ".xml":
    raise Exception("XML input could not be found")

filename = os.path.splitext(os.path.basename(input_path))[0]
if args.output_folder is None:
    output_folder = os.path.dirname(input_path)
else:
    output_folder = args.output_folder

# Now parse the XML file:
tree = ET.parse(input_path)
root = tree.getroot()
tuple_list = []

for image in root:
    # Later fix this to account for scenes (Note need to record separately for real data because this doesn't seem to be accounted for in metadata)
    filename = image.find('Filename').text
    bounds = image.find('Bounds')
    relative_z = float(image.find("Z").text)
    t = image.find("T")
    if t is None:
        t = 0.0
    else:
        t = float(t.text)
    channel = int(bounds.get("StartC")) 
    t_index = int(bounds.get("StartT"))

    tup = (filename, channel, t_index, t, relative_z)
    tuple_list.append(tup)

df = pd.DataFrame(tuple_list, columns = ["Filename", "Channel", "TimeIndex", "TimeS", "RelZ"])
df.to_csv(os.path.join(output_folder, "%s_plane_table.csv" % expt_name), index=False)