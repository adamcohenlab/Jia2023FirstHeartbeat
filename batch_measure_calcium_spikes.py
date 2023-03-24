#! /usr/bin/python3
import argparse
import os
import subprocess
import pandas as pd
from spikecounter import utils
# axes are T, (Z), (C), (Y), (X) 

parser = argparse.ArgumentParser()
parser.add_argument("input", help="Input folder")
parser.add_argument("--output_folder", help="Output folder for results", default=None)

args = parser.parse_args()
# Locate the XML file
input_path = args.input

dataframes = []

for p in os.listdir(input_path):
    full_path = os.path.join(input_path, p)
    if not os.path.isdir(full_path) and os.path.splitext(full_path)[1] == ".tif":
        command = ["measure_calcium_spikes.py", full_path, "1"]
        print(command)
        subprocess.run(command)
        expt_name = utils.extract_experiment_name(full_path)
        dataframes.append(pd.read_csv(os.path.join(input_path, expt_name, "%s_data.csv" % expt_name)))

df = pd.concat(dataframes)
df.to_csv(os.path.join(input_path, "full_data.csv"), index=False)
