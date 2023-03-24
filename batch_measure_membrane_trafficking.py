#! /usr/bin/python3
import argparse
import os
import subprocess
import pandas as pd
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
    if os.path.isdir(full_path):
        command = ["measure_membrane_trafficking.py", full_path, "1", "--reuse_masks", "1"]
        subprocess.run(command)
        print(command)
        dataframes.append(pd.read_csv(os.path.join(full_path, "results.csv")))

df = pd.concat(dataframes)
df.to_csv(os.path.join(input_path, "full_data.csv"), index=False)
