#! /usr/bin/python3
""" Batch register experiments with dual camera using register_dualcam.py

"""
import argparse
from pathlib import Path
import subprocess
import multiprocessing
import os


parser = argparse.ArgumentParser()
parser.add_argument("rootdir", help="Root directory")
parser.add_argument("tform_path", type=str)
parser.add_argument("--subfolder", default="", type=str)
parser.add_argument("--output_dir", default="None", type=str)

args = parser.parse_args()
rootdir = Path(args.rootdir)
max_n_threads = multiprocessing.cpu_count()
print(f"Threads available: {max_n_threads}")
cmd_strings = []

for subdir in rootdir.iterdir():
    if subdir.is_dir() and (subdir/"output_data_py.mat").exists():
        cmd_string = ["python3", f"{os.getenv('SPIKECOUNTER_PATH')}/register_dualcam.py", rootdir,
                      str(subdir.name), args.tform_path, "--subfolder", args.subfolder,
                      "--output_dir", args.output_dir]
        print(cmd_string)
        cmd_strings.append(cmd_string)
pool_size = min(len(cmd_strings), max_n_threads)
print(f"pool_size: {pool_size}")
p = multiprocessing.Pool(pool_size)
p.map(subprocess.run, cmd_strings)
p.close()
