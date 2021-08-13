import os
import subprocess
import argparse
import json
from parse import *

parser = argparse.ArgumentParser()
parser.add_argument("filepath", type=str)
args = parser.parse_args()
filepath = args.filepath

failed_jobs = []
for p in os.listdir("~"):
    if ".err" in p:
        err_file = os.path.join("~", p)
        with open(err_file) as f:
            if 'limit' in f.read().lower():
                res = parse("myerrors_{:d}.err", p)
                failed_jobs.append(res[0])
        with open(err_file) as f:
            if 'mem' in f.read().lower():
                res = parse("myerrors_{:d}.err", p)
                failed_jobs.append(res[0])
print(failed_jobs)
with open(filepath, "r") as jobs_file:
    while True:
        l = jobs_file.readline()
        if not l:
            break
        cmd = jobs_file.readline()
        if str(failed_jobs[0]) in l:
            subprocess.run(json.loads(cmd))
            print(cmd)
            failed_jobs = failed_jobs[1:]