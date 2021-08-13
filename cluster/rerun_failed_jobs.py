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
failed_job_lines = []
with open(filepath, "r") as jobs_file:
    while True:
        l = jobs_file.readline()
        line_counter = 0
        line_counter2 = 0
        if "Submitted" in l:
            if str(failed_jobs[0]) in l:
                failed_job_lines.append(line_counter)
            line_counter += 1
        else:
            if line_counter2 == failed_job_lines[0]:
                print(l)
                subprocess.run(json.loads(l))
                failed_job_lines = failed_job_lines[1:]
            line_counter2 +=1