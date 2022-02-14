import os
import subprocess
import argparse
import ast
from parse import *

parser = argparse.ArgumentParser()
parser.add_argument("filepath", type=str)
args = parser.parse_args()
filepath = args.filepath
causes = []
failed_jobs = []
for p in sorted(os.listdir(".")):
    if ".err" in p:
        err_file = os.path.join(".", p)
        with open(err_file) as f:
            if 'limit' in f.read().lower():
                res = parse("myerrors_{:d}.err", p)
                failed_jobs.append(res[0])
                causes.append("time")
        with open(err_file) as f:
            if 'mem' in f.read().lower():
                res = parse("myerrors_{:d}.err", p)
                failed_jobs.append(res[0])
                causes.append("mem")
failed_jobs = set(failed_jobs)
failed_job_lines = []
print(causes)
line_counter = 0
line_counter2 = 0
with open(filepath, "r") as jobs_file:
    for l in jobs_file:
        res = search("job {:d}", l)
        if res is not None:
            if res[0] in failed_jobs:
                failed_job_lines.append(line_counter)
                failed_jobs.remove(res[0])
            line_counter +=1
        elif "sbatch" in l:
            if len(failed_job_lines) == 0:
                break 
            if line_counter2 == 0:
                print(failed_job_lines)
                print(len(failed_job_lines))
            if line_counter2 == failed_job_lines[0]:
                print(l)
                subprocess.run(ast.literal_eval(l))
                failed_job_lines = failed_job_lines[1:]
            line_counter2 +=1
