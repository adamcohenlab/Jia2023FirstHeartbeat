import os
import subprocess
import argparse
import ast
from parse import *

parser = argparse.ArgumentParser()
parser.add_argument("filepath", type=str)
args = parser.parse_args()
filepath = args.filepath

failed_jobs = []
for p in sorted(os.listdir(".")):
    if ".err" in p:
        err_file = os.path.join(".", p)
        with open(err_file) as f:
            if 'limit' in f.read().lower():
                res = parse("myerrors_{:d}.err", p)
                failed_jobs.append(res[0])
        with open(err_file) as f:
            if 'mem' in f.read().lower():
                res = parse("myerrors_{:d}.err", p)
                failed_jobs.append(res[0])
print(failed_jobs)
print(len(failed_jobs))
failed_job_lines = []
with open(filepath, "r") as jobs_file:
    line_counter = 0
    line_counter2 = 0 
    while True:
        l = jobs_file.readline()
        if "Submitted" in l:
            if len(failed_jobs) > 0 and str(failed_jobs[0]) in l:
                failed_job_lines.append(line_counter)
                failed_jobs = failed_jobs[1:] 
            line_counter += 1
        else: 
            if len(failed_job_lines) == 0:
                break 
            if line_counter2 == 0:
                print(failed_job_lines) 
            if line_counter2 == failed_job_lines[0]:
                print(l)
                subprocess.run(ast.literal_eval(l))
                failed_job_lines = failed_job_lines[1:]
            line_counter2 +=1
