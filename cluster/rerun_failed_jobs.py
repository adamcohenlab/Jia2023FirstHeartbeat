import os
import subprocess
import argparse
import ast
from parse import *

parser = argparse.ArgumentParser()
parser.add_argument("filepath", type=str)
parser.add_argument("log_root", type=str)
parser.add_argument("--exec", type=int, default=1)
args = parser.parse_args()
filepath = args.filepath
causes = {}
# failed_jobs = []
# for p in sorted(os.listdir(".")):
#     if ".err" in p:
#         err_file = os.path.join(".", p)
#         with open(err_file) as f:
#             if 'limit' in f.read().lower():
#                 res = parse("myerrors_{:d}.err", p)
#                 failed_jobs.append(res[0])
#                 causes[res[0]] = "time"
#         with open(err_file) as f:
#             if 'mem' in f.read().lower():
#                 res = parse("myerrors_{:d}.err", p)
#                 failed_jobs.append(res[0])
#                 causes[res[0]] = "mem"
# failed_jobs = set(failed_jobs)
# print(failed_jobs)
failed_job_lines = []
successful_count = 0
line_counter = 0
line_counter2 = 0
with open(filepath, "r") as jobs_file:
    for l in jobs_file:
        res = search("job {:d}", l)
        if res is not None:
            err_file = os.path.join(args.log_root, "myerrors_%d.err" % res[0])
            failed = False
            try:
                with open(err_file) as f:
                    if 'limit' in f.read().lower():
                        failed = True
                        causes[res[0]] = "time"
                with open(err_file) as f:
                    if 'mem' in f.read().lower():
                        failed = True
                        causes[res[0]] = "mem"
            except FileNotFoundError:
                if os.path.exists(os.path.join(args.log_root, "myoutput_%d.out" % res[0])):
                    pass
                else:
                    failed = True
                    causes[res[0]] = "prolog"
            if failed:
                failed_job_lines.append(line_counter)
            else:
                successful_count += 1
                try:
                    os.remove(os.path.join(args.log_root, "myerrors_%d.err" % res[0]))
                    # os.remove(os.path.join(args.log_root, "myoutput_%d.out" % res[0]))
                except FileNotFoundError:
                    # print("Could not find error file: %s" % os.path.join(args.log_root, "myerrors_%d.err" % res[0]))
                    pass
            line_counter +=1
        elif "sbatch" in l:
            if len(failed_job_lines) == 0:
                break
            if line_counter2 == failed_job_lines[0]:
                print(l)
                failed_job_lines = failed_job_lines[1:]
                if args.exec == 1:
                    subprocess.run(ast.literal_eval(l))
            line_counter2 +=1

print(causes)
print(len(causes),"/",(len(causes)+successful_count))