import argparse
import subprocess
import os
import shutil
import re
import numpy as np
import time
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("rootpath", type=str)

args = parser.parse_args()
rootpath = args.rootpath


for f in os.listdir(rootpath):
    if re.match("E\d+", f):
        embryo_path = os.path.join(rootpath, f)
        im_names = []
        for fpath in os.listdir(embryo_path):
            im_name = re.search("%s_\w+(?=.tif)" % f, fpath)
            if im_name is not None:
                im_names.append(im_name.group(0))
                
        # print(im_names)
        for im_name in im_names:
            sh_line = ["sbatch", "SpikeCounter/cluster/clicky_calcium.sh", os.path.join(embryo_path, "%s.tif" % im_name), os.path.join(embryo_path, "analysis/automasks/%s_mask.tif" % im_name), os.path.join(embryo_path, "analysis")]
            print(sh_line)
            subprocess.run(sh_line)
            time.sleep(0.5)
            # break
    # break
