import os
import subprocess
import argparse
import skimage.io as skio
import shutil
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("rootpath", type=str)
parser.add_argument("framerate", type=float)
args = parser.parse_args()
rootpath = args.rootpath
framerate = args.framerate

for folder in os.listdir(rootpath):
    if os.path.isdir(os.path.join(rootpath, folder)) and "sta.tif" in os.listdir(os.path.join(rootpath, folder)):
        print(folder)
        os.makedirs(os.path.join(rootpath, folder, "vidframes"), exist_ok=True)
        stack = skio.imread(os.path.join(rootpath, folder, "sta.tif"))
        
        pct5 = np.percentile(stack, 5)
        pct99 = np.percentile(stack, 99.5)
        
        stack[stack < pct5] = pct5
        stack[stack > pct99] = pct99
        
        stack = (stack - pct5)/(pct99-pct5)*255
        
        y = stack.shape[1]
        x = stack.shape[2]
        
        for t in range(stack.shape[0]):
            skio.imsave(os.path.join(rootpath, folder, "vidframes/vid%04d.tif" % t), stack[t,:,:].astype(np.uint8))
        
        sh_line = ["ffmpeg", "-r", "%d" % framerate, "-f", "image2", "-i" ,os.path.join(rootpath, folder, "vidframes/vid%04d.tif"), \
                "-vcodec", "libx264", "-crf", "10", "-pix_fmt", "yuv420p", os.path.join(rootpath, folder, "vid.mp4")]
        subprocess.run(sh_line)
        
        shutil.rmtree(os.path.join(rootpath, folder, "vidframes"))
        