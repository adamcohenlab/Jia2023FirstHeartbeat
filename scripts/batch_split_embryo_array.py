import sys
from pathlib import Path
import os
SPIKECOUNTER_PATH = os.getenv("SPIKECOUNTER_PATH")
sys.path.append(SPIKECOUNTER_PATH)

from spikecounter.analysis import images
from spikecounter import utils
import logging
import skimage.io as skio
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("expt_info_path")
parser.add_argument("data_folder")
parser.add_argument("--expt_index", default="None", type=str)
parser.add_argument("--block_size", default=375, type=int)
parser.add_argument("--offset", default=0.05, type=float)

args = parser.parse_args()
output_root = Path(args.data_folder, "analysis", "individual_fish_recordings")
data_folder = Path(args.data_folder)
os.makedirs(output_root, exist_ok=True)
logging.basicConfig(
    filename=output_root / "debug.log",
    level=logging.DEBUG,
    encoding="utf-8",
    filemode="w",
)

expt_info = pd.read_csv(args.expt_info_path).sort_values("start_time")

if args.expt_index == "None":
    expt_info["placeholder_index"] = ""
    expt_index = "placeholder_index"
else:
    expt_index = args.expt_index
expt_info = expt_info.reset_index().set_index(expt_index)


n_embryos = None
for idx in expt_info.index.unique():
    idx_string = "_".join([str(f) for f in utils.make_iterable(idx)])
    output_path = output_root/idx_string
    os.makedirs(output_path, exist_ok=True)
    curr_batch_info = expt_info.loc[idx]
    segmentation_mask = []
    # for i in range(2):
    for i in range(curr_batch_info.shape[0]):
        file_name = curr_batch_info["file_name"].iloc[i]
        img = skio.imread(data_folder/f"{file_name}.tif")
        ri, rp, rm, _ = images.split_embryos(
            img, offset=args.offset, block_size=args.block_size
        )

        for j in range(rp.shape[0]):
            embryo = j + 1
            embryo_directory = output_root/idx_string/f"E{embryo}"
            os.makedirs(embryo_directory, exist_ok=True)
            skio.imsave(
                embryo_directory/f"E{embryo}_{file_name}.tif", ri[j]
            )
        segmentation_mask.append(rm)

        logging.info(f"{rp.shape[0]} embryos detected for file {file_name}")
        if n_embryos is None:
            n_embryos = rp.shape[0]
        elif n_embryos != rp.shape[0]:
            logging.warning(f"Mismatch in number of embryos at file {file_name}")
    segmentation_mask = np.array(segmentation_mask, dtype=np.int32)
    skio.imsave(
        os.path.join(output_root, idx_string, "segmentation_mask.tif"),
        segmentation_mask,
    )
