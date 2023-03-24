from spikecounter import utils
import argparse
import os

## Get file information
parser = argparse.ArgumentParser()
parser.add_argument("source", help="Input file or folder")
parser.add_argument("dest")
args = parser.parse_args()
utils.transferjob(args.source, args.dest)