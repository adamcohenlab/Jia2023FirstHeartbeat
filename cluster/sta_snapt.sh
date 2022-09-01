#!/bin/bash
#SBATCH -n 1                # Number of cores (-n)
#SBATCH -N 1                # Ensure that all cores are on one Node (-N)
#SBATCH -t 0-00:20          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p serial_requeue   # Partition to submit to
#SBATCH --mem=8000          # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid

conda init bash
conda activate bjia2
python3 SpikeCounter/sta_snapt.py $1 $2 --subfolder $3 --s $4 --n_knots $5 --sta_before_s $6 --sta_after_s $7 --normalize_height $8 --bootstrap_n $9