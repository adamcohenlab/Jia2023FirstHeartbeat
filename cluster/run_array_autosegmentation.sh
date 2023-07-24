#!/bin/bash
#SBATCH -n 1                # Number of cores (-n)
#SBATCH -N 1                # Ensure that all cores are on one Node (-N)
#SBATCH -t 0-04:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p test   # Partition to submit to
#SBATCH --mem=32000          # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid
conda init bash
conda activate $CONDA_ENV
python3 $SPIKECOUNTER_PATH/scripts/run_array_autosegmentation.py $1 $2 --start_idx $3 --end_idx $4 --opening_size $5 --dilation_size $6 --block_size $7 --offset $8  --band_threshold $9 --corr_threshold ${10} --band_min ${11} --band_max ${12}
