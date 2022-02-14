#!/bin/bash
#SBATCH -n 1                # Number of cores (-n)
#SBATCH -N 1                # Ensure that all cores are on one Node (-N)
#SBATCH -t 0-04:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p test   # Partition to submit to
#SBATCH --mem=32000          # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid
conda init bash
conda activate bjia
python3 SpikeCounter/run_array_autosegmentation.py $1 $2 $3 --start_idx $4 --end_idx $5 --time_remove_from_start $6 --time_remove_from_end $7 --opening_size $8 --dilation_size $9 --intensity_threshold ${10} --band_threshold ${11} --corr_threshold ${12} --band_min ${13} --band_max ${14}
