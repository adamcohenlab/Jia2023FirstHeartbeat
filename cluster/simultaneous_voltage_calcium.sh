#!/bin/bash
#SBATCH -n 1                # Number of cores (-n)
#SBATCH -N 1                # Ensure that all cores are on one Node (-N)
#SBATCH -t 0-00:20          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p serial_requeue   # Partition to submit to
#SBATCH --mem=32000          # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o job_files/myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e job_files/myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid
conda init bash
conda activate bjia2
python3 /n/home11/bjia/SpikeCounter/analyze_simultaneous_voltage_calcium.py $1 $2 --um_per_px $3 --hard_cutoff $4 --downsample_factor $5 --window_size $6 --sta_before $7 --sta_after $8