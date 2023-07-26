#!/bin/bash
#SBATCH -n 1                # Number of cores (-n)
#SBATCH -N 1                # Ensure that all cores are on one Node (-N)
#SBATCH -t 0-04:30          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p serial_requeue   # Partition to submit to
#SBATCH --mem=4000          # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o job_files/myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e job_files/myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid
conda init bash
conda activate bjia2
python3 $SPIKECOUNTER_PATH/simulations/MorrisLecar/single_sim.py $1 0 0 $2 0 $3 --n_repeats $4 --state_path ${5} --burn_in 0