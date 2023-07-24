#!/bin/bash
#SBATCH -n 1                # Number of cores (-n)
#SBATCH -N 1                # Ensure that all cores are on one Node (-N)
#SBATCH -t 0-00:10          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p shared   # Partition to submit to
#SBATCH --mem=4000          # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o job_files/myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e job_files/myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid
conda init bash
conda activate  $CONDA_ENV
python3 $SPIKECOUNTER_PATH/scripts/clicky_calcium_traces_nojvm.py $1 0 --path_to_regions $2 --output_folder $3
