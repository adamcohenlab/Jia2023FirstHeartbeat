#!/bin/bash
#SBATCH -n 1                # Number of cores (-n)
#SBATCH -N 1                # Ensure that all cores are on one Node (-N)
#SBATCH -t 0-04:30          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p serial_requeue   # Partition to submit to
#SBATCH --mem=1200          # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid
module load matlab/R2018b-fasrc01
matlab -nojvm -nosplash -nodesktop -nodisplay -r "addpath('$SPIKECOUNTER_PATH/simulations/QIF');batch_run_LIF_white($1,$2,$3,$4,$5,$6,$7,'$8');"
