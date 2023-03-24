#!/bin/bash
#SBATCH -n 1                # Number of cores (-n)
#SBATCH -N 1                # Ensure that all cores are on one Node (-N)
#SBATCH -t 0-00:15          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p shared   # Partition to submit to
#SBATCH --mem=64000          # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid
module load matlab/R2020b-fasrc01
matlab -nojvm -nosplash -nodesktop -nodisplay -r "addpath(genpath('/n/cohen_lab/Lab/Computer Code/2020RigControl'));addpath('/n/cohen_lab/Lab/Computer Code/Image Processing/');batch_vm_to_tiff2('$1','$2', 1, 1);"
