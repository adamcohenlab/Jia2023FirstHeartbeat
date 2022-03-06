#!/bin/bash
#SBATCH -n 1                # Number of cores (-n)
#SBATCH -N 1                # Ensure that all cores are on one Node (-N)
#SBATCH -t 0-00:20          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p shared   # Partition to submit to
#SBATCH --mem=32000          # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid
conda init bash
conda activate bjia
python3 SpikeCounter/preprocess_widefield_stim_experiment.py $1 $2 --output_folder $3 --remove_from_start $4 --remove_from_end $5 --scale_factor $6 --zsc_threshold $7 --upper $8 --fs $9 --start_from_downsampled ${10} --expected_stim_width ${11} --fallback_mask_path ${12} --n_pcs ${13} --skewness_threshold ${14}
