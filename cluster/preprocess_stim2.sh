#!/bin/bash
#SBATCH -n 1                # Number of cores (-n)
#SBATCH -N 1                # Ensure that all cores are on one Node (-N)
#SBATCH -t 0-00:20          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p serial_requeue   # Partition to submit to
#SBATCH --mem=32000          # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o job_files/myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e job_files/myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid
conda init bash
conda activate  $CONDA_ENV
python3 $SPIKECOUNTER_PATH/scripts/preprocess_stim_experiment2.py $1 $2 $3 --output_folder $4 --remove_from_start $5 --remove_from_end $6 --scale_factor $7 --start_from_downsampled $8 --n_pcs $9 --skewness_threshold ${10} --left_shoulder ${11} --right_shoulder ${12} --invert ${13} --pb_correct_method ${14} --lpad ${15} --rpad ${16} --decorr_pct ${17} --pb_correct_mask ${18} --denoise ${19} --decorrelate ${20} --bg_const ${21} --initial_subfolder ${22}