#!/bin/bash
#SBATCH --output dsq-joblist_smooth_coarse_1_0_1-%A_%4a-%N.out
#SBATCH --array 0-9045
#SBATCH --job-name dsq-joblist_smooth_coarse_1_0_1
#SBATCH --mem-per-cpu 8g --cpus-per-task 1 -t 01:00:00 --mail-type=ALL --mail-user=yuhan.tseng@yale.edu --partition=day

# DO NOT EDIT LINE BELOW
/vast/palmer/apps/avx2/software/dSQ/1.05/dSQBatch.py --job-file /vast/palmer/home.grace/yt388/microspheres/dm_nanospheres/jobfiles/joblist_smooth_coarse_1_0_1.txt --status-dir /vast/palmer/home.grace/yt388/microspheres/dm_nanospheres/jobfiles

