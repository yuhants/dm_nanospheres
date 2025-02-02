#!/bin/bash
#SBATCH --output dsq-joblist_smooth_coarse_0_01-%A_%4a-%N.out
#SBATCH --array 0-4522
#SBATCH --job-name dsq-joblist_smooth_coarse_0_01
#SBATCH --mem-per-cpu 8g --cpus-per-task 1 -t 01:00:00 --mail-type=ALL --mail-user=yuhan.tseng@yale.edu --partition=day

# DO NOT EDIT LINE BELOW
/vast/palmer/apps/avx2/software/dSQ/1.05/dSQBatch.py --job-file /vast/palmer/home.grace/yt388/microspheres/dm_nanospheres/jobfiles/joblist_smooth_coarse_0_01.txt --status-dir /vast/palmer/home.grace/yt388/microspheres/dm_nanospheres/jobfiles

