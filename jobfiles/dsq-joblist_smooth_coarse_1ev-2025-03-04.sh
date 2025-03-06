#!/bin/bash
#SBATCH --output dsq-joblist_smooth_coarse_1ev-%A_%2a-%N.out
#SBATCH --array 0-24
#SBATCH --job-name dsq-joblist_smooth_coarse_1ev
#SBATCH --mem-per-cpu 32g --cpus-per-task 1 -t 00:30:00 --mail-type=ALL --mail-user=yuhan.tseng@yale.edu --partition=scavenge

# DO NOT EDIT LINE BELOW
/vast/palmer/apps/avx2/software/dSQ/1.05/dSQBatch.py --job-file /vast/palmer/home.grace/yt388/microspheres/dm_nanospheres/jobfiles/joblist_smooth_coarse_1ev.txt --status-dir /vast/palmer/home.grace/yt388/microspheres/dm_nanospheres/jobfiles

