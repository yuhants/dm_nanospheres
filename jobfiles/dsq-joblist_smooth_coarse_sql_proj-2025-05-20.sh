#!/bin/bash
#SBATCH --output dsq-joblist_smooth_coarse_sql_proj-%A_%1a-%N.out
#SBATCH --array 0
#SBATCH --job-name dsq-joblist_smooth_coarse_sql_proj
#SBATCH --cpus-per-task=4 --mem=1G -t 01:00:00 --mail-type=ALL --mail-user=yuhan.tseng@yale.edu --partition=scavenge --requeue

# DO NOT EDIT LINE BELOW
/vast/palmer/apps/avx2/software/dSQ/1.05/dSQBatch.py --job-file /vast/palmer/home.grace/yt388/microspheres/dm_nanospheres/jobfiles/joblist_smooth_coarse_sql_proj.txt --status-dir /vast/palmer/home.grace/yt388/microspheres/dm_nanospheres/jobfiles

