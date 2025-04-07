#!/bin/bash
#SBATCH --output dsq-joblist_smooth_all_20250407-%A_%1a-%N.out
#SBATCH --array 0-5
#SBATCH --job-name dsq-joblist_smooth_all_20250407
#SBATCH --cpus-per-task=32 --mem=16G -t 00:45:00 --mail-type=ALL --mail-user=yuhan.tseng@yale.edu

# DO NOT EDIT LINE BELOW
/vast/palmer/apps/avx2/software/dSQ/1.05/dSQBatch.py --job-file /vast/palmer/home.grace/yt388/microspheres/dm_nanospheres/jobfiles/joblist_smooth_all_20250407.txt --status-dir /vast/palmer/home.grace/yt388/microspheres/dm_nanospheres/jobfiles

