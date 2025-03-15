#!/bin/bash
#SBATCH --output dsq-joblist_smooth_25mevthr_250314-%A_%1a-%N.out
#SBATCH --array 0-2
#SBATCH --job-name dsq-joblist_smooth_25mevthr_250314
#SBATCH --cpus-per-task=32 --mem=16G -t 01:00:00 --mail-type=ALL --mail-user=yuhan.tseng@yale.edu

# DO NOT EDIT LINE BELOW
/vast/palmer/apps/avx2/software/dSQ/1.05/dSQBatch.py --job-file /vast/palmer/home.grace/yt388/microspheres/dm_nanospheres/jobfiles/joblist_smooth_25mevthr_250314.txt --status-dir /vast/palmer/home.grace/yt388/microspheres/dm_nanospheres/jobfiles

