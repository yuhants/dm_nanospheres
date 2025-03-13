#!/bin/bash
#SBATCH --output dsq-joblist_nlls_extended_right_250312-%A_%1a-%N.out
#SBATCH --array 0-3
#SBATCH --job-name dsq-joblist_nlls_extended_right_250312
#SBATCH --cpus-per-task=32 --mem=4G -t 02:00:00 --mail-type=ALL --mail-user=yuhan.tseng@yale.edu --partition=scavenge --requeue

# DO NOT EDIT LINE BELOW
/vast/palmer/apps/avx2/software/dSQ/1.05/dSQBatch.py --job-file /vast/palmer/home.grace/yt388/microspheres/dm_nanospheres/jobfiles/joblist_nlls_extended_right_250312.txt --status-dir /vast/palmer/home.grace/yt388/microspheres/dm_nanospheres/jobfiles

