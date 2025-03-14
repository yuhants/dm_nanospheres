#!/bin/bash
#SBATCH --output dsq-joblist_nlls_extended_250314-%A_%2a-%N.out
#SBATCH --array 0-11
#SBATCH --job-name dsq-joblist_nlls_extended_250314
#SBATCH --cpus-per-task=32 --mem=16G -t 01:00:00 --mail-type=ALL --mail-user=yuhan.tseng@yale.edu

# DO NOT EDIT LINE BELOW
/vast/palmer/apps/avx2/software/dSQ/1.05/dSQBatch.py --job-file /vast/palmer/home.grace/yt388/microspheres/dm_nanospheres/jobfiles/joblist_nlls_extended_250314.txt --status-dir /vast/palmer/home.grace/yt388/microspheres/dm_nanospheres/jobfiles

