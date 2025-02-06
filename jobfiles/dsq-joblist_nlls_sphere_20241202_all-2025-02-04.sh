#!/bin/bash
#SBATCH --output dsq-joblist_nlls_sphere_20241202_all-%A_%2a-%N.out
#SBATCH --array 0-12
#SBATCH --job-name dsq-joblist_nlls_sphere_20241202_all
#SBATCH --mem-per-cpu 1g --cpus-per-task 32 -t 2:00:00 --mail-type=ALL --mail-user=yuhan.tseng@yale.edu --partition=scavenge

# DO NOT EDIT LINE BELOW
/vast/palmer/apps/avx2/software/dSQ/1.05/dSQBatch.py --job-file /vast/palmer/home.grace/yt388/microspheres/dm_nanospheres/jobfiles/joblist_nlls_sphere_20241202_all.txt --status-dir /vast/palmer/home.grace/yt388/microspheres/dm_nanospheres/jobfiles

