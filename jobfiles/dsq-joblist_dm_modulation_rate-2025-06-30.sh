#!/bin/bash
#SBATCH --output dsq-joblist_dm_modulation_rate-%A_%3a-%N.out
#SBATCH --array 0-941
#SBATCH --job-name dsq-joblist_dm_modulation_rate
#SBATCH --mem=4G -t 00:15:00 --mail-type=ALL --mail-user=yuhan.tseng@yale.edu --partition=scavenge --requeue

# DO NOT EDIT LINE BELOW
/vast/palmer/apps/avx2/software/dSQ/1.05/dSQBatch.py --job-file /vast/palmer/home.grace/yt388/microspheres/dm_nanospheres/jobfiles/joblist_dm_modulation_rate.txt --status-dir /vast/palmer/home.grace/yt388/microspheres/dm_nanospheres/jobfiles

