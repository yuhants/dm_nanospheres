#!/bin/bash
#SBATCH --output dsq-joblist_veryfinemx_right_0_1-%A_%3a-%N.out
#SBATCH --array 0-797
#SBATCH --job-name dsq-joblist_veryfinemx_right_0_1
#SBATCH --mem-per-cpu 1g --cpus-per-task 32 -t 01:00:00 --mail-type=ALL --mail-user=yuhan.tseng@yale.edu --partition=scavenge --requeue

# DO NOT EDIT LINE BELOW
/vast/palmer/apps/avx2/software/dSQ/1.05/dSQBatch.py --job-file /vast/palmer/home.grace/yt388/microspheres/dm_nanospheres/dm_xsec_calc/dm_xsec/jobfiles/joblist_veryfinemx_right_0_1.txt --status-dir /vast/palmer/home.grace/yt388/microspheres/dm_nanospheres/dm_xsec_calc/dm_xsec/jobfiles

