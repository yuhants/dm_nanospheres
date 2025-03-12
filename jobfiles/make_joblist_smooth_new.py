import os
import numpy as np

R_um       = 0.083
mphi_list  = [10, 1, 0.1, 0.01]
datasets = ['fine_left']

outfile = 'joblist_smooth_all_250311.txt'

print(f'Writing file {outfile}')
job_file = open(outfile, "wt")
for dataset in datasets:
    for mphi in mphi_list:
        job_str = f'module load miniconda; conda activate microsphere; python ../project_dm_smooth_ana.py {mphi} {dataset}\n'
        job_file.write( job_str )

job_file.close()
