import os
import numpy as np

R_um       = 0.083
datasets = ['coarse_extended_alpha_right']
mphi_lists  = [[10]]

outfile = 'joblist_smooth_100mevthr_remain_250317.txt'

print(f'Writing file {outfile}')
job_file = open(outfile, "wt")
for i, dataset in enumerate(datasets):
    for mphi in mphi_lists[i]:
        job_str = f'module load miniconda; conda activate microsphere; python ../project_dm_smooth_ana.py {mphi} {dataset}\n'
        job_file.write( job_str )

job_file.close()
