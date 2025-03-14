import os
import numpy as np

R_um       = 0.083
datasets = ['coarse_extended_alpha_left']#, 'coarse_extended_alpha_right']
mphi_lists  = [[0]]#, 0.01, 0.1, 1, 10], [10]]

# datasets = ['coarse_extended_right']
# mphi_list  = [0.1, 0.01]

outfile = 'joblist_smooth_extended_massless_250314.txt'

print(f'Writing file {outfile}')
job_file = open(outfile, "wt")
for i, dataset in enumerate(datasets):
    for mphi in mphi_lists[i]:
        job_str = f'module load miniconda; conda activate microsphere; python ../project_dm_smooth_ana.py {mphi} {dataset}\n'
        job_file.write( job_str )

job_file.close()
