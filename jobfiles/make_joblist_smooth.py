import os
import numpy as np

R_um       = 0.083
datasets = ['coarse', 'coarse_extended_right']
mphi_list  = [0]

# datasets = ['coarse_extended_right']
# mphi_list  = [0.1, 0.01]

outfile = 'joblist_all_smooth_massless_250312.txt'

print(f'Writing file {outfile}')
job_file = open(outfile, "wt")
for dataset in datasets:
    for mphi in mphi_list:
        job_str = f'module load miniconda; conda activate microsphere; python ../project_dm_smooth_ana.py {mphi} {dataset}\n'
        job_file.write( job_str )

job_file.close()
