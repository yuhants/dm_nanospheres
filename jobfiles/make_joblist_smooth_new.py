import os
import numpy as np

R_um       = 0.083
mphi_list  = [10, 1, 0.1, 0.01]
dataset = 'coarse'

outfile = 'joblist_smooth_coarse_all.txt'

print(f'Writing file {outfile}')
job_file = open(outfile, "wt")
for mphi in mphi_list:
    job_str = f'module load miniconda; conda activate microsphere; python ../project_dm_smooth.py {mphi} coarse\n'
    job_file.write( job_str )

job_file.close()
