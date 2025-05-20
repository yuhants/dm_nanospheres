import os
import numpy as np

R_um       = 0.083

datasets = ['coarse']
mphi_lists  = [[1]]

# datasets = ['thermalized_dm']
# mphi_lists  = [[1e-2, 1e-3, 1e-4, 1e-5]]

# datasets = ['coarse', 'coarse_extended_right']
# mphi_lists  = [[10, 1, 0.1, 0], [1, 0.1, 0]]
# datasets = ['coarse_extended_right']
# mphi_lists  = [[1, 0.1, 0]]

outfile = 'joblist_smooth_coarse_sql_proj.txt'

print(f'Writing file {outfile}')
job_file = open(outfile, "wt")
for i, dataset in enumerate(datasets):
    for mphi in mphi_lists[i]:
        job_str = f'module load miniconda; conda activate microsphere; python ../project_dm_smooth_ana.py {mphi} {dataset}\n'
        job_file.write( job_str )

job_file.close()
