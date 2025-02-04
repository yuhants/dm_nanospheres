import os
import numpy as np

# sphere = 'sphere_20250103'
sphere = 'sphere_20241202'

datasets = ['coarse', 'fine_left', 'veryfine_bottom', 'fine_side']
mphi_lists  = [[0.01, 0.1, 1, 10], [0.01, 0.1, 1, 10], [0.01, 0.1, 1], [0.01, 0.1]]

outfile = 'joblist_nlls_sphere_20241202_all.txt'

job_file = open(outfile, "wt")
print(f'Writing file {outfile}')
for i, dataset in enumerate(datasets):
    for mphi in mphi_lists[i]:
        data_dir = '/home/yt388/microspheres/dm_nanospheres/data_processed'
        outfile = f'{data_dir}/profile_nlls/{sphere}/profile_nlls_{sphere}_{mphi:.0e}_{dataset}.npz'
        # if( os.path.isfile(outfile) ):
        #     print("Skipping: ", outfile)
        #     continue

        job_str = f'module load miniconda; conda activate microsphere; python ../calc_profile_nlls.py {mphi} {sphere} {dataset}\n'
        job_file.write( job_str )
        
job_file.close()
