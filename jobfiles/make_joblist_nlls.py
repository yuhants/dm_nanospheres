import os
import numpy as np

tail_only = False
dm_only = False

spheres = ['sphere_20241202', 'sphere_20250103']

datasets = ['coarse', 'coarse_extended_right']
mphi_lists  = [[10, 1, 0.1, 0.01, 0], [0.1, 0.01, 0]]

outfile = 'joblist_nlls_all_250313.txt'

job_file = open(outfile, "wt")
print(f'Writing file {outfile}')
for sphere in spheres:
    for i, dataset in enumerate(datasets):
        for mphi in mphi_lists[i]:
            # data_dir = '/home/yt388/microspheres/dm_nanospheres/data_processed'
            # outfile = f'{data_dir}/profile_nlls/{sphere}/profile_nlls_{sphere}_{mphi:.0e}_{dataset}.npz'
            # if( os.path.isfile(outfile) ):
            #     print("Skipping: ", outfile)
            #     continue

            if tail_only:
                job_str = f'module load miniconda; conda activate microsphere; python ../calc_profile_nlls_tail_only.py {sphere} {mphi} {dataset}\n'
            elif dm_only:
                job_str = f'module load miniconda; conda activate microsphere; python ../calc_profile_nlls_dm_only.py {sphere} {mphi} {dataset}\n'
            else:
                job_str = f'module load miniconda; conda activate microsphere; python ../calc_profile_nlls.py {sphere} {mphi} {dataset}\n'
            job_file.write( job_str )
        
job_file.close()
