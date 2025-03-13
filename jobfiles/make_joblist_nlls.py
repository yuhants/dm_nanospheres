import os
import numpy as np

spheres = ['sphere_20241202', 'sphere_20250103']

datasets = ['coarse', 'coarse_extended_right']
mphi_lists  = [[0], [0]]

# datasets = ['coarse_extended_right']
# mphi_lists  = [[0.1, 0.01]]

outfile = 'joblist_nlls_massless_250312.txt'

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

            job_str = f'module load miniconda; conda activate microsphere; python ../calc_profile_nlls.py {sphere} {mphi} {dataset}\n'
            job_file.write( job_str )
        
job_file.close()
