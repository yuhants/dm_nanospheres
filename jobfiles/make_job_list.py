import os
import numpy as np

R_um       = 0.083 
# mx_list    = np.logspace(-1, 4, 40)
# alpha_list = np.logspace(-8, -3, 80)

# mx_list = np.logspace(-1, 4, 39)
# alpha_list = np.logspace(-7, -3, 40)

mx_list = np.logspace(-1, 4, 77)
alpha_list = np.logspace(-7, -3, 79)

mphi_list  = [10]

job_file = open("job_list_smooth_10.txt", "wt")

for mx in mx_list:
    for alpha in alpha_list:
        for mphi in mphi_list:
            outdir = f'/home/yt388/palmer_scratch/data/dm_rate/mphi_{mphi:.0e}'
            outfile = outdir + f'/drdqz_nanosphere_{R_um:.2e}_{mx:.5e}_{alpha:.5e}_{mphi:.0e}.npz'
            if( os.path.isfile(outfile) ):
                 print("Skipping: ", outfile)
                 continue
            job_str = f'module load miniconda; conda activate microsphere; python ../project_dm_smooth.py {mx} {alpha} {mphi}\n'
            job_file.write( job_str )

job_file.close()
