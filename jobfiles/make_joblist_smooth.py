import os
import numpy as np

R_um       = 0.083

#### Old Parameters ################
####################################
## Parameters that has been calculated
# mx_list    = np.logspace(-1, 4, 40)
# alpha_list = np.logspace(-8, -3, 80)

# mx_list = np.logspace(-1, 4, 39)
# alpha_list = np.logspace(-7, -3, 40)
#####################################
#####################################

## Coarse search over a larger range
mx_list_coarse = np.logspace(-1, 4, 77)
alpha_list_coarse = np.logspace(-7, -3, 79)

mx_list_fine = np.logspace(-1, 4, 153)
alpha_list_fine = np.logspace(-7, -3, 157)

alpha_list_veryfine = np.logspace(-7, -3, 625)

# ## For finer search on the left end
# mx_list = mx_list_fine[np.logical_and(mx_list_fine > 2, mx_list_fine < 5)]
# alpha_list = alpha_list_fine

## Very fine search at the bottom (1, 0.1, 0.01 eV)
mx_list = mx_list_fine[np.logical_and(mx_list_fine > 4, mx_list_fine < 30)]
alpha_list = alpha_list_veryfine[alpha_list_veryfine < 1e-6]

## Further fine search for 0.1 and 0.01 eV on the side
# mx_list = mx_list_fine[np.logical_and(mx_list_fine > 30, mx_list_fine < 1000)]
# alpha_list = alpha_list_fine[alpha_list_fine < 1e-4]

mphi_list  = [0.01, 0.1, 1]

job_file = open("joblist_smooth_veryfine_bottom.txt", "wt")

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
