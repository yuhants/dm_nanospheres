import os
import numpy as np

R_um       = 0.083 

#### Old parameters ####
#######################
# mx_list    = np.logspace(-1, 4, 40)
# alpha_list = np.logspace(-8, -3, 80)

# mx_list = np.logspace(-1, 4, 39)
# alpha_list = np.logspace(-7, -3, 40)
#### End of old params ####
###########################

mx_list_coarse = np.logspace(-1, 4, 77)
mx_list_fine = np.logspace(-1, 4, 153)
mx_list_veryfine = np.logspace(-1, 4, 609)

alpha_list_coarse = np.logspace(-7, -3, 79)
alpha_list_fine = np.logspace(-7, -3, 157)
alpha_list_veryfine = np.logspace(-7, -3, 625)

## For coarse overall search
# mx_list = mx_list_coarse
# alpha_list = alpha_list_coarse

## For finer search on the left end
# mx_list = mx_list_fine[np.logical_and(mx_list_fine > 2, mx_list_fine < 5)]
# alpha_list = alpha_list_fine

## Further fine search for 0.1 and 0.01 eV on the side
# mx_list = mx_list_fine[np.logical_and(mx_list_fine > 30, mx_list_fine < 1000)]
# alpha_list = alpha_list_fine[alpha_list_fine < 1e-4]

## Very fine search at the bottom (1, 0.1, 0.01 eV)
mx_list = mx_list_fine[np.logical_and(mx_list_fine > 4, mx_list_fine < 30)]
# alpha_list = alpha_list_veryfine[alpha_list_veryfine < 1e-6]
alpha_list = alpha_list_veryfine[alpha_list_veryfine < 5e-6]

## For finer search on the right end (10 eV)
# mx_list = mx_list_veryfine[np.logical_and(mx_list_veryfine > 100, mx_list_veryfine < 130)]
# alpha_list = alpha_list_fine[alpha_list_fine > 1e-5]

## For finer search on the right end (1 eV)
# mx_list = mx_list_veryfine[np.logical_and(mx_list_veryfine > 550, mx_list_veryfine < 750)]
# alpha_list = alpha_list_fine[alpha_list_fine > 2.5e-5]

## For finer search on the right end (0.1 eV)
# mx_list = mx_list_veryfine[np.logical_and(mx_list_veryfine > 4000, mx_list_veryfine < 6000)]
# alpha_list = alpha_list_fine[alpha_list_fine > 1e-4]

mphi_list  = [1, 0.1, 0.01]

job_file = open("joblist_veryfine_bottom_all_remain.txt", "wt")

for mphi in mphi_list:
    for mx in mx_list:
        for alpha in alpha_list:
            outdir = f'/home/yt388/palmer_scratch/data/dm_rate/mphi_{mphi:.0e}'
            outfile = outdir + f'/drdq_nanosphere_{R_um:.2e}_{mx:.5e}_{alpha:.5e}_{mphi:.0e}.npz'
            if( os.path.isfile(outfile) ):
                 print("Skipping: ", outfile)
                 continue

            job_str = f'module load miniconda; conda activate microsphere; python ../rate_massive_mediator.py {R_um} {mx} {alpha} {mphi}\n'
            job_file.write( job_str )

job_file.close()
