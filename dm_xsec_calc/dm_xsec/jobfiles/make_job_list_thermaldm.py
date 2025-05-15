import os
import numpy as np

R_um       = 0.083 

# alpha_list_extended = np.logspace(-12, 4, 159)
# alpha_list = alpha_list_extended[alpha_list_extended<1e-2]
# mx_list = np.logspace(1, 8, 80)

alpha_list = np.logspace(-8, 0, 40)
mx_list = np.logspace(2, 8, 60)

mphi_list  = [0.1, 1, 10]

job_file = open("joblist_thermaldm_halo.txt", "wt")

for mphi in mphi_list:
    for mx in mx_list:
        for alpha in alpha_list:
            outdir = f'/home/yt388/palmer_scratch/data/dm_rate/thermalized_dm_halo_density/mphi_{mphi:.0e}'
            outfile = outdir + f'/drdq_100mevthr_thermaldm_halodensity_nanosphere_{R_um:.2e}_{mx:.5e}_{alpha:.5e}_{mphi:.0e}.npz'
            if( os.path.isfile(outfile) ):
                print("Skipping: ", outfile)
                continue

            job_str = f'module load miniconda; conda activate microsphere; python ../rate_massive_mediator_thermaldm.py {R_um} {mx} {alpha} {mphi}\n'
            job_file.write( job_str )

job_file.close()
