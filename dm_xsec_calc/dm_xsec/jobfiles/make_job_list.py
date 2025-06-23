import os
import numpy as np

R_um       = 0.083 

mx_list_proj = np.logspace(-3, 7, 79)
alpha_list_proj = np.logspace(-11, -4, 79)

mx_list_coarser_extended = np.logspace(4, 9, 39)
mx_list_coarse = np.logspace(-1, 4, 77)
mx_list_fine = np.logspace(-1, 4, 153)
mx_list_veryfine = np.logspace(-1, 4, 609)

alpha_list_coarse = np.logspace(-7, -3, 79)
alpha_list_coarse_extended= np.logspace(-7, 1, 157)

alpha_list_fine = np.logspace(-7, -3, 157)
alpha_list_veryfine = np.logspace(-7, -3, 625)

## For projection
# mx_list = mx_list_proj
# alpha_list = alpha_list_proj

## For coarse overall search
mx_list = mx_list_coarse
alpha_list = alpha_list_coarse

## Coarser extended search on the right end (1, 0.1, 0 eV)
# mx_list = mx_list_coarser_extended[np.logical_and(mx_list_coarser_extended >1e4, mx_list_coarser_extended < 1e8)]
# alpha_list = alpha_list_coarse

## Coarse search on the right end (10 eV)
# mx_list = mx_list_coarse[np.logical_and(mx_list_coarse > 50, mx_list_coarse < 500)]
# alpha_list = alpha_list_coarse_extended
# alpha_list = alpha_list_coarse_extended[alpha_list_coarse_extended > 1e-3]

# mx_list = mx_list_coarse[mx_list_coarse < 50]
# alpha_list = alpha_list_coarse

## Coarse search on the left end
# mx_list = mx_list_coarse[np.logical_and(mx_list_coarse > 0.1, mx_list_coarse < 1)]
# alpha_list = alpha_list_coarse_extended

## For finer search on the left end
# mx_list = mx_list_fine[np.logical_and(mx_list_fine > 0.1, mx_list_fine < 1)]
# alpha_list = alpha_list_coarse_extended

## Further fine search for 0.1 and 0.01 eV on the side
# mx_list = mx_list_fine[np.logical_and(mx_list_fine > 30, mx_list_fine < 1000)]
# alpha_list = alpha_list_fine[alpha_list_fine < 1e-4]

## Further fine search for 1 eV on the side
# mx_list = mx_list_fine[np.logical_and(mx_list_fine > 30, mx_list_fine < 1000)]
# alpha_list = alpha_list_fine[alpha_list_fine < 1e-3]

## Very fine search at the bottom (1, 0.1, 0.01 eV)
# mx_list = mx_list_fine[np.logical_and(mx_list_fine > 4, mx_list_fine < 30)]
# alpha_list = alpha_list_veryfine[alpha_list_veryfine < 1e-6]
# alpha_list = alpha_list_veryfine[alpha_list_veryfine < 5e-6]

## For finer search on the right end (10 eV)
# mx_list = mx_list_veryfine[np.logical_and(mx_list_veryfine > 100, mx_list_veryfine < 130)]
# alpha_list = alpha_list_fine[alpha_list_fine > 1e-5]

## For finer search on the right end (1 eV)
# mx_list = mx_list_veryfine[np.logical_and(mx_list_veryfine > 550, mx_list_veryfine < 750)]
# alpha_list = alpha_list_fine[alpha_list_fine > 2.5e-5]

## For finer search on the right end (0.1 eV)
# mx_list = mx_list_veryfine[np.logical_and(mx_list_veryfine > 4000, mx_list_veryfine < 6000)]
# alpha_list = alpha_list_fine[alpha_list_fine > 1e-4]

## Fine search for 10 eV
# mx_list = mx_list_fine[np.logical_and(mx_list_fine > 1, mx_list_fine < 200)]
# alpha_list = alpha_list_fine[alpha_list_fine > 1e-6]

mphi_list  = [1]

job_file = open("joblist_1ev_100mevthr_20250623.txt", "wt")

for mphi in mphi_list:
    for mx in mx_list:
        for alpha in alpha_list:
            if mx < 1880 or mx > 1900:
                continue
            if alpha < 1e-5 or alpha > 1e-4:
                continue
            # outdir = f'/home/yt388/palmer_scratch/data/dm_rate/mphi_{mphi:.0e}'
            outdir = f'/home/yt388/data/data/dm_rate/mphi_{mphi:.0e}'

            outfile = outdir + f'/drdq_100mevthr_nanosphere_{R_um:.2e}_{mx:.5e}_{alpha:.5e}_{mphi:.0e}.npz'
            if( os.path.isfile(outfile) ):
                print("Skipping: ", outfile)
                continue

            job_str = f'module load miniconda; conda activate microsphere; python ../rate_massive_mediator.py {R_um} {mx} {alpha} {mphi}\n'
            job_file.write( job_str )

job_file.close()
