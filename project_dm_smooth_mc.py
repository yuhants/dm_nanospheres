"""
Project the calculate DM scattering rate to the
z axis and add a Gaussian noise

THE IMPLEMENTATION IS INCORRECT
The angular distirbution is not uniform on a sphere
will fix later
"""

import numpy as np
import sys, os
from scipy.signal import savgol_filter
from scipy.signal import medfilt
from multiprocessing import Pool

def get_random_q_samples(qq, drdq, rr):
    norm_factor = np.trapz(drdq, qq)

    f_drdq_norm = drdq / norm_factor       # PDF of q
    Fc_drdq_norm = np.cumsum(f_drdq_norm)  # CDF of q

    qq_sampled = np.interp(rr, Fc_drdq_norm, qq, left=0, right=0)
    return qq_sampled, norm_factor

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
alpha_list_coarse = np.logspace(-7, -3, 79)

mx_list_fine = np.logspace(-1, 4, 153)
alpha_list_fine = np.logspace(-7, -3, 157)

## For coarse overall search
# mx_list = mx_list_coarse
# alpha_list = alpha_list_coarse

## For finer search on the left end
# mx_list = mx_list_fine[np.logical_and(mx_list_fine > 2, mx_list_fine < 5)]
# alpha_list = alpha_list_fine

## Fine search at the bottom
# mx_list = mx_list_fine[np.logical_and(mx_list_fine > 4, mx_list_fine < 30)]
# alpha_list = alpha_list_fine[alpha_list_fine < 1e-6]

## Further fine search for 0.1 and 0.01 eV on the side
mx_list = mx_list_fine[np.logical_and(mx_list_fine > 30, mx_list_fine < 1000)]
alpha_list = alpha_list_fine[alpha_list_fine < 1e-4]

#### Generate Monte Carlo Gaussian noise
# sigma_gaus = 180  # keV/c
# n_mc = int(1e8)
# rand_seeds = [34324, 6666, 88873, 9984, 1123, 322235, 4442343, 5434234, 66453453, 22040403]

# for i, rand_seed in enumerate(rand_seeds):
#     rng = np.random.default_rng(rand_seed)

#     rr = rng.uniform(0, 1, n_mc)
#     phiphi = rng.uniform(0, np.pi, n_mc)
#     noise_gaussian = rng.normal(0, sigma_gaus, n_mc)

#     np.savez(f'/home/yt388/palmer_scratch/data/dm_rate/noise_mc_1e8_{i}.npz', rr=rr, phiphi=phiphi, noise_gaussian=noise_gaussian)

def project_smooth(mphi, mx, alpha, outfile):
    file = f'{data_dir}/drdq_nanosphere_{R_um:.2e}_{mx:.5e}_{alpha:.5e}_{mphi:.0e}.npz'
    drdq_npz = np.load(file)

    qq = drdq_npz['q_kev']
    drdq = drdq_npz['drdq_hz_kev']

    # If we have zero sensitivity
    if np.sum(drdq) == 0:
        bins = np.arange(0, 10000, 50)
        bc = 0.5 * (2 * bins + 50)
        hh, hhz, hhzn = [np.zeros_like(bc) for i in range(3)]
        norm = 0

    else:
        # Smooth with a polynomial filter in the log space
        _qq, _drdq_smoothed = qq, np.nan_to_num( np.exp(savgol_filter(np.log(drdq), 50, 1)), nan=0 )
        if np.sum(_drdq_smoothed) == 0:
            _drdq_smoothed = drdq

        qmax = max(50 * (_qq[np.nonzero(_drdq_smoothed > 0)[0][-1]] // 50 + 2), 10000)
        bins_sample = np.arange(0, qmax, 50)
        bc = 0.5 * (bins_sample[1:] + bins_sample[:-1])

        hh_zns = np.empty((10, bc.size), dtype=np.int64)
        norm = None
        for i in range(10):
            # Read in the MC noise (1e9 samples in total)
            noise_file = f'/home/yt388/palmer_scratch/data/dm_rate/noise_mc/noise_mc_1e8_{i}.npz'
            noise_npz = np.load(noise_file)
            rr, phiphi, noise_gaussian = noise_npz['rr'], noise_npz['phiphi'], noise_npz['noise_gaussian']

            qq_sampled, norm = get_random_q_samples(_qq, _drdq_smoothed, rr)

            # hh, be   = np.histogram(qq_sampled, bins=np.arange(0, qmax, 50), density=True)
            # hhz, be  = np.histogram(qq_sampled*np.abs(np.cos(phiphi)), bins=np.arange(0, qmax, 50), density=True)
            hhzn, be = np.histogram(qq_sampled*np.abs(np.cos(phiphi)) + noise_gaussian, bins=bins_sample, density=True)
            hh_zns[i] = hhzn

    # All the projected rates are normalized so take the average value
    rate_zn = np.mean(hh_zns, axis=0) * norm
    rate_smoothed = np.nan_to_num( np.exp(savgol_filter(np.log(rate_zn), 5, 1)), nan=0)
    rate_smoothed_med = medfilt(rate_zn, 15)

    print(f'Writing file {outfile}')
    np.savez(outfile, bc_kev=bc, drdqzn=rate_zn, drdqzn_smoothed=rate_smoothed, drdqzn_smoothed_med=rate_smoothed_med)

def get_projected_spectrum(mphi):
    for i, mx in enumerate(mx_list):

        # pool = Pool(16)
        # n_alpha = alpha_list.size
        # params = list(np.vstack((np.full(n_alpha, mphi), np.full(n_alpha, mx), alpha_list)).T)
        # pool.starmap(project_smooth, params)

        for j, alpha in enumerate(alpha_list):
            project_smooth(mphi, mx, alpha)

if __name__ == "__main__":
    mx    = float(sys.argv[1])  # DM mass in GeV
    alpha = float(sys.argv[2])  # Single neutron coupling
    mphi  = float(sys.argv[3])   # Mediator mass in eV

    data_dir = f'/home/yt388/palmer_scratch/data/dm_rate/mphi_{mphi:.0e}'

    outfile_name = f'drdqz_nanosphere_{R_um:.2e}_{mx:.5e}_{alpha:.5e}_{mphi:.0e}.npz'
    outfile = os.path.join(data_dir, outfile_name)

    if os.path.isfile(outfile):
        print(f'Skipping {outfile_name}')

    else:    
        print(f'Working on mx = {mx:.5e} GeV, alpha = {alpha:.5e} m_phi = {mphi:.0e} eV')
        project_smooth(mphi, mx, alpha, outfile)
        # get_projected_spectrum(m_phi)
