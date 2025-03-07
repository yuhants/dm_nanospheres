import numpy as np
import h5py

import os
import sys
sys.path.append('../')
import analysis_utils as utils

from multiprocessing import Pool

R_um = 0.083

mx_list_coarse = np.logspace(-1, 4, 77)
alpha_list_coarse = np.logspace(-7, -3, 79)

bins = np.arange(0, 10000, 50)  # keV
bc = 0.5 * (bins[:-1] + bins[1:])

def get_drdqz(qq, drdq):
    ret = np.empty_like(drdq)
    drdq_iso = drdq / (4 * np.pi * qq**2)

    for i, q in enumerate(qq):
        xx = qq[qq >= q]
        integrand = drdq_iso[qq >= q]

        # Another factor of two because we want rate
        # for both +z and -z
        ret[i] = 2 * 2 * np.pi * np.trapz(integrand*xx, xx)

    return ret

def smear_drdqz_gauss(qq, drdqz, sigma_kev=180):
    """Convolve spectrum with a Gaussian kernel"""
    dq = qq[1] - qq[0]
    qq_gauss = np.arange(-1000, 1000, dq)
    gauss_kernel = utils.gauss(qq_gauss, A=1, mu=0, sigma=180)

    # Pad the array to minimize edge effect
    # Pad the first two indices with edge value
    # to get the rising tail when q -> 0
    # then pad with mirror image
    pad_len = gauss_kernel.size
    if pad_len > 2:
        padded_drdqz = np.pad(drdqz, (2, 0), mode='edge')
        padded_drdqz = np.pad(padded_drdqz, (pad_len-2, 0), mode='reflect')
    else:
        padded_drdqz = np.pad(drdqz, (pad_len, 0), mode='reflect')
    padded_drdqz = np.pad(padded_drdqz, (0, pad_len), mode='edge')

    convolved = np.convolve(padded_drdqz, gauss_kernel, mode='valid')
    idx_start = (convolved.size - drdqz.size) // 2
    ret = convolved[idx_start : idx_start + drdqz.size] / np.sum(gauss_kernel)

    return ret

def get_final_drdqz(mphi, mx, alpha, sigma_kev):
    file = f'{data_dir}/drdq_nanosphere_{R_um:.2e}_{mx:.5e}_{alpha:.5e}_{mphi:.0e}.npz'
    drdq_npz = np.load(file)

    qq = drdq_npz['q_kev']
    drdq = drdq_npz['drdq_hz_kev']

    drdqz = get_drdqz(qq, drdq)
    drdqz_smeared = smear_drdqz_gauss(qq, drdqz, sigma_kev)
    drdqz_smeared_resampled = np.interp(bc, qq, drdqz_smeared)

    return drdqz_smeared_resampled

if __name__ == '__main__':
    mphi = 0.01
    mx_list = mx_list_coarse
    alpha_list = alpha_list_coarse

    data_dir = f'/home/yt388/palmer_scratch/data/dm_rate/mphi_{mphi:.0e}'

    drdqzn_all = np.empty(shape=(mx_list.size, alpha_list.size, bc.size), dtype=np.float64)
    for i, mx in enumerate(mx_list):

        pool = Pool(32)
        n_alpha = alpha_list.size
        params = list(np.vstack((np.full(n_alpha, mphi), 
                                 np.full(n_alpha, mx), 
                                 alpha_list,
                                 np.full(n_alpha, 180))).T)

        res = pool.starmap(get_final_drdqz, params)
        drdqzn_all[i] = np.asarray(res)

        # for j, alpha in enumerate(alpha_list):
        #     drdqzn_all[i, j] = get_final_drdqz(mphi, mx, alpha, 180, bc)

    outfile_name = f'drdqz_all_coarse_nanosphere_{R_um:.2e}_{mphi:.0e}.npz'
    outfile = os.path.join(data_dir, outfile_name)
    print(f'Saving file {outfile}')
    np.savez(outfile, bc_kev=bc, drdqzn=drdqzn_all, mx_list=mx_list, alpha_list=alpha_list)