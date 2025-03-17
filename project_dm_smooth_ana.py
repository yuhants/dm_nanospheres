import numpy as np
import h5py

import os
import sys
sys.path.append('../')
import analysis_utils as utils

from multiprocessing import Pool

R_um = 0.083
qmax = 25000

mx_list_coarser_extended = np.logspace(4, 9, 39)
mx_list_coarse = np.logspace(-1, 4, 77)
mx_list_fine = np.logspace(-1, 4, 153)

alpha_list_coarse = np.logspace(-7, -3, 79)
alpha_list_coarse_extended= np.logspace(-7, 1, 157)
alpha_list_fine = np.logspace(-7, -3, 157)

if qmax == 25000:
    bins = np.arange(0, qmax, 50)  # keV
    bc = 0.5 * (bins[:-1] + bins[1:])
    prefix = '_25mevthr'
else:
    bins = np.arange(0, 10000, 50)  # keV
    bc = 0.5 * (bins[:-1] + bins[1:])
    prefix = ''


def iter_smooth_drdq(drdq):
    ret = np.copy(drdq)
        
    # Assume DM rate to be strickly decreasing
    # Find local bumps and interpolate
    i = 1
    while i < drdq.size - 1:
        if drdq[i] > drdq[i - 1]:

            start = i - 1
            while i < drdq.size - 1 and drdq[i] > drdq[start]:
                i += 1
            end = i if i < drdq.size - 1 else drdq.size - 1

            x = np.array([start, end])
            y = np.array([drdq[start], drdq[end]])
            ret[start:end] = np.interp(range(start, end), x, y)
        i += 1

    return ret

def get_drdqz(qq, drdq):
    drdq = iter_smooth_drdq(drdq)

    if (qq[1] - qq[0] > 25 and np.max(qq) > qmax):
        qq_old, drdq_old = qq, drdq
        good_idx = drdq_old > 0
        qq = np.arange(start=25, stop=np.max(qq_old[good_idx]), step=25)
        drdq = np.exp(np.interp(qq, qq_old[good_idx], np.log(drdq_old[good_idx])))

    elif (qq[1] - qq[0] < 10 and np.max(qq) < qmax):
        qq_old, drdq_old = qq, drdq
        good_idx = drdq_old > 0
        qq = np.arange(start=5, stop=qmax, step=5)
        drdq = np.exp(np.interp(qq, qq_old[good_idx], np.log(drdq_old[good_idx]), right=-np.inf))

    qq_out = qq[qq < qmax]
    ret = np.empty_like(drdq[qq < qmax])

    drdq_iso = drdq / (4 * np.pi * qq**2)

    for i, q in enumerate(qq_out):
        xx = qq[qq >= q]
        integrand = drdq_iso[qq >= q]

        # Another factor of two because we want rate
        # for both +z and -z
        ret[i] = 2 * 2 * np.pi * np.trapz(integrand*xx, xx)

    return qq_out, ret

def smear_drdqz_gauss(qq, drdqz, sigma_kev=180):
    """Convolve spectrum with a Gaussian kernel"""
    dq = qq[1] - qq[0]
    qq_gauss = np.arange(-1000, 1000, dq)
    gauss_kernel = utils.gauss(qq_gauss, A=1, mu=0, sigma=180)

    # Pad the array to minimize edge effect
    # to get the rising tail when q -> 0
    # then pad with mirror image
    pad_len = gauss_kernel.size
    if qq[0] >= dq:
        padded_drdqz = np.pad(drdqz, (pad_len, 0), mode='symmetric')
    else:
        padded_drdqz = np.pad(drdqz, (pad_len, 0), mode='reflect')
    padded_drdqz = np.pad(padded_drdqz, (0, pad_len), mode='constant', constant_values=0)

    convolved = np.convolve(padded_drdqz, gauss_kernel, mode='valid')
    idx_start = (convolved.size - drdqz.size) // 2
    ret = convolved[idx_start : idx_start + drdqz.size] / np.sum(gauss_kernel)

    return qq, ret

def get_final_drdqz(mphi, mx, alpha, sigma_kev, return_bc=False):
    if mphi == 0:
        file = f'{data_dir}/drdq{prefix}_nanosphere_{R_um:.2e}_{mx:.5e}_{alpha:.5e}_massless.npz'
    else:
        file = f'{data_dir}/drdq{prefix}_nanosphere_{R_um:.2e}_{mx:.5e}_{alpha:.5e}_{mphi:.0e}.npz'
    drdq_npz = np.load(file)

    qq = drdq_npz['q_kev']
    drdq = drdq_npz['drdq_hz_kev']

    if np.sum(drdq > 0) < 2:
        return np.zeros_like(bc)

    _qq, _drdqz = get_drdqz(qq, drdq)
    _qq, _drdqz_smeared = smear_drdqz_gauss(_qq, _drdqz, sigma_kev)
    drdqz_smeared_resampled = np.interp(bc, _qq, _drdqz_smeared)

    if return_bc:
        return bc, drdqz_smeared_resampled
    else:
        return drdqz_smeared_resampled

if __name__ == '__main__':
    mphi  = float(sys.argv[1])   # Mediator mass in eV
    dataset = sys.argv[2]

    if dataset == 'coarse':
        mx_list = mx_list_coarse
        alpha_list = alpha_list_coarse

    elif dataset == 'coarse_extended_right':
        mx_list = mx_list_coarser_extended[mx_list_coarser_extended < 1e8]
        alpha_list = alpha_list_coarse

    elif dataset == 'coarse_extended_alpha_left':
        mx_list = mx_list_coarse[np.logical_and(mx_list_coarse > 0.1, mx_list_coarse < 1)]
        alpha_list = alpha_list_coarse_extended
        
    elif dataset == 'coarse_extended_alpha_right':
        mx_list = mx_list_coarse[np.logical_and(mx_list_coarse > 50, mx_list_coarse < 150)]
        alpha_list = alpha_list_coarse_extended

    elif dataset == 'coarse_10ev_left':
        mx_list = mx_list_coarse[mx_list_coarse < 50]
        alpha_list = alpha_list_coarse

    if mphi == 0:
        data_dir = f'/home/yt388/palmer_scratch/data/dm_rate/massless_mediator'
    else:
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
        #     print(j, alpha)
        #     drdqzn_all[i, j] = get_final_drdqz(mphi, mx, alpha, 180)

    if mphi == 0:
        outfile_name = f'drdqz{prefix}_nanosphere_{R_um:.2e}_{dataset}_massless.npz'
    else:
        outfile_name = f'drdqz{prefix}_nanosphere_{R_um:.2e}_{dataset}_{mphi:.0e}.npz'
    outfile = os.path.join(data_dir, outfile_name)
    print(f'Saving file {outfile}')
    np.savez(outfile, bc_kev=bc, drdqzn=drdqzn_all, mx_list=mx_list, alpha_list=alpha_list)