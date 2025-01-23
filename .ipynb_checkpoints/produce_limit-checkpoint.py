import h5py
import sys

import numpy as np
from scipy.special import erf
from scipy.optimize import minimize

from multiprocessing import Pool

# Fit params for no dark matter
params_nodm = np.array([9.99999411e-01, 3.63789735e+02, 5.82216279e-02, 1.41641008e+02, 1.47480619e+03, 1.23523645e+02])
NLL_OFFSET = 4304827608.245811

data_dir = '/home/yt388/microspheres/impulse_analysis/data_processed'
rate_dir = '/home/yt388/palmer_scratch/data/dm_rate'

# Read in reconstruction histogram and signal efficiency
# file = '/Users/yuhan/work/nanospheres/data/dm_data_processed/sphere_20250103/sphere_20250103_recon_all.h5py'
file_dm = f'{data_dir}/sphere_20250103_recon_all.h5py'
with h5py.File(file_dm, 'r') as fout:
    g = fout['recon_data_all']
    hist = g['hist'][:]
    n_window = g['hist'].attrs['n_windows']
    scaling = g['hist'].attrs['scaling']

    rate_all = g['rate_hist'][:]
    rate_all_err = g['rate_hist_err'][:]
    bc = g['bc'][:]

    time_all = g.attrs['time_hours']

    fout.close()

hist_norm = n_window * scaling

# file = '/Users/yuhan/work/nanospheres/data/pulse_calibration_processed/sphere_20250103_calibration_all.h5py'
file_cal = f'{data_dir}/sphere_20250103_calibration_all.h5py'
with h5py.File(file_cal, 'r') as fout:
    g = fout['calibration_data_processed']
    eff_coefs = g['sig_efficiency_fit_params'][:]

    fout.close()


def func2(x, z, f):
    return 0.5 * erf((x - z) * f) + 0.5

def expo_corrected(x, b, xi, eff_coefs=None):
    # Re-normalize exponential after applying efficiency correction 
    # and truncate from below
    xx = np.linspace(0, 50000, 50000)

    if eff_coefs is not None:
        eff_xx = func2(xx, *eff_coefs)
        expo_eff_truncated = eff_xx * np.exp(-1 * (xx) / xi) / xi
    else:
        expo_eff_truncated = np.exp(-1 * (xx) / xi) / xi
    expo_eff_truncated[xx < b] = 0

    expo_corrected_norm = np.trapz(expo_eff_truncated, xx)

    x = np.asarray(x)
    if eff_coefs is not None:
        eff_x = func2(x, *eff_coefs)
        ret = eff_x * np.exp(-1 * (x) / xi) / xi
    else:
        ret = np.exp(-1 * (x) / xi) / xi
    ret[x < b] = 0

    if ret.size == 1:
        return ret[0] / expo_corrected_norm
    else:
        return ret / expo_corrected_norm

def half_gaus_mod(x, mu, m, n):
    xx = np.linspace(0, 50000, 50000)
    sigma = m * xx + n
    _norm = np.trapz((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-1 * (xx - mu)**2 / (2 * sigma**2)), xx)

    sigma_x = m * x + n
    return (1 / (np.sqrt(2 * np.pi) * sigma_x)) * np.exp(-1 * (x - mu)**2 / (2 * sigma_x**2)) / _norm

def read_dm_rate(mphi, mx, alpha):
    R_um       = 0.083
    # file = f'/Users/yuhan/work/nanospheres/data/dm_rate/mphi_{mphi:.0e}/drdqz_nanosphere_{R_um:.2e}_{mx:.5e}_{alpha:.5e}_{mphi:.0e}.npz'
    file = f'{rate_dir}/mphi_{mphi:.0e}/drdqz_nanosphere_{R_um:.2e}_{mx:.5e}_{alpha:.5e}_{mphi:.0e}.npz'
    drdq_npz = np.load(file)

    qq = drdq_npz['bc_kev']
    drdqzn = drdq_npz['drdqzn']
    
    return qq, drdqzn

def nll_dm_scaled(alpha, mu, m, n, b, xi, q_scale, n_scale,
                  bc, hist, eff_coefs, mphi, mx, alpha_n, hist_norm):
    
    # Rescale DM model to account for uncertainties in
    # E field and neutron number
    # Multiply `drdqzn` by `q_scale` to account for normalization
    # against the scaled bin width
    # Assume dr/dq scales with n_neutron**2
    qq, drdqzn = read_dm_rate(mphi, mx, alpha_n)
    qq_scaled = qq * q_scale
    drdqzn_scaled = np.interp(qq_scaled, qq, drdqzn*q_scale, left=0, right=0) * n_scale**2

    eff_qq = func2(qq_scaled, *eff_coefs)
    hist_dm = eff_qq * drdqzn_scaled * hist_norm

    # For large dm couplings that give large kicks
    # zero pad the actual measured histogram
    if qq.size > bc.size:
        hist = np.pad(hist, (0, qq.size - bc.size))
        bc = qq

    idx = qq > 800

    bi = bc[idx]
    ni = hist[idx]

    # Total number of count in the entire range
    # DM events are fixed so profile over other parameters
    ntot = np.sum(hist) - np.sum(hist_dm)

    # Use only the central value of pdf
    # faster and avoid numerical issues from integration
    # No correctiion for efficiency for the background
    joint_pdf = alpha * half_gaus_mod(bi, mu, m, n) + (1 - alpha) * expo_corrected(bi, b, xi, eff_coefs=None)
    mui = ntot * joint_pdf * 50 + hist_dm[idx]

    sigma_q = 0.05
    sigma_n = np.sqrt(3) * 0.054
    gaus_term = (q_scale - 1)**2 / (2 * sigma_q**2)  # 5% uncertainty as sigma
    neut_term = (n_scale - 1)**2 / (2 * sigma_n**2)

    return np.sum(np.nan_to_num(mui - ni * np.log(mui))) + gaus_term + neut_term + NLL_OFFSET

def minimize_nll(mphi, mx, alpha):
    args = (bc, hist, eff_coefs, mphi, mx, alpha, hist_norm)
    res = minimize(fun=lambda x: nll_dm_scaled(*x, *args), x0=[*params_nodm, 1, 1],
            method='Nelder-Mead',
            bounds=[(0.99, 1), (0, 500), (0, 0.1), (100, 300), (1000, 1500), (90, 160), (0.9, 1.1), (0.8, 1.2)],
            options={'disp' : False,
                    'maxiter': 50000,
                    'maxfev': 50000,
                    'adaptive': True,
                    'fatol': 0.001,
                    }
            )
    return res

def calc_profile_nlls(mphi, mx_list, alpha_list):
    nlls = np.empty((mx_list.size, alpha_list.size))

    for i, mx in enumerate(mx_list):
        print(fr'Working on $M_x=$ {mx:.2f} GeV')
        
        pool = Pool(4)
        n_alpha = alpha_list.size
        params = list(np.vstack((np.full(n_alpha, mphi), np.full(n_alpha, mx), alpha_list)).T)
        res_pool = pool.starmap(minimize_nll, params)

        for j in range(n_alpha):
            if res_pool[j].success:
                nlls[i, j] = res_pool[j].fun
            else:
                nlls[i, j] = np.nan

    return nlls


if __name__ == "__main__":
    mx_list_0    = np.logspace(-2, 5, 40)
    alpha_list_0 = np.logspace(-10, -4, 40)

    mx_list_1    = np.logspace(0, 1, 10)
    alpha_list_1 = np.logspace(-7, -3, 20)

    mx_list_2    = np.logspace(-1, 4, 39)
    alpha_list_2 = np.logspace(-7, -3, 40)

    mphi      = float(sys.argv[1])  # Mediator mass in eV
    print(f'Working on m_phi = {mphi:.0e} eV')

    # Calculate profile NLLs for each DM parameter
    idx = 2
    mx_list, alpha_list = mx_list_2, alpha_list_2
    
    nlls_all = calc_profile_nlls(mphi, mx_list, alpha_list)

    # file_out = f'/Users/yuhan/work/nanospheres/impulse_analysis/profile_nlls/profile_nlls_{mphi:.0e}_{idx}.npz'
    file_out = f'{data_dir}/profile_nlls/profile_nlls_{mphi:.0e}_{idx}.npz'
    print(f'Writing file {file_out}')
    np.savez(file_out, mx=mx_list, alpha=alpha_list, nll=nlls_all)