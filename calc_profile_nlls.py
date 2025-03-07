import h5py
import sys

import numpy as np
from scipy.special import erf
from scipy.optimize import minimize

from multiprocessing import Pool

R_um = 0.083

ana_threshold = 1000.  # Analysis threhsold in keV/c
length_search_window = 5e-5  # We perform search every 50 us

data_dir = '/home/yt388/microspheres/dm_nanospheres/data_processed'
rate_dir = '/home/yt388/palmer_scratch/data/dm_rate'

## Coarse search over a larger range
mx_list_coarse = np.logspace(-1, 4, 77)
alpha_list_coarse = np.logspace(-7, -3, 79)

mx_list_fine = np.logspace(-1, 4, 153)
alpha_list_fine = np.logspace(-7, -3, 157)

mx_list_veryfine = np.logspace(-1, 4, 609)
alpha_list_veryfine = np.logspace(-7, -3, 625)

def load_sphere_data(sphere):
    # Very bad coding...but will be passed to the pooled nll calculation
    global params_nodm, nll_offset, bounds_params, bc, hist, eff_coefs

    if sphere == 'sphere_20241202':
        # Params for Sphere 20241202
        params_nodm = np.array([[0.38447462, 215.97561887, 239.31177006]])
        nll_offset = 4900399.764535
        bounds_params = [(0.2, 1), (100, 300), (100, 500), (0.8, 1.2), (0.8, 1.2)]

    elif sphere == 'sphere_20250103':
        # Fit params with no dark matter (Sphere 20250103)
        params_nodm = np.array([0.94784728, 223.00764832, 145.00866937])
        nll_offset = 12101712.701790
        bounds_params = [(0.7, 1), (100, 300), (100, 500), (0.8, 1.2), (0.8, 1.2)]

    # Read in reconstruction histogram and signal efficiency
    file_dm = f'{data_dir}/sphere_data/{sphere}_recon_all.h5py'
    with h5py.File(file_dm, 'r') as fout:
        g = fout['recon_data_all']
        bc = g['bc'][:]
        hist = g['hist_all'][:]
        fout.close()

    file_cal = f'{data_dir}/sphere_data/sphere_combined_calibration.h5py'
    with h5py.File(file_cal, 'r') as fout:
        g = fout[f'calibration_{sphere}']
        eff_coefs = g['sig_efficiency_fit_params'][:]
        fout.close()

def func2(x, z, f):
    return 0.5 * erf((x - z) * f) + 0.5

def gaus(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-1 * (x - mu)**2 / (2 * sigma**2))

def gaus_normalized(x, mu, sigma, cutoff=1000):
    # xx = np.linspace(0, 50000, 5000)
    # func_val = gaus(xx, mu, sigma)
    # norm = np.trapz(func_val, xx)
    norm = 1 - (0.5 + 0.5 * erf((cutoff - mu)/(np.sqrt(2)*sigma)))

    x = np.asarray(x)
    if x.size == 1:
        return gaus(x, mu, sigma)[0] / norm
    else:
        return gaus(x, mu, sigma) / norm

def expo_corrected(x, cutoff, xi):
    # Re-normalize exponential after applying efficiency correction 
    # and truncate from below
    xx = np.linspace(0, 50000, 5000)

    expo_eff_truncated = np.exp(-1 * (xx) / xi) / xi
    expo_eff_truncated[xx < cutoff] = 0

    expo_corrected_norm = np.trapz(expo_eff_truncated, xx)

    x = np.asarray(x)
    ret = np.exp(-1 * (x) / xi) / xi
    ret[x < cutoff] = 0

    if ret.size == 1:
        return ret[0] / expo_corrected_norm
    else:
        return ret / expo_corrected_norm

## ==================== Functions not currently used ====================== ##
# def half_gaus_mod(x, mu, m, n):
#     xx = np.linspace(0, 50000, 50000)
#     sigma = m * xx + n
#     _norm = np.trapz((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-1 * (xx - mu)**2 / (2 * sigma**2)), xx)

#     sigma_x = m * x + n
#     return (1 / (np.sqrt(2 * np.pi) * sigma_x)) * np.exp(-1 * (x - mu)**2 / (2 * sigma_x**2)) / _norm

# def crystal_ball_rev(x, alpha, n, mu, sigma):
#     # Modified from https://arxiv.org/pdf/1603.08591
#     # and https://en.wikipedia.org/wiki/Crystal_Ball_function

#     x = np.asarray(x)
#     ret = np.empty_like(x)

#     A = np.power(n / np.abs(alpha), n) * np.exp(-1 * alpha**2 / 2)
#     B = n / np.abs(alpha) - np.abs(alpha)

#     # Flip the direction to get the tail on the positive side
#     idx_gaus = ((x - mu) / sigma) < alpha
#     idx_other = ((x - mu) / sigma) > alpha

#     # Flip `B - ...` to `B + ...` to reverse the power law tail 
#     ret[idx_gaus] = np.exp(-1 * (x[idx_gaus] - mu)**2 / (2 * sigma**2))
#     ret[idx_other] = A * np.power((B + (x[idx_other] - mu) / sigma), (-1 * n))

#     return ret

# def crystal_ball_rev_normalized(x, alpha, n, mu, sigma):
#     xx = np.linspace(0, 50000, 5000)
#     func_val = crystal_ball_rev(xx, alpha, n, mu, sigma)
#     norm = np.trapz(func_val, xx)

#     x = np.asarray(x)
#     if x.size == 1:
#         return crystal_ball_rev(x, alpha, n, mu, sigma)[0] / norm
#     else:
#         return crystal_ball_rev(x, alpha, n, mu, sigma) / norm

# def read_dm_rate(mphi, mx, alpha):
#     R_um       = 0.083
#     file = f'{rate_dir}/mphi_{mphi:.0e}/drdqz_nanosphere_{R_um:.2e}_{mx:.5e}_{alpha:.5e}_{mphi:.0e}.npz'
#     drdq_npz = np.load(file)

#     qq = drdq_npz['bc_kev']
#     drdqzn = drdq_npz['drdqzn']
    
#     return qq, drdqzn

## ==================== End of functions not currently used =================== ##

def nll_dm_scaled(a, sigma, xi_b, q_scale, n_scale,
                  drdqzn, bc, hist, eff_coefs, nll_offset):
    # DM scattering rate has already resampled to the same bins
    # Rescale DM model to account for uncertainties in
    # E field and neutron number
    # Multiply `drdqzn` by `q_scale` to account for normalization
    # against the scaled bin width
    # Assume dr/dq scales with n_neutron**2
    qq_scaled = bc * q_scale
    drdqzn_scaled = np.interp(qq_scaled, bc, drdqzn*q_scale, left=0, right=0) * n_scale**2

    # DM contribution that accounts for live time and bin width
    hist_norm = np.sum(hist) * length_search_window * (bc[1] - bc[0])
    eff_qq = func2(qq_scaled, *eff_coefs)
    hist_dm = eff_qq * drdqzn_scaled * hist_norm

    idx = bc > ana_threshold
    bi = bc[idx]
    ni = hist[idx]

    # Modified 20250306: only use count above threshold
    # Total number of count above analysis threhsold
    # DM events are fixed so profile over other parameters
    # Only count DM events above the analysis threshold
    # Modified 20250306: include DM contribution above threshold
    # and ignore the undetectable part below
    dm_contribution = np.sum(hist_dm[idx])
    ntot = np.sum(ni) - dm_contribution

    # Use only the central value of pdf
    # faster and avoid numerical issues from integration
    # No correctiion for efficiency for background
    joint_pdf = a * gaus_normalized(bi, 0, sigma, 1000) + (1 - a) * expo_corrected(bi, 1000, xi_b)
    mui = ntot * joint_pdf * 50 + hist_dm[idx]

    # Nusance parameters to account for uncertainties in
    # calibration and neutron number
    sigma_q = 0.05
    sigma_n = np.sqrt(3) * 0.054

    # Also truncate `gaus_term` at central 3 sigma
    # the calibration is unlikely to be off by more than 15%
    if np.abs(q_scale - 1) > 1.5 * sigma_q:
        gaus_term = (q_scale - 1)**2 / (2 * sigma_q**2) + (q_scale - 1)**2 / (1e-2 * sigma_q**2)
    else:
        gaus_term = (q_scale - 1)**2 / (2 * sigma_q**2)

    # Following the 2020 paper, truncate `neut_term` at central 1 sigma
    # to avoid the profile NLL be driven to physically impossible values
    if np.abs(n_scale - 1) > 0.5 * sigma_n:
        neut_term = (n_scale - 1)**2 / (2 * sigma_n**2) + ((n_scale - 1)**2) / (1e-2 * sigma_n**2)
    else:
        neut_term = (n_scale - 1)**2 / (2 * sigma_n**2)

    return np.sum(np.nan_to_num(mui - ni * np.log(mui))) + gaus_term + neut_term + nll_offset


def minimize_nll(drdqzn, x0_bg=None, bounds=None):
    if x0_bg is None:
        x0_bg = params_nodm
    if bounds is None:
        bounds = bounds_params

    args = (drdqzn, bc, hist, eff_coefs, nll_offset)
    res = minimize(fun=lambda x: nll_dm_scaled(*x, *args), x0=[*x0_bg, 1, 1],
            method='Nelder-Mead',
            bounds=bounds,
            options={'disp' : False,
                    'maxiter': 50000,
                    'maxfev': 50000,
                    'adaptive': True,
                    'fatol': 0.001,
                    }
            )
    return res

def calc_profile_nlls(mphi, dataset='coarse'):

    if dataset == 'coarse':
        mx_list, alpha_list = mx_list_coarse, alpha_list_coarse
    elif dataset == 'fine_left':
        # For finer search on the left end
        mx_list = mx_list_fine[np.logical_and(mx_list_fine > 2, mx_list_fine < 5)]
        alpha_list = alpha_list_fine
    elif dataset == 'veryfine_bottom':
        ## Very fine search at the bottom (1, 0.1, 0.01 eV)
        mx_list = mx_list_fine[np.logical_and(mx_list_fine > 4, mx_list_fine < 30)]
        alpha_list = alpha_list_veryfine[alpha_list_veryfine < 1e-6]
    elif dataset == 'fine_side':
        ## Further fine search for 0.1 and 0.01 eV on the side
        mx_list = mx_list_fine[np.logical_and(mx_list_fine > 30, mx_list_fine < 1000)]
        alpha_list = alpha_list_fine[alpha_list_fine < 1e-4]

    rate_file = f'{rate_dir}/mphi_{mphi:.0e}/drdqz_all_{dataset}_nanosphere_{R_um:.2e}_{mphi:.0e}.npz'
    drdqzn_npz = np.load(rate_file)
    drdqzn = drdqzn_npz['drdqzn']

    nlls = np.empty((drdqzn.shape[0:2]))

    for i, mx in enumerate(mx_list):
        print(fr'Working on $M_x=$ {mx:.2f} GeV')
        
        pool = Pool(32)
        params = [ [drdqzn[i, j]] for j in range(alpha_list.size) ]
        res_pool = pool.starmap(minimize_nll, params)

        for j in range(alpha_list.size):
            if res_pool[j].success:
                nlls[i, j] = res_pool[j].fun
            else:
                nlls[i, j] = np.nan
    return mx_list, alpha_list, nlls

if __name__ == "__main__":
    ## Start calculation
    sphere = sys.argv[1]
    mphi = float(sys.argv[2])  # Mediator mass in eV
    dataset = sys.argv[3]

    print(f'Working on m_phi = {mphi:.0e} eV; sphere = {sphere}, dataset = {dataset}')
    load_sphere_data(sphere)

    # Calculate profile NLLs for each DM parameter
    mx_list, alpha_list, nlls_all = calc_profile_nlls(mphi, dataset)

    file_out = f'{data_dir}/profile_nlls/{sphere}/profile_nlls_{sphere}_{mphi:.0e}_{dataset}.npz'
    print(f'Writing file {file_out}')
    np.savez(file_out, mx=mx_list, alpha=alpha_list, nll=nlls_all)
