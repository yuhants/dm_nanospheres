import h5py
import sys

import numpy as np
from scipy.special import erf
from scipy.optimize import minimize

from multiprocessing import Pool

R_um = 0.083

ana_threshold = 1500.        # Analysis threhsold in keV/c
length_search_window = 5e-5  # We perform search every 50 us

data_dir = '/home/yt388/microspheres/dm_nanospheres/data_processed'
rate_dir = '/home/yt388/palmer_scratch/data/dm_rate'

mx_list_coarser_extended = np.logspace(4, 9, 39)
mx_list_coarse = np.logspace(-1, 4, 77)
mx_list_fine = np.logspace(-1, 4, 153)

alpha_list_coarse = np.logspace(-7, -3, 79)
alpha_list_coarse_extended= np.logspace(-7, -1, 118)
alpha_list_fine = np.logspace(-7, -3, 157)

def load_sphere_data(sphere):
    # Very bad coding...but will be passed to the pooled nll calculation
    global params_nodm_noxi, nll_offset, bounds_params, bc, hist, eff_coefs, eff_chi2

    if sphere == 'sphere_20241202':
        # Params for Sphere 20241202
        nll_offset = 269225.577261
        bounds_params = [(0.1, 1000), (0.8, 1.2), (0.8, 1.2)]

        # Signal efficiency for chi2 cut
        eff_chi2 = 0.9620145113102859

    elif sphere == 'sphere_20250103':
        # Fit params with no dark matter (Sphere 20250103)
        nll_offset = 6357.498234
        bounds_params = [(0.1, 1000), (0.8, 1.2), (0.8, 1.2)]

        # Signal efficiency for chi2 cut
        eff_chi2 = 0.9882030178326474

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
        if expo_corrected_norm == 0:
            return 0
        return ret[0] / expo_corrected_norm
    else:
        if expo_corrected_norm == 0:
            return np.zeros_like(ret)
        return ret / expo_corrected_norm

def nll_dm_scaled_tail_only(xi_b, q_scale, n_scale,
                            drdqzn, bc, hist, eff_coefs, nll_offset, eff_chi2):
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

    # Correct for signal efficiency (search and chi2 cut)
    eff_qq = func2(qq_scaled, *eff_coefs)
    hist_dm = eff_chi2 * eff_qq * drdqzn_scaled * hist_norm

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

    if ntot > 0:
        # Use only the central value of pdf
        # faster and avoid numerical issues from integration
        # No correctiion for efficiency for background
        joint_pdf = expo_corrected(bi, ana_threshold, xi_b)
        mui = ntot * joint_pdf * 50 + hist_dm[idx]
    else:
        mui = hist_dm[idx]

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

    return np.sum(np.nan_to_num(mui - ni * np.log(mui, where=(mui>0)))) + gaus_term + neut_term + nll_offset

def minimize_nll_tail_only(drdqzn, bounds=None):
    if bounds is None:
        bounds = bounds_params

    xi_b_try = [20, 100, 150, 300, 400, 500]
    nlls_try = []
    res_x_try = []

    args = (drdqzn, bc, hist, eff_coefs, nll_offset, eff_chi2)
    for xi in xi_b_try:
        x0_bg = [xi]
        res = minimize(fun=lambda x: nll_dm_scaled_tail_only(*x, *args), x0=[*x0_bg, 1, 1],
                method='Nelder-Mead',
                bounds=bounds,
                options={'disp' : False,
                        'maxiter': 50000,
                        'maxfev': 50000,
                        'adaptive': True,
                        'fatol': 0.001,
                        }
                )
        if res.success:
            nlls_try.append(res.fun)
            res_x_try.append(res.x)
        else:
            nlls_try.append(np.nan)
            res_x_try.append(np.full(3, np.nan))
    try:
        min_idx = np.nanargmin(np.asarray(nlls_try))
    except ValueError:    # would raise ValueError if all elements are nan
        return np.nan, np.full(3, np.nan)

    return nlls_try[min_idx], res_x_try[min_idx]

def calc_profile_nlls(mphi, dataset='coarse'):

    if dataset == 'coarse':
        mx_list, alpha_list = mx_list_coarse, alpha_list_coarse
    elif dataset == 'fine_left':
        # For finer search on the left end
        mx_list = mx_list_fine[np.logical_and(mx_list_fine > 0.1, mx_list_fine < 1)]
        alpha_list = alpha_list_coarse_extended
    elif dataset == 'coarse_right':
        mx_list = mx_list_coarse[np.logical_and(mx_list_coarse > 1e2, mx_list_coarse < 1e3)]
        alpha_list = alpha_list_coarse_extended
    elif dataset == 'coarse_extended_right':
        # The calculated rate doesn't make sense after idx 20
        mx_list = mx_list_coarser_extended[mx_list_coarser_extended < 1e8]
        alpha_list = alpha_list_coarse

    if mphi == 0:
        rate_file = f'{rate_dir}/massless_mediator/drdqz_nanosphere_{R_um:.2e}_{dataset}_massless.npz'
    else:
        rate_file = f'{rate_dir}/mphi_{mphi:.0e}/drdqz_nanosphere_{R_um:.2e}_{dataset}_{mphi:.0e}.npz'
    drdqzn_npz = np.load(rate_file)
    drdqzn = drdqzn_npz['drdqzn']

    nlls = np.empty((drdqzn.shape[0:2]))
    res_xs = np.empty((drdqzn.shape[0], drdqzn.shape[1], 3))

    for i, mx in enumerate(mx_list):
        print(fr'Working on $M_x=$ {mx:.2f} GeV')
        
        pool = Pool(32)
        params = [ [drdqzn[i, j]] for j in range(alpha_list.size) ]
        res_pool = pool.starmap(minimize_nll_tail_only, params)

        for j in range(alpha_list.size):
            nlls[i, j] = res_pool[j][0]
            res_xs[i, j] = res_pool[j][1]

    return mx_list, alpha_list, nlls, res_xs

if __name__ == "__main__":
    ## Start calculation
    sphere = sys.argv[1]
    mphi = float(sys.argv[2])  # Mediator mass in eV
    dataset = sys.argv[3]

    print(f'Working on m_phi = {mphi:.0e} eV; sphere = {sphere}, dataset = {dataset}')
    load_sphere_data(sphere)

    # Calculate profile NLLs for each DM parameter
    mx_list, alpha_list, nlls_all, res_x_all = calc_profile_nlls(mphi, dataset)

    if mphi == 0:
        file_out = f'{data_dir}/profile_nlls/{sphere}/profile_nlls_tail_only_{sphere}_massless_{dataset}.npz'
    else:
        file_out = f'{data_dir}/profile_nlls/{sphere}/profile_nlls_tail_only_{sphere}_{mphi:.0e}_{dataset}.npz'
    print(f'Writing file {file_out}')
    np.savez(file_out, mx=mx_list, alpha=alpha_list, nll=nlls_all, res_x=res_x_all)
