import h5py
import sys

import numpy as np
from scipy.special import erf
from scipy.optimize import minimize

from multiprocessing import Pool

ana_threshold = 1000.  # Analysis threhsold in keV/c
data_dir = '/home/yt388/microspheres/dm_nanospheres/data_processed'
rate_dir = '/home/yt388/palmer_scratch/data/dm_rate'

######## Previous parameters ########
#####################################
# mx_list_0    = np.logspace(-2, 5, 40)
# alpha_list_0 = np.logspace(-10, -4, 40)

# mx_list_1    = np.logspace(0, 1, 10)
# alpha_list_1 = np.logspace(-7, -3, 20)

# mx_list_2    = np.logspace(-1, 4, 39)
# alpha_list_2 = np.logspace(-7, -3, 40)
######## End of previous parameters####

## Coarse search over a larger range
mx_list_coarse = np.logspace(-1, 4, 77)
alpha_list_coarse = np.logspace(-7, -3, 79)

mx_list_fine = np.logspace(-1, 4, 153)
alpha_list_fine = np.logspace(-7, -3, 157)

mx_list_veryfine = np.logspace(-1, 4, 609)
alpha_list_veryfine = np.logspace(-7, -3, 625)

def load_sphere_data(sphere):
    # Very bad coding...but will be passed to the pooled nll calculation
    global params_nodm, nll_offset, bounds_params, bc, hist, hist_norm, eff_coefs

    if sphere == 'sphere_20241202':
        # Params for Sphere 20241202
        params_nodm = np.array([9.99996691e-01, 3.47276545e+00, 1.79826520e+02, 5.93489570e-05, 2.88315304e+02, 1.43999985e+03, 2.46167595e+02])
        nll_offset = 152178767.020342
        bounds_params = [(0.9999, 1), (0, 5), (0, 100), (0, 500), (100, 350), (1200, 2000), (200, 300), (0.8, 1.2), (0.8, 1.2)]

    elif sphere == 'sphere_20250103':
        # Fit params for no dark matter (Sphere 20250103)
        params_nodm = np.array([9.99999963e-01, 3.88844938e+00, 1.88930977e+01, 1.64962666e+02, 2.55454974e+02, 1.70317112e+03, 3.73851449e+02])
        nll_offset = 325398931.400860
        bounds_params = [(0.9999, 1), (0, 5), (0, 100), (0, 500), (100, 350), (1400, 2000), (200, 300), (0.8, 1.2), (0.8, 1.2)]
        # bounds_params = [(0.9999, 1), (0, 5), (0, 30), (0, 500), (100, 350), (1400, 2000), (200, 300), (0.8, 1.2), (0.8, 1.2)]

    # Read in reconstruction histogram and signal efficiency
    file_dm = f'{data_dir}/sphere_data/{sphere}_recon_all.h5py'
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

    file_cal = f'{data_dir}/sphere_data/{sphere}_calibration_all.h5py'
    with h5py.File(file_cal, 'r') as fout:
        g = fout['calibration_data_processed']
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
        return ret[0] / expo_corrected_norm
    else:
        return ret / expo_corrected_norm

# def half_gaus_mod(x, mu, m, n):
#     xx = np.linspace(0, 50000, 50000)
#     sigma = m * xx + n
#     _norm = np.trapz((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-1 * (xx - mu)**2 / (2 * sigma**2)), xx)

#     sigma_x = m * x + n
#     return (1 / (np.sqrt(2 * np.pi) * sigma_x)) * np.exp(-1 * (x - mu)**2 / (2 * sigma_x**2)) / _norm

def crystal_ball_rev(x, alpha, n, mu, sigma):
    # Modified from https://arxiv.org/pdf/1603.08591
    # and https://en.wikipedia.org/wiki/Crystal_Ball_function

    x = np.asarray(x)
    ret = np.empty_like(x)

    A = np.power(n / np.abs(alpha), n) * np.exp(-1 * alpha**2 / 2)
    B = n / np.abs(alpha) - np.abs(alpha)

    # Flip the direction to get the tail on the positive side
    idx_gaus = ((x - mu) / sigma) < alpha
    idx_other = ((x - mu) / sigma) > alpha

    # Flip `B - ...` to `B + ...` to reverse the power law tail 
    ret[idx_gaus] = np.exp(-1 * (x[idx_gaus] - mu)**2 / (2 * sigma**2))
    ret[idx_other] = A * np.power((B + (x[idx_other] - mu) / sigma), (-1 * n))

    return ret

def crystal_ball_rev_normalized(x, alpha, n, mu, sigma):
    xx = np.linspace(0, 50000, 5000)
    func_val = crystal_ball_rev(xx, alpha, n, mu, sigma)
    norm = np.trapz(func_val, xx)

    x = np.asarray(x)
    if x.size == 1:
        return crystal_ball_rev(x, alpha, n, mu, sigma)[0] / norm
    else:
        return crystal_ball_rev(x, alpha, n, mu, sigma) / norm

def read_dm_rate(mphi, mx, alpha):
    R_um       = 0.083
    file = f'{rate_dir}/mphi_{mphi:.0e}/drdqz_nanosphere_{R_um:.2e}_{mx:.5e}_{alpha:.5e}_{mphi:.0e}.npz'
    drdq_npz = np.load(file)

    qq = drdq_npz['bc_kev']
    drdqzn = drdq_npz['drdqzn']
    
    return qq, drdqzn

def nll_dm_scaled(a, alpha, n, mu, sigma, cutoff, xi, q_scale, n_scale,
                  bc, hist, eff_coefs, mphi, mx, alpha_n, hist_norm, nll_offset):
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

    idx = bc > ana_threshold

    bi = bc[idx]
    ni = hist[idx]

    # Total number of count in the entire range
    # DM events are fixed so profile over other parameters
    # Only count DM events above the analysis threshold
    dm_contribution = np.sum(hist_dm)
    ntot = np.sum(hist) - dm_contribution

    # Use only the central value of pdf
    # faster and avoid numerical issues from integration
    # No correctiion for efficiency for the background
    # joint_pdf = alpha * half_gaus_mod(bi, mu, m, n) + (1 - alpha) * expo_corrected(bi, b, xi, eff_coefs=None)
    joint_pdf = a * crystal_ball_rev_normalized(bi, alpha, n, mu, sigma) + (1 - a) * expo_corrected(bi, cutoff, xi)
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

def minimize_nll(mphi, mx, alpha, x0_bg=None, bounds=None):
    if x0_bg is None:
        x0_bg = params_nodm
    if bounds is None:
        bounds = bounds_params

    args = (bc, hist, eff_coefs, mphi, mx, alpha, hist_norm, nll_offset)
    res = minimize(fun=lambda x: nll_dm_scaled(*x, *args), x0=[*x0_bg, 1, 1],
            method='Nelder-Mead',
            bounds=bounds,
            options={'disp' : False,
                    'maxiter': 50000,
                    'maxfev': 50000,
                    'adaptive': True,
                    'fatol': 0.01,
                    }
            )
    return res

def calc_profile_nlls(mphi, mx_list, alpha_list):
    nlls = np.empty((mx_list.size, alpha_list.size))

    for i, mx in enumerate(mx_list):
        print(fr'Working on $M_x=$ {mx:.2f} GeV')
        
        pool = Pool(32)
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
    ## Start calculation
    mphi = float(sys.argv[1])  # Mediator mass in eV
    sphere = sys.argv[2]
    dataset = sys.argv[3]

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

    print(f'Working on m_phi = {mphi:.0e} eV; sphere = {sphere}, dataset = {dataset}')

    load_sphere_data(sphere)

    # Calculate profile NLLs for each DM parameter
    nlls_all = calc_profile_nlls(mphi, mx_list, alpha_list)

    file_out = f'{data_dir}/profile_nlls/{sphere}/profile_nlls_{sphere}_{mphi:.0e}_{dataset}.npz'
    print(f'Writing file {file_out}')
    np.savez(file_out, mx=mx_list, alpha=alpha_list, nll=nlls_all)
