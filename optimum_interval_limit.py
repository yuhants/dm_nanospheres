import numpy as np
import matplotlib.pyplot as plt

import h5py

import os, sys
import contextlib
from pathlib import Path
from scipy import interpolate, integrate

sys.path.append('./upper/')
import upper as _upper

## Parameters for analysis
amp2kev_sphere_20241202 = 5945.245097647231
amp2kev_sphere_20250103 = 5923.2059527417405

# Update 20250505: not doing anti-coincidence cut
exposure_sphere_20241202 = 947860.623
exposure_sphere_20250103 = 1476235.3273   # s

chi2_cut_eff = 0.9538018099684543

## Functions
@contextlib.contextmanager
def _working_directory(path):
    """
    Changes working directory and returns to previous on exit.

    Parameters
    ----------
    path : str
        The directory that the current working directory will temporarily be switched to.

    """
    prev_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)

def upper(fc, cl=0.95):
    """
    Fortran wrapper function for Steve Yellin's Optimum Interval code `Upper.f`. In this case,
    it calls a version of `UpperLim.f` that allows a larger range of confidence levels.

    Parameters
    ----------
    fc : array_like
        Given the foreground distribution whose shape is known, but whose normalization is
        to have its upper limit total expected number of events determined, fc(0) to fc(N+1),
        with fc(0)=0, fc(N+1)=1, and with  fc(i) the increasing ordered set of cumulative
        probabilities for the foreground distribution for event i, i=1 to N.
    cl : float, optional
        The confidence level desired for the upper limit. Default is 0.9. Can be any value
        between 0.00001 and 0.99999. However, the algorithm requires less than 100 upper
        limit events when outside the range 0.8 to 0.995 in order to work, so an error may
        be raised.

    Returns
    -------
    ulout : float
        The output of the Upper Fortran code, corresponding to the upper limit expected number of
        events. To convert to cross section, the output should be divided by the total rate of the
        signal and multiplied by the expected cross section for that rate.
    endpoints0 : int
        An integer giving the index of FC at which the optimum interval started.
    endpoints1 : int
        An integer giving the index of FC at which the optimum interval ended.

    Notes
    -----
    This is a wrapper around Steve Yellin's Optimum Interval Fortran code, which was compiled via f2py to
    be callable by Python. Because the Fortran code expects look-up tables in the current working directory,
    we need to use a context manager to switch directories to where the look-up tables are when running the
    algorithm.

    Read more about Steve Yellin's Optimum Interval code here:
        - http://titus.stanford.edu/Upper/
        - https://arxiv.org/abs/physics/0203002
        - https://arxiv.org/abs/0709.2701

    """

    file_path = os.path.dirname(os.path.realpath(__file__))

    # make sure fc starts with 0 and ends with 1
    fc_new = fc
    if fc[0]!=0:
        fc_new = np.concatenate(([0], fc_new))
    if fc[-1]!=1:
        fc_new = np.concatenate((fc_new, [1]))

    method = 0
    nexp = 1
    maxp1 = len(fc_new) - 1
    nevts = np.array([maxp1 - 1])
    mu = 1
    icode = 0

    with _working_directory(f"{file_path}/upper/"):
        ulout = _upper.upper(
            method=method,
            cl=cl,
            nexp=nexp,
            maxp1=maxp1,
            nevts=nevts,
            mu=np.asarray([mu]),
            fc=fc_new[:, np.newaxis],
            icode=icode,
        )

    endpoints = _upper.upperlimcom.endpoints

    return ulout, endpoints[0], endpoints[1]

def get_fc(q_events, qq, drdqzn, exposure):
    q_events = np.sort(q_events)

    q_low = np.min(q_events)
    q_high = np.max(q_events)

    q_interp = np.linspace(q_low, q_high, 1000)

    rate = np.interp(q_interp, qq, drdqzn) * exposure
    
    integ_rate = integrate.cumulative_trapezoid(rate, x=q_interp, initial=0)
    tot_rate = integ_rate[-1]

    x_val_fcn = interpolate.interp1d(
        q_interp,
        integ_rate,
        kind="linear",
        bounds_error=False,
        fill_value=(0, tot_rate),
    )

    x_vals = x_val_fcn(q_events)

    if tot_rate != 0:
        fc = x_vals/tot_rate
        fc[fc > 1] = 1

        cdf_max = 1 - 1e-6
        possiblewimp = fc <= cdf_max
        fc = fc[possiblewimp]

        if len(fc) == 0:
            fc = np.asarray([0, 1])
    else:
        fc = None

    return fc

def optimum_interval(q_events, qq, drdqzn, exposure, cl=0.95):
    q_events = np.sort(q_events)

    q_low = np.min(q_events)
    q_high = np.max(q_events)

    q_interp = np.linspace(q_low, q_high, 1000)

    rate = np.interp(q_interp, qq, drdqzn) * exposure
    integ_rate = integrate.cumulative_trapezoid(rate, x=q_interp, initial=0)
    tot_rate = integ_rate[-1]

    x_val_fcn = interpolate.interp1d(
        q_interp,
        integ_rate,
        kind="linear",
        bounds_error=False,
        fill_value=(0, tot_rate),
    )

    x_vals = x_val_fcn(q_events)

    uloutput, endpoint0, endpoint1 = None, None, None
    if tot_rate != 0:
        fc = x_vals/tot_rate
        fc[fc > 1] = 1

        cdf_max = 1 - 1e-6
        possiblewimp = fc <= cdf_max
        fc = fc[possiblewimp]

        if len(fc) == 0:
            fc = np.asarray([0, 1])

        uloutput, endpoint0, endpoint1 = upper(fc, cl=cl)

    return uloutput, endpoint0, endpoint1

def upper_combined(fcs, cl=0.95):
    file_path = os.path.dirname(os.path.realpath(__file__))

    fc_0, fc_1 = fcs[0], fcs[1]
    if fc_0 is None or fc_1 is None:
        return None

    method = 4
    nexp = 2
    maxp1 = max(fc_0.size, fc_1.size) + 1
    nevts = np.array([fc_0.size, fc_1.size])
    mu = np.array([1, 1])
    icode = 0

    # make sure fc starts with 0 and ends with 1
    fc_in = np.full((nexp, maxp1+1), np.nan)
    fc_in[0, 0] = 0
    fc_in[1, 0] = 0

    fc_in[0][1 : 1+fc_0.size] = fc_0
    fc_in[1][1 : 1+fc_1.size] = fc_1

    fc_in[0][1+fc_0.size] = 1
    fc_in[0][1+fc_1.size] = 1
    fc_in = fc_in.T

    with _working_directory(f"{file_path}/upper/"):
        ulout = _upper.upper(
            method=method,
            cl=cl,
            nexp=nexp,
            maxp1=maxp1,
            nevts=nevts,
            mu=mu,
            fc=fc_in,
            icode=icode,
        )
    return ulout

if __name__ == '__main__':
    # sphere = 'sphere_20241202'
    # sphere = 'sphere_20250103'
    sphere = 'sphere_combined'
    minlim = False

    datasets = ['coarse', 'coarse_extended_right']
    mphi_lists = [[0, 0.1, 1, 10], [1, 0, 0.1]]

    # qmin, qmax = 2500, 10000  # For Sphere 20241202
    # qmin, qmax = 2000, 10000  # For Sphere 20250103 and combined dataset (minimum limit)
    qmin, qmax = 1250, 10000  # For combined dataset (simple merging)

    if sphere == 'sphere_20241202' or sphere == 'sphere_20250103':
        if sphere == 'sphere_20241202':
            amp2kev = amp2kev_sphere_20241202
            exposure = exposure_sphere_20241202
        elif sphere == 'sphere_20250103':
            amp2kev = amp2kev_sphere_20250103
            exposure = exposure_sphere_20250103

        file = h5py.File(rf'/Users/yuhan/work/nanospheres/dm_nanospheres/data_processed/sphere_data/{sphere}_unbinned_amps.h5py')
        amps = file['unbinned_amps']['amplitude'][:]
        file.close()
        amps_kev = np.abs(amps * amp2kev)

    elif sphere == 'sphere_combined':
        exposure = exposure_sphere_20241202 + exposure_sphere_20250103

        file0 = h5py.File(rf'/Users/yuhan/work/nanospheres/dm_nanospheres/data_processed/sphere_data/sphere_20241202_unbinned_amps.h5py')
        file1 = h5py.File(rf'/Users/yuhan/work/nanospheres/dm_nanospheres/data_processed/sphere_data/sphere_20250103_unbinned_amps.h5py')

        amps0 = file0['unbinned_amps']['amplitude'][:]
        amps1 = file1['unbinned_amps']['amplitude'][:]
        file0.close()
        file1.close()

        amps_kev_0 = np.abs(amps0 * amp2kev_sphere_20241202)
        amps_kev_1 = np.abs(amps1 * amp2kev_sphere_20250103)
        amps_kev = np.concatenate([amps_kev_0, amps_kev_1])

    for i, dataset in enumerate(datasets):
        for mphi in mphi_lists[i]:
            print(f'Working on mphi = {mphi} eV')

            mphi_prefix = f'{mphi:.0e}'
            drdqzn_file = rf'/Users/yuhan/work/nanospheres/dm_nanospheres/data_processed/dm_rate/drdqz_100mevthr_nanosphere_8.30e-02_{dataset}_ampdepsigma_{mphi_prefix}.npz'
            drdqzn_npz = np.load(drdqzn_file)

            q_kev = drdqzn_npz['bc_kev']
            drdqzn = drdqzn_npz['drdqzn']
            mx = drdqzn_npz['mx_list']
            alpha = drdqzn_npz['alpha_list']

            alpha_lim = np.full_like(mx, fill_value=np.nan)
            i_mx = 0

            for i_mx, _mx in enumerate(mx):
                uu = np.empty_like(alpha)
                mu = np.empty_like(alpha)

                for i_alpha in range(alpha.size):
                    # Efficiency corrected DM rate
                    # No need to correct for reconstruction efficiency because it is 1
                    # above the minimum analysis threshold here
                    drdqzn_mx_alpha = drdqzn[i_mx, i_alpha] * chi2_cut_eff

                    if sphere == 'sphere_combined' and minlim is True:
                        fc_0 = get_fc(amps_kev_0[amps_kev_0 > qmin], q_kev, drdqzn_mx_alpha, exposure_sphere_20241202)
                        fc_1 = get_fc(amps_kev_1[amps_kev_1 > qmin], q_kev, drdqzn_mx_alpha, exposure_sphere_20250103)
                        u = upper_combined(fcs=[fc_0, fc_1], cl=0.95)
                    else:
                        u, e0, e1 = optimum_interval(amps_kev[amps_kev > qmin], q_kev, drdqzn_mx_alpha, exposure, cl=0.95)

                    uu[i_alpha] = u

                    q_idx = np.logical_and(q_kev > qmin, q_kev < qmax)
                    mu[i_alpha] = np.trapz(drdqzn_mx_alpha[q_idx] * exposure, q_kev[q_idx])

                alpha_lim[i_mx] = np.interp(0, mu - uu, alpha, left=1e6, right=1e6)

            if minlim:
                outfile = fr'/Users/yuhan/work/nanospheres/dm_nanospheres/data_processed/alpha_lim_optimum/alpha_lim_minlim_{sphere}_{dataset}_{mphi_prefix}.npz'
            else:
                outfile = fr'/Users/yuhan/work/nanospheres/dm_nanospheres/data_processed/alpha_lim_optimum/alpha_lim_{sphere}_{dataset}_{mphi_prefix}_1_25mevthr.npz'
            print(f'Saving file {outfile}')
            np.savez(outfile, mx_gev=mx, alpha_lim=alpha_lim)
