
import numpy as np
import scipy.io as sio
import os, glob
import h5py
from datetime import datetime

import matplotlib.pyplot as plt
from cycler import cycler

from scipy.signal import welch
from scipy.signal import butter, sosfilt, iirnotch, filtfilt
from scipy.optimize import curve_fit, minimize
from scipy.linalg import solve
from scipy.fft import rfft, irfft, rfftfreq

yale_colors = ['#00356b', '#286dc0', '#63aaff', '#4a4a4a']

c = 299792458    # m / s
SI2ev = (1 / 1.6e-19) * c

m = 2000 * (83.5e-9)**3 * 4 * np.pi / 3
hbar = 6.626e-34
kb = 1.380649e-23

#### File processing
def load_timestreams(file, channels=['C']):
    timestreams = []
    delta_t = None
    if file[-4:] == '.mat':
        data = sio.loadmat(file)
        delta_t = data['Tinterval'][0,0]

        for c in channels:
            timestreams.append(data[c][:,0])

    if file[-5:] == '.hdf5':
        f = h5py.File(file, 'r')
        for c in channels:
            # Convert mv to V
            adc2mv = f['data'][f'channel_{c.lower()}'].attrs['adc2mv']
            timestreams.append(f['data'][f'channel_{c.lower()}'][:] * adc2mv / 1000)

        if delta_t is None:
                delta_t = f['data'].attrs['delta_t']
            
    return delta_t, timestreams

def get_psd(dt=None, tt=None, zz=None, nperseg=None):
    if dt is not None:
        fs = int(np.round(1 / dt))
    elif tt is not None:
        fs = int(np.ceil(1 / (tt[1] - tt[0])))
    else:
        raise SyntaxError('Need to supply either `dt` or `tt`.')
    
    if nperseg is None:
        nperseg = fs / 10
    ff, pp = welch(zz, fs=fs, nperseg=nperseg)
    return ff, pp

def get_pulse_idx(drive_sig, trigger_val=0.5, positive=True):
    if positive:
        return np.flatnonzero((drive_sig[:-1] < trigger_val) & (drive_sig[1:] > trigger_val))+1
    else:
        return np.flatnonzero((drive_sig[:-1] > trigger_val) & (drive_sig[1:] < trigger_val))+1

def get_drive_window(tt, pulse_idx, length_idx=14):
    window = np.full(tt.size, True)
    window[:pulse_idx-3] = False
    window[pulse_idx+length_idx:] = False
    
    return window

def get_drive_amp(charge, tt, drive_sig, pulse_window):
    sig = drive_sig[pulse_window]
    
    # Subtract background and integrate
    bg = 0.5 * (sig[0] + sig[-1])
    _sig = sig - bg
    area = np.abs(np.trapz(_sig, tt[pulse_window]))

    amp_kev = (charge*1.6e-19) * 120 * area * SI2ev / 1000
    return amp_kev

#### Filtering
def notch_filtered(data, fs, f0=93000, q=50):
    b, a = iirnotch(f0, q, fs)
    filtered = filtfilt(b, a, data)
    return filtered

def bandpass_filtered(data, fs, f_low=10000, f_high=100000, order=3): 
    sos_bp = butter(order, [f_low, f_high], 'bandpass', fs=fs, output='sos')
    filtered = sosfilt(sos_bp, data)
    return filtered

def lowpass_filtered(tod, fs, f_lp=50000, order=3):
    sos_lp = butter(order, f_lp, 'lp', fs=fs, output='sos')
    filtered = sosfilt(sos_lp, tod)
    return filtered

def highpass_filtered(tod, fs, f_hp=50000, order=3):
    sos_hp = butter(order, f_hp, 'hp', fs=fs, output='sos')
    filtered = sosfilt(sos_hp, tod)
    return filtered

#### Fitting
def log_gauss(x, A, mu, sigma):
    return np.log(A*np.exp(-(x-mu)**2/(2*sigma**2)))

def gauss(x, A, mu, sigma):
    return A*np.exp(-(x-mu)**2/(2*sigma**2))

from scipy.special import voigt_profile
def log_voigt(x, amp, x0, sigma, gamma):
    return np.log(amp * voigt_profile(x-x0, sigma, gamma))

def log_lorentzian_with_const(x, amp, x0, gamma, c):
    """A Lorentzian line shape"""
    return np.log(c + amp * gamma / ( ( x0**2 - x**2)**2 + gamma**2 * x**2 ))

def fit_peak(x, y, peak_func, p0=None):
    popt, pcov = curve_fit(peak_func, x, y, p0=p0, maxfev=50000)
    
    # Return central frequency and gamma
    return popt, x, peak_func(x, *popt)

def fit_z_peak(ff, pp, peak_func=log_voigt, passband=(60000, 70000), p0=[2e9, 62500*2*np.pi, 800, 50], plot=True):
    all_idx = np.logical_and(ff > passband[0], ff < passband[1])
    
    popt, omega_fit, p_fit = fit_peak(ff[all_idx]*2*np.pi, np.log(pp[all_idx]), peak_func, p0=p0)

    if plot:
        if len(p0) == 4:    # fitted by voigt
            label = (fr'$f_0$ = {popt[1]/(2*np.pi*1000):.2f} kHz,\n'
                     fr'$\gamma$ = {popt[3]/(2*np.pi):.1f} Hz, \n'
                     fr'$\sigma$ = {popt[2]/(2*np.pi):.1f} Hz')
        else:
            label = {}

        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(ff[all_idx], pp[all_idx], '.')
        ax.plot(omega_fit/(2*np.pi), np.exp(p_fit), label=label)
        ax.set_yscale('log')
        ax.legend()

    # amp, omega0, (sigma,) gamma
    return popt

#### Plotting
def load_plotting_setting():
    # colors=['#fe9f6d', '#de4968', '#8c2981', '#3b0f70', '#000004']
    colors = plt.colormaps.get_cmap('tab20b').resampled(6).colors
    default_cycler = cycler(color=colors)
    
    params = {'figure.figsize': (7, 5),
              'axes.prop_cycle': default_cycler,
              'axes.titlesize': 14,
              'legend.fontsize': 12,
              'axes.labelsize': 14,
              'axes.titlesize': 14,
              'xtick.labelsize': 12,
              'ytick.labelsize': 12,
              'xtick.direction': 'in',
              'ytick.direction': 'in',
              'xtick.top': True,
              'ytick.right': True
              }
    plt.rcParams.update(params)

#### Pulse reconstruction
def get_analysis_window(tt, pulse_idx, length):
    window = np.full(tt.size, True)
    pulse_idx_in_window = length

    if length < pulse_idx:
        window[:pulse_idx-length] = False
    else:
        # Pulse happens at the beginning of the file
        # so it's not in the middle of the window
        pulse_idx_in_window = pulse_idx

    if (pulse_idx + length) < (tt.size-1):
        window[pulse_idx+length:] = False
    
    return window, pulse_idx_in_window

def get_prepulse_window(tt, pulse_idx, length):
    window = np.full(tt.size, True)
    window[:pulse_idx-int(length/2)] = False
    window[pulse_idx+int(length/2):] = False
    
    return window

def lorentzian_for_fitting(x, amp, x0, gamma):
    return amp / ((1 + (x - x0)/gamma)**2)

def fit_lorentzian(ff, pp, passband=(45000, 100000)):
    all_idx = np.logical_and(ff > passband[0], ff < passband[1])

    xi, Yi = ff[all_idx]*2*np.pi, pp[all_idx]
    aa = np.array([[np.sum(xi**4 * Yi**3), np.sum(xi**3 * Yi**3), np.sum(xi**2 * Yi**3)],
                  [np.sum(xi**3 * Yi**3), np.sum(xi**2 * Yi**3), np.sum(xi * Yi**3)],
                  [np.sum(xi**2 * Yi**3), np.sum(xi**1 * Yi**3), np.sum(Yi**3)]])
    bb = np.array([np.sum(xi**2 * Yi**2), np.sum(xi * Yi**2), np.sum(Yi**2)])
    x = solve(aa, bb)

    omega0 = -0.5 * x[1] / x[0]
    gamma  = np.sqrt(x[2] / x[0] - omega0**2)
    amp    = (1 / x[0]) / (gamma**2)
    return amp, omega0, gamma

def get_susceptibility(omega, omega0, gamma):
    ## Note that this is *not* how susceptibility is usually
    ## defined.
    ## DO NOT use this function for other calculations
    chi = 1 / (omega0**2 - omega**2 - 1j*gamma*omega)
    return chi

def get_force_noise(omega0, dt, zz_bp, pre_window):
    ft = get_pulse_amp(dt, zz_bp[pre_window], omega0, 1*2*np.pi)
    return np.sum(np.log(ft*ft))

def get_omega0(dt, zz_bp, omega_window, omega0_guess):
    # omega0 = minimize(get_force_noise, x0=[78000*2*np.pi], args=(dt, zz_bp, pre_window),)

    # Make an array of possible omega's
    # and use the one that has the smallest force noise
    omegas = np.linspace(omega0_guess-500*2*np.pi, omega0_guess+500*2*np.pi, 20)
    nn = np.empty_like(omegas)
    for i, omega in enumerate(omegas):
        nn[i] = get_force_noise(omega, dt, zz_bp, omega_window)

    return omegas[np.argmin(nn)]

def get_search_window(amp, pulse_idx_in_window, search_window_length, pulse_length=20):
    search_window = np.full(amp.size, False)
    left  = pulse_idx_in_window + pulse_length
    right = left + search_window_length

    if right > amp.size:
        print('Skipping pulse too close to the edge of search window')
        return None

    search_window[left:right] = True
    return search_window

def get_cal_window(amp, pulse_idx_in_window, cal_window_length=50):
    cal_window = np.full(amp.size, False)
    left  = pulse_idx_in_window - cal_window_length
    right = pulse_idx_in_window
    
    cal_window[left:right] = True
    return cal_window

def get_pulse_amp(dt, zz, omega0, gamma):
    zzk = rfft(zz)
    ff = rfftfreq(zz.size, dt)
    omega = ff * 2 * np.pi

    chi_omega = get_susceptibility(omega, omega0, gamma)
    filter_output = irfft(zzk / chi_omega)

    return filter_output

def recon_force(dtt, zz_bp, c_mv=None):
    fs = int(np.ceil(1 / dtt))

    zzk = rfft(zz_bp)
    ff = rfftfreq(zz_bp.size, dtt)
    pp = np.abs(zzk)**2 / (zz_bp.size / dtt)

    omega0_fit = ff[np.argmax(pp)] * 2 * np.pi
    amp = get_pulse_amp(dtt, zz_bp, omega0_fit, (ff[1]-ff[0])*2*np.pi)    
    amp_lp = lowpass_filtered(amp, fs, 80000, 3)

    if c_mv is not None:
        in_band = np.logical_and(ff>30000, ff<80000)
        temp = m * omega0_fit**2 * np.trapz(pp[in_band], ff[in_band]) * c_mv**2 / kb
    else:
        temp = None

    return amp/1e9, amp_lp/1e9, temp

def recon_pulse(idx, dtt, zz_bp, dd, plot=False, fname=None, 
                analysis_window_length=100000,
                prepulse_window_length=5000,
                search_window_length=20,
                pulse_length=20):

    if idx < prepulse_window_length:
        print('Skipping pulse too close to the beginning of the file')
        return None, None, None, np.nan

    fs = int(np.ceil(1 / dtt))

    window, pulse_idx_in_window = get_analysis_window(dd, idx, analysis_window_length)
    prepulse_window = get_prepulse_window(dd, idx, prepulse_window_length)

    zzk = rfft(zz_bp[prepulse_window])
    ff = rfftfreq(zz_bp[prepulse_window].size, dtt)
    pp = np.abs(zzk)**2 / (zz_bp[prepulse_window].size / dtt)
    
    ##
    ## Archived methods for frequency estimation
    ##
    ## Method 1: fit with a Lorentzian
    # psd_fit_scale_factor = (zz_bp[prepulse_window].size / dtt)
    # amp_fit, omega0_fit, gamma_fit = fit_lorentzian(ff, pp*psd_fit_scale_factor, (45000, 120000))
    #
    ## Method 2: minimize prepulse force noise
    # omega_window = get_prepulse_window(tt, idx, 500)
    # omega0_fit = get_omega0(dtt, zz_bp, omega_window, omega0_guess) 
    ##

    # Now just take the fft frequency
    omega0_guess = ff[np.argmax(pp)] * 2 * np.pi
    omega0_fit = omega0_guess

    # Use a fixed damping (2 pi * 1 Hz) to reconstruct pulse amp
    # Actual damping doesn't matter as long as gamma << omega0
    # amp = get_pulse_amp(dtt, zz_bp[window], omega0_fit, 1*2*np.pi)
    amp = get_pulse_amp(dtt, zz_bp[window], omega0_fit, (ff[1]-ff[0])*2*np.pi)

    ## Modified 20241104
    ## Change lowpass from 100 to 80 kHz
    amp_lp = lowpass_filtered(amp, fs, 80000, 3)

    # cal_window = get_cal_window(amp, pulse_idx_in_window, cal_window_length=20000)
    # recon_amp = np.abs(np.min(amp_lp[search_window])/np.var(amp_lp[cal_window]))
    search_window = get_search_window(amp, pulse_idx_in_window, search_window_length, pulse_length)
    # recon_amp = np.abs(np.max(amp_lp[search_window])/1e9)
    recon_amp = np.max(np.abs(amp_lp[search_window])/1e9)

    # amp_lp_search = amp_lp[search_window]
    # min_idx = np.argmax(np.abs(amp_lp_search))
    # recon_amp = np.sum(np.abs( amp_lp_search[min_idx-10:min_idx+10] ) )

    if plot:
        fig, axes = plt.subplots(2, 1, figsize=(8, 6))

        ax0_twinx = axes[0].twinx()
        sig0 = axes[0].plot((tt[window]-tt[idx])*1e3, zz_bp[window], color=yale_colors[1], alpha=0.7, label='$z$ signal')
        sig1 = ax0_twinx.plot((tt[window]-tt[idx])*1e3, dd[window], color=yale_colors[3], alpha=0.6, label='Drive signal')
        labs = [s.get_label() for s in (sig0+sig1)]
        
        axes[0].set_ylabel('$z$ signal (V)')
        ax0_twinx.set_ylabel('Drive signal (V)')
        axes[0].legend((sig0+sig1), labs, fontsize=12, loc='upper right')
        
        #axes[1].plot((tt[window][100:-100]-tt[idx])*1e3, amp[100:-100]/1e9, label='Recon. force')
        axes[1].plot((tt[window][100:-100]-tt[idx])*1e3, amp_lp[100:-100]/1e9, label='Recon. force (filtered)', color=yale_colors[0])
        axes[1].set_ylabel('Reconstruction (a. u.)')
        axes[1].set_xlabel('Time (ms)')
        axes[1].legend(fontsize=12, loc='upper right')
        
        for ax in axes:
            ax.set_xlim(-0.25, 0.25)
        # axes[0].set_ylim(-0.04, 0.04)
        # axes[1].set_ylim(-0.75, 1)
            
        fig.tight_layout()

        if fname is not None:
            plt.savefig(fname, format='png', transparent=True, dpi=400)
    return window, amp, amp_lp, recon_amp

def get_unnormalized_amps(data_files, noise=False):
    amps = []
    for file in data_files:
        dtt, nn = load_timestreams(file, ['D', 'G'])
        fs = int(np.ceil(1 / dtt))

        ## Modified 20241104
        ## Change bandpass filter upper bound from 100 to 80 kHz
        zz, dd = nn[0], nn[1]
        zz_bp = bandpass_filtered(zz, fs, 30000, 80000)
        pulse_idx = get_pulse_idx(dd, 0.5, True)  # use this for positive pulses
        # pulse_idx = get_pulse_idx(dd, -0.5, False) 
        
        if noise:
            # Fit noise away from the pulses
            pulse_idx = np.ceil(0.5 * (pulse_idx[:-1] + pulse_idx[1:])).astype(np.int64)

        for i, idx in enumerate(pulse_idx):
            if idx < 100000 or idx > zz.size-100000:
                continue

            # 20241205: use a much narrower search window (25 indices; 50 us)
            # to be consistent with DM analysis
            window, f, f_lp, amp = recon_pulse(idx, dtt, zz_bp, dd, False, None, 40000, 40000, 25, 80)

            if noise:
                # No search, just take th middle value
                # amps.append(np.abs(f_lp[np.ceil(f_lp.size/2).astype(np.int64)])/1e9)
                if np.isnan(amp):
                    pass
                else:
                    amps.append(amp)
            else:
                amps.append(amp)

    amps = np.asarray(amps)
    return amps

def get_all_unnormalized_amps(folder, datasets, pulseamps, noise=False):
    unnormalized_amps = []
    for i, dataset in enumerate(datasets):
        print(dataset)
        # combined_path = os.path.join(folder, dataset, '**/*.mat')
        # combined_path = os.path.join(folder, dataset, '*.hdf5')
        combined_path = os.path.join(folder, f'{dataset}*.hdf5')
        data_files = glob.glob(combined_path)

        unnormalized_amps.append(get_unnormalized_amps(data_files, noise))
        
    return unnormalized_amps

def fit_amps_gaus(normalized_amps, bins=None, noise=False, return_bins=False):
    hhs, bcs, gps = [], [], []
    bins_ret = []
    for amp in normalized_amps:
        if bins is None:
            bin = np.linspace(0, np.max(amp)*1.5, 50)
        else:
            bin = bins
        hh, be = np.histogram(amp, bins=bin)
        bc = 0.5 * (be[1:] + be[:-1])
        
        if noise:
            gp, gcov = curve_fit(gauss, bc, hh, p0=[np.max(hh), 0, np.std(np.abs(amp))], maxfev=100000)
        else:
            gp, gcov = curve_fit(gauss, bc, hh, p0=[np.max(hh), np.mean(np.abs(amp)), np.std(np.abs(amp))], maxfev=50000)

        hhs.append(hh)
        bcs.append(bc)
        gps.append(gp)
        bins_ret.append(bin)
    
    if return_bins:
        return hhs, bcs, gps, bins_ret
    else:
        return hhs, bcs, gps
    
def plot_gaus_fit(pulseamps, normalized_amps, hhs, bcs, gps, amp2kev=None, noise=False, title=None, fig=None, ax=None, colors=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    xx = np.linspace(0, np.max(np.asarray(bcs).flatten()), 1000)

    if colors is None:
        colors = plt.colormaps.get_cmap('tab20b').resampled(len(pulseamps)).colors
    for i, _ in enumerate(normalized_amps):
        color = colors[i]

        if amp2kev is not None:
            ax.errorbar(bcs[i]*amp2kev, hhs[i], yerr=np.sqrt(hhs[i]), fmt='.', markersize=10, color=color)
            gps_normalized = [gps[i][0], gps[i][1]*amp2kev, gps[i][2]*amp2kev]
            if noise:
                ax.plot(xx*amp2kev, gauss(xx*amp2kev, *gps_normalized), label=f'{pulseamps[i]} keV (noise), $\sigma$ = {gps_normalized[2]:.1f} keV', color=color)
            else:
                ax.plot(xx*amp2kev, gauss(xx*amp2kev, *gps_normalized), label=f'{pulseamps[i]} keV, $\sigma$ = {gps_normalized[2]:.1f} keV', color=color)

        else:
            ax.errorbar(bcs[i], hhs[i], yerr=np.sqrt(hhs[i]), fmt='o', color=color)
            if noise:
                ax.plot(xx, gauss(xx, *gps[i]), label=f'{pulseamps[i]} keV (noise), $\sigma$ = {gps[i][2]:.1f} keV', color=color)
            else:
                ax.plot(xx, gauss(xx, *gps[i]), label=f'{pulseamps[i]} keV, $\sigma$ = {gps[i][2]:.1f} keV', color=color)

    if title is not None:
        ax.set_title(title, fontsize=16)
    ax.set_xlabel('Reconstruced pulse (keV/c)', fontsize=14)
    ax.set_ylabel(f'Counts', fontsize=14)
    ax.legend(fontsize=14)
    
    return fig, ax

## Calibration
def get_area_driven_peak(ffd, ppd, passband=(88700, 89300), noise_floor=None, plot=False):
    """Integrate power in PSD over passband"""
    if noise_floor is None:
        noise_idx = np.logical_and(ffd > 100000, ffd < 105000)
        noise_floor = np.mean(ppd[noise_idx])
    
    all_idx = np.logical_and(ffd > passband[0], ffd < passband[1])
    area_all = np.trapz(ppd[all_idx]-noise_floor, ffd[all_idx]*2*np.pi)
    v2_drive = area_all / (2 * np.pi)

    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        ax.plot(ffd[all_idx], ppd[all_idx])
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Spectral density ($V^2 / Hz$)')
        ax.set_yscale('log')

    if plot:
        plt.show()

    return v2_drive

def get_c_mv(data_files_ordered, vp2p, omegad, passband, charge=3, n_chunk=10):
    m = 2000 * (83.5e-9**3) * (4 / 3) * np.pi  # sphere mass
    
    ffss, ppss = [], []
    for file in data_files_ordered:
        dtt, nn = load_timestreams(file, ['D'])
        zz = nn[0]

        size_per_chunk = int(zz.size / n_chunk)
        ffs, pps = [], []

        for i in range(n_chunk):
            ff, pp = get_psd(dt=dtt, zz=zz[i*size_per_chunk : (i+1)*size_per_chunk], nperseg=2**16)
            ffs.append(ff)
            pps.append(pp)

        ffss.append(ffs)
        ppss.append(pps)
        
    c_cals = []
    for i, vpp in enumerate(vp2p):
        fd0 = (vpp / 2) * 120 * charge * 1.6e-19

        c_cal = []
        for j, ff in enumerate(ffss[i]):
            pp = ppss[i][j]
            v2_drive = get_area_driven_peak(ff, pp, passband=passband, plot=False)

            idx_band = np.logical_and(ff > 40000, ff < 80000)
            omega0 = 2 * np.pi * ff[idx_band][np.argmax(pp[idx_band])]
            z2_drive = (fd0**2 / 2) / ((m * (omega0**2 - omegad**2))**2)

            c_cal.append(v2_drive / z2_drive)
        c_cals.append(c_cal)
    
    return np.sqrt(1 / np.asarray(c_cals))

## After processing
def load_histograms(data_dir, data_prefix, n_file):
    bc = None
    hhs, good_dets, noise_levels = [], [], []

    for i in range(n_file):
        file = os.path.join(data_dir, f'{data_prefix}{i}_processed.hdf5')
        f = h5py.File(file, 'r')

        if bc is None:
            bc = f['data_processed'].attrs['bin_center_kev']
        
        hhs.append(f['data_processed']['histogram'][:])
        good_dets.append(f['data_processed']['good_detection'][:])
        noise_levels.append(f['data_processed']['noise_level_kev'][:])

        f.close()
    
    hhs = np.asarray(hhs)
    noise_levels = np.asarray(noise_levels)
    good_dets = np.array(good_dets)

    return bc, hhs, good_dets, noise_levels

def check_excess_event(hhs, bc, thr=1500):
    no_excess_events = np.full(shape=hhs.shape[0:2], fill_value=True)
    for i, _hh_file in enumerate(hhs):
        for j, _hh in enumerate(_hh_file):
            if np.sum(_hh[bc > thr]) > 1 or np.sum(_hh) < 50:
                no_excess_events[i, j] = False

    return no_excess_events

def check_noise_level(noise_all, thr=400):
    return (noise_all < thr)

def load_data_hists(data_dir, data_prefix, n_file, excess_thr=1500, noise_thr=400):
    bc, hhs, good_dets, noise_levels = load_histograms(data_dir, data_prefix, n_file)
    good_noise_level = check_noise_level(noise_levels, noise_thr)
    no_excess_events = check_excess_event(hhs, bc, thr=excess_thr)

    hh_cut_det   = hhs[good_dets]
    hh_cut_noise = hhs[np.logical_and(good_dets, good_noise_level)]
    hh_cut_all   = hhs[np.logical_and(np.logical_and(good_dets, good_noise_level), no_excess_events)]

    return [bc, hhs, hh_cut_det, hh_cut_noise, hh_cut_all, good_dets, good_noise_level, no_excess_events]

def get_events_after_cut(hists, thr=4000):
    bc, hhs, good_dets, good_noise_level, no_excess_events = hists[0], hists[1], hists[5], hists[6], hists[7]

    events_after_cut = []
    for i, _hh_file in enumerate(hhs):
        for j, _hh in enumerate(_hh_file):
            if not good_dets[i, j]:  continue
            if not good_noise_level[i, j]: continue
            if not no_excess_events[i, j]: continue

            n_large_events = np.sum(_hh[bc > thr])
            if n_large_events > 0:
                events_after_cut.append((i, j))
    return events_after_cut

def get_summed_rates(data_dir, dataset):
    f = h5py.File(os.path.join(data_dir, f'{dataset}_summed_histograms.hdf5'), 'r')

    bc = f['summed_histograms'].attrs['bin_center_kev']
    scaling = f['summed_histograms'].attrs['scaling']

    hhs, rates_hz_kev, n_windows = [], [], []
    for hist in ['hh_all_sum', 'hh_cut_det_sum', 'hh_cut_noise_sum', 'hh_cut_all_sum']:
        hh = f['summed_histograms'][hist][:]
        n_window = f['summed_histograms'][hist].attrs['n_window']

        hhs.append(hh)
        rates_hz_kev.append(hh / (n_window * scaling))
        n_windows.append(n_window)

    f.close()

    return bc, hhs, rates_hz_kev, n_windows, scaling

def plot_hist(dataset, data_prefix, n_file, hists):
    data_dir = rf'/Volumes/LaCie/dm_data/{dataset}'
    file = os.path.join(data_dir, f'{data_prefix}0.hdf5')
    f = h5py.File(file, 'r')
    start_time = str(datetime.fromtimestamp(f['data'].attrs['timestamp']))
    f.close()

    hhs, hh_cut_det, hh_cut_noise, hh_cut_all = hists[1], hists[2], hists[3], hists[4]
    hh_all_sum = np.sum(np.sum(hhs, axis=0), axis=0)
    hh_cut_det_sum = np.sum(hh_cut_det, axis=0)
    hh_cut_noise_sum = np.sum(hh_cut_noise, axis=0)
    hh_cut_all_sum = np.sum(hh_cut_all, axis=0)

    n_search_per_win = (5000 - 150) / 25
    time_per_search = 2e-6 * 25
    scaling = n_search_per_win * time_per_search * (hists[0][1] - hists[0][0])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(hists[0], hh_all_sum / (hhs.shape[0] * hhs.shape[1] * scaling), '-', color=yale_colors[0], label='All', alpha=1)
    ax.plot(hists[0], hh_cut_det_sum / (hh_cut_det.shape[0] * scaling), '-', color=yale_colors[1], label='Detection quality cut', alpha=1)
    ax.plot(hists[0], hh_cut_noise_sum / (hh_cut_noise.shape[0] * scaling), '-', color=yale_colors[2], label='Detection quality + noise cut', alpha=1)
    ax.plot(hists[0], hh_cut_all_sum / (hh_cut_all.shape[0] * scaling), '-', color=yale_colors[3], label='Detection quality + noise + anti-coincidence cut', alpha=1)

    ax.legend(frameon=False, fontsize=12)

    ax.set_yscale('log')
    # ax.set_xscale('log')
    ax.set_xlim(0, 10000)
    ax.set_ylim(1e-7, 100)

    ax.set_xlabel('Recon. amplitude (keV)')
    ax.set_ylabel('Differential count (Hz/keV)')
    ax.set_title(f'Data ({n_file/60:.1f} hours), begins {start_time}')

    return fig, ax

def plot_hist_events(data_dir, data_prefix, file_idx, idx, window_length, bins, bc, c_mv, amp2kev, acce_micro=False, zoom=True):
    # data_dir = rf'/Volumes/LaCie/dm_data/{dataset}'

    file = os.path.join(data_dir, f'{data_prefix}{file_idx}.hdf5')
    print(file)

    f = h5py.File(file, "r")
    zz = f['data']['channel_d'][:] * f['data']['channel_d'].attrs['adc2mv'] / 1e3

    dtt = f['data'].attrs['delta_t']
    fs = int(np.ceil(1 / dtt))

    # If the sphere is charged and driven, apply a notch filter
    try:
        if f['data'].attrs['channel_e_mean_mv'] > 50:
            zz = notch_filtered(zz, fs, 93000, 100)
    except KeyError:
        pass

    zz_bp = bandpass_filtered(zz, fs, 30000, 80000)

    # Long window    
    zz_long = np.reshape(zz, (int(zz.size / window_length), window_length))
    zz_bp_long = np.reshape(zz_bp, (int(zz_bp.size / window_length), window_length))

    if acce_micro:
        ff = f['data']['channel_f'][:] * f['data']['channel_f'].attrs['adc2mv'] / 1e3
        gg = f['data']['channel_g'][:] * f['data']['channel_g'].attrs['adc2mv'] / 1e3

        ff_lp = lowpass_filtered(ff, fs, 10000)
        gg_lp = lowpass_filtered(gg, fs, 10000)

        ff_lp_long = np.reshape(ff_lp, (int(ff.size / window_length), window_length))
        gg_lp_long = np.reshape(gg_lp, (int(ff.size / window_length), window_length))

    idx_window = np.full(zz.size, True)
    idx_window[0:window_length*idx] = False
    idx_window[window_length*(idx+1):] = False

    amp, amp_lp, temp = recon_force(dtt, zz_bp_long[idx], c_mv)

    amp_search = np.abs(amp_lp[100:-50])
    amp_reshaped = np.reshape(amp_search, (int(amp_search.size/25), 25))
    amp_searched = np.max(amp_reshaped, axis=1)

    hh = np.histogram(amp_searched*amp2kev, bins=bins)[0]
    # hh = np.histogram(amp_lp[500:-500], bins=bins)[0]
    
    if not acce_micro:
        fig, ax = plt.subplots(1, 3, figsize=(10, 3))
    else:
        fig, ax = plt.subplots(1, 5, figsize=(15, 3))

    ax[0].errorbar(bc, hh, np.sqrt(hh), fmt='o', markersize=2)
    ax[0].set_yscale('log')
    # ax[0].set_xlim(0, 5000)
#     ax[0].set_ylim(1, 5e4)
    ax[0].set_xlabel('Reconstructed amp. (keV/c)', fontsize=12)
    ax[0].set_ylabel('Count', fontsize=12)

    ax[1].plot(dtt*1e6*np.arange(0, amp_lp.size), amp_lp*amp2kev/1000, color='grey')
    ax[1].set_ylim(-5, 5)
    # ax[1].set_ylim(-3, 3)
    ax[1].set_xlabel('Time ($\mu s$)', fontsize=12)
    ax[1].set_ylabel('Amp. (MeV/c)', fontsize=12)
    
    ax[2].plot(dtt*1e6*np.arange(0, zz_bp_long[idx].size), c_mv*zz_bp_long[idx]*1e9)
    ax[2].set_ylabel('Z homodyne (nm)', fontsize=12)
    ax[2].set_xlabel('Time ($\mu s$)', fontsize=12)

    if zoom:
        arg = np.argmax(np.abs(amp_lp[100:-50])) + 100
        lb = max(0, arg-100)
        ub = min(amp_lp.size, arg+100)

        ax[1].set_xlim(dtt*1e6*lb, dtt*1e6*ub)
        ax[2].set_xlim(dtt*1e6*lb, dtt*1e6*ub)

    if acce_micro:
        ax[3].plot(dtt*1e6*np.arange(0, ff_lp_long[idx].size), ff_lp_long[idx])
        ax[3].set_ylabel('Accelerometer (V)', fontsize=12)
        ax[3].set_xlabel('Time ($\mu s$)', fontsize=12)

        ax[4].plot(dtt*1e6*np.arange(0, gg_lp_long[idx].size), gg_lp_long[idx])
        ax[4].set_ylabel('Microphone (V)', fontsize=12)
        ax[4].set_xlabel('Time ($\mu s$)', fontsize=12)
    
    fig.suptitle(f'Event (file_{file_idx}, window_{idx})')
    fig.tight_layout()
    
    f.close()
    return amp_lp, hh, zz_long[idx], zz_bp_long[idx], fig, ax