import numpy as np
import scipy.io as sio
from scipy.signal import welch
from scipy.signal import butter, sosfilt
from scipy.optimize import curve_fit, minimize
from scipy.linalg import solve
from scipy.fft import rfft, irfft, rfftfreq

import matplotlib.pyplot as plt
from cycler import cycler

yale_colors = ['#00356b', '#286dc0', '#63aaff', '#4a4a4a']

c = 299792458    # m / s
SI2ev = (1 / 1.6e-19) * c

#### File processing
def load_timestreams(file, channels=['C']):
    data = sio.loadmat(file)
    length = data['Length'][0,0]
    delta_t = data['Tinterval'][0,0]

    tt = np.arange(length*delta_t, step=delta_t)
    timestreams = []
    for c in channels:
        timestreams.append(data[c][:,0])

    return delta_t, tt, np.asarray(timestreams)

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
def gauss(x, A, mu, sigma):
    return A*np.exp(-(x-mu)**2/(2*sigma**2))

from scipy.special import voigt_profile
def log_voigt(x, amp, x0, sigma, gamma):
    return np.log(amp * voigt_profile(x-x0, sigma, gamma))

def log_lorentzian(x, amp, x0, gamma):
    """A Lorentzian line shape"""
    return np.log(amp * gamma / ( ( x0**2 - x**2)**2 + gamma**2 * x**2 ))

def fit_peak(x, y, peak_func, p0=None):
    popt, pcov = curve_fit(peak_func, x, y, p0=p0, maxfev=50000)
    
    # Return central frequency and gamma
    return popt, x, peak_func(x, *popt)

def fit_z_peak(ff, pp, peak_func=log_voigt, passband=(60000, 70000), p0=[2e9, 62500*2*np.pi, 800, 50], plot=True):
    all_idx = np.logical_and(ff > passband[0], ff < passband[1])
    
    popt, omega_fit, p_fit = fit_peak(ff[all_idx]*2*np.pi, np.log(pp[all_idx]), peak_func, p0=p0)

    if plot:
        if len(p0) == 4:    # fitted by voigt
            label = (f'$f_0$ = {popt[1]/(2*np.pi*1000):.2f} kHz,\n'
                     f'$\gamma$ = {popt[3]/(2*np.pi):.1f} Hz, \n'
                     f'$\sigma$ = {popt[2]/(2*np.pi):.1f} Hz')
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
    
    params = {'figure.figsize': (7, 3),
              'axes.prop_cycle': default_cycler,
              'axes.titlesize': 14,
              'legend.fontsize': 12,
              'axes.labelsize': 14,
              'axes.titlesize': 14,
              'xtick.labelsize': 12,
              'ytick.labelsize': 12,
              'xtick.direction': 'in',
              'ytick.direction': 'in'
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
    window[:pulse_idx-length] = False
    window[pulse_idx-1:] = False
    
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
    chi = 1 / (omega0**2 - omega**2 - 1j*gamma*omega)
    return chi

def get_pulse_amp(dt, zz, omega0, gamma):
    zzk = rfft(zz)
    ff = rfftfreq(zz.size, dt)
    omega = ff * 2 * np.pi

    chi_omega = get_susceptibility(omega, omega0, gamma)
    filter_output = irfft(zzk / chi_omega)

    return filter_output

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
    # TODO
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

def recon_pulse(idx, dtt, tt, zz_bp, dd, plot=False, fname=None, 
                analysis_window_length=100000,
                prepulse_window_length=5000,
                search_window_length=20,
                pulse_length=20):

    if idx < prepulse_window_length:
        print('Skipping pulse too close to the beginning of the file')
        return None, None, None, np.nan

    fs = int(np.ceil(1 / dtt))

    window, pulse_idx_in_window = get_analysis_window(tt, idx, analysis_window_length)
    prepulse_window = get_prepulse_window(tt, idx, prepulse_window_length)

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
    amp = get_pulse_amp(dtt, zz_bp[window], omega0_fit, 1*2*np.pi)
    amp_lp = lowpass_filtered(amp, fs, 120000, 3)


    # cal_window = get_cal_window(amp, pulse_idx_in_window, cal_window_length=20000)
    # recon_amp = np.abs(np.min(amp_lp[search_window])/np.var(amp_lp[cal_window]))
    search_window = get_search_window(amp, pulse_idx_in_window, search_window_length, pulse_length)
    recon_amp = np.abs(np.min(amp_lp[search_window])/1e9)

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
        dtt, tt, nn = load_timestreams(file, ['D', 'G'])
        fs = int(np.ceil(1/dtt))

        zz, dd = nn[0], nn[1]
        zz_bp = bandpass_filtered(zz, fs, 40000, 130000)
        pulse_idx = get_pulse_idx(dd, -0.5, False)
        
        if noise:
            pulse_idx = np.ceil(0.5 * (pulse_idx[:-1] + pulse_idx[1:])).astype(np.int64)

        for i, idx in enumerate(pulse_idx):
            if idx < 100000 or idx > tt.size-100000:
                continue
            window, f, f_lp, amp = recon_pulse(idx, dtt, tt, zz_bp, dd, False, None,
                                               500000, 20000, 40, 30)
            
            if noise:
                amps.append(np.abs(f_lp[np.ceil(f_lp.size/2+20).astype(np.int64)])/1e9)
            else:
                amps.append(amp)

    amps = np.asarray(amps)
    return amps


