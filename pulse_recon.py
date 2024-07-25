import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import flattop

from analysis_utils import get_psd

def log_peak_func(x, amp, omega_0, gamma):
    """A Lorentzian line shape"""
    return np.log(amp * gamma / ( ( omega_0**2 - x**2)**2 + gamma**2 * x**2 ))

def fit_peak(x, y, peak_func, p0=None):
    popt, pcov = curve_fit(peak_func, x, y, p0=p0, maxfev=10000)
    
    # Return central frequency and gamma
    return popt, x, peak_func(x, *popt)

def fit_z_peak(ff, pp, peak_func, passband=(60000, 70000), p0=[2e9, 62500*2*np.pi, 50], plot=True):
    all_idx = np.logical_and(ff > passband[0], ff < passband[1])
    
    popt, omega_fit, p_fit = fit_peak(ff[all_idx]*2*np.pi, np.log(pp[all_idx]), peak_func, p0=p0)

    if plot:
        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(ff[all_idx], pp[all_idx])
        ax.plot(omega_fit/(2*np.pi), np.exp(p_fit))
        ax.set_yscale('log')

    # amp, omega0, gamma
    return popt

def get_susceptibility(omega, omega0, gamma):
    chi = 1 / (omega0**2 - omega**2 - 1j*gamma*omega)
    return chi

def get_pulse_amp(dt, zz, passband_recon=(45000, 80000)):
    # First fit a Lorentzian to the peak
    # to crudely estimate frequency and damping
    # ff, pp = get_psd(dt, None, zz, nperseg=10000)
    # a0, omega0, gamma0 = fit_z_peak(ff, pp, log_peak_func, passband=(59000, 70000))
    
    # # Define lenght of search window
    # window_length_s = 0.2  # second
    # window_length = int(np.ceil(window_length_s / dt))

    # # Calculate FT in each window
    # fs = int(np.ceil(1 / dt))    
    # w = flattop(window_length, sym=True)
    # SFT = ShortTimeFFT(w, hop=20, fs=fs, fft_mode='twosided', mfft=None, scale_to='magnitude')
    # Szz = SFT.stft(zz)

    # ff = fftfreq(window_length, dt)
    # omega = ff * 2 * np.pi
    # chi_omega = get_susceptibility(omega, omega0, gamma0)

    # filter_output = np.empty(Szz.T.shape[0])
    # good_idx = np.logical_or(np.logical_and(ff > passband_recon[0], ff < passband_recon[1]),
    #                          np.logical_and(ff > -1*passband_recon[1], ff < -1*passband_recon[0]))

    # for i, sz in enumerate(Szz.T):    
    #     ffk = sz  / chi_omega
    #     # filter_output[i] = np.sum(np.abs(ffk[good_idx]))
    #     filter_output[i] = np.sum(np.real(ffk[good_idx]))

    omega0 = 63000 * 2 * np.pi
    gamma0 = 30

    zzk = fft(zz)
    ff = fftfreq(zz.size, dt)
    omega = ff * 2 * np.pi
    chi_omega = get_susceptibility(omega, omega0, gamma0)

    good_idx = np.logical_or(np.logical_and(ff > passband_recon[0], ff < passband_recon[1]),
                             np.logical_and(ff > -1*passband_recon[1], ff < -1*passband_recon[0]))
    zzk[np.logical_not(good_idx)] = 0
    filter_output = ifft(zzk / chi_omega)

    return filter_output
