import numpy as np
import scipy.io as sio
from scipy.signal import welch
from scipy.signal import butter, sosfilt

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

#### Filtering
def bandpass_filtered(data, fs, f_low=10000, f_high=100000, order=4): 
    sos_bp = butter(order, [f_low, f_high], 'bandpass', fs=fs, output='sos')
    filtered = sosfilt(sos_bp, data)
    return filtered

def lowpass_filtered(tod, fs, f_lp=50000, order=4):
    sos_lp = butter(order, f_lp, 'lp', fs=fs, output='sos')
    filtered = sosfilt(sos_lp, tod)
    return filtered

def highpass_filtered(tod, fs, f_hp=50000, order=4):
    sos_hp = butter(order, f_hp, 'hp', fs=fs, output='sos')
    filtered = sosfilt(sos_hp, tod)
    return filtered
