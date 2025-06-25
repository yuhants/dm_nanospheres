import numpy as np
import glob
from scipy.signal import decimate

from analysis_utils import load_timestreams
from pulse_recon import get_pulse_amp

def main():
    files = glob.glob(f"/Volumes/LaCie/20240723_minus_one_e_data/20240723_10v_2e-8mbar/*/*")
    recon_amp = []

    for file in files:
        print(f'Processing file {file}')
        dtt, tt, nn = load_timestreams(file, ['D'])

        amp = get_pulse_amp(dtt, nn[0])
        recon_amp.append(amp)

    amp_file = './recon_amp/20240723_1e/recon_amp_10v.npy'
    print(f'Saving file {amp_file}')
    np.save(amp_file, np.asarray(recon_amp)) 

if __name__ == '__main__':
    main()

