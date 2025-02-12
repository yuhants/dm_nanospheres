import numpy as np

import os
import h5py

from scipy.signal import decimate
import analysis_utils as utils

# amp2kev = 7381.747090660193  # Sphere 20241202; averaged over 5 calibration datasets
# amp2kev = 6927.379154444802  # Sphere 20241219; calibration 20241220
# amp2kev = 6844.611961407297  # Sphere 20241221; calibration 20241222
# amp2kev =  7034.799462287863  # Sphere 20241226; calibration 20241228

# amp2kev = 7396.062147743912  # Sphere 20250103; averaged over 8 calibration datasets
# sigma_p = 193.80085102332893  # Sphere 20250103; averaged over 8 calibration datasets

c_mv = 8.263269630174246e-08   # Sphere 20250103; calibration 20250114

window_length = 5000  # 10 ms analysis window, assume dt=2 us
search_window_length = 25  # 50 us search window

# If saving histograms
# bins = np.arange(0, 10000, 50)  # keV
# bc = 0.5 * (bins[:-1] + bins[1:])

## Calibration data used to correct for search bias
## Not using at the moment (20250209)
# cal_file = '/Users/yuhan/work/nanospheres/data/pulse_calibration_processed/sphere_20250103_calibration_all.h5py'
# with h5py.File(cal_file, 'r') as fout:
#     amp_true = fout['calibration_data_processed']['amp_true'][:]
#     amp_after_search = fout['calibration_data_processed']['amp_search'][:]
#     fout.close()
# search_bias_amp_true = amp_after_search - amp_true

def get_normalized_template(sphere, voltage, downsampled=False):
    pulse_shape_file = np.load(rf'/Users/yuhan/work/nanospheres/dm_nanospheres/data_processed/pulse_shape/{sphere}_pulse_shape_template_{voltage}v.npz')
    pulse_shapes = pulse_shape_file['pulse_shape']
    pulse_shape_template = np.mean(pulse_shapes, axis=0)

    normalized_template = pulse_shape_template / np.max(pulse_shape_template)
    # Take the central 0.2 ms around the peak
    ret = normalized_template[1000:2000]

    # Downsample to 500 kHz (so the 200 us template has 100 indices)
    if downsampled:
        ret_downsampled = decimate(ret, 10)
        return ret_downsampled / np.max(ret_downsampled)
    else:
        return ret
    
def get_idx_in_window(amp_searched_idx, search_length, lb):
    ret = np.empty_like(amp_searched_idx)

    for i, amp_idx in enumerate(amp_searched_idx):
        ret[i] = amp_idx + lb + search_length*i
    
    return ret

def calc_chisquares(amp_lp, indices_in_window, normalized_template, sigma_amp):
    ret = np.empty(indices_in_window.shape, np.float64)

    for i, idx in enumerate(indices_in_window):
        amp = amp_lp[idx]
        waveform = amp_lp[idx-50 : idx+50]

        # Amplitude can be negative so no need to adjust for polarity
        template_scaled = amp * normalized_template

        # Sigma should be in amplitude (not keV)
        ret[i] = np.sum( ((waveform - template_scaled)/sigma_amp)**2 )
        
    return ret

def bad_detection_quality(zz_windowed, zz_bp_windowed):
    # Z signal out of balance, meaning that homodyne losing lock
    if np.abs(np.mean(zz_windowed)) > 0.25:
        return True
    
    if np.max(np.abs(zz_windowed)) > 0.95:
        return True

    # Check the sum over 10 indices to see if there
    # is a consecutive period of very small signal after bandpass
    convolved = np.convolve(np.abs(zz_bp_windowed),np.ones(10, dtype=int), 'valid')
    if np.sum(convolved < 1e-3) > 0:
        return True
    


def process_dataset(sphere, dataset, data_prefix, nfile, idx_start):
    # data_dir = rf'/Volumes/LaCie/dm_data/{sphere}/{dataset}'
    data_dir = rf'/Volumes/Expansion/dm_data/{sphere}/{dataset}'

    # out_dir = rf'/Users/yuhan/work/nanospheres/data/dm_data_processed/{sphere}/{dataset}'
    out_dir = rf'/Volumes/Expansion/dm_data_processed_amp_chisquare/{sphere}/{dataset}'

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    normalized_template_downsampled = get_normalized_template(sphere, voltage=20, downsampled=True)

    for i in range(nfile):
        file = os.path.join(data_dir, f'{data_prefix}{i+idx_start}.hdf5')
        print(file)

        outfile_name = f'{data_prefix}{i+idx_start}_processed.hdf5'

        if os.path.isfile(os.path.join(out_dir, outfile_name)):
            print(f'Skipping {outfile_name}')
            continue

        f = h5py.File(file, "r")

        dtt = f['data'].attrs['delta_t']
        fs = int(np.ceil(1 / dtt))   # Sampling rate at Hz
        zz = f['data']['channel_d'][:] * f['data']['channel_d'].attrs['adc2mv'] / 1e3  # Signal in V
        
        # If the sphere is charged and driven, apply a notch filter at 93 kHz
        try:
            if f['data'].attrs['channel_e_mean_mv'] > 35:
                zz = utils.notch_filtered(zz, fs, 93000, 100)
        except KeyError:
            pass

        zz_bp = utils.bandpass_filtered(zz, fs, 30000, 80000)

        ## Reshape filtered z signal into 10 ms chunks and reconstruct
        zz_shaped = np.reshape(zz, (int(zz.size / window_length), window_length))
        zz_bp_shaped = np.reshape(zz_bp, (int(zz_bp.size / window_length), window_length))

        ## Not saving histograms but full amplitudes for now (20250210)
        # hh_all = np.empty(shape=(zz_bp_shaped.shape[0], bc.size), dtype=np.int16)
        # # hh_all_debiased = np.empty(shape=(zz_bp_shaped.shape[0], bc.size), dtype=np.int16)

        amp_all         = np.empty(shape=(zz_bp_shaped.shape[0], 194), dtype=np.float64)
        chisquare_all   = np.empty(shape=(zz_bp_shaped.shape[0], 194), dtype=np.float64)
        idx_in_window   = np.empty(shape=(zz_bp_shaped.shape[0], 194), dtype=np.int16)
        noise_level_all = np.empty(shape=zz_bp_shaped.shape[0])
        good_detection  = np.full(shape=zz_bp_shaped.shape[0], fill_value=True)

        for j, _zz_bp in enumerate(zz_bp_shaped):
            _amp, amp_lp, temp = utils.recon_force(dtt, _zz_bp, c_mv)

            # Divide the reconstructed amplitude in 25 (50 us) index chunks and search
            # for max absolute amplitude
            # Throw out the edge to avoid Fourier transform artifact
            # and leave enough indices to perform chisquare test
            lb, ub = 100, -50

            amp_search = np.abs(amp_lp[lb:ub])
            amp_reshaped = np.reshape(amp_search, (int(amp_search.size/25), 25))
            # amp_searched = np.max(amp_reshaped, axis=1)

            # Find the index of each searched pulse in the 10 ms window
            amp_searched_idx = np.argmax(amp_reshaped, axis=1)
            amp_searched_idx_in_window = get_idx_in_window(amp_searched_idx, 25, lb)

            ## Save the indices, amplitudes, and chisquares
            idx_in_window[j] = amp_searched_idx_in_window
            amp_all[j] = amp_lp[amp_searched_idx_in_window]
            chisquare_all[j] = calc_chisquares(amp_lp, amp_searched_idx_in_window, normalized_template_downsampled, sigma_amp=sigma_p/amp2kev)

            # Noise level in amplitude (not keV) in each window
            noise_level_all[j] = np.std(amp_lp[100:-50])
            
            # Identify period of poor detection quality
            if bad_detection_quality(zz_shaped[j], zz_bp_shaped[j]):
                good_detection[j] = False

            # Correct for search bias (not using at the moment, 20250209)
            # # bias_amp_searched = np.interp(amp_searched*amp2kev, amp_true, search_bias_amp_true, left=0)
            # bias_amp_searched = 178.1
            # amp_searched_debiased_kev = amp_searched*amp2kev - bias_amp_searched
            # amp_searched_debiased_kev[amp_searched_debiased_kev < 0] = 0
            # 
            # hh_all_debiased[j] = np.histogram(amp_searched_debiased_kev, bins=bins)[0]

            ## Save all amplitudes instead of histogram only (20250210)
            # hh_all[j] = np.histogram(amp_searched*amp2kev, bins=bins)[0]
            # if save_nosearch:
            #     hh_all_nosearch[j] = np.histogram(amp_search*amp2kev, bins=bins)[0]

        with h5py.File(os.path.join(out_dir, outfile_name), 'w') as fout:
            print(f'Writing file {outfile_name}')

            g = fout.create_group('data_processed')
            g.attrs['timestamp'] = f['data'].attrs['timestamp']
            g.attrs['amp2kev'] = amp2kev

            # g.attrs['bin_center_kev'] = bc
            # g.create_dataset('histogram', data=hh_all, dtype=np.int16)

            # g.create_dataset('histogram_debiased', data=hh_all_debiased, dtype=np.int16)
            # if save_nosearch:
            #     g.create_dataset('histogram_nosearch', data=hh_all_nosearch, dtype=np.int16)

            g.create_dataset('amplitude', data=amp_all, dtype=np.float64)
            g.create_dataset('chisquare', data=chisquare_all, dtype=np.float64)
            g.create_dataset('idx_in_window', data=idx_in_window, dtype=np.int16)
            g.create_dataset('noise_level_amp', data=noise_level_all, dtype=np.float64)
            g.create_dataset('good_detection', data=good_detection, dtype=np.bool_)

            fout.close()

        f.close()

if __name__ == '__main__':
    sphere = 'sphere_20250103'
    datasets = ['20250104_4e-8mbar_alignment0_long',
                '20250105_2e-8mbar_alignment0_long',
                '20250106_2e-8mbar_8e_alignment0_long',
                '20250107_1e-8mbar_8e_alignment0_long',
                '20250108_1e-8mbar_8e_alignment0_long',
                '20250109_1e-8mbar_8e_alignment1_long',
                '20250110_1e-8mbar_8e_alignment1_long',
                '20250111_1e-8mbar_8e_alignment1_long',
                '20250112_9e-9mbar_8e_alignment1_long',
                '20250113_5e-8mbar_8e_alignment1_long',
                '20250114_1e-8mbar_1e_alignment1_long',
                '20250115_8e-9mbar_0e_alignment1_long',
                '20250116_8e-9mbar_0e_alignment1_long_wrong_lo',
                '20250117_8e-9mbar_0e_alignment1_long',
                '20250118_8e-9mbar_1e_alignment1_long',
                '20250120_8e-9mbar_1e_alignment1_long_wbackscat',
                '20250121_8e-9mbar_1e_alignment1_long',
                '20250122_8e-9mbar_1e_alignment1_long',
                '20250123_7e-9mbar_1e_alignment1_long',
                '20250124_7e-9mbar_1e_alignment1_long',
                '20250125_7e-9mbar_1e_alignment1_long'    
            ]

    data_prefixs = ['20250104_d_',
                    '20250105_d_',
                    '20250106_d_',
                    '20250107_d_',
                    '20250108_d_',
                    '20250109_d_',
                    '20250110_d_',
                    '20250111_d_',
                    '20250112_d_',
                    '20250113_d_',
                    '20250114_d_',
                    '20250115_d_',
                    '20250116_d_',
                    '20250117_d_',
                    '20250118_d_',
                    '20250119_d_',
                    '20250120_d_',
                    '20250121_d_',
                    '20250122_d_',
                    '20250123_d_',
                    '20250124_d_',
                    '20250125_d_',
                    ]

    n_files = [1440, 900, 1440, 1440, 1440, 1440, 1440, 1440, 780, 1440, 1440, 1440, 1440, 1440, 1983, 1463, 1440, 1440, 1440, 1440, 1121]
    idx_start = 0

    for idx, dataset in enumerate(datasets):
        process_dataset(sphere, dataset, data_prefixs[idx], n_files[idx], idx_start)