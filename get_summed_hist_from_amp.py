import numpy as np
import analysis_utils as utils
import h5py
import os

amp2kev = 7157.624533259538     # Sphere 20250103
bins = np.arange(0, 10000, 50)  # keV
bc = 0.5 * (bins[:-1] + bins[1:])

excess_thr = 1500
noise_thr = 400  # keV/c
chi2_thr_short = 150

window_length = 5000

def throw_away_doublecounts(amplitude_short, good_det_noise, idx_in_window, amp_thr_kev=1000, double_count_idx_thr=7):
    ret = np.copy(amplitude_short)

    # Pulse above threshold in good time windows
    large_pulses = np.logical_and(np.abs(ret) * amp2kev > amp_thr_kev, np.tile(good_det_noise, (194, 1)).T)

    # Identify pulses less than 7 indices apart
    small_diff_large_pulses = np.abs(np.diff(idx_in_window[large_pulses], append=5000)) < double_count_idx_thr

    large_pulses_pos = np.asarray(np.nonzero(large_pulses)).T
    amplitude_large_pulses = ret[large_pulses]

    for i, indices in enumerate(large_pulses_pos):
        # Keep the larger value of two nearby pulses above threshold
        # and set the other to nan
        if small_diff_large_pulses[i]:
            if np.isnan(amplitude_large_pulses[i]):
                continue

            # If the two amplitudes are from the same pulse, we expect them
            # to be different in sign
            if amplitude_large_pulses[i] * amplitude_large_pulses[i+1] > 1:
                continue

            if np.abs(amplitude_large_pulses[i]) < np.abs(amplitude_large_pulses[i+1]):
                ret[ *large_pulses_pos[i] ] = np.nan
            else:
                ret[ *large_pulses_pos[i+1] ] = np.nan
    
    return ret

def get_pulse_time(timestamp, idx_in_window, pulse_window_idx, window_length=5000, dtt=2e-6):
    pulse_idx_in_window = idx_in_window[pulse_window_idx]

    ret = (pulse_window_idx[0] * window_length + pulse_idx_in_window) * dtt
    return timestamp + ret

def get_summed_hist_from_amp(sphere, dataset, data_prefix, nfile):
    # data_dir = f'/Volumes/LaCie/dm_data_processed_amp_chisquare/{sphere}/{dataset}'
    raw_data_dir = f'/Volumes/LaCie/dm_data/{sphere}/{dataset}'
    data_dir = f'/Users/yuhan/work/nanospheres/data/dm_data_processed_amp_chisquare/{sphere}/{dataset}'

    outfile_name = f'{dataset}_summed_histograms.hdf5'
    outfile_pulse_waveform = f'{dataset}_pulse_waveforms.hdf5'

    hhs_0, hhs_1, hhs_2, hhs_3 = [np.zeros((nfile, bc.size), dtype=np.int64) for i in range(4)]
    file_idx, pulse_amp, pulse_time, pulse_waveform = [], [], [], []

    for i in range(nfile):
        if i % 100 == 0:
            print(i)

        file = os.path.join(data_dir, f'{data_prefix}{i}_processed.hdf5')
        f = h5py.File(file, 'r')
        
        timestamp = f['data_processed'].attrs['timestamp']

        amplitude = f['data_processed']['amplitude'][:]
        idx_in_window = f['data_processed']['idx_in_window'][:]

        good_detection = f['data_processed']['good_detection'][:]
        noise_level_amp = f['data_processed']['noise_level_amp'][:]
        chisquare_short = f['data_processed']['chisquare_short'][:]

        f.close()

        good_noise = (noise_level_amp * amp2kev) < noise_thr
        good_det_noise = np.logical_and(good_detection, good_noise)

        amplitude_short = np.copy(amplitude)
        bad_chi2_short = (chisquare_short > chi2_thr_short)
        amplitude_short[bad_chi2_short] = np.nan

        amplitude_short_corrected = throw_away_doublecounts(amplitude_short, good_det_noise, idx_in_window, 1000, 7)

        # Save pulse time and waveform
        amp_thr = 1000
        pulse_window_idx = np.nonzero(np.logical_and(np.abs(amplitude_short_corrected) * amp2kev > amp_thr, np.tile(good_det_noise, (194, 1)).T))
        if pulse_window_idx[0].size == 0:
            continue
        else:
            f = h5py.File(f'{raw_data_dir}/{data_prefix}{i}.hdf5', 'r')

            dtt = f['data'].attrs['delta_t']
            fs = int(np.ceil(1 / dtt))   # Sampling rate at Hz
            zz = f['data']['channel_d'][:] * f['data']['channel_d'].attrs['adc2mv'] / 1e3  # Signal in V

            zz_bp = utils.bandpass_filtered(zz, fs, 30000, 80000)
            zz_bp_shaped = np.reshape(zz_bp, (int(zz_bp.size / window_length), window_length))
            f.close()

            searched_idx_in_window = idx_in_window[pulse_window_idx]

            waveforms = np.empty((pulse_window_idx[0].size, 100))

            for idx, i_window in enumerate(pulse_window_idx[0]):
                _zz_bp = zz_bp_shaped[i_window]
                _amp, amp_lp, temp = utils.recon_force(dtt, _zz_bp, c_mv=None)

                idx_pulse = searched_idx_in_window[idx]
                waveforms[idx] = amp_lp[idx_pulse - 50 : idx_pulse + 50]

            file_idx.append(np.full(pulse_window_idx[0].size, i))
            pulse_amp.append(amplitude[pulse_window_idx])
            pulse_time.append(get_pulse_time(timestamp, idx_in_window, pulse_window_idx, 5000, 2e-6))
            pulse_waveform.append(waveforms)

        # Save histograms at each step per file
        hh_all, _ = np.histogram(amplitude * amp2kev, bins)
        hh_det, _ = np.histogram(amplitude[good_detection] * amp2kev, bins)
        hh_det_noise, _ = np.histogram(amplitude[good_det_noise] * amp2kev, bins)
        hh_det_noise_chi2_short, _ = np.histogram(amplitude_short_corrected[good_det_noise] * amp2kev, bins)

        hhs_0[i] = hh_all
        hhs_1[i] = hh_det
        hhs_2[i] = hh_det_noise
        hhs_3[i] = hh_det_noise_chi2_short

    with h5py.File(os.path.join(data_dir, outfile_name), 'w') as fout:
        print(f'Writing file {os.path.join(data_dir, outfile_name)}')

        g = fout.create_group('summed_histograms')
        g.attrs['bc_kev'] = bc

        g0 = g.create_dataset('hh_all_sum', data=hhs_0, dtype=np.int64)
        g1 = g.create_dataset('hh_det_sum', data=hhs_1, dtype=np.int64)
        g2 = g.create_dataset('hh_det_noise_sum', data=hhs_2, dtype=np.int64)
        g3 = g.create_dataset('hh_det_noise_chi2_short_sum', data=hhs_3, dtype=np.int64)

        fout.close()

    with h5py.File(os.path.join(data_dir, outfile_pulse_waveform), 'w') as fout:
        print(f'Writing file {os.path.join(data_dir, outfile_pulse_waveform)}')

        g = fout.create_group('pulses')

        g0 = g.create_dataset('file_idx', data=np.concatenate(file_idx, axis=0), dtype=np.int32)
        g1 = g.create_dataset('pulse_amp', data=np.concatenate(pulse_amp, axis=0), dtype=np.float64)
        g2 = g.create_dataset('pulse_time', data=np.concatenate(pulse_time, axis=0), dtype=np.float64)
        g3 = g.create_dataset('pulse_waveform', data=np.concatenate(pulse_waveform, axis=0), dtype=np.float64)

        fout.close()

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

    for idx, dataset in enumerate(datasets):
        get_summed_hist_from_amp(sphere, dataset, data_prefixs[idx], n_files[idx])