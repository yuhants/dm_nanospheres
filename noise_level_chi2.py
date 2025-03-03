import numpy as np
import h5py
import os
import analysis_utils as utils

window_length = 5000

# amp2kev = 7187.368332843102     # Sphere 20241202
# sphere = 'sphere_20241202'
# datasets = [
#             '20241202_8e-8mbar_long',
#             '20241204_2e-8mbar_8e_aftercal_long',
#             '20241205_2e-8mbar_0e_aftercal_long',
#             '20241206_1e-8mbar_0e_aftercal_long',
#             '20241207_1e-8mbar_1e_aftercal_long',
#             '20241208_1e-8mbar_1e_aftercal_long',
#             '20241210_1e-8mbar_8e_alignment1_long',
#             '20241210_1e-8mbar_8e_alignment2_long_withdrive',
#             '20241210_1e-8mbar_8e_alignment2_long_nodrive',
#             '20241211_1e-8mbar_8e_alignment2_long_nodrive',
#             '20241212_1e-8mbar_8e_alignment2_long_nodrive',
#             '20241213_1e-8mbar_0e_alignment2_long',
#             '20241214_1e-8mbar_0e_alignment2_long',
#             '20241215_9e-9mbar_0e_alignment2_long',
#             '20241216_5e-8mbar_0e_alignment2_long',
#             '20241217_6e-8mbar_0e_alignment3_long',
#             ]
# data_prefixs = [
#                 '20241202_abcd_',
#                 '20241204_abcd_',
#                 '20241205_d_',
#                 '20241206_d_',
#                 '20241207_d_',
#                 '20241208_d_',
#                 '20241210_d_',
#                 '20241210_d_',
#                 '20241210_d_',
#                 '20241211_d_',
#                 '20241212_d_',
#                 '20241213_d_',
#                 '20241214_d_',
#                 '20241215_d_',
#                 '20241216_d_',
#                 '20241217_d_',
#                 ]
# n_files = [1440, 1440, 1440, 1440, 1440, 821, 640, 181, 1440, 1418, 917, 1169, 1565, 1440, 1164, 601]

amp2kev = 7157.624533259538     # Sphere 20250103
sphere = 'sphere_20250103'
datasets = [
            # '20250104_4e-8mbar_alignment0_long',
            # '20250105_2e-8mbar_alignment0_long',
            # '20250106_2e-8mbar_8e_alignment0_long',
            # '20250107_1e-8mbar_8e_alignment0_long',
            # '20250108_1e-8mbar_8e_alignment0_long',
            # '20250109_1e-8mbar_8e_alignment1_long',
            # '20250110_1e-8mbar_8e_alignment1_long',
            # '20250111_1e-8mbar_8e_alignment1_long',
            # '20250112_9e-9mbar_8e_alignment1_long',
            # '20250113_5e-8mbar_8e_alignment1_long',
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
data_prefixs = [
                # '20250104_d_',
                # '20250105_d_',
                # '20250106_d_',
                # '20250107_d_',
                # '20250108_d_',
                # '20250109_d_',
                # '20250110_d_',
                # '20250111_d_',
                # '20250112_d_',
                # '20250113_d_',
                '20250114_d_',
                '20250115_d_',
                '20250116_d_',
                '20250117_d_',
                '20250118_d_',
                '20250120_d_',
                '20250121_d_',
                '20250122_d_',
                '20250123_d_',
                '20250124_d_',
                '20250125_d_',
                ]
n_files = [1440, 1440, 1440, 1440, 1983, 1463, 1440, 1440, 1440, 1440, 1121]
# n_files = [1440, 900, 1440, 1440, 1440, 1440, 1440, 1440, 780, 1440, 1440, 1440, 1440, 1440, 1983, 
#            1463, 1440, 1440, 1440, 1440, 1121]

def get_noise_level_hist(sphere, dataset, data_prefix, n_files):
    data_dir = rf'E:\dm_data_processed_amp_chisquare\{sphere}\{dataset}'

    bins = np.arange(0, 1000, 10)
    hh_ret = np.zeros(bins.size-1, dtype=np.int64)

    print(dataset)
    for i in range(n_files):
        if i % 100 == 0:
            print(i)

        f = h5py.File(fr'{data_dir}\{data_prefix}{i}_processed.hdf5')
        good_detection = f['data_processed']['good_detection'][:]
        noise_level = f['data_processed']['noise_level_amp'][:]
        f.close()

        noise_level[np.logical_not(good_detection)] = np.nan
        hh, _ = np.histogram(np.ravel(noise_level) * amp2kev, bins)
        hh_ret += hh
    
    outfile = rf'{data_dir}\{dataset}_noise_level_all.npz'
    print(f'Saving file {outfile}')
    np.savez(outfile, bins=bins, hh_noise_kev=hh_ret)

def get_amp_chi2_hist(sphere, dataset, data_prefix, n_files, noise_thr=250):
    data_dir = rf'E:\dm_data_processed_amp_chisquare\{sphere}\{dataset}'

    bins_amp  = np.arange(0, 10000, 50)
    bins_chi2 = np.arange(0, 10000, 10)
    hh_ret = np.zeros((bins_amp.size-1, bins_chi2.size-1), dtype=np.float64)

    print(dataset)
    for i in range(n_files):
        if i % 100 == 0:
            print(i)

        f = h5py.File(fr'{data_dir}\{data_prefix}{i}_processed.hdf5')
        amplitude = f['data_processed']['amplitude'][:]
        good_detection = f['data_processed']['good_detection'][:]
        noise_level = f['data_processed']['noise_level_amp'][:]
        chisquare_short = f['data_processed']['chisquare_short'][:]
        f.close()

        good_noise = (noise_level * amp2kev) < noise_thr
        good_det_noise = np.logical_and(good_detection, good_noise)

        good_amp  = amplitude[good_det_noise]
        good_chi2 = chisquare_short[good_det_noise]

        hh, _, _ = np.histogram2d(x=np.abs(good_amp.flatten())*amp2kev,
                                  y=good_chi2.flatten(),
                                  bins=[bins_amp, bins_chi2])
        hh_ret += hh
    
    outfile = rf'{data_dir}\{dataset}_amp_chi2_all.npz'
    print(f'Saving file {outfile}')
    np.savez(outfile, bins_amp=bins_amp, bins_chi2=bins_chi2, hh_amp_chi2=hh_ret)

def get_pulse_time(timestamp, idx_in_window, pulse_window_idx, window_length=5000, dtt=2e-6):
    pulse_idx_in_window = idx_in_window[pulse_window_idx]

    ret = (pulse_window_idx[0] * window_length + pulse_idx_in_window) * dtt
    return timestamp + ret

def get_large_pulses(sphere, dataset, data_prefix, n_files, noise_thr=250, amp_thr_kev=2000):
    raw_data_dir = fr'E:\dm_data\{sphere}\{dataset}'
    data_dir = rf'E:\dm_data_processed_amp_chisquare\{sphere}\{dataset}'
    outfile = fr'{dataset}_large_pulse_waveforms.hdf5'

    file_idx, pulse_amp, pulse_chi2, pulse_time, pulse_waveform = [], [], [], [], []
    for i in range(n_files):
        if i % 100 == 0:
            print(i)

        f = h5py.File(fr'{data_dir}\{data_prefix}{i}_processed.hdf5')
        timestamp = f['data_processed'].attrs['timestamp']
        amplitude = f['data_processed']['amplitude'][:]
        idx_in_window = f['data_processed']['idx_in_window'][:]
        good_detection = f['data_processed']['good_detection'][:]
        noise_level = f['data_processed']['noise_level_amp'][:]
        chisquare_short = f['data_processed']['chisquare_short'][:]
        f.close()

        good_noise = (noise_level * amp2kev) < noise_thr
        good_det_noise = np.logical_and(good_detection, good_noise)

        pulse_window_idx = np.nonzero(np.logical_and(np.abs(amplitude) * amp2kev > amp_thr_kev, np.tile(good_det_noise, (194, 1)).T))
        if pulse_window_idx[0].size == 0:
            continue
        else:
            f = h5py.File(fr'{raw_data_dir}\{data_prefix}{i}.hdf5', 'r')

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
            pulse_chi2.append(chisquare_short[pulse_window_idx])
            pulse_time.append(get_pulse_time(timestamp, idx_in_window, pulse_window_idx, 5000, 2e-6))
            pulse_waveform.append(waveforms)

    with h5py.File(os.path.join(data_dir, outfile), 'w') as fout:
        print(f'Writing file {os.path.join(data_dir, outfile)}')

        g = fout.create_group('pulses')

        g0 = g.create_dataset('file_idx', data=np.concatenate(file_idx, axis=0), dtype=np.int32)
        g1 = g.create_dataset('pulse_amp', data=np.concatenate(pulse_amp, axis=0), dtype=np.float64)
        g2 = g.create_dataset('pulse_chi2', data=np.concatenate(pulse_chi2, axis=0), dtype=np.float64)
        g3 = g.create_dataset('pulse_time', data=np.concatenate(pulse_time, axis=0), dtype=np.float64)
        g4 = g.create_dataset('pulse_waveform', data=np.concatenate(pulse_waveform, axis=0), dtype=np.float64)

        fout.close()

if __name__ == '__main__':
    for j, dataset in enumerate(datasets):
        # get_noise_level_hist(sphere, dataset, data_prefixs[j], n_files[j])
        # get_amp_chi2_hist(sphere, dataset, data_prefixs[j], n_files[j], 250)
        get_large_pulses(sphere, dataset, data_prefixs[j], n_files[j], noise_thr=250, amp_thr_kev=2000)