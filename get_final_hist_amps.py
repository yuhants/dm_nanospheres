import numpy as np

import os
import h5py

time_per_search = 50e-6  # sec

# Timestamps used to perform background rate cut
# Note that these timestamps are approximate
# See `t_cut_start_actual` and `t_cut_end_actual` below
# Cut data after around 2024/12/17 14:00:00 (Nanosphere 0)
# Cut data before around 2025/01/05 08:00:00 (Nanosphere 1)
# Cut data after around 2025/01/24 22:00:00 (Nanosphere 1)
t0_cut_end   = 1734462000.0
t1_cut_start = 1736082000.0
t1_cut_end   = 1737774000.0

sphere_0 = 'sphere_20241202'
datasets_0 = ['20241202_8e-8mbar_long',
            '20241204_2e-8mbar_8e_aftercal_long',
            '20241205_2e-8mbar_0e_aftercal_long',
            '20241206_1e-8mbar_0e_aftercal_long',
            '20241207_1e-8mbar_1e_aftercal_long',
            '20241208_1e-8mbar_1e_aftercal_long',
            '20241210_1e-8mbar_8e_alignment1_long',
            '20241210_1e-8mbar_8e_alignment2_long_withdrive',
            '20241210_1e-8mbar_8e_alignment2_long_nodrive',
            '20241211_1e-8mbar_8e_alignment2_long_nodrive',
            '20241212_1e-8mbar_8e_alignment2_long_nodrive',
            '20241213_1e-8mbar_0e_alignment2_long',
            '20241214_1e-8mbar_0e_alignment2_long',
            '20241215_9e-9mbar_0e_alignment2_long',
            '20241216_5e-8mbar_0e_alignment2_long',
            '20241217_6e-8mbar_0e_alignment3_long'
            ]
data_prefixs_0 = [
                '20241202_abcd_',
                '20241204_abcd_',
                '20241205_d_',
                '20241206_d_',
                '20241207_d_',
                '20241208_d_',
                '20241210_d_',
                '20241210_d_',
                '20241210_d_',
                '20241211_d_',
                '20241212_d_',
                '20241213_d_',
                '20241214_d_',
                '20241215_d_',
                '20241216_d_',
                '20241217_d_',
                ]
nfiles_0 = [1440, 1440, 1440, 1440, 1440, 821, 640, 181, 1440, 1418, 917, 1169, 1565, 1440, 1164, 601]

sphere_1 = 'sphere_20250103'
datasets_1 = ['20250104_4e-8mbar_alignment0_long',
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
data_prefixs_1 = ['20250104_d_',
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
                '20250120_d_',
                '20250121_d_',
                '20250122_d_',
                '20250123_d_',
                '20250124_d_',
                '20250125_d_',
                ]
nfiles_1 = [1440, 900, 1440, 1440, 1440, 1440, 1440, 1440, 780, 1440, 1440, 1440, 1440, 1440, 1983, 1463, 1440, 1440, 1440, 1440, 1121]

# data_dir = r'E:\dm_data_hist_waveform'
# out_dir = r'C:\Users\yuhan\dm_nanospheres\data_processed\sphere_data'
data_dir = r'/Volumes/Expansion/dm_data_hist_waveform'
out_dir = r'/Users/yuhan/work/nanospheres/dm_nanospheres/data_processed/sphere_data'

def get_file_timestamps(sphere, datasets, data_prefixs, nfiles):
    timestamps_all = []
    for i, dataset in enumerate(datasets):
        print(dataset)

        data_dir = fr'/Volumes/Expansion/dm_data_processed_amp_chisquare/{sphere}/{dataset}'
        data_prefix, nfile = data_prefixs[i], nfiles[i]

        timestamps = np.empty(nfile)
        for j in range(nfile):
            file = os.path.join(data_dir, f'{data_prefix}{j}_processed.hdf5')

            f = h5py.File(file, 'r')        
            timestamps[j] = f['data_processed'].attrs['timestamp']
            f.close()
        timestamps_all.append(timestamps)

    np.savez(f'{out_dir}/{sphere}_file_timestamps.npz', *timestamps_all)

def get_bg_rate_cut_mask(sphere, datasets, data_prefixs, nfiles, t_cut_start, t_cut_end):
    try:
        timestamps_npz = np.load(f'{out_dir}/{sphere}_file_timestamps.npz')
    except:
        get_file_timestamps(sphere, datasets, data_prefixs, nfiles)
        timestamps_npz = np.load('{out_dir}/{sphere}_file_timestamps.npz')

    t_cut_start_actual, t_cut_end_actual = None, None

    mask_all = []
    for i, dataset in enumerate(datasets):
        timestamps = timestamps_npz[f'arr_{i}']

        nfile = nfiles[i]
        mask = np.full(nfile, True)

        if t_cut_start is not None:
            mask[timestamps < t_cut_start] = False

            if np.sum(mask) > 0 and np.sum(mask) < mask.size:
                idx_file = np.where(mask == False)[0][-1] + 1
                t_cut_start_actual = timestamps[idx_file]

        if t_cut_end is not None:
            mask[timestamps > t_cut_end] = False
            if np.sum(mask) > 0 and np.sum(mask) < mask.size:
                idx_file = np.where(mask == False)[0][0]
                t_cut_end_actual = timestamps[idx_file]

        mask_all.append(mask)

    return mask_all, t_cut_start_actual, t_cut_end_actual

def get_summed_histograms(sphere, datasets, data_prefixs, nfiles, t_cut_start=None, t_cut_end=None):
    hhs_all, hhs_det, hhs_det_noise, hhs_det_noise_chi2_short = [], [], [], []
    bg_rate_cut_mask, t_cut_start_actual, t_cut_end_actual = get_bg_rate_cut_mask(sphere, datasets, data_prefixs, nfiles, t_cut_start, t_cut_end)

    # Read in the reconstructed histograms
    for i, dataset in enumerate(datasets):
        with h5py.File(f'{data_dir}/{sphere}/{dataset}_all_histograms.hdf5', 'r') as f:
            hhs_all.append(f['all_histograms']['hh_all'][:])
            hhs_det.append(f['all_histograms']['hh_det'][:])
            hhs_det_noise.append(f['all_histograms']['hh_det_noise'][:])
            hhs_det_noise_chi2_short.append(f['all_histograms']['hh_det_noise_chi2_short'][:])

            f.close()

    # Apply the background rate cut
    # The bg rate cut is also applied before the chi2 cut
    # to get the correct livetime
    hhs_det_noise_bg, hhs_det_noise_chi2_bg = [], []
    for i, hh_det_noise in enumerate(hhs_det_noise):
        mask = bg_rate_cut_mask[i]

        _hh_det_noise_bg = hh_det_noise[mask]
        _hh_det_noise_chi2_bg = hhs_det_noise_chi2_short[i][mask]

        hhs_det_noise_bg.append(_hh_det_noise_bg)
        hhs_det_noise_chi2_bg.append(_hh_det_noise_chi2_bg)

    # Get the summed histograms
    hist_all = np.sum(np.asarray([ np.sum(hhs_all[i], axis=0) for i in range(len(datasets))]), axis=0)
    hist_det = np.sum(np.asarray([ np.sum(hhs_det[i], axis=0) for i in range(len(datasets))]), axis=0)
    hist_det_noise = np.sum(np.asarray([ np.sum(hhs_det_noise[i], axis=0) for i in range(len(datasets))]), axis=0)
    hist_det_noise_chi2 = np.sum(np.asarray([ np.sum(hhs_det_noise_chi2_short[i], axis=0) for i in range(len(datasets))]), axis=0)
    hist_det_noise_bg = np.sum(np.asarray([ np.sum(hhs_det_noise_bg[i], axis=0) for i in range(len(datasets))]), axis=0)
    hist_det_noise_chi2_bg = np.sum(np.asarray([ np.sum(hhs_det_noise_chi2_bg[i], axis=0) for i in range(len(datasets))]), axis=0)

    return hist_all, hist_det, hist_det_noise, hist_det_noise_chi2, hist_det_noise_bg, hist_det_noise_chi2_bg, t_cut_start_actual, t_cut_end_actual

def save_summed_hists(sphere, datasets, data_prefixs, nfiles, t_cut_start=None, t_cut_end=None):
    hist_all, hist_det, hist_det_noise, hist_det_noise_chi2, hist_det_noise_bg, hist_det_noise_chi2_bg, t_cut_start_actual, t_cut_end_actual = get_summed_histograms(sphere, datasets, data_prefixs, nfiles, t_cut_start, t_cut_end)

    bins = np.arange(0, 10000, 50)  # keV
    bc = 0.5 * (bins[:-1] + bins[1:])

    outfile_name = f'{sphere}_recon_all_bg.h5py'
    print(f'Saving file {outfile_name}')
    with h5py.File(os.path.join(out_dir, outfile_name), 'w') as fout:
        g = fout.create_group('recon_data_all')
        g.attrs['livetime_sec'] = np.sum(hist_det_noise_bg) * time_per_search

        d = g.create_dataset('bc', data=bc, dtype=np.float64)
        d.attrs['unit'] = 'keV'

        d = g.create_dataset('hist_all', data=(hist_all), dtype=np.int64)
        d = g.create_dataset('hist_det', data=(hist_det), dtype=np.int64)
        d = g.create_dataset('hist_det_noise', data=(hist_det_noise), dtype=np.int64)
        d = g.create_dataset('hist_det_noise_chi2', data=(hist_det_noise_chi2), dtype=np.int64)
        d = g.create_dataset('hist_det_noise_bg', data=(hist_det_noise_bg), dtype=np.int64)
        d = g.create_dataset('hist_det_noise_chi2_bg', data=(hist_det_noise_chi2_bg), dtype=np.int64)

        d.attrs['unit'] = 'count/50keV'
        fout.close()
    
    return np.sum(hist_det_noise_bg) * time_per_search, t_cut_start_actual, t_cut_end_actual

def save_unbinned_amps_time(sphere, datasets, t_cut_start=None, t_cut_end=None, livetime_sec=None):
    # Write the unbinned amplitudes
    all_amps, all_amps_time = [], []

    for i, dataset in enumerate(datasets):
        with h5py.File(f'{data_dir}/{sphere}/{dataset}_pulse_waveforms.hdf5', 'r') as f:
            pulse_amp = f['pulses']['pulse_amp'][:]
            pulse_time = f['pulses']['pulse_time'][:]
            f.close()
        
            all_amps.append(pulse_amp)
            all_amps_time.append(pulse_time)

    all_amps = np.concatenate(all_amps)
    all_amps_time = np.concatenate(all_amps_time)

    # Apply bg rate cut
    if t_cut_start is not None:
        all_amps = all_amps[all_amps_time > t_cut_start]
        all_amps_time = all_amps_time[all_amps_time > t_cut_start]
    if t_cut_end is not None:
        all_amps = all_amps[all_amps_time < t_cut_end]
        all_amps_time = all_amps_time[all_amps_time < t_cut_end]

    outfile_name = f'{sphere}_unbinned_amps_bg.h5py'
    print(f'Saving file {outfile_name}')
    with h5py.File(os.path.join(out_dir, outfile_name), 'w') as fout:
        g = fout.create_group('unbinned_amps')
        g.attrs['livetime_sec'] = livetime_sec
        d = g.create_dataset('amplitude', data=all_amps, dtype=np.float64)
        d = g.create_dataset('time', data=all_amps_time, dtype=np.float64)
        fout.close()

if __name__ == '__main__':
    livetime_sec_0, t_cut_start_actual_0, t_cut_end_actual_0 = save_summed_hists(sphere_0, datasets_0, data_prefixs_0, nfiles_0, t_cut_start=None, t_cut_end=t0_cut_end)
    livetime_sec_1, t_cut_start_actual_1, t_cut_end_actual_1 = save_summed_hists(sphere_1, datasets_1, data_prefixs_1, nfiles_1, t_cut_start=t1_cut_start, t_cut_end=t1_cut_end)
    save_unbinned_amps_time(sphere_0, datasets_0, t_cut_start=None, t_cut_end=t_cut_end_actual_0, livetime_sec=livetime_sec_0)
    save_unbinned_amps_time(sphere_1, datasets_1, t_cut_start=t_cut_start_actual_1, t_cut_end=t_cut_end_actual_1, livetime_sec=livetime_sec_1)

    print(t_cut_start_actual_0, t_cut_end_actual_0)
    print(t_cut_start_actual_1, t_cut_end_actual_1)