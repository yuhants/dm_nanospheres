import numpy as np

import os
import h5py

time_per_search = 50e-6  # sec

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

data_dir = r'E:\dm_data_hist_waveform'
out_dir = r'C:\Users\yuhan\dm_nanospheres\data_processed\sphere_data'

hhs_all_0, hhs_det_0, hhs_det_noise_0, hhs_det_noise_chi2_short_0 = [], [], [], []
hhs_all_1, hhs_det_1, hhs_det_noise_1, hhs_det_noise_chi2_short_1 = [], [], [], []

for i, dataset in enumerate(datasets_0):
    with h5py.File(f'{data_dir}/{sphere_0}/{dataset}_all_histograms.hdf5', 'r') as f:
        hhs_all_0.append(f['all_histograms']['hh_all'][:])
        hhs_det_0.append(f['all_histograms']['hh_det'][:])
        hhs_det_noise_0.append(f['all_histograms']['hh_det_noise'][:])
        hhs_det_noise_chi2_short_0.append(f['all_histograms']['hh_det_noise_chi2_short'][:])

        f.close()

for i, dataset in enumerate(datasets_1):
    with h5py.File(f'{data_dir}/{sphere_1}/{dataset}_all_histograms.hdf5', 'r') as f:
        hhs_all_1.append(f['all_histograms']['hh_all'][:])
        hhs_det_1.append(f['all_histograms']['hh_det'][:])
        hhs_det_noise_1.append(f['all_histograms']['hh_det_noise'][:])
        hhs_det_noise_chi2_short_1.append(f['all_histograms']['hh_det_noise_chi2_short'][:])

        f.close()

bins = np.arange(0, 10000, 50)  # keV
bc = 0.5 * (bins[:-1] + bins[1:])

# Write the summed histograms
hist_all_0 = np.sum(np.asarray([ np.sum(hhs_all_0[i], axis=0) for i in range(len(datasets_0))]), axis=0)
hist_det_noise_0 = np.sum(np.asarray([ np.sum(hhs_det_noise_0[i], axis=0) for i in range(len(datasets_0))]), axis=0)
hist_det_noise_chi2_0 = np.sum(np.asarray([ np.sum(hhs_det_noise_chi2_short_0[i], axis=0) for i in range(len(datasets_0))]), axis=0)

hist_all_1 = np.sum(np.asarray([ np.sum(hhs_all_1[i], axis=0) for i in range(len(datasets_1))]), axis=0)
hist_det_noise_1 = np.sum(np.asarray([ np.sum(hhs_det_noise_1[i], axis=0) for i in range(len(datasets_1))]), axis=0)
hist_det_noise_chi2_1 = np.sum(np.asarray([ np.sum(hhs_det_noise_chi2_short_1[i], axis=0) for i in range(len(datasets_1))]), axis=0)

outfile_name = 'sphere_20241202_recon_all.h5py'
with h5py.File(os.path.join(out_dir, outfile_name), 'w') as fout:
    g = fout.create_group('recon_data_all')
    g.attrs['livetime_sec'] = np.sum(hist_det_noise_0) * time_per_search

    d = g.create_dataset('bc', data=bc, dtype=np.float64)
    d.attrs['unit'] = 'keV'

    d = g.create_dataset('hist_all', data=(hist_all_0), dtype=np.int64)
    d = g.create_dataset('hist_det_noise', data=(hist_det_noise_0), dtype=np.int64)
    d = g.create_dataset('hist_det_noise_chi2', data=(hist_det_noise_chi2_0), dtype=np.int64)
    d.attrs['unit'] = 'count/50keV'

    fout.close()

outfile_name = 'sphere_20250103_recon_all.h5py'
with h5py.File(os.path.join(out_dir, outfile_name), 'w') as fout:
    g = fout.create_group('recon_data_all')
    g.attrs['livetime_sec'] = np.sum(hist_det_noise_1) * time_per_search

    d = g.create_dataset('bc', data=bc, dtype=np.float64)
    d.attrs['unit'] = 'keV'

    d = g.create_dataset('hist_all', data=(hist_all_1), dtype=np.int64)
    d = g.create_dataset('hist_det_noise', data=(hist_det_noise_1), dtype=np.int64)
    d = g.create_dataset('hist_det_noise_chi2', data=(hist_det_noise_chi2_1), dtype=np.int64)
    d.attrs['unit'] = 'count/50keV'

    fout.close()

# Write the unbinned amplitudes
all_amps_0, all_amps_1 = [], []
all_amps_time_0, all_amps_time_1 = [], []

for i, dataset in enumerate(datasets_0):
    with h5py.File(f'{data_dir}/{sphere_0}/{dataset}_pulse_waveforms.hdf5', 'r') as f:
        pulse_amp = f['pulses']['pulse_amp'][:]
        pulse_time = f['pulses']['pulse_time'][:]
        file_idx = f['pulses']['file_idx'][:]
        f.close()
    
        all_amps_0.append(pulse_amp)
        all_amps_time_0.append(pulse_time)

for i, dataset in enumerate(datasets_1):
    with h5py.File(f'{data_dir}/{sphere_1}/{dataset}_pulse_waveforms.hdf5', 'r') as f:
        pulse_amp = f['pulses']['pulse_amp'][:]
        pulse_time = f['pulses']['pulse_time'][:]
        file_idx = f['pulses']['file_idx'][:]
        f.close()
    
        all_amps_1.append(pulse_amp)
        all_amps_time_1.append(pulse_time)

all_amps_0 = np.concatenate(all_amps_0)
all_amps_time_0 = np.concatenate(all_amps_time_0)

all_amps_1 = np.concatenate(all_amps_1)
all_amps_time_1 = np.concatenate(all_amps_time_1)

outfile_name = 'sphere_20241202_unbinned_amps.h5py'
with h5py.File(os.path.join(out_dir, outfile_name), 'w') as fout:
    g = fout.create_group('unbinned_amps')
    g.attrs['livetime_sec'] = np.sum(hist_det_noise_0) * time_per_search
    d = g.create_dataset('amplitude', data=all_amps_0, dtype=np.float64)
    d = g.create_dataset('time', data=all_amps_time_0, dtype=np.float64)
    fout.close()

outfile_name = 'sphere_20250103_unbinned_amps.h5py'
with h5py.File(os.path.join(out_dir, outfile_name), 'w') as fout:
    g = fout.create_group('unbinned_amps')
    g.attrs['livetime_sec'] = np.sum(hist_det_noise_1) * time_per_search
    d = g.create_dataset('amplitude', data=all_amps_1, dtype=np.float64)
    d = g.create_dataset('time', data=all_amps_time_1, dtype=np.float64)
    fout.close()