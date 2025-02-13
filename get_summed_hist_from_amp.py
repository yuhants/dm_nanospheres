import numpy as np
import analysis_utils as utils
import h5py
import os

amp2kev = 7157.624533259538   # Sphere 20250103
bins = np.arange(0, 10000, 50)  # keV
bc = 0.5 * (bins[:-1] + bins[1:])

def get_summed_hist_from_amp(sphere, dataset, data_prefix, nfile):
    data_dir = f'/Users/yuhan/work/nanospheres/data/dm_data_processed_chisquare/{sphere}/{dataset}'
    outfile_name = f'{dataset}_summed_histograms.hdf5'

    excess_thr = 1600
    noise_thr = 400  # keV/c
    chi2_thr_long  = 200
    chi2_thr_short = 150

    hhs_0, hhs_1, hhs_2, hhs_3, hhs_4 = [np.zeros(bc.size, dtype=np.int64) for i in range(5)]
    for i in range(nfile):
        file = os.path.join(data_dir, f'{data_prefix}{i}_processed.hdf5')
        f = h5py.File(file, 'r')
        
        amplitude = f['data_processed']['amplitude'][:]
        good_detection = f['data_processed']['good_detection'][:]
        noise_level_amp = f['data_processed']['noise_level_amp'][:]
        chisquare_long = f['data_processed']['chisquare_long'][:]
        chisquare_short = f['data_processed']['chisquare_short'][:]

        f.close()

        good_noise = (noise_level_amp * amp2kev) < noise_thr
        good_det_noise = np.logical_and(good_detection, good_noise)
    
        amplitude_long = np.copy(amplitude)
        bad_chi2_long = (chisquare_long > chi2_thr_long)
        amplitude_long[bad_chi2_long] = np.nan

        amplitude_short = np.copy(amplitude)
        bad_chi2_short = (chisquare_short > chi2_thr_short)
        amplitude_short[bad_chi2_short] = np.nan

        hh_all, _ = np.histogram(amplitude * amp2kev, bins)
        hh_det, _ = np.histogram(amplitude[good_detection] * amp2kev, bins)
        hh_det_noise, _ = np.histogram(amplitude[good_det_noise] * amp2kev, bins)

        hh_det_noise_chi2_long, _ = np.histogram(amplitude_long[good_det_noise] * amp2kev, bins)
        hh_det_noise_chi2_short, _ = np.histogram(amplitude_short[good_det_noise] * amp2kev, bins)

        hhs_0 += hh_all
        hhs_1 += hh_det
        hhs_2 += hh_det_noise
        hhs_3 += hh_det_noise_chi2_long
        hhs_4 += hh_det_noise_chi2_short

    with h5py.File(os.path.join(data_dir, outfile_name), 'w') as fout:
        print(f'Writing file {os.path.join(data_dir, outfile_name)}')

        g = fout.create_group('summed_histograms')
        g.attrs['bc_kev'] = bc

        g0 = g.create_dataset('hh_all_sum', data=hhs_0, dtype=np.int64)
        g1 = g.create_dataset('hh_det_sum', data=hhs_1, dtype=np.int64)
        g2 = g.create_dataset('hh_det_noise_sum', data=hhs_2, dtype=np.int64)
        g3 = g.create_dataset('hh_det_noise_chi2_long_sum', data=hhs_3, dtype=np.int64)
        g3 = g.create_dataset('hh_det_noise_chi2_short_sum', data=hhs_4, dtype=np.int64)

        fout.close()

if __name__ == '__main__':
    sphere = 'sphere_20250103'

    datasets = ['20250104_4e-8mbar_alignment0_long',
               ]

    data_prefixs = ['20250104_d_',
                    ]

    n_files = [720]

    for idx, dataset in enumerate(datasets):
        get_summed_hist_from_amp(sphere, dataset, data_prefixs[idx], n_files[idx])