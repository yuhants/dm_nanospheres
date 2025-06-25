import numpy as np
import analysis_utils as utils
import h5py
import os

bins = np.arange(0, 10000, 50)  # keV
bc = 0.5 * (bins[:-1] + bins[1:])

def get_summed_hist(sphere, dataset):
    out_dir = r'C:\Users\yuhan\dm_nanospheres\data_processed\sphere_data\for_background'
    outfile_name = f'{dataset}_summed_histograms.hdf5'

    data_dir = fr'E:\dm_data_hist_waveform\{sphere}'
    histfile_name = f'{dataset}_all_histograms.hdf5'

    _file = h5py.File(os.path.join(data_dir, histfile_name), 'r')
    hh_all = _file['all_histograms']['hh_all'][:]
    hh_det = _file['all_histograms']['hh_det'][:] 
    hh_det_noise = _file['all_histograms']['hh_det_noise'][:]
    hh_det_noise_chi2_short = _file['all_histograms']['hh_det_noise_chi2_short'][:] 
    _file.close()

    hh_all_sum = np.sum(hh_all, axis=0)
    hh_det_sum = np.sum(hh_det, axis=0)
    hh_det_noise_sum = np.sum(hh_det_noise, axis=0)
    hh_det_noise_chi2_sum = np.sum(hh_det_noise_chi2_short, axis=0)

    with h5py.File(os.path.join(out_dir, outfile_name), 'w') as fout:
        print(f'Writing file {os.path.join(out_dir, outfile_name)}')

        g = fout.create_group('summed_histograms')
        g.attrs['bc'] = bc

        g0 = g.create_dataset('hh_all_sum', data=hh_all_sum, dtype=np.int64)
        g1 = g.create_dataset('hh_det_sum', data=hh_det_sum, dtype=np.int64)
        g2 = g.create_dataset('hh_det_noise_sum', data=hh_det_noise_sum, dtype=np.int64)
        g3 = g.create_dataset('hh_det_noise_chi2_sum', data=hh_det_noise_chi2_sum, dtype=np.int64)

        fout.close()

if __name__ == '__main__':
    sphere = 'sphere_20241202'
    datasets = [
                '20241202_8e-8mbar_long',
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
                '20241217_6e-8mbar_0e_alignment3_long',
                ]

    # sphere = 'sphere_20250103'
    # datasets = ['20250104_4e-8mbar_alignment0_long',
    #             '20250105_2e-8mbar_alignment0_long',
    #             '20250106_2e-8mbar_8e_alignment0_long',
    #             '20250107_1e-8mbar_8e_alignment0_long',
    #             '20250108_1e-8mbar_8e_alignment0_long',
    #             '20250109_1e-8mbar_8e_alignment1_long',
    #             '20250110_1e-8mbar_8e_alignment1_long',
    #             '20250111_1e-8mbar_8e_alignment1_long',
    #             '20250112_9e-9mbar_8e_alignment1_long',
    #             '20250113_5e-8mbar_8e_alignment1_long',
    #             '20250114_1e-8mbar_1e_alignment1_long',
    #             '20250115_8e-9mbar_0e_alignment1_long',
    #             '20250116_8e-9mbar_0e_alignment1_long_wrong_lo',
    #             '20250117_8e-9mbar_0e_alignment1_long',
    #             '20250118_8e-9mbar_1e_alignment1_long',
    #             '20250120_8e-9mbar_1e_alignment1_long_wbackscat',
    #             '20250121_8e-9mbar_1e_alignment1_long',
    #             '20250122_8e-9mbar_1e_alignment1_long',
    #             '20250123_7e-9mbar_1e_alignment1_long',
    #             '20250124_7e-9mbar_1e_alignment1_long',
    #             '20250125_7e-9mbar_1e_alignment1_long'    
    #         ]

    # sphere = 'sphere_20241221'
    # datasets = ['20241222_5e-8mbar_10e_alignment0_long']

    # sphere = 'sphere_20241226'
    # datasets = ['20241227_6e-8mbar_alignment0_long']

    for idx, dataset in enumerate(datasets):
        get_summed_hist(sphere, dataset)