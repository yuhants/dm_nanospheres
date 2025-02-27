import numpy as np
import os, glob

import h5py
import analysis_utils as utils

# data_dir = r'/Volumes/Expansion/pulse_calibration'
data_dir = r'/Volumes/LaCie/pulse_calibration'
out_dir = '/Users/yuhan/work/nanospheres/data/pulse_calibration_processed'

# sphere = 'sphere_20250103'
# # For Sphere 20250103, exclude two calibration datasets (20240109*)
# # because they have unusually low amplitudes; the charge could be off
# datasets = ['20250106_8e_alignment0_2e-8mbar_0', '20250106_8e_alignment0_2e-8mbar_1', '20250107_8e_alignment0_1e-8mbar_0', '20250107_8e_alignment0_1e-8mbar_1', 
#                '20250108_8e_alignment0_1e-8mbar_0', '20250108_8e_alignment0_1e-8mbar_1', '20250117_8e_alignment1_8e-9mbar_0', '20250117_8e_alignment1_8e-9mbar_1']
# dataset_prefixs = ['20250116_dg_8e_200ns_', '20250116_dg_8e_200ns_', '20250107_dg_8e_200ns_', '20250107_dg_8e_200ns_', 
#                    '20250108_dg_8e_200ns_', '20250108_dg_8e_200ns_', '20250117_dg_8e_200ns_', '20250117_dg_8e_200ns_']

# sphere = 'sphere_20241202'
# datasets = ['20241204_8e', '20241205_8e', '20241209_8e_alignment1_1', '20241213_8e_alignment2_4', '20241213_8e_alignment2_5']
# dataset_prefixs = ['20241204_dg_8e_200ns_', '20241205_dg_8e_200ns_', '20241209_dg_8e_200ns_', '20241213_dg_8e_200ns_', '20241213_dg_8e_200ns_']

# sphere = 'sphere_20241221'
# datasets = ['20241222_10e_alignment0_5e-8mbar', '20241222_10e_alignment0_5e-8mbar_1']
# dataset_prefixs = ['20241222_dg_10e_200ns_', '20241222_dg_10e_200ns_']
# voltages = [2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20]

sphere = 'sphere_20241226'
datasets = ['20241228_12e_alignment0_4e-8mbar_0']
dataset_prefixs = ['20241228_dg_12e_200ns_']
voltages = [2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20]

unnormalized_amps_all = [[] for i in range(len(voltages))]
unnormalized_amps_noise = []

for i, v in enumerate(voltages):
    for j, f in enumerate(datasets):
        folder = f'{data_dir}/{sphere}/{f}'
        dataset = dataset_prefixs[j] + f'{v}v'

        combined_path = os.path.join(folder, f'{dataset}*.hdf5')
        data_files = glob.glob(combined_path)
        print(combined_path)

        unnormalized_amps_all[i].append(utils.get_unnormalized_amps(data_files, 
                                                                    noise=False,
                                                                    positive_pulse=True,
                                                                    passband=(30000, 80000),
                                                                    analysis_window_length=50000,
                                                                    prepulse_window_length=50000,
                                                                    search_window_length=250,
                                                                    search_offset_length=20
                                                                    ))
        if v == 5:
            unnormalized_amps_noise.append(utils.get_unnormalized_amps(data_files, 
                                                                       noise=True,
                                                                       positive_pulse=True,
                                                                       passband=(30000, 80000),
                                                                       analysis_window_length=50000,
                                                                       prepulse_window_length=50000,
                                                                       search_window_length=250,
                                                                       search_offset_length=20
                                                                       ))

unnormalized_amps_all_flattened = []
for i in range(len(voltages)):
    unnormalized_amps_all_flattened.append(np.concatenate(unnormalized_amps_all[i]))
unnormalized_amps_noise_flattened = np.concatenate(unnormalized_amps_noise)

outfile_name = f'{sphere}_calibration_unnormalized_amps.h5py'
print(f'Writing file {outfile_name}')
with h5py.File(os.path.join(out_dir, outfile_name), 'w') as fout:
    g = fout.create_group('processed_amplitudes')
    for i, v in enumerate(voltages):
        g.create_dataset(f'unnormalized_amps_{str(v)}v', data=unnormalized_amps_all_flattened[i], dtype=np.float64)

    g.create_dataset(f'unnormalized_amps_noise_5v', data=unnormalized_amps_noise_flattened, dtype=np.float64)
    fout.close()