import numpy as np

import os, glob
import h5py

import analysis_utils as utils

c_mv = 5.522e-08
# amp2kev = 7381.747090660193  # Sphere 20241202; averaged over 5 calibration datasets
# amp2kev = 6927.379154444802    # Sphere 20241219; calibration 20241220
amp2kev = 6844.611961407297    # Sphere 20241221; calibration 20241222

window_length = 5000  # 10 ms analysis window, assume dt=2 us
bins = np.arange(0, 10000, 50)  # keV
bc = 0.5 * (bins[:-1] + bins[1:])

idx_start = 0
nfile = 1331 - idx_start

sphere = 'sphere_20241221'
dataset = '20241221_3e-7mbar_16e_alignment0_long'
data_prefix = r'20241221_d_'
data_dir = rf'/Volumes/LaCie/dm_data/{sphere}/{dataset}'
out_dir = rf'/Users/yuhan/work/nanospheres/data/dm_data_processed/{sphere}/{dataset}'

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

def main():
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    for i in range(nfile):
        file = os.path.join(data_dir, f'{data_prefix}{i+idx_start}.hdf5')
        print(file)

        outfile_name = f'{data_prefix}{i+idx_start}_processed.hdf5'

        # if os.path.isfile(os.path.join(out_dir, outfile_name)):
        #     print(f'Skipping {outfile_name}')
        #     continue

        f = h5py.File(file, "r")

        dtt = f['data'].attrs['delta_t']
        fs = int(np.ceil(1 / dtt))

        zz = f['data']['channel_d'][:] * f['data']['channel_d'].attrs['adc2mv'] / 1e3
        
        # If the sphere is charged and driven, apply a notch filter
        try:
            if f['data'].attrs['channel_e_mean_mv'] > 35:
                zz = utils.notch_filtered(zz, fs, 93000, 100)
        except KeyError:
            pass

        zz_bp = utils.bandpass_filtered(zz, fs, 30000, 80000)

        zz_shaped = np.reshape(zz, (int(zz.size / window_length), window_length))
        zz_bp_shaped = np.reshape(zz_bp, (int(zz_bp.size / window_length), window_length))

        hh_all = np.empty(shape=(zz_bp_shaped.shape[0], bc.size), dtype=np.int16)
        noise_level_all = np.empty(shape=zz_bp_shaped.shape[0])
        good_detection = np.full(shape=zz_bp_shaped.shape[0], fill_value=True)

        for j, _zz_bp in enumerate(zz_bp_shaped):
            _amp, amp_lp, temp = utils.recon_force(dtt, _zz_bp, c_mv)

            # Divide the reconstructed amplitude in 25 (50 us) index chunks and search
            amp_search = np.abs(amp_lp[100:-50])
            amp_reshaped = np.reshape(amp_search, (int(amp_search.size/25), 25))
            amp_searched = np.max(amp_reshaped, axis=1)

            hh_all[j] = np.histogram(amp_searched*amp2kev, bins=bins)[0]
            noise_level_all[j] = np.std(amp_lp[100:-50]*amp2kev)
            
            # Identify period of poor detection quality
            if bad_detection_quality(zz_shaped[j], zz_bp_shaped[j]):
                good_detection[j] = False

        with h5py.File(os.path.join(out_dir, outfile_name), 'w') as fout:
            print(f'Writing file {outfile_name}')

            g = fout.create_group('data_processed')
            g.attrs['timestamp'] = f['data'].attrs['timestamp']
            g.attrs['bin_center_kev'] = bc
            g.attrs['amp2kev'] = amp2kev
            g.create_dataset('histogram', data=hh_all, dtype=np.int16)
            g.create_dataset('noise_level_kev', data=noise_level_all)
            g.create_dataset('good_detection', data=good_detection, dtype=np.bool_)

            fout.close()

        f.close()

if __name__ == '__main__':
    main()